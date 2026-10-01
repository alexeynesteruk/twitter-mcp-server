import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
import uvicorn
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from starlette.middleware.cors import CORSMiddleware
from twikit import Client, errors
from twikit.constants import DOMAIN
from twikit.tweet import Tweet, tweet_from_data
from twikit.x_client_transaction import ClientTransaction

from config import HOST, PORT

logger = logging.getLogger(__name__)

mcp = MCPServer(name="twitter-mcp-server")
# httpx logs every X request URL at INFO; keep the MCP client's stderr readable.
logging.getLogger("httpx").setLevel(logging.WARNING)

HTTP_MODE = bool(PORT)
MAX_COUNT = 50
AUTH_REQUIRED = "Authentication required: AUTH_REQUIRED"

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True)
# One tool per toggle pair (like/unlike, follow/unfollow, retweet/undo), so
# the hint has to cover the removing half: destructive, but idempotent.
TOGGLE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=True
)
PUBLISH = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
)


@dataclass(frozen=True)
class AuthContext:
    auth_token: str
    ct0: str


def parse_bearer(header: str) -> AuthContext:
    """Parse `Authorization: Bearer <auth_token>:<ct0>`."""
    parts = header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise ToolError(
            f"{AUTH_REQUIRED} (expected 'Authorization: Bearer <auth_token>:<ct0>')"
        )
    auth_token, sep, ct0 = parts[1].partition(":")
    if not sep or not auth_token or not ct0:
        raise ToolError(f"{AUTH_REQUIRED} (bearer token must be <auth_token>:<ct0>)")
    return AuthContext(auth_token, ct0)


def get_auth_context(
    headers: Mapping[str, str] | None, http_mode: bool | None = None
) -> AuthContext:
    """Resolve the X session cookies for one tool call.

    In HTTP mode (APP_PORT set) the caller must send its own cookies in the
    Authorization header and the server never falls back to its own env
    cookies, so anything that can reach the port cannot act as the server
    owner. The mode is decided by config, not by whether the transport
    happened to expose headers. Over stdio the cookies come from
    TWITTER_AUTH_TOKEN/TWITTER_CT0.
    """
    if http_mode is None:
        http_mode = HTTP_MODE
    if http_mode:
        header = (headers or {}).get("authorization")
        if not header:
            raise ToolError(AUTH_REQUIRED)
        return parse_bearer(header)
    auth_token = os.getenv("TWITTER_AUTH_TOKEN")
    ct0 = os.getenv("TWITTER_CT0")
    if not auth_token or not ct0:
        raise ToolError(f"{AUTH_REQUIRED} (set TWITTER_AUTH_TOKEN and TWITTER_CT0)")
    return AuthContext(auth_token, ct0)


def is_transaction_rejection(e: errors.NotFound) -> bool:
    # twikit formats these as: status: 404, message: "<response body>"
    message = str(e)
    return '"code":34' in message or message.endswith('message: ""')


class XClient(Client):
    """twikit Client with a serialized, retryable x.com handshake.

    twikit derives each request's X-Client-Transaction-Id from keys scraped
    off the x.com home page during a lazy handshake on the first request.
    Two problems with that, both seen live:
    - The handshake marks itself done before it finishes, so concurrent first
      requests race on a half-built ClientTransaction, and a handshake that
      fails halfway leaves the client broken for good.
    - Some handshakes yield keys that X rejects on stricter endpoints
      (settings.json, FavoriteTweet, CreateRetweet) with a 404: either code
      34 "page does not exist" or an empty body. Re-sending with the same
      keys keeps failing; a fresh handshake fixes it. The rejection comes
      back before the request is processed, so re-sending a write is safe.
    """

    RETRIES = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._handshake_lock = asyncio.Lock()

    async def handshake(self, stale: ClientTransaction | None = None) -> None:
        """Make sure a complete ClientTransaction is in place.

        With `stale`, replace it unless another task already has. The new
        transaction is only published once fully built, so twikit's own
        lazy-init check never sees a half-built one.
        """
        async with self._handshake_lock:
            current = self.client_transaction
            if current is not stale and current.home_page_response:
                return
            fresh = ClientTransaction()
            cookies = self.get_cookies().copy()
            # Same headers twikit's Client.request uses for its own handshake.
            await fresh.init(
                self.http,
                {
                    "Accept-Language": f"{self.language},{self.language.split('-')[0]};q=0.9",
                    "Cache-Control": "no-cache",
                    "Referer": f"https://{DOMAIN}",
                    "User-Agent": self._user_agent,
                },
            )
            self.set_cookies(cookies, clear_cookies=True)
            self.client_transaction = fresh

    async def request(self, method, url, *args, **kwargs):
        for attempt in range(self.RETRIES + 1):
            await self.handshake()
            transaction = self.client_transaction
            try:
                return await super().request(method, url, *args, **kwargs)
            except errors.NotFound as e:
                if not is_transaction_rejection(e) or attempt == self.RETRIES:
                    raise
                logger.info(
                    "X rejected the transaction id for %s (404); redoing the handshake",
                    urlparse(url).path,
                )
                await self.handshake(stale=transaction)


# One twikit Client per cookie pair, so the x.com transaction keys are fetched
# once instead of on every call.
_clients: dict[AuthContext, Client] = {}
_clients_lock = asyncio.Lock()
_MAX_CLIENTS = 32


async def get_client(ctx: Context | None) -> Client:
    auth = get_auth_context(ctx.headers if ctx is not None else None)
    client = _clients.get(auth)
    if client is not None:
        return client
    async with _clients_lock:
        client = _clients.get(auth)
        if client is not None:
            return client
        client = XClient("en-US")
        client.set_cookies({"auth_token": auth.auth_token, "ct0": auth.ct0})
        # Cache a client only once its handshake worked, so a failed one is
        # never reused and bad cookies (logged-out page, no ondemand.s) never
        # take a cache slot.
        try:
            await client.handshake()
        except BaseException:
            await client.http.aclose()
            raise
        if len(_clients) >= _MAX_CLIENTS:
            oldest = next(iter(_clients))
            await _clients.pop(oldest).http.aclose()
        _clients[auth] = client
        return client


@contextmanager
def twitter_errors() -> Iterator[None]:
    """Turn twikit exceptions into short, actionable tool errors."""
    try:
        yield
    except ToolError:
        raise
    except errors.Unauthorized:
        raise ToolError(
            f"{AUTH_REQUIRED} (X rejected the session cookies; refresh auth_token/ct0)"
        ) from None
    except errors.Forbidden as e:
        raise ToolError(f"{AUTH_REQUIRED} (X returned 403: {e})") from None
    except errors.TooManyRequests as e:
        reset = ""
        if e.rate_limit_reset:
            reset = f"; resets at {datetime.fromtimestamp(e.rate_limit_reset, UTC).isoformat()}"
        raise ToolError(f"Rate limited by X{reset}") from None
    except (errors.UserNotFound, errors.UserUnavailable) as e:
        raise ToolError(f"User not found or unavailable: {e}") from None
    except (errors.TweetNotAvailable, errors.NotFound) as e:
        raise ToolError(f"Not found: {e}") from None
    except errors.TwitterException as e:
        raise ToolError(f"X request failed: {type(e).__name__}: {e}") from None
    except httpx.HTTPError as e:
        raise ToolError(
            f"Network error talking to X: {type(e).__name__}: {e}"
        ) from None
    except (KeyError, IndexError, AttributeError) as e:
        # X reshapes its GraphQL responses every few months and twikit parses
        # them by fixed keys; say so instead of a bare "Error executing tool".
        logger.exception("Unexpected X response shape")
        raise ToolError(
            f"Unexpected X response shape ({type(e).__name__}: {e}); "
            "twikit likely needs re-patching"
        ) from None
    except Exception as e:
        # twikit raises a bare Exception when x.com serves a page without the
        # ondemand.s script: a logged-out page (bad cookies) or a format change.
        if "KEY_BYTE" not in str(e):
            raise
        raise ToolError(
            f"{AUTH_REQUIRED} (X client handshake failed: {e}). Usually invalid or "
            "expired auth_token/ct0; if the cookies are fresh, X changed its "
            "ondemand.s format and twikit needs re-patching"
        ) from None


@asynccontextmanager
async def x_client(ctx: Context | None) -> AsyncIterator[Client]:
    """Yield the caller's twikit Client; any failure in the block, including
    parsing twikit's lazily-read Tweet fields, becomes a ToolError."""
    with twitter_errors():
        yield await get_client(ctx)


def check_write(response: httpx.Response) -> None:
    """twikit only raises for a few GraphQL error codes; X often answers a
    rejected like/retweet with 200 and an errors array."""
    try:
        data = response.json()
    except ValueError:
        return
    if isinstance(data, dict) and data.get("errors"):
        first = data["errors"][0]
        message = first.get("message") if isinstance(first, dict) else first
        raise ToolError(f"X rejected the action: {message}")


def parse_count(count: int | str) -> int:
    try:
        count_int = int(count)
    except (TypeError, ValueError):
        raise ToolError("Invalid argument (count)") from None
    if count_int > MAX_COUNT:
        raise ToolError(f"Invalid argument (count): max value is {MAX_COUNT}")
    if count_int <= 0:
        raise ToolError("Invalid argument (count): must be at least 1")
    return count_int


def tweet_to_dict(tweet: Tweet) -> dict[str, Any]:
    return {
        "id": tweet.id,
        "in_reply_to": tweet.in_reply_to,
        "author_username": tweet.user.screen_name,
        "text": tweet.text,
        "lang": tweet.lang,
        "created_at": tweet.created_at,
        "view_count": tweet.view_count,
        "favorite_count": tweet.favorite_count,
        "reply_count": tweet.reply_count,
        "retweet_count": tweet.retweet_count,
    }


def tweets_json(tweets, count: int) -> str:
    # X treats count as a hint and often returns a few pages' worth.
    return json.dumps([tweet_to_dict(t) for t in list(tweets)[:count]])


@mcp.tool(description="Get recent tweets from a user", annotations=READ_ONLY)
async def get_tweets(username: str, count: int = 30, ctx: Context | None = None) -> str:
    """
    Args:
      username: Username of the user (without @)
      count: Number of tweets to retrieve (default: 30, max: 50)
    """
    count_int = parse_count(count)
    async with x_client(ctx) as client:
        user = await client.get_user_by_screen_name(username)
        tweets = await client.get_user_tweets(user.id, "Tweets", count=count_int)
        return tweets_json(tweets, count_int)


@mcp.tool(description="Get a Twitter user's profile information", annotations=READ_ONLY)
async def get_profile(username: str, ctx: Context | None = None) -> str:
    """
    Args:
      username: Username of the user (without @)
    """
    async with x_client(ctx) as client:
        user = await client.get_user_by_screen_name(username)
        return json.dumps(
            {
                "id": user.id,
                "name": user.name,
                "username": user.screen_name,
                "created_at": user.created_at,
                "profile_image_url": user.profile_image_url,
                "url": user.url,
                "location": user.location,
                "description": user.description,
                "description_urls": user.description_urls,
                "is_blue_verified": user.is_blue_verified,
                "verified": user.verified,
                "possibly_sensitive": user.possibly_sensitive,
                "can_dm": user.can_dm,
                "followers_count": user.followers_count,
                "fast_followers_count": user.fast_followers_count,
                "normal_followers_count": user.normal_followers_count,
                "following_count": user.following_count,
            }
        )


@mcp.tool(description="Search for tweets by hashtag or keyword", annotations=READ_ONLY)
async def search_tweets(
    query: str,
    mode: Literal["Latest", "Top"] = "Top",
    count: int = 30,
    ctx: Context | None = None,
) -> str:
    """
    Args:
      query: Search query (hashtag or keyword). For hashtags, include the # symbol
      mode: 'Latest' for most recent tweets or 'Top' for most relevant tweets (default: 'Top')
      count: Number of tweets to retrieve (default: 30, max: 50)
    """
    count_int = parse_count(count)
    async with x_client(ctx) as client:
        tweets = await client.search_tweet(query, mode, count=count_int)
        return tweets_json(tweets, count_int)


@mcp.tool(description="Like or unlike a tweet", annotations=TOGGLE)
async def like_tweet(
    tweet_id: str,
    action: Literal["like", "unlike"] = "like",
    ctx: Context | None = None,
) -> str:
    """
    Args:
      tweet_id: ID of the tweet to like/unlike
      action: Whether to "like" or "unlike" the tweet
    """
    async with x_client(ctx) as client:
        if action == "like":
            check_write(await client.favorite_tweet(tweet_id))
        else:
            check_write(await client.unfavorite_tweet(tweet_id))
        return json.dumps({"status": "success"})


@mcp.tool(description="Retweet or undo retweet of a tweet", annotations=TOGGLE)
async def retweet(
    tweet_id: str,
    action: Literal["retweet", "undo"] = "retweet",
    ctx: Context | None = None,
) -> str:
    """
    Args:
      tweet_id: ID of the tweet to retweet/undo retweet
      action: Whether to "retweet" or "undo" the retweet
    """
    async with x_client(ctx) as client:
        if action == "retweet":
            check_write(await client.retweet(tweet_id))
        else:
            check_write(await client.delete_retweet(tweet_id))
        return json.dumps({"status": "success"})


@mcp.tool(description="Post a new tweet, optionally as a reply", annotations=PUBLISH)
async def post_tweet(
    text: str,
    reply_to_tweet_id: str = "",
    ctx: Context | None = None,
) -> str:
    """
    Args:
      text: The text content of the tweet limited to 280 characters
      reply_to_tweet_id: Optional ID of the tweet to reply to
    """
    async with x_client(ctx) as client:
        tweet = await client.create_tweet(text=text, reply_to=reply_to_tweet_id or None)
        return json.dumps({"status": "success", "id": getattr(tweet, "id", None)})


@mcp.tool(description="Get current trending topics on Twitter", annotations=READ_ONLY)
async def get_trends(
    category: Literal[
        "trending", "for-you", "news", "sports", "entertainment"
    ] = "trending",
    count: int = 30,
    ctx: Context | None = None,
) -> str:
    """
    Args:
      category: 'trending' for overall trends, 'for-you', 'news', 'sports', 'entertainment' for more specific trends
      count: Number of trends to retrieve (default: 30, max: 50)
    """
    count_int = parse_count(count)
    async with x_client(ctx) as client:
        trends = await client.get_trends(category, count=count_int, retry=False)

        result = [
            {
                "name": trend.name,
                "tweet_count": trend.tweets_count,
                "grouped_trends": trend.grouped_trends,
                "domain_context": trend.domain_context,
            }
            for trend in list(trends)[:count_int]
        ]
        return json.dumps(result)


@mcp.tool(
    description="Get tweets from a user's home personalized timeline",
    annotations=READ_ONLY,
)
async def get_timeline(
    category: Literal["for-you", "following"] = "for-you",
    count: int = 40,
    ctx: Context | None = None,
) -> str:
    """
    Args:
      category: 'for-you' for personalized home for-you feed, 'following' for your following timeline
      count: Number of tweets to retrieve (default: 40, max: 50)
    """
    count_int = parse_count(count)
    async with x_client(ctx) as client:
        if category == "for-you":
            tweets = await client.get_timeline(count=count_int)
        else:
            tweets = await client.get_latest_timeline(count=count_int)
        return tweets_json(tweets, count_int)


@mcp.tool(description="Follow or unfollow a Twitter user", annotations=TOGGLE)
async def follow_user(
    username: str,
    action: Literal["follow", "unfollow"] = "follow",
    ctx: Context | None = None,
) -> str:
    """
    Args:
      username: Username of the user to follow/unfollow (without @)
      action: Whether to follow or unfollow the user
    """
    async with x_client(ctx) as client:
        user = await client.get_user_by_screen_name(username)
        if action == "follow":
            await client.follow_user(user.id)
        else:
            await client.unfollow_user(user.id)
        return json.dumps({"status": "success"})


def replies_from_tweet_detail(
    client: Client, tweet_id: str, response: dict
) -> list[Tweet]:
    """Pull the direct replies out of a TweetDetail GraphQL response.

    twikit's own get_tweet_by_id breaks on the current response shape (the
    bottom cursor moved from content.itemContent.value to content.value), so
    this reads only what get_replies needs. Each conversationthread-* module
    starts with a direct reply; the rest of the module is its sub-thread.
    """
    conversation = (response.get("data") or {}).get(
        "threaded_conversation_with_injections_v2"
    )
    not_found = ToolError(
        f"Not found: tweet {tweet_id} is deleted, protected or unavailable"
    )
    if not conversation:
        raise not_found
    focal_found = False
    replies = []
    for instruction in conversation.get("instructions", []):
        for entry in instruction.get("entries", []):
            entry_id = entry.get("entryId", "")
            if entry_id == f"tweet-{tweet_id}":
                # X still sends the focal entry for a missing tweet, just empty.
                focal_found = tweet_from_data(client, entry) is not None
                continue
            if not entry_id.startswith("conversationthread-"):
                continue
            items = entry.get("content", {}).get("items") or []
            reply = tweet_from_data(client, items[0]) if items else None
            if reply is not None:
                replies.append(reply)
    if not focal_found:
        raise not_found
    return replies


@mcp.tool(description="Read replies under post", annotations=READ_ONLY)
async def get_replies(
    tweet_id: str, count: int = 30, ctx: Context | None = None
) -> str:
    """
    Args:
        tweet_id: ID of the tweet to get replies of
        count: Maximum number of replies to return (default: 30, max: 50)
    """
    count_int = parse_count(count)
    async with x_client(ctx) as client:
        response, _ = await client.gql.tweet_detail(tweet_id, None)
        return tweets_json(
            replies_from_tweet_detail(client, tweet_id, response), count_int
        )


def main() -> None:
    if HTTP_MODE:
        app = mcp.streamable_http_app(host=HOST)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["mcp-session-id"],
        )
        uvicorn.run(app, host=HOST, port=int(PORT))
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()
