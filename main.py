import asyncio
import html
import json
import logging
import math
import operator
import os
import re
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import reduce
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import httpx
import uvicorn
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field
from starlette.middleware.cors import CORSMiddleware
from twikit import Client, errors
from twikit.constants import DOMAIN
from twikit.tweet import Tweet, tweet_from_data
from twikit.utils import Result, find_dict
from twikit.x_client_transaction import ClientTransaction
from twikit.x_client_transaction.cubic_curve import Cubic
from twikit.x_client_transaction.interpolate import interpolate
from twikit.x_client_transaction.rotation import convert_rotation_to_matrix
from twikit.x_client_transaction.utils import float_to_hex, is_odd

from config import HOST, PORT

logger = logging.getLogger(__name__)

mcp = MCPServer(name="twitter-mcp-server")
# httpx logs every X request URL at INFO; keep the MCP client's stderr readable.
logging.getLogger("httpx").setLevel(logging.WARNING)

HTTP_MODE = bool(PORT)
MAX_COUNT = 50
# X serves timelines in pages of ~10-20; fetch at most this many per call.
MAX_PAGES = 5
# X error bodies can be whole HTML pages; keep tool errors readable.
MAX_ERROR_TEXT = 300
AUTH_REQUIRED = "Authentication required: AUTH_REQUIRED"
# X error codes meaning the session itself is bad, as opposed to X refusing
# one action (161 follow limit, 162 blocked, ...).
AUTH_ERROR_CODES = frozenset({32, 89, 215, 353})
# Cookie values are hex; anything else is a paste error and must never be
# echoed back (httpx puts illegal header values into its error text).
COOKIE_VALUE = re.compile(r"[A-Za-z0-9_-]{1,512}")

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True)
# One tool per toggle pair (like/unlike, follow/unfollow, retweet/undo), so
# the hint has to cover the removing half: destructive, but idempotent.
TOGGLE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=True
)
PUBLISH = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
)

# Tool parameters. Descriptions live here: MCPServer uses `description=` as
# the whole tool description and never reads the functions' docstrings.
Username = Annotated[
    str, Field(description="X username, with or without @; a profile URL also works")
]
TweetId = Annotated[
    str,
    Field(description="Tweet ID, or a tweet URL like https://x.com/user/status/123"),
]


def count_param(default_what: str) -> Any:
    return Field(
        ge=1,
        le=MAX_COUNT,
        description=f"Number of {default_what} to return (1-{MAX_COUNT}); "
        "fetched across X's pages as needed",
    )


@dataclass(frozen=True)
class AuthContext:
    auth_token: str
    ct0: str


def checked_cookies(auth_token: str, ct0: str) -> AuthContext:
    if not COOKIE_VALUE.fullmatch(auth_token) or not COOKIE_VALUE.fullmatch(ct0):
        raise ToolError(f"{AUTH_REQUIRED} (malformed auth_token or ct0 value)")
    return AuthContext(auth_token, ct0)


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
    return checked_cookies(auth_token, ct0)


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
    return checked_cookies(auth_token, ct0)


def username_arg(value: str) -> str:
    """Accept "name", "@name" or a profile URL."""
    value = value.strip()
    if "/" in value:
        path = urlparse(value if "://" in value else f"https://{value}").path
        value = path.strip("/").split("/")[0]
    value = value.removeprefix("@")
    if not re.fullmatch(r"[A-Za-z0-9_]{1,50}", value):
        raise ToolError(f"Invalid argument (username): {value[:60]!r}")
    return value


def tweet_id_arg(value: str) -> str:
    """Accept a numeric tweet ID or a tweet URL."""
    value = value.strip()
    if value.isdigit():
        return value
    match = re.search(r"/status(?:es)?/(\d+)", value)
    if not match:
        raise ToolError(f"Invalid argument (tweet_id): {value[:60]!r}")
    return match.group(1)


def brief(e: BaseException) -> str:
    text = str(e)
    return text if len(text) <= MAX_ERROR_TEXT else text[:MAX_ERROR_TEXT] + "..."


def x_error_code(e: BaseException) -> int | None:
    match = re.search(r'"code"\s*:\s*(\d+)', str(e))
    return int(match.group(1)) if match else None


def is_transaction_rejection(e: errors.NotFound) -> bool:
    # twikit formats these as: status: 404, message: "<response body>"
    message = str(e)
    return '"code":34' in message or message.endswith('message: ""')


class HandshakeError(Exception):
    """The x.com handshake (transaction keys scraped off the home page) failed.

    twikit raises bare Exceptions here ("Couldn't get KEY_BYTE indices",
    "Couldn't get key from the page source", "invalid response"). x.com
    serves a page without the keys when the cookies are logged out, when it
    rate-limits or challenges the home page, or after a format change that
    twikit needs re-patching for.
    """


class TransactionRejected(errors.NotFound):
    """X kept answering 404 even after fresh handshakes."""


def js_round(x: float) -> int:
    """JavaScript's Math.round (halves round up), not Python's banker's round."""
    return math.floor(x + 0.5)


class XTransaction(ClientTransaction):
    """twikit's ClientTransaction with the upstream animation-key fixes.

    twikit vendored iSarabjitDhiman/XClientTransaction before two fixes to
    the animation key that every X-Client-Transaction-Id is built from
    (reference 1.0.3, MIT): the frame time is rounded to a multiple of 10
    the way the browser does, and interpolated colors are clamped to 255.
    Both depend on the random key bytes of each handshake, so without them
    about 1 handshake in 10 produced ids that X rejects with 404 code 34 on
    stricter endpoints. Measured 2026-10-02: on every such handshake the
    old id got 404 and the corrected one 200.
    """

    def get_animation_key(self, key_bytes, response):
        row_index = key_bytes[self.DEFAULT_ROW_INDEX] % 16
        frame_time = reduce(
            operator.mul, (key_bytes[i] % 16 for i in self.DEFAULT_KEY_BYTES_INDICES)
        )
        frame_time = js_round(frame_time / 10) * 10
        frame_row = self.get_2d_array(key_bytes, response)[row_index]
        return self.animate(frame_row, frame_time / 4096)

    def animate(self, frames, target_time):
        from_color = [float(v) for v in [*frames[:3], 1]]
        to_color = [float(v) for v in [*frames[3:6], 1]]
        to_rotation = [self.solve(float(frames[6]), 60.0, 360.0, True)]
        curves = [
            self.solve(float(v), is_odd(i), 1.0, False)
            for i, v in enumerate(frames[7:])
        ]
        value = Cubic(curves).get_value(target_time)
        color = [max(0, min(255, c)) for c in interpolate(from_color, to_color, value)]
        rotation = interpolate([0.0], to_rotation, value)
        parts = [format(round(c), "x") for c in color[:-1]]
        for m in convert_rotation_to_matrix(rotation[0]):
            hex_value = float_to_hex(abs(round(m, 2)))
            parts.append(
                f"0{hex_value}".lower()
                if hex_value.startswith(".")
                else hex_value or "0"
            )
        parts.extend(["0", "0"])
        return re.sub(r"[.-]", "", "".join(parts))


def clean_home_timeline(response: dict) -> dict:
    """Drop ads and flatten conversation modules in a home timeline page.

    twikit keeps every entry that has itemContent, which includes promoted
    tweets, and skips modules, which is how X groups a conversation in the
    home feeds. Rewrites the entries in place, before twikit parses them;
    cursor entries keep their positions (twikit reads the last one).
    """
    found = find_dict(response, "entries", find_one=True)
    if not found:
        return response
    entries = found[0]
    cleaned = []
    for entry in entries:
        content = entry.get("content", {})
        item_content = content.get("itemContent", {})
        if entry.get("entryId", "").startswith("promoted") or (
            "promotedMetadata" in item_content
        ):
            continue
        if "items" in content and not item_content:
            for item in content["items"]:
                inner = item.get("item", {}).get("itemContent", {})
                if inner.get("itemType") == "TimelineTweet" and (
                    "promotedMetadata" not in inner
                ):
                    cleaned.append(
                        {
                            "entryId": item.get("entryId", ""),
                            "content": {"itemContent": inner},
                        }
                    )
            continue
        cleaned.append(entry)
    entries[:] = cleaned
    return response


def _cleaned(fetch):
    async def fetch_clean(*args, **kwargs):
        response, raw = await fetch(*args, **kwargs)
        return clean_home_timeline(response), raw

    return fetch_clean


# Set while XClient checks the account state after a 429, see _get_user_state.
_checking_user_state: ContextVar[bool] = ContextVar(
    "checking_user_state", default=False
)


class XClient(Client):
    """twikit Client with a serialized, retryable x.com handshake.

    twikit derives each request's X-Client-Transaction-Id from keys scraped
    off the x.com home page during a lazy handshake on the first request.
    Problems with that, all seen live:
    - The handshake marks itself done before it finishes, so concurrent first
      requests race on a half-built ClientTransaction, and a handshake that
      fails halfway leaves the client broken for good.
    - twikit's transaction math was missing two upstream fixes (XTransaction).
    - X still occasionally rejects an id with 404 code 34 or an empty 404;
      re-sending with a fresh handshake fixes it. The rejection comes back
      before the request is processed, so re-sending a write is safe.
    """

    RETRIES = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._handshake_lock = asyncio.Lock()
        # Calls using this client right now, and whether the cache dropped it;
        # the last call out of an evicted client closes it.
        self.inflight = 0
        self.evicted = False
        self.gql.home_timeline = _cleaned(self.gql.home_timeline)
        self.gql.home_latest_timeline = _cleaned(self.gql.home_latest_timeline)

    async def handshake(self, stale: ClientTransaction | None = None) -> None:
        """Make sure a complete ClientTransaction is in place.

        With `stale`, replace it unless another task already has. The new
        transaction is only published once fully built, so twikit's own
        lazy-init check never sees a half-built one, and the common case
        needs no lock.
        """
        current = self.client_transaction
        if current is not stale and current.home_page_response:
            return
        async with self._handshake_lock:
            current = self.client_transaction
            if current is not stale and current.home_page_response:
                return
            fresh = XTransaction()
            cookies = self.get_cookies().copy()
            try:
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
            except (errors.TwitterException, httpx.HTTPError):
                raise
            except Exception as e:
                raise HandshakeError(str(e)) from e
            finally:
                self.set_cookies(cookies, clear_cookies=True)
            self.client_transaction = fresh

    async def request(self, method, url, *args, **kwargs):
        for attempt in range(self.RETRIES + 1):
            await self.handshake()
            transaction = self.client_transaction
            try:
                return await super().request(method, url, *args, **kwargs)
            except errors.NotFound as e:
                if not is_transaction_rejection(e):
                    raise
                if attempt == self.RETRIES:
                    raise TransactionRejected(
                        f"{urlparse(url).path}: {brief(e)}", headers=e.headers
                    ) from e
                logger.info(
                    "X rejected the transaction id for %s (404); redoing the handshake",
                    urlparse(url).path,
                )
                await self.handshake(stale=transaction)

    async def _get_user_state(self):
        # twikit's request() answers every 429 by fetching the account state,
        # which goes through request() again; if that is rate limited too it
        # recurses until RecursionError, hammering X the whole way down.
        if _checking_user_state.get():
            return "normal"
        token = _checking_user_state.set(True)
        try:
            return await super()._get_user_state()
        except errors.TooManyRequests:
            return "normal"
        finally:
            _checking_user_state.reset(token)


# One twikit Client per cookie pair, so the x.com transaction keys are fetched
# once instead of on every call. Least recently used first.
_clients: dict[AuthContext, XClient] = {}
# Handshakes in progress, one per cookie pair, shared by concurrent callers.
_pending: dict[AuthContext, asyncio.Future] = {}
_MAX_CLIENTS = 32


async def _retire(client: XClient) -> None:
    client.evicted = True
    if client.inflight == 0:
        await client.http.aclose()


async def _create_client(auth: AuthContext) -> XClient:
    client = XClient("en-US")
    client.set_cookies({"auth_token": auth.auth_token, "ct0": auth.ct0})
    # Cache a client only once its handshake worked, so a failed one is never
    # reused and bad cookies (logged-out page, no ondemand.s) never take a slot.
    try:
        await client.handshake()
    except BaseException:
        await client.http.aclose()
        raise
    _clients[auth] = client
    while len(_clients) > _MAX_CLIENTS:
        await _retire(_clients.pop(next(iter(_clients))))
    return client


async def get_client(ctx: Context | None) -> XClient:
    auth = get_auth_context(ctx.headers if ctx is not None else None)
    client = _clients.pop(auth, None)
    if client is not None:
        _clients[auth] = client  # now the most recently used
        return client
    # A per-pair handshake, so one caller's slow handshake (HTTP mode) never
    # holds up another caller's.
    task = _pending.get(auth)
    if task is None or (task.done() and (task.cancelled() or task.exception())):
        task = asyncio.ensure_future(_create_client(auth))
        _pending[auth] = task

        def done(finished: asyncio.Future) -> None:
            if _pending.get(auth) is finished:
                del _pending[auth]
            if not finished.cancelled():
                finished.exception()  # retrieved, even if every caller left

        task.add_done_callback(done)
    return await asyncio.shield(task)


@contextmanager
def twitter_errors() -> Iterator[None]:
    """Turn twikit exceptions into short, actionable tool errors."""
    try:
        yield
    except ToolError:
        raise
    except errors.AccountLocked as e:
        raise ToolError(
            f"The X account is locked; unlock it at x.com in a browser ({brief(e)})"
        ) from None
    except errors.AccountSuspended as e:
        raise ToolError(f"The X account is suspended ({brief(e)})") from None
    except errors.Unauthorized:
        raise ToolError(
            f"{AUTH_REQUIRED} (X rejected the session cookies; refresh auth_token/ct0)"
        ) from None
    except errors.Forbidden as e:
        code = x_error_code(e)
        if code is None or code in AUTH_ERROR_CODES:
            raise ToolError(f"{AUTH_REQUIRED} (X returned 403: {brief(e)})") from None
        raise ToolError(
            f"X refused the request (403, code {code}): {brief(e)}"
        ) from None
    except errors.TooManyRequests as e:
        reset = ""
        if e.rate_limit_reset:
            reset = f"; resets at {datetime.fromtimestamp(e.rate_limit_reset, UTC).isoformat()}"
        raise ToolError(f"Rate limited by X{reset}") from None
    except (errors.UserNotFound, errors.UserUnavailable) as e:
        raise ToolError(f"User not found or unavailable: {brief(e)}") from None
    except TransactionRejected as e:
        raise ToolError(
            f"X kept rejecting the request after {XClient.RETRIES} fresh handshakes "
            f"({brief(e)}). Most likely the twikit endpoint is stale (X rotated its "
            "query id) or X changed its transaction check: twikit needs re-patching"
        ) from None
    except (errors.TweetNotAvailable, errors.NotFound) as e:
        raise ToolError(f"Not found: {brief(e)}") from None
    except errors.TwitterException as e:
        raise ToolError(f"X request failed: {type(e).__name__}: {brief(e)}") from None
    except httpx.LocalProtocolError as e:
        # Its text can contain request header values, i.e. the cookies.
        raise ToolError(
            f"Could not send the request to X ({type(e).__name__})"
        ) from None
    except httpx.HTTPError as e:
        raise ToolError(
            f"Network error talking to X: {type(e).__name__}: {brief(e)}"
        ) from None
    except HandshakeError as e:
        raise ToolError(
            f"X client handshake failed (HANDSHAKE_FAILED: {brief(e)}). Most often "
            "the auth_token/ct0 cookies are expired or invalid (x.com served a "
            "logged-out page); it can also be X rate-limiting or challenging the "
            "home page. If it persists with fresh cookies, twikit needs re-patching"
        ) from None
    except (KeyError, IndexError, AttributeError) as e:
        # X reshapes its GraphQL responses every few months and twikit parses
        # them by fixed keys; say so instead of a bare "Error executing tool".
        logger.exception("Unexpected X response shape")
        raise ToolError(
            f"Unexpected X response shape ({type(e).__name__}: {brief(e)}); "
            "twikit likely needs re-patching"
        ) from None


@asynccontextmanager
async def x_client(ctx: Context | None) -> AsyncIterator[XClient]:
    """Yield the caller's twikit Client; any failure in the block, including
    parsing twikit's lazily-read Tweet fields, becomes a ToolError."""
    with twitter_errors():
        client = await get_client(ctx)
        client.inflight += 1
        try:
            yield client
        finally:
            client.inflight -= 1
            if client.evicted and client.inflight == 0:
                await client.http.aclose()


def check_write(
    response: httpx.Response, already_done: frozenset[int] = frozenset()
) -> bool:
    """twikit only raises for a few GraphQL error codes; X answers most
    rejected likes/retweets with 200 and an errors array. Returns True when
    X said the action was already in place (e.g. 139 already liked)."""
    try:
        data = response.json()
    except ValueError:
        return False
    if isinstance(data, dict) and data.get("errors"):
        first = data["errors"][0]
        if isinstance(first, dict) and first.get("code") in already_done:
            return True
        message = first.get("message") if isinstance(first, dict) else first
        raise ToolError(f"X rejected the action: {brief(Exception(message))}")
    return False


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


def iso_time(value: str | None) -> str | None:
    """X's "Thu Oct 01 17:31:51 +0000 2026" as ISO 8601."""
    try:
        return datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y").isoformat()
    except (TypeError, ValueError):
        return value


def as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def expand_urls(text: str | None, urls) -> str | None:
    for url in urls or []:
        if isinstance(url, dict) and url.get("url") and url.get("expanded_url"):
            text = text.replace(url["url"], url["expanded_url"]) if text else text
    return text


def tweet_text(tweet: Tweet) -> str:
    """Full text: long-form (note) tweets in full, t.co links expanded, the
    t.co links of attached media dropped (they are listed under "media"),
    and X's HTML escaping (&amp; &lt; &gt;) undone."""
    text = expand_urls(tweet.full_text, tweet.urls)
    for media in tweet.media:
        if media.url:
            text = text.replace(media.url, "")
    return html.unescape(text).strip()


def tweet_to_dict(tweet: Tweet) -> dict[str, Any]:
    # A retweet's own text is a truncated "RT @user: ..." and its counts are
    # zero; report the original post instead and say whose it is.
    original = tweet.retweeted_tweet
    source = original or tweet
    data = {
        "id": tweet.id,
        "url": f"https://x.com/{tweet.user.screen_name}/status/{tweet.id}",
        "in_reply_to": tweet.in_reply_to,
        "author_username": tweet.user.screen_name,
        "text": tweet_text(source),
        "lang": source.lang,
        "created_at": iso_time(tweet.created_at),
        "view_count": as_int(source.view_count),
        "favorite_count": source.favorite_count,
        "reply_count": source.reply_count,
        "retweet_count": source.retweet_count,
    }
    if source.media:
        data["media"] = [{"type": m.type, "url": m.expanded_url} for m in source.media]
    if original is not None:
        data["retweet_of"] = {
            "id": original.id,
            "author_username": original.user.screen_name,
            "created_at": iso_time(original.created_at),
        }
    quoted = source.quote
    if quoted is not None:
        data["quoted"] = {
            "id": quoted.id,
            "author_username": quoted.user.screen_name,
            "text": tweet_text(quoted),
        }
    return data


def tweets_json(tweets, count: int) -> str:
    # X treats count as a hint and often returns a few pages' worth.
    return json.dumps([tweet_to_dict(t) for t in list(tweets)[:count]])


async def collect(result: Result | list, count: int) -> list:
    """Follow X's page cursors until `count` items (X serves ~10-20 a page)."""
    items = list(result)
    seen = {getattr(item, "id", None) for item in items}
    for _ in range(MAX_PAGES - 1):
        if len(items) >= count or not getattr(result, "next_cursor", None):
            break
        try:
            result = await result.next()
        except (IndexError, KeyError):
            # twikit indexes the last entries of a page; past the end of a
            # timeline X sends none.
            break
        new = [item for item in result if getattr(item, "id", None) not in seen]
        if not new:
            break
        seen.update(item.id for item in new)
        items.extend(new)
    return items[:count]


def with_thread_follow_ups(tweets: Iterable[Tweet], author: str) -> list[Tweet]:
    """twikit returns the first tweet of each of the author's threads and
    tucks the follow-ups into .replies, so the newest posts of a thread
    would be missing. Put the author's follow-ups back, newest first (tweet
    ids grow with time)."""
    out, seen = [], set()
    for tweet in tweets:
        follow_ups = tweet.replies if isinstance(tweet.replies, Iterable) else []
        for item in [tweet, *follow_ups]:
            if item.id in seen:
                continue
            if item is not tweet and item.user.screen_name.lower() != author.lower():
                continue
            seen.add(item.id)
            out.append(item)
    return sorted(out, key=lambda t: as_int(t.id) or 0, reverse=True)


@mcp.tool(
    description="Recent posts by a user, newest first: their tweets, retweets (the "
    "original post's text, with retweet_of), and the follow-ups of their own threads. "
    "Each tweet has id, url, text (full long-form text, links expanded), created_at "
    "(ISO 8601), counts, and media/quoted when present.",
    annotations=READ_ONLY,
    structured_output=False,
)
async def get_tweets(
    username: Username,
    count: Annotated[int, count_param("tweets")] = 30,
    ctx: Context | None = None,
) -> str:
    count_int = parse_count(count)
    username = username_arg(username)
    async with x_client(ctx) as client:
        user = await client.get_user_by_screen_name(username)
        if user.protected and user.following is not True:
            raise ToolError(f"@{user.screen_name} is protected and not followed")
        if not user.statuses_count:
            return "[]"
        tweets = await client.get_user_tweets(user.id, "Tweets", count=count_int)
        tweets = await collect(tweets, count_int)
        return tweets_json(with_thread_follow_ups(tweets, user.screen_name), count_int)


@mcp.tool(
    description="A user's profile: name, bio (links expanded), counts, and whether "
    "the logged-in account follows them (following) or is followed by them.",
    annotations=READ_ONLY,
    structured_output=False,
)
async def get_profile(username: Username, ctx: Context | None = None) -> str:
    username = username_arg(username)
    async with x_client(ctx) as client:
        user = await client.get_user_by_screen_name(username)
        return json.dumps(
            {
                "id": user.id,
                "name": user.name,
                "username": user.screen_name,
                "created_at": iso_time(user.created_at),
                "profile_image_url": user.profile_image_url,
                "url": expand_urls(user.url, user.urls),
                "location": user.location,
                "description": expand_urls(user.description, user.description_urls),
                "description_urls": user.description_urls,
                "is_blue_verified": user.is_blue_verified,
                "verified": user.verified,
                "possibly_sensitive": user.possibly_sensitive,
                "can_dm": user.can_dm,
                "followers_count": user.followers_count,
                "fast_followers_count": user.fast_followers_count,
                "normal_followers_count": user.normal_followers_count,
                "following_count": user.following_count,
                "statuses_count": user.statuses_count,
                "protected": user.protected,
                "following": user.following,
                "followed_by": user.followed_by,
            }
        )


@mcp.tool(
    description="Search tweets by keyword, hashtag or X search operators "
    "(from:user, since:2026-10-01, filter:links, ...).",
    annotations=READ_ONLY,
    structured_output=False,
)
async def search_tweets(
    query: Annotated[str, Field(description="Search query; include # for hashtags")],
    mode: Annotated[
        Literal["Latest", "Top"],
        Field(description="'Latest' for newest first, 'Top' for most relevant"),
    ] = "Top",
    count: Annotated[int, count_param("tweets")] = 30,
    ctx: Context | None = None,
) -> str:
    count_int = parse_count(count)
    async with x_client(ctx) as client:
        tweets = await client.search_tweet(query, mode, count=count_int)
        return tweets_json(await collect(tweets, count_int), count_int)


@mcp.tool(
    description="Like or unlike a tweet as the logged-in account",
    annotations=TOGGLE,
    structured_output=False,
)
async def like_tweet(
    tweet_id: TweetId,
    action: Annotated[
        Literal["like", "unlike"], Field(description="like or unlike")
    ] = "like",
    ctx: Context | None = None,
) -> str:
    tweet_id = tweet_id_arg(tweet_id)
    async with x_client(ctx) as client:
        if action == "like":
            already = check_write(
                await client.favorite_tweet(tweet_id), already_done=frozenset({139})
            )
        else:
            already = check_write(await client.unfavorite_tweet(tweet_id))
        return json.dumps({"status": "success", "already_done": already})


@mcp.tool(
    description="Retweet or undo a retweet as the logged-in account",
    annotations=TOGGLE,
    structured_output=False,
)
async def retweet(
    tweet_id: TweetId,
    action: Annotated[
        Literal["retweet", "undo"], Field(description="retweet, or undo a retweet")
    ] = "retweet",
    ctx: Context | None = None,
) -> str:
    tweet_id = tweet_id_arg(tweet_id)
    async with x_client(ctx) as client:
        if action == "retweet":
            already = check_write(
                await client.retweet(tweet_id), already_done=frozenset({327})
            )
        else:
            already = check_write(await client.delete_retweet(tweet_id))
        return json.dumps({"status": "success", "already_done": already})


@mcp.tool(
    description="Post a tweet as the logged-in account, optionally as a reply. "
    "Returns the new tweet's id.",
    annotations=PUBLISH,
    structured_output=False,
)
async def post_tweet(
    text: Annotated[
        str,
        Field(
            min_length=1,
            description="Tweet text, up to 280 characters as X counts them "
            "(a link counts as 23, an emoji or CJK character as 2)",
        ),
    ],
    reply_to_tweet_id: Annotated[
        str, Field(description="Optional tweet ID or URL to reply to")
    ] = "",
    ctx: Context | None = None,
) -> str:
    reply_to = tweet_id_arg(reply_to_tweet_id) if reply_to_tweet_id.strip() else None
    async with x_client(ctx) as client:
        tweet = await client.create_tweet(text=text, reply_to=reply_to)
        return json.dumps({"status": "success", "id": getattr(tweet, "id", None)})


@mcp.tool(
    description="Current trending topics",
    annotations=READ_ONLY,
    structured_output=False,
)
async def get_trends(
    category: Annotated[
        Literal["trending", "for-you", "news", "sports", "entertainment"],
        Field(description="Which trends tab to read"),
    ] = "trending",
    count: Annotated[int, count_param("trends")] = 30,
    ctx: Context | None = None,
) -> str:
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
    description="Tweets from the logged-in account's own home timeline: 'for-you' "
    "(algorithmic) or 'following' (accounts it follows, newest first). Ads removed.",
    annotations=READ_ONLY,
    structured_output=False,
)
async def get_timeline(
    category: Annotated[
        Literal["for-you", "following"], Field(description="Which home timeline")
    ] = "for-you",
    count: Annotated[int, count_param("tweets")] = 40,
    ctx: Context | None = None,
) -> str:
    count_int = parse_count(count)
    async with x_client(ctx) as client:
        if category == "for-you":
            tweets = await client.get_timeline(count=count_int)
        else:
            tweets = await client.get_latest_timeline(count=count_int)
        return tweets_json(await collect(tweets, count_int), count_int)


@mcp.tool(
    description="Follow or unfollow a user as the logged-in account",
    annotations=TOGGLE,
    structured_output=False,
)
async def follow_user(
    username: Username,
    action: Annotated[
        Literal["follow", "unfollow"], Field(description="follow or unfollow")
    ] = "follow",
    ctx: Context | None = None,
) -> str:
    username = username_arg(username)
    async with x_client(ctx) as client:
        user = await client.get_user_by_screen_name(username)
        if action == "follow":
            await client.follow_user(user.id)
        else:
            await client.unfollow_user(user.id)
        return json.dumps({"status": "success"})


def detail_entries(response: dict) -> list[dict] | None:
    conversation = (response.get("data") or {}).get(
        "threaded_conversation_with_injections_v2"
    )
    if not conversation:
        return None
    return [
        entry
        for instruction in conversation.get("instructions", [])
        for entry in instruction.get("entries", [])
    ]


def bottom_cursor(response: dict) -> str | None:
    """Cursor to the next page of replies (content.value nowadays,
    content.itemContent.value in older responses)."""
    for entry in detail_entries(response) or []:
        if entry.get("entryId", "").startswith("cursor-bottom"):
            content = entry.get("content", {})
            return content.get("value") or content.get("itemContent", {}).get("value")
    return None


def replies_from_tweet_detail(
    client: Client, tweet_id: str, response: dict, require_focal: bool = True
) -> list[Tweet]:
    """Pull the direct replies out of a TweetDetail GraphQL response.

    twikit's own get_tweet_by_id breaks on the current response shape (the
    bottom cursor moved from content.itemContent.value to content.value), so
    this reads only what get_replies needs. Each conversationthread-* module
    starts with a direct reply; the rest of the module is its sub-thread.
    """
    entries = detail_entries(response)
    not_found = ToolError(
        f"Not found: tweet {tweet_id} is deleted, protected or unavailable"
    )
    if entries is None:
        raise not_found
    focal_found = False
    replies = []
    for entry in entries:
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
    # Only the first page carries the focal tweet.
    if require_focal and not focal_found:
        raise not_found
    return replies


@mcp.tool(
    description="Direct replies under a tweet (the first reply of each reply thread)",
    annotations=READ_ONLY,
    structured_output=False,
)
async def get_replies(
    tweet_id: TweetId,
    count: Annotated[int, count_param("replies")] = 30,
    ctx: Context | None = None,
) -> str:
    count_int = parse_count(count)
    tweet_id = tweet_id_arg(tweet_id)
    async with x_client(ctx) as client:
        response, _ = await client.gql.tweet_detail(tweet_id, None)
        replies = replies_from_tweet_detail(client, tweet_id, response)
        seen = {reply.id for reply in replies}
        cursor = bottom_cursor(response)
        for _ in range(MAX_PAGES - 1):
            if len(replies) >= count_int or not cursor:
                break
            response, _ = await client.gql.tweet_detail(tweet_id, cursor)
            new = [
                reply
                for reply in replies_from_tweet_detail(
                    client, tweet_id, response, require_focal=False
                )
                if reply.id not in seen
            ]
            if not new:
                break
            seen.update(reply.id for reply in new)
            replies.extend(new)
            cursor = bottom_cursor(response)
        return tweets_json(replies, count_int)


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
