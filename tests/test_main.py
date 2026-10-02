"""Unit tests for Twitter MCP server."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from mcp import Client as MCPClient
from mcp.server.mcpserver.exceptions import ToolError
from twikit import errors

import main
from main import (
    AuthContext,
    check_write,
    get_auth_context,
    get_client,
    get_profile,
    get_replies,
    get_tweets,
    like_tweet,
    parse_bearer,
    parse_count,
    post_tweet,
    replies_from_tweet_detail,
    search_tweets,
    twitter_errors,
)


@pytest.fixture(autouse=True)
def stdio_cookies(monkeypatch):
    monkeypatch.setenv("TWITTER_AUTH_TOKEN", "env_token")
    monkeypatch.setenv("TWITTER_CT0", "env_ct0")
    monkeypatch.setattr(main, "HTTP_MODE", False)
    main._clients.clear()
    yield
    main._clients.clear()


def make_tweet(i) -> Mock:
    tweet = Mock()
    tweet.id = str(i)
    tweet.in_reply_to = None
    tweet.user.screen_name = "testuser"
    tweet.text = f"tweet {i}"
    tweet.lang = "en"
    tweet.created_at = "2024-01-01"
    tweet.view_count = 100
    tweet.favorite_count = 10
    tweet.reply_count = 5
    tweet.retweet_count = 2
    return tweet


def make_user(username: str = "testuser") -> SimpleNamespace:
    return SimpleNamespace(
        id="123",
        name="Test User",
        screen_name=username,
        created_at="2020-01-01",
        profile_image_url="https://example.com/image.jpg",
        url="https://example.com",
        location="Test City",
        description="Test description",
        description_urls=[],
        is_blue_verified=False,
        verified=False,
        possibly_sensitive=False,
        can_dm=True,
        followers_count=1000,
        fast_followers_count=0,
        normal_followers_count=1000,
        following_count=500,
    )


def new_twikit_mock() -> Mock:
    client = Mock()
    client.handshake = AsyncMock()
    client.http.aclose = AsyncMock()
    return client


@pytest.fixture
def twikit_client():
    with patch("main.XClient") as client_class:
        client = new_twikit_mock()
        client_class.return_value = client
        yield client


# ============================================================================
# Auth
# ============================================================================


def test_parse_bearer_valid():
    assert parse_bearer("Bearer tok:csrf") == AuthContext("tok", "csrf")
    assert parse_bearer("bearer tok:csrf") == AuthContext("tok", "csrf")


@pytest.mark.parametrize(
    "header",
    ["tok:csrf", "Basic tok:csrf", "Bearer tok", "Bearer :csrf", "Bearer tok:"],
)
def test_parse_bearer_rejects_malformed(header):
    with pytest.raises(ToolError, match="AUTH_REQUIRED"):
        parse_bearer(header)


def test_stdio_uses_env_cookies():
    assert get_auth_context(None, http_mode=False) == AuthContext(
        "env_token", "env_ct0"
    )


def test_stdio_without_env_cookies(monkeypatch):
    monkeypatch.delenv("TWITTER_AUTH_TOKEN")
    with pytest.raises(ToolError, match="AUTH_REQUIRED"):
        get_auth_context(None, http_mode=False)


@pytest.mark.parametrize("headers", [{}, None])
def test_http_mode_never_falls_back_to_env(headers):
    # Even if a transport ever reports no headers, HTTP mode must not act as owner.
    with pytest.raises(ToolError, match="AUTH_REQUIRED"):
        get_auth_context(headers, http_mode=True)


def test_http_uses_header_cookies():
    headers = {"authorization": "Bearer tok:csrf"}
    assert get_auth_context(headers, http_mode=True) == AuthContext("tok", "csrf")


# ============================================================================
# Client cache and code-34 retry
# ============================================================================


def code34() -> errors.NotFound:
    return errors.NotFound(
        'status: 404, message: "{"errors":[{"message":"Sorry, that page does not exist","code":34}]}"'
    )


class FakeTransaction:
    """Stands in for twikit's ClientTransaction; counts handshakes."""

    made = 0
    fail_next = 0

    def __init__(self):
        self.home_page_response = None

    async def init(self, session, headers):
        await asyncio.sleep(0.01)
        if FakeTransaction.fail_next:
            FakeTransaction.fail_next -= 1
            raise Exception("Couldn't get KEY_BYTE indices")
        FakeTransaction.made += 1
        self.home_page_response = object()


@pytest.fixture
def fake_transaction(monkeypatch):
    FakeTransaction.made = 0
    FakeTransaction.fail_next = 0
    monkeypatch.setattr(main, "ClientTransaction", FakeTransaction)
    return FakeTransaction


def empty404() -> errors.NotFound:
    return errors.NotFound('status: 404, message: ""')


@pytest.mark.parametrize("rejection", [code34, empty404])
async def test_xclient_rejection_redoes_handshake(fake_transaction, rejection):
    ok = ({"ok": True}, Mock())
    side = [rejection(), rejection(), ok]
    with patch.object(main.Client, "request", AsyncMock(side_effect=side)) as req:
        client = main.XClient("en-US")
        assert await client.request("GET", "https://x.com/i/api/x") == ok
    assert req.await_count == 3
    assert fake_transaction.made == 3  # initial + one fresh handshake per 404


async def test_xclient_gives_up_after_retries(fake_transaction):
    with patch.object(main.Client, "request", AsyncMock(side_effect=code34())) as req:
        with pytest.raises(errors.NotFound):
            await main.XClient("en-US").request("GET", "https://x.com/i/api/x")
    assert req.await_count == main.XClient.RETRIES + 1


async def test_xclient_does_not_retry_other_404s(fake_transaction):
    other = errors.NotFound('status: 404, message: "{"errors":[{"code":144}]}"')
    with patch.object(main.Client, "request", AsyncMock(side_effect=other)) as req:
        with pytest.raises(errors.NotFound):
            await main.XClient("en-US").request("GET", "https://x.com/i/api/x")
    assert req.await_count == 1
    assert fake_transaction.made == 1


async def test_xclient_concurrent_handshakes_run_once(fake_transaction):
    client = main.XClient("en-US")
    await asyncio.gather(*(client.handshake() for _ in range(5)))
    assert fake_transaction.made == 1


async def test_xclient_failed_handshake_is_not_published(fake_transaction):
    client = main.XClient("en-US")
    before = client.client_transaction
    fake_transaction.fail_next = 1
    with pytest.raises(main.HandshakeError, match="KEY_BYTE"):
        await client.handshake()
    assert client.client_transaction is before
    await client.handshake()
    assert client.client_transaction.home_page_response is not None


@pytest.mark.parametrize(
    "message",
    [
        "Couldn't get KEY_BYTE indices",
        "Couldn't get key from the page source",
        "invalid response",
    ],
)
async def test_xclient_wraps_every_bare_handshake_failure(monkeypatch, message):
    class Broken(FakeTransaction):
        async def init(self, session, headers):
            raise Exception(message)

    monkeypatch.setattr(main, "ClientTransaction", Broken)
    with pytest.raises(main.HandshakeError, match=message):
        await main.XClient("en-US").handshake()


async def test_xclient_handshake_keeps_network_and_x_errors(monkeypatch):
    class Offline(FakeTransaction):
        async def init(self, session, headers):
            raise httpx.ConnectError("offline")

    monkeypatch.setattr(main, "ClientTransaction", Offline)
    with pytest.raises(httpx.ConnectError):
        await main.XClient("en-US").handshake()


async def test_xclient_stale_handshake_replaced_once(fake_transaction):
    client = main.XClient("en-US")
    await client.handshake()
    stale = client.client_transaction
    # Two requests hit code 34 on the same keys; only one re-handshake runs.
    await asyncio.gather(client.handshake(stale=stale), client.handshake(stale=stale))
    assert fake_transaction.made == 2
    assert client.client_transaction is not stale


async def test_client_is_reused_per_cookie_pair(twikit_client):
    assert await get_client(None) is await get_client(None)
    twikit_client.set_cookies.assert_called_once_with(
        {"auth_token": "env_token", "ct0": "env_ct0"}
    )
    twikit_client.handshake.assert_awaited_once()


async def test_failed_handshake_is_not_cached():
    broken, healthy = new_twikit_mock(), new_twikit_mock()
    broken.handshake.side_effect = main.HandshakeError("Couldn't get KEY_BYTE indices")
    healthy.get_user_by_screen_name = AsyncMock(return_value=make_user())
    with patch("main.XClient", side_effect=[broken, healthy]):
        with pytest.raises(ToolError, match="handshake failed"):
            await get_profile("testuser")
        broken.http.aclose.assert_awaited_once()
        assert main._clients == {}
        json.loads(await get_profile("testuser"))
    assert list(main._clients.values()) == [healthy]


async def test_parallel_cold_calls_warm_one_client():
    clients = []

    def make(_lang):
        client = new_twikit_mock()

        async def slow_handshake():
            await asyncio.sleep(0.05)

        client.handshake = AsyncMock(side_effect=slow_handshake)
        client.get_user_by_screen_name = AsyncMock(return_value=make_user())
        clients.append(client)
        return client

    with patch("main.XClient", side_effect=make):
        await asyncio.gather(*(get_profile("testuser") for _ in range(5)))
    assert len(clients) == 1
    clients[0].handshake.assert_awaited_once()


async def test_cache_evicts_oldest_and_closes_it(monkeypatch):
    monkeypatch.setattr(main, "_MAX_CLIENTS", 2)
    made = []

    def make(_lang):
        made.append(new_twikit_mock())
        return made[-1]

    with patch("main.XClient", side_effect=make):
        for i in range(3):
            monkeypatch.setenv("TWITTER_AUTH_TOKEN", f"tok{i}")
            await get_client(None)
    assert len(main._clients) == 2
    made[0].http.aclose.assert_awaited_once()
    assert made[0] not in main._clients.values()


# ============================================================================
# Input validation
# ============================================================================


@pytest.mark.parametrize("count,expected", [(5, 5), ("5", 5), (50, 50), (1, 1)])
def test_parse_count_valid(count, expected):
    assert parse_count(count) == expected


@pytest.mark.parametrize(
    "count,message",
    [
        ("invalid", "Invalid argument \\(count\\)"),
        (51, "max value is 50"),
        (0, "must be at least 1"),
        (-3, "must be at least 1"),
    ],
)
async def test_tools_reject_bad_count(count, message):
    with pytest.raises(ToolError, match=message):
        await get_tweets("testuser", count=count)
    with pytest.raises(ToolError, match=message):
        await search_tweets("python", count=count)


# ============================================================================
# Error mapping
# ============================================================================


@pytest.mark.parametrize(
    "exc,message",
    [
        (errors.Unauthorized("401"), "AUTH_REQUIRED.*refresh"),
        (errors.Forbidden("403"), "AUTH_REQUIRED.*403"),
        (errors.UserNotFound("gone"), "User not found"),
        (errors.TweetNotAvailable("gone"), "Not found"),
        (errors.NotFound("404"), "Not found"),
        (errors.ServerError("500"), "X request failed: ServerError"),
        (httpx.ConnectError("boom"), "Network error.*ConnectError"),
        (KeyError("itemContent"), "Unexpected X response shape.*itemContent"),
        (AttributeError("json"), "Unexpected X response shape"),
        (
            main.HandshakeError("Couldn't get key from the page source"),
            "AUTH_REQUIRED.*handshake",
        ),
    ],
)
def test_twitter_errors_mapping(exc, message):
    with pytest.raises(ToolError, match=message), twitter_errors():
        raise exc


def test_twitter_errors_rate_limit_reset():
    exc = errors.TooManyRequests("429", headers={"x-rate-limit-reset": "1700000000"})
    with pytest.raises(ToolError, match="Rate limited.*2023-11-14"), twitter_errors():
        raise exc


def test_twitter_errors_passes_tool_errors_through():
    with pytest.raises(ToolError, match="^mine$"), twitter_errors():
        raise ToolError("mine")


@pytest.mark.parametrize("exc", [ValueError("ours"), TypeError("ours")])
def test_twitter_errors_reraises_unrelated_exceptions(exc):
    with pytest.raises(type(exc)), twitter_errors():
        raise exc


async def test_tool_maps_unauthorized(twikit_client):
    twikit_client.get_user_by_screen_name = AsyncMock(
        side_effect=errors.Unauthorized("401")
    )
    with pytest.raises(ToolError, match="AUTH_REQUIRED"):
        await get_profile("testuser")


async def test_lazy_tweet_field_shape_error_is_tool_error(twikit_client):
    # twikit reads Tweet fields lazily, so a reshaped response only fails
    # while serializing; that must still be inside the error mapping.
    class BrokenTweet:
        id = "1"
        in_reply_to = None

        @property
        def user(self):
            raise KeyError("core")

    twikit_client.search_tweet = AsyncMock(return_value=[BrokenTweet()])
    with pytest.raises(ToolError, match="Unexpected X response shape.*core"):
        await search_tweets("python")


# ============================================================================
# Write checks
# ============================================================================


def test_check_write_raises_on_graphql_errors():
    response = httpx.Response(
        200, json={"errors": [{"message": "already favorited", "code": 139}]}
    )
    with pytest.raises(ToolError, match="X rejected the action: already favorited"):
        check_write(response)


def test_check_write_accepts_success_and_non_json():
    check_write(httpx.Response(200, json={"data": {"favorite_tweet": "Done"}}))
    check_write(httpx.Response(200, text="not json"))


async def test_like_tweet_reports_rejection(twikit_client):
    twikit_client.favorite_tweet = AsyncMock(
        return_value=httpx.Response(200, json={"errors": [{"message": "nope"}]})
    )
    with pytest.raises(ToolError, match="rejected the action: nope"):
        await like_tweet("1")


# ============================================================================
# Tools
# ============================================================================


async def test_get_tweets_success_and_truncates(twikit_client):
    twikit_client.get_user_by_screen_name = AsyncMock(return_value=Mock(id="123"))
    twikit_client.get_user_tweets = AsyncMock(
        return_value=[make_tweet(i) for i in range(14)]
    )

    data = json.loads(await get_tweets("testuser", count=5))

    assert [t["id"] for t in data] == ["0", "1", "2", "3", "4"]
    assert data[0]["author_username"] == "testuser"
    twikit_client.get_user_tweets.assert_called_once_with("123", "Tweets", count=5)


async def test_get_profile_success(twikit_client):
    twikit_client.get_user_by_screen_name = AsyncMock(return_value=make_user())

    data = json.loads(await get_profile("testuser"))

    assert data["username"] == "testuser"
    assert data["followers_count"] == 1000


async def test_post_tweet_returns_id(twikit_client):
    twikit_client.create_tweet = AsyncMock(return_value=Mock(id="999"))

    data = json.loads(await post_tweet("Hello"))

    assert data == {"status": "success", "id": "999"}
    twikit_client.create_tweet.assert_called_once_with(text="Hello", reply_to=None)


async def test_post_tweet_reply(twikit_client):
    twikit_client.create_tweet = AsyncMock(return_value=Mock(id="999"))
    await post_tweet("Hi", reply_to_tweet_id="42")
    twikit_client.create_tweet.assert_called_once_with(text="Hi", reply_to="42")


@pytest.mark.parametrize(
    "action,method", [("like", "favorite_tweet"), ("unlike", "unfavorite_tweet")]
)
async def test_like_tweet(twikit_client, action, method):
    ok = httpx.Response(200, json={"data": {}})
    setattr(twikit_client, method, AsyncMock(return_value=ok))
    data = json.loads(await like_tweet("123", action=action))
    assert data["status"] == "success"
    getattr(twikit_client, method).assert_called_once_with("123")


# ============================================================================
# Replies (current TweetDetail shape that breaks twikit's get_tweet_by_id)
# ============================================================================


def tweet_detail_response() -> dict:
    def module(reply_id, sub_id):
        return {
            "entryId": f"conversationthread-{reply_id}",
            "content": {
                "__typename": "TimelineTimelineModule",
                "items": [
                    {
                        "entryId": f"conversationthread-{reply_id}-tweet-{reply_id}",
                        "item": {"itemContent": {"id": reply_id}},
                    },
                    {
                        "entryId": f"conversationthread-{reply_id}-tweet-{sub_id}",
                        "item": {"itemContent": {"id": sub_id}},
                    },
                ],
            },
        }

    return {
        "data": {
            "threaded_conversation_with_injections_v2": {
                "instructions": [
                    {
                        "type": "TimelineAddEntries",
                        "entries": [
                            {
                                "entryId": "tweet-1",
                                "content": {"itemContent": {"id": "1"}},
                            },
                            module("2", "2a"),
                            module("3", "3a"),
                            {
                                "entryId": "tweetdetailrelatedtweets-1",
                                "content": {"items": []},
                            },
                            {
                                "entryId": "cursor-bottom-9",
                                "content": {
                                    "__typename": "TimelineTimelineCursor",
                                    "value": "abc",
                                },
                            },
                        ],
                    }
                ]
            }
        }
    }


def fake_tweet_from_data(_client, data):
    content = data["item"] if "item" in data else data["content"]
    tweet_id = content["itemContent"].get("id")
    return make_tweet(tweet_id) if tweet_id else None


def test_replies_from_tweet_detail_takes_first_item_per_thread():
    with patch("main.tweet_from_data", side_effect=fake_tweet_from_data):
        replies = replies_from_tweet_detail(Mock(), "1", tweet_detail_response())
    assert [r.id for r in replies] == ["2", "3"]


def test_replies_from_tweet_detail_no_conversation():
    with pytest.raises(ToolError, match="Not found"):
        replies_from_tweet_detail(Mock(), "1", {"data": {}})


def test_replies_from_tweet_detail_empty_focal_tweet():
    # Shape X returns for a deleted/nonexistent tweet: focal entry, no result.
    response = tweet_detail_response()
    instructions = response["data"]["threaded_conversation_with_injections_v2"][
        "instructions"
    ]
    instructions[0]["entries"][0]["content"]["itemContent"] = {"tweet_results": {}}
    with patch("main.tweet_from_data", side_effect=fake_tweet_from_data):
        with pytest.raises(ToolError, match="Not found: tweet 1"):
            replies_from_tweet_detail(Mock(), "1", response)


async def test_get_replies_tool(twikit_client):
    twikit_client.gql.tweet_detail = AsyncMock(
        return_value=(tweet_detail_response(), None)
    )
    with patch("main.tweet_from_data", side_effect=fake_tweet_from_data):
        data = json.loads(await get_replies("1", count=1))
    assert [t["id"] for t in data] == ["2"]
    twikit_client.gql.tweet_detail.assert_called_once_with("1", None)


# ============================================================================
# Through the MCP protocol (in-process)
# ============================================================================


async def test_tools_listed_with_annotations_and_integer_count():
    async with MCPClient(main.mcp) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}

    assert set(tools) == {
        "get_tweets",
        "get_profile",
        "search_tweets",
        "like_tweet",
        "retweet",
        "post_tweet",
        "get_trends",
        "get_timeline",
        "follow_user",
        "get_replies",
    }
    assert tools["get_profile"].annotations.read_only_hint is True
    assert tools["post_tweet"].annotations.read_only_hint is False
    assert tools["follow_user"].annotations.read_only_hint is False
    assert tools["follow_user"].annotations.destructive_hint is True
    assert tools["get_tweets"].input_schema["properties"]["count"]["type"] == "integer"
    assert "ctx" not in tools["get_tweets"].input_schema["properties"]


@pytest.mark.parametrize("count", [3, "3"])
async def test_call_tool_accepts_int_or_numeric_string(twikit_client, count):
    twikit_client.get_trends = AsyncMock(
        return_value=[
            SimpleNamespace(
                name=f"t{i}", tweets_count=i, grouped_trends=[], domain_context=""
            )
            for i in range(30)
        ]
    )
    async with MCPClient(main.mcp) as client:
        result = await client.call_tool("get_trends", {"count": count})

    assert not result.is_error
    assert len(json.loads(result.content[0].text)) == 3


async def test_call_tool_error_is_tool_error(twikit_client, monkeypatch):
    monkeypatch.delenv("TWITTER_CT0")
    async with MCPClient(main.mcp) as client:
        result = await client.call_tool("get_profile", {"username": "x"})
    assert result.is_error
    assert "AUTH_REQUIRED" in result.content[0].text


# ============================================================================
# HTTP mode, through the real Streamable HTTP app (in-process ASGI)
# ============================================================================


async def http_call(app, headers: dict, name: str, arguments: dict) -> dict:
    base = {
        "content-type": "application/json",
        "accept": "application/json, text/event-stream",
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://127.0.0.1:3000"
    ) as c:
        initialize = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "0"},
            },
        }
        r = await c.post("/mcp", headers=base | headers, json=initialize)
        h = base | headers | {"mcp-protocol-version": "2025-06-18"}
        if sid := r.headers.get("mcp-session-id"):
            h["mcp-session-id"] = sid
        await c.post(
            "/mcp",
            headers=h,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        call = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        r = await c.post("/mcp", headers=h, json=call)
    if r.headers.get("content-type", "").startswith("text/event-stream"):
        data = [ln[5:] for ln in r.text.splitlines() if ln.startswith("data:")]
        return json.loads(data[-1])["result"]
    return r.json()["result"]


@asynccontextmanager
async def http_app(monkeypatch):
    # Entered inside the test, not as a fixture: anyio's cancel scope must be
    # exited in the same task that entered it.
    monkeypatch.setattr(main, "HTTP_MODE", True)
    app = main.mcp.streamable_http_app(host="127.0.0.1")
    async with app.router.lifespan_context(app):
        yield app


async def test_http_without_bearer_never_uses_owner_cookies(monkeypatch, twikit_client):
    async with http_app(monkeypatch) as app:
        result = await http_call(app, {}, "get_profile", {"username": "x"})
    assert result["isError"] is True
    assert "AUTH_REQUIRED" in result["content"][0]["text"]
    twikit_client.set_cookies.assert_not_called()


async def test_http_bearer_cookies_are_used(monkeypatch, twikit_client):
    twikit_client.get_user_by_screen_name = AsyncMock(return_value=make_user())
    async with http_app(monkeypatch) as app:
        result = await http_call(
            app, {"Authorization": "Bearer tok:csrf"}, "get_profile", {"username": "x"}
        )
    assert result["isError"] is False
    twikit_client.set_cookies.assert_called_once_with(
        {"auth_token": "tok", "ct0": "csrf"}
    )


# ============================================================================
# Replies against twikit's real tweet_from_data (shapes from live X responses)
# ============================================================================

LEGACY_USER = {
    "created_at": "Mon Jan 01 00:00:00 +0000 2020",
    "name": "N",
    "screen_name": "u",
    "profile_image_url_https": "x",
    "location": "",
    "description": "",
    "entities": {},
    "verified": False,
    "possibly_sensitive": False,
    "can_dm": False,
    "can_media_tag": False,
    "want_retweets": False,
    "default_profile": True,
    "default_profile_image": False,
    "has_custom_timelines": False,
    "followers_count": 1,
    "fast_followers_count": 0,
    "normal_followers_count": 1,
    "friends_count": 1,
    "favourites_count": 0,
    "listed_count": 0,
    "media_count": 0,
    "statuses_count": 1,
    "is_translator": False,
    "translator_type": "none",
}


def raw_tweet(tid, reply_to=None, quoted=None) -> dict:
    t = {
        "__typename": "Tweet",
        "rest_id": tid,
        "core": {
            "user_results": {
                "result": {
                    "__typename": "User",
                    "rest_id": "9",
                    "is_blue_verified": False,
                    "legacy": dict(LEGACY_USER),
                }
            }
        },
        "views": {"count": "5", "state": "EnabledWithCount"},
        "legacy": {
            "created_at": "Mon Jan 01 00:00:00 +0000 2024",
            "full_text": f"t{tid}",
            "lang": "en",
            "in_reply_to_status_id_str": reply_to,
            "reply_count": 0,
            "favorite_count": 0,
            "retweet_count": 0,
            "is_quote_status": False,
        },
    }
    if quoted:
        t["quoted_status_result"] = {"result": quoted}
    return t


def raw_item(entry_id, result) -> dict:
    return {
        "entryId": entry_id,
        "item": {
            "itemContent": {
                "itemType": "TimelineTweet",
                "__typename": "TimelineTweet",
                "tweet_results": {"result": result},
            }
        },
    }


def raw_module(rid, subs=(), first=None) -> dict:
    items = [
        raw_item(f"conversationthread-{rid}-tweet-{rid}", first or raw_tweet(rid, "1"))
    ]
    items += [
        raw_item(f"conversationthread-{rid}-tweet-{s}", raw_tweet(s, rid)) for s in subs
    ]
    items.append(
        {
            "entryId": f"conversationthread-{rid}-cursor-showmore-77",
            "item": {
                "itemContent": {
                    "itemType": "TimelineTimelineCursor",
                    "value": "c",
                    "cursorType": "ShowMore",
                }
            },
        }
    )
    return {
        "entryId": f"conversationthread-{rid}",
        "content": {"__typename": "TimelineTimelineModule", "items": items},
    }


def raw_detail(focal_result, entries_after=()) -> dict:
    entries = [
        # Ancestor: the focal tweet is itself a reply.
        {
            "entryId": "tweet-0",
            "content": {"itemContent": {"tweet_results": {"result": raw_tweet("0")}}},
        },
        {
            "entryId": "tweet-1",
            "content": {"itemContent": {"tweet_results": {"result": focal_result}}},
        },
        *entries_after,
        {
            "entryId": "cursor-bottom-1",
            "content": {"__typename": "TimelineTimelineCursor", "value": "abc"},
        },
    ]
    return {
        "data": {
            "threaded_conversation_with_injections_v2": {
                "instructions": [
                    {"type": "TimelineClearCache"},
                    {"type": "TimelineAddEntries", "entries": entries},
                ]
            }
        }
    }


def test_replies_real_parser_handles_live_shapes():
    response = raw_detail(
        raw_tweet("1", "0"),
        [
            raw_module("2", ["2a"]),
            raw_module("3", first={"__typename": "TweetTombstone", "tombstone": {}}),
            raw_module(
                "4",
                first={
                    "__typename": "TweetWithVisibilityResults",
                    "tweet": raw_tweet("4", "1"),
                },
            ),
            raw_module("5", first=raw_tweet("5", "1", quoted=raw_tweet("Q"))),
            {
                "entryId": "tweetdetailrelatedtweets-1",
                "content": {"items": [raw_item("x-tweet-R", raw_tweet("R"))]},
            },
        ],
    )
    replies = replies_from_tweet_detail(Mock(), "1", response)
    # Ancestor, tombstone, sub-replies, show-more cursor and related tweets are skipped.
    assert [r.id for r in replies] == ["2", "4", "5"]
    assert json.loads(main.tweets_json(replies, 50))[0]["text"] == "t2"


@pytest.mark.parametrize(
    "focal",
    [
        {"__typename": "TweetTombstone"},
        {"__typename": "TweetUnavailable", "reason": "Protected"},
    ],
)
def test_replies_real_parser_unavailable_focal(focal):
    with pytest.raises(ToolError, match="Not found: tweet 1"):
        replies_from_tweet_detail(Mock(), "1", raw_detail(focal))


def test_replies_real_parser_errors_response():
    response = {
        "errors": [{"message": "No status found with that ID.", "code": 144}],
        "data": {},
    }
    with pytest.raises(ToolError, match="Not found"):
        replies_from_tweet_detail(Mock(), "1", response)


async def test_get_replies_shape_break_and_non_json_are_tool_errors(twikit_client):
    broken = raw_detail(raw_tweet("1"), [raw_module("2")])
    entries = broken["data"]["threaded_conversation_with_injections_v2"][
        "instructions"
    ][1]["entries"]
    del entries[2]["content"]["items"][0]["item"]["itemContent"]["tweet_results"][
        "result"
    ]["legacy"]["full_text"]
    twikit_client.gql.tweet_detail = AsyncMock(return_value=(broken, None))
    with pytest.raises(ToolError, match="Unexpected X response shape"):
        await get_replies("1")

    twikit_client.gql.tweet_detail = AsyncMock(return_value=("<html>oops</html>", None))
    with pytest.raises(ToolError, match="Unexpected X response shape"):
        await get_replies("1")


@pytest.mark.parametrize(
    "action,method", [("retweet", "retweet"), ("undo", "delete_retweet")]
)
async def test_retweet(twikit_client, action, method):
    ok = httpx.Response(200, json={"data": {}})
    setattr(twikit_client, method, AsyncMock(return_value=ok))
    data = json.loads(await main.retweet("123", action=action))
    assert data["status"] == "success"
    getattr(twikit_client, method).assert_called_once_with("123")


async def test_retweet_reports_daily_limit(twikit_client):
    # Real 200 body seen live on 2026-10-01.
    body = {
        "data": {},
        "errors": [
            {
                "code": 344,
                "message": "Authorization: You have reached your daily limit for "
                "sending Tweets and messages. Please try again later.",
            }
        ],
    }
    twikit_client.retweet = AsyncMock(return_value=httpx.Response(200, json=body))
    with pytest.raises(ToolError, match="rejected the action.*daily limit"):
        await main.retweet("123")


@pytest.mark.parametrize(
    "action,method", [("follow", "follow_user"), ("unfollow", "unfollow_user")]
)
async def test_follow_user(twikit_client, action, method):
    twikit_client.get_user_by_screen_name = AsyncMock(return_value=make_user())
    setattr(twikit_client, method, AsyncMock())
    data = json.loads(await main.follow_user("testuser", action=action))
    assert data["status"] == "success"
    getattr(twikit_client, method).assert_called_once_with("123")


@pytest.mark.parametrize(
    "category,method",
    [("for-you", "get_timeline"), ("following", "get_latest_timeline")],
)
async def test_get_timeline(twikit_client, category, method):
    setattr(
        twikit_client,
        method,
        AsyncMock(return_value=[make_tweet(i) for i in range(35)]),
    )
    data = json.loads(await main.get_timeline(category=category, count=5))
    assert len(data) == 5
    getattr(twikit_client, method).assert_called_once_with(count=5)
