"""Unit tests for Twitter MCP server."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from mcp import Client as MCPClient
from mcp.server.mcpserver.exceptions import ToolError
from twikit import errors

import main
from main import (
    AuthContext,
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
    main._clients.clear()
    yield
    main._clients.clear()


def make_tweet(i: int) -> Mock:
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


@pytest.fixture
def twikit_client():
    with patch("main.Client") as client_class:
        client = Mock()
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
    assert get_auth_context(None) == AuthContext("env_token", "env_ct0")


def test_stdio_without_env_cookies(monkeypatch):
    monkeypatch.delenv("TWITTER_AUTH_TOKEN")
    with pytest.raises(ToolError, match="AUTH_REQUIRED"):
        get_auth_context(None)


def test_http_without_header_never_falls_back_to_env():
    with pytest.raises(ToolError, match="AUTH_REQUIRED"):
        get_auth_context({})


def test_http_uses_header_cookies():
    headers = {"authorization": "Bearer tok:csrf"}
    assert get_auth_context(headers) == AuthContext("tok", "csrf")


def test_client_is_reused_per_cookie_pair(twikit_client):
    assert get_client(None) is get_client(None)
    twikit_client.set_cookies.assert_called_once_with(
        {"auth_token": "env_token", "ct0": "env_ct0"}
    )


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
    ],
)
def test_twitter_errors_mapping(exc, message):
    with pytest.raises(ToolError, match=message), twitter_errors():
        raise exc


def test_twitter_errors_reports_response_shape_change():
    with (
        pytest.raises(ToolError, match="Unexpected X response shape.*itemContent"),
        twitter_errors(),
    ):
        raise KeyError("itemContent")


def test_twitter_errors_handshake_failure():
    with pytest.raises(ToolError, match="AUTH_REQUIRED.*handshake"), twitter_errors():
        raise Exception("Couldn't get KEY_BYTE indices")


def test_twitter_errors_reraises_unrelated_exceptions():
    with pytest.raises(ValueError), twitter_errors():
        raise ValueError("ours")


def test_twitter_errors_rate_limit_reset():
    exc = errors.TooManyRequests("429", headers={"x-rate-limit-reset": "1700000000"})
    with (
        pytest.raises(ToolError, match="Rate limited.*2023-11-14"),
        twitter_errors(),
    ):
        raise exc


async def test_tool_maps_unauthorized(twikit_client):
    twikit_client.get_user_by_screen_name = AsyncMock(
        side_effect=errors.Unauthorized("401")
    )
    with pytest.raises(ToolError, match="AUTH_REQUIRED"):
        await get_profile("testuser")


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
    user = Mock()
    user.id = "123"
    user.name = "Test User"
    user.screen_name = "testuser"
    user.created_at = "2020-01-01"
    user.profile_image_url = "https://example.com/image.jpg"
    user.url = "https://example.com"
    user.location = "Test City"
    user.description = "Test description"
    user.description_urls = []
    user.is_blue_verified = False
    user.verified = False
    user.possibly_sensitive = False
    user.can_dm = True
    user.followers_count = 1000
    user.fast_followers_count = 0
    user.normal_followers_count = 1000
    user.following_count = 500
    twikit_client.get_user_by_screen_name = AsyncMock(return_value=user)

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
    setattr(twikit_client, method, AsyncMock())
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
    entries = response["data"]["threaded_conversation_with_injections_v2"][
        "instructions"
    ][0]["entries"]
    entries[0]["content"]["itemContent"] = {"tweet_results": {}}
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
# Through the MCP protocol
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
