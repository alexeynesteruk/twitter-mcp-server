## twitter-mcp-server

Twitter/X client MCP server that works off your logged-in session cookies
(`auth_token` + `ct0`), no paid API key. Built on
[twikit](https://github.com/d60/twikit), pinned to a community patch because
upstream twikit has been broken against X since March 2026.

Fork of [touchmeangel/twitter-mcp-server](https://github.com/touchmeangel/twitter-mcp-server)
with stdio support, mcp SDK 2.x, and the fixes listed in the git log.

## Tools
### Reading Tools (read-only)
- `get_tweets` - A user's recent posts, newest first, including retweets and the
  follow-ups of their own threads
- `get_profile` - Profile details, plus whether you follow them (`following`) and
  they follow you (`followed_by`)
- `search_tweets` - Search by keyword, hashtag or X search operators (`Top` or `Latest`)
- `get_replies` - Direct replies under a tweet

### Timeline Tools (read-only)
- `get_timeline` - Your own home timeline (`for-you` or `following`), ads removed
- `get_trends` - Trending topics (`trending`, `for-you`, `news`, `sports`, `entertainment`)

### Write Tools (act on your account)
- `like_tweet` - Like or unlike a tweet
- `retweet` - Retweet or undo a retweet
- `post_tweet` - Post a tweet or a reply; returns the new tweet's id
- `follow_user` - Follow or unfollow a user

Arguments: usernames work with or without `@` (a profile URL also works), tweet
IDs can be tweet URLs, and `count` (1-50) is filled across X's pages (X serves
about 10-20 items a page; at most 5 pages per call). Tools carry MCP
`readOnlyHint` / `destructiveHint` annotations, so clients can auto-approve
reads and prompt on writes. Liking or retweeting something already
liked/retweeted succeeds with `"already_done": true`.

### Tweet fields

Every tool that returns tweets returns a JSON list of:

| Field | |
|---|---|
| `id`, `url`, `author_username`, `in_reply_to` | |
| `text` | Full text: long-form posts in full, `t.co` links expanded, HTML entities decoded. For a retweet, the original post's text |
| `created_at` | ISO 8601, UTC |
| `lang`, `view_count`, `favorite_count`, `reply_count`, `retweet_count` | Of the original post for retweets |
| `media` | `[{type, url}]` when the post has photos/videos/GIFs |
| `retweet_of` | `{id, author_username, created_at}` of the original, for retweets |
| `quoted` | `{id, author_username, text}` of the quoted post, for quote posts |

## Usage

### stdio (local)

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements.lock
cp .env.example .env   # fill in TWITTER_AUTH_TOKEN and TWITTER_CT0
chmod 600 .env
```

```json
{
  "mcpServers": {
    "twitter": {
      "command": "/path/to/twitter-mcp-server/.venv/bin/python",
      "args": ["/path/to/twitter-mcp-server/main.py"]
    }
  }
}
```

`.env` is read from the script's directory, so the client's working directory
does not matter. `TWITTER_AUTH_TOKEN`/`TWITTER_CT0` can also be passed as env vars.

`requirements.txt` lists the direct dependencies; `requirements.lock` is the
full tested set (`uv pip compile requirements.txt --python-version 3.14 -o
requirements.lock` after changing it).

### Streamable HTTP

Set `APP_PORT` (and optionally `APP_HOST`, default `127.0.0.1`):

```bash
APP_PORT=3000 .venv/bin/python main.py
```

Connect to `http://localhost:3000/mcp`. Every request must carry its own cookies:

```
Authorization: Bearer <auth_token>:<ct0>
```

In HTTP mode (decided by `APP_PORT`, not by the request) the server never falls
back to the cookies in its own `.env`, so anything that can reach the port
cannot act as the server owner. On localhost, requests with a foreign `Host` or
`Origin` are rejected (DNS-rebinding protection).

The cookies travel in plaintext, so put a TLS proxy in front of anything that
is not localhost. Binding `APP_HOST=0.0.0.0` (needed inside Docker) turns the
Host/Origin checks off.

### Docker

```bash
docker build -t twitter-mcp-server .
docker run --rm -e APP_PORT=3000 -e APP_HOST=0.0.0.0 -p 127.0.0.1:3000:3000 twitter-mcp-server
```

The image installs from `requirements.lock`, contains only `main.py` and
`config.py`, and runs as a non-root user.

## How it talks to X

X requires an `X-Client-Transaction-Id` on API calls, computed from keys on the
x.com home page (a "handshake"). twikit's copy of that algorithm was missing two
fixes from the reference implementation
([XClientTransaction](https://github.com/iSarabjitDhiman/XClientTransaction)),
so about 1 handshake in 10 produced ids X rejected with a 404. `XTransaction`
in `main.py` carries the fixes; the server also redoes the handshake and resends
if X still rejects an id. Handshakes are done once per cookie pair and shared.

## Errors

Tool errors come back as MCP tool errors with an actionable message:
- `AUTH_REQUIRED` - missing, malformed, invalid or expired cookies
- `X client handshake failed (HANDSHAKE_FAILED ...)` - usually expired cookies;
  can also be X rate-limiting or challenging the home page
- `X refused the request (403, code N)` - X refused this action (follow limit,
  blocked, ...), the session itself is fine
- `Rate limited by X; resets at <time>`
- `User not found or unavailable` / `Not found`
- `X kept rejecting the request after 2 fresh handshakes` /
  `Unexpected X response shape` - X changed something; twikit needs re-patching

## Known limitations

- twikit imports Js2Py at load time, which uses `co_lnotab`; Python deprecated
  that, so a future Python release may need a twikit update before the server
  starts.
- `search_tweets`: twikit silently skips tweets it cannot parse, so an X format
  change shows up as fewer results rather than an error.

## Development

```bash
uv pip install --python .venv/bin/python -r requirements.lock -r requirements-dev.txt
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python -m pytest
```

Tests never touch the network: an autouse fixture fails any real request.
