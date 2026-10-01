## twitter-mcp-server

Twitter/X client MCP server that works off your logged-in session cookies
(`auth_token` + `ct0`), no paid API key. Built on
[twikit](https://github.com/d60/twikit), pinned to a community patch because
upstream twikit has been broken against X since March 2026.

Fork of [touchmeangel/twitter-mcp-server](https://github.com/touchmeangel/twitter-mcp-server)
with stdio support, mcp SDK 2.x, and the fixes listed in the git log.

## Tools
### Reading Tools (read-only)
- `get_tweets` - Latest tweets from a user
- `get_profile` - Profile details of a user
- `search_tweets` - Search by hashtag or keyword (`Top` or `Latest`)
- `get_replies` - Direct replies under a tweet

### Timeline Tools (read-only)
- `get_timeline` - Your home timeline (`for-you` or `following`)
- `get_trends` - Trending topics (`trending`, `for-you`, `news`, `sports`, `entertainment`)

### Write Tools (act on your account)
- `like_tweet` - Like or unlike a tweet
- `retweet` - Retweet or undo a retweet
- `post_tweet` - Post a tweet or a reply; returns the new tweet's id
- `follow_user` - Follow or unfollow a user

`count` is capped at 50 everywhere. Tools carry MCP `readOnlyHint` annotations,
so clients can auto-approve reads and prompt on writes.

## Usage

### stdio (local)

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements.txt
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

### Streamable HTTP

Set `APP_PORT` (and optionally `APP_HOST`, default `127.0.0.1`):

```bash
APP_PORT=3000 .venv/bin/python main.py
```

Connect to `http://localhost:3000/mcp`. Every request must carry its own cookies:

```
Authorization: Bearer <auth_token>:<ct0>
```

In HTTP mode the server never falls back to the cookies in its own `.env`, so
anything that can reach the port cannot act as the server owner. On localhost,
requests with a foreign `Origin` are rejected (DNS-rebinding protection).

## Errors

Tool errors come back as MCP tool errors with an actionable message:
- `AUTH_REQUIRED` - missing, malformed, invalid or expired cookies
- `Rate limited by X; resets at <time>`
- `User not found or unavailable` / `Not found`
- `Unexpected X response shape ... twikit likely needs re-patching` - X changed
  its GraphQL responses again

## Development

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python -m pytest
```
