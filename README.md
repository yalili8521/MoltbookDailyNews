# MoltbookDailyNews

Automated daily digest that fetches trending posts from [Moltbook](https://www.moltbook.com) and [Hacker News](https://news.ycombinator.com), scores them, detects multi-day trends, and writes a **Daily Brief** page to a Notion database.

## Features

- **Dual-source digest** — Moltbook + Hacker News in one Notion page
- **HOT ranking** — engagement score (`score + 2 × comments`)
- **RISING detection** — snapshot-based velocity with recency boost
- **Comment surge alerts** — identifies threads with rapid comment growth
- **Multi-day streaks ("持续升温")** — posts/topics trending across 2-3+ days
- **Topic extraction** — keyword map + bigram fallback, combined across both sources
- **Builder Notes** — actionable signals, distribution plays, what to copy
- **DST-safe scheduling** — runs at 8:00 AM Pacific regardless of daylight saving
- **Duplicate prevention** — only posts once per day

## Prerequisites

- Python 3.11+
- [Moltbook API key](https://www.moltbook.com/developers)
- [Notion integration token](https://www.notion.so/my-integrations)

## Notion Database Setup

Create a Notion database with **exactly** these properties:

| Property | Type |
|----------|------|
| `Name` | Title |
| `Date` | Date |
| `Top Topics` | Multi-select |
| `TL;DR` | Text (rich text) |
| `Top Posts` | Text (rich text) |
| `Builder Notes` | Text (rich text) |

Then share the database with your Notion integration.

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MOLTBOOK_API_KEY` | Yes | — | Moltbook bearer token |
| `NOTION_TOKEN` | Yes* | — | Notion integration token |
| `NOTION_DATABASE_ID` | Yes* | — | Target Notion database ID |
| `HOT_LIMIT` | No | `30` | Hot posts to fetch |
| `NEW_LIMIT` | No | `50` | New posts to fetch |
| `RISING_LIMIT` | No | `30` | Rising posts to fetch |
| `TOP_HOT_N` | No | `7` | Hot picks in digest |
| `TOP_RISING_N` | No | `5` | Rising picks in digest |
| `TOP_SURGE_N` | No | `3` | Comment surges in digest |
| `TOP_STREAK_POSTS_N` | No | `5` | Max streak posts shown |
| `TOP_STREAK_TOPICS_N` | No | `5` | Max streak topics shown |
| `DRY_RUN` | No | `0` | `1` = print digest to console, skip Notion |
| `ENABLE_COMMENT_SAMPLING` | No | `0` | `1` = fetch top comments for surge posts |
| `ENABLE_HN` | No | `1` | `1` = include Hacker News section in digest |
| `HN_LIMIT` | No | `30` | Max HN stories to fetch |
| `TOP_HN_N` | No | `7` | HN hot picks in digest |
| `HN_MIN_SCORE` | No | `10` | Minimum HN score to include |
| `HN_MIN_COMMENTS` | No | `3` | Minimum HN comments to include |

\* Not required when `DRY_RUN=1`

## Usage

### Local (dry run — Moltbook + Hacker News)

```bash
export MOLTBOOK_API_KEY="moltbook_xxx_your_key"
ENABLE_HN=1 DRY_RUN=1 python moltbook_digest_to_notion.py
```

### Local (dry run — Moltbook only)

```bash
export MOLTBOOK_API_KEY="moltbook_xxx_your_key"
ENABLE_HN=0 DRY_RUN=1 python moltbook_digest_to_notion.py
```

### Local (post to Notion)

```bash
export MOLTBOOK_API_KEY="moltbook_xxx_your_key"
export NOTION_TOKEN="secret_xxx"
export NOTION_DATABASE_ID="abc123..."
python moltbook_digest_to_notion.py
```

### GitHub Actions

1. Push this repo to GitHub
2. Add these repository secrets: `MOLTBOOK_API_KEY`, `NOTION_TOKEN`, `NOTION_DATABASE_ID`
3. The workflow runs every hour; the script only posts at 8:00 AM Pacific

You can also trigger manually from the Actions tab via **workflow_dispatch**.

## How It Works

### Scoring

- **HOT** = `score + 2 × comment_count` — pure engagement ranking
- **RISING** = `velocity × recency_boost` where:
  - `velocity = (Δscore + 3 × Δcomments) / Δhours` (from snapshot deltas)
  - `recency_boost`: 1.5× for posts ≤24h old, 1.2× for ≤72h, 1.0× otherwise
  - Fallback (no snapshot): `(score + 2 × comments) / age_hours`
- **Comment surge** = `Δcomments / Δhours` — top 3 by comment velocity

### Snapshots & Streaks

The script maintains `.cache/` files across runs:

- `moltbook_snapshot.json` — per-post score/comment snapshots for velocity calculation
- `moltbook_history.json` — 14-day rolling history for streak detection
- `hn_snapshot.json` — Hacker News score/comment snapshots
- `hn_history.json` — Hacker News 14-day rolling history
- `state.json` — last posted date to prevent duplicates

A post is "持续升温" if it appears in HOT/RISING for ≥2 of the last 3 days.
A topic is "持续升温" if it appears in topic tags for ≥3 of the last 5 days.

## Project Structure

```
MoltbookDailyNews/
├── moltbook_digest_to_notion.py   # Main script (all logic)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── .gitignore
├── .github/workflows/
│   └── moltbook_digest.yml        # GitHub Actions workflow
└── .cache/                        # Created at runtime (git-ignored)
    ├── moltbook_snapshot.json
    ├── moltbook_history.json
    ├── hn_snapshot.json
    ├── hn_history.json
    └── state.json
```

## License

MIT
