#!/usr/bin/env python3
"""
Moltbook Daily Brief → Notion

Fetches HOT / RISING / NEW posts from Moltbook, scores them,
detects comment surges and multi-day streaks, then writes a
daily digest page to a Notion database.
"""

import json
import logging
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("moltbook_digest")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """All tunables, loaded from env with sane defaults."""

    moltbook_api_key: str = ""
    notion_token: str = ""
    notion_database_id: str = ""

    hot_limit: int = 30
    new_limit: int = 50
    rising_limit: int = 30

    top_hot_n: int = 7
    top_rising_n: int = 5
    top_surge_n: int = 3
    top_streak_posts_n: int = 5
    top_streak_topics_n: int = 5

    dry_run: bool = False
    force_run: bool = False
    enable_comment_sampling: bool = False

    cache_dir: str = ".cache"

    enable_hn: bool = True
    hn_limit: int = 30
    top_hn_n: int = 7
    hn_min_score: int = 10
    hn_min_comments: int = 3

    @classmethod
    def from_env(cls) -> "Config":
        c = cls(
            moltbook_api_key=os.getenv("MOLTBOOK_API_KEY", ""),
            notion_token=os.getenv("NOTION_TOKEN", ""),
            notion_database_id=os.getenv("NOTION_DATABASE_ID", ""),
            hot_limit=int(os.getenv("HOT_LIMIT", "30")),
            new_limit=int(os.getenv("NEW_LIMIT", "50")),
            rising_limit=int(os.getenv("RISING_LIMIT", "30")),
            top_hot_n=int(os.getenv("TOP_HOT_N", "7")),
            top_rising_n=int(os.getenv("TOP_RISING_N", "5")),
            top_surge_n=int(os.getenv("TOP_SURGE_N", "3")),
            top_streak_posts_n=int(os.getenv("TOP_STREAK_POSTS_N", "5")),
            top_streak_topics_n=int(os.getenv("TOP_STREAK_TOPICS_N", "5")),
            dry_run=os.getenv("DRY_RUN", "0") == "1",
            force_run=os.getenv("FORCE_RUN", "0") == "1",
            enable_comment_sampling=os.getenv("ENABLE_COMMENT_SAMPLING", "0") == "1",
            cache_dir=os.getenv("CACHE_DIR", ".cache"),
            enable_hn=os.getenv("ENABLE_HN", "1") == "1",
            hn_limit=int(os.getenv("HN_LIMIT", "30")),
            top_hn_n=int(os.getenv("TOP_HN_N", "7")),
            hn_min_score=int(os.getenv("HN_MIN_SCORE", "10")),
            hn_min_comments=int(os.getenv("HN_MIN_COMMENTS", "3")),
        )
        return c

    def validate(self) -> None:
        if not self.moltbook_api_key:
            _die("MOLTBOOK_API_KEY is required")
        if not self.dry_run:
            if not self.notion_token:
                _die("NOTION_TOKEN is required (set DRY_RUN=1 to skip Notion)")
            if not self.notion_database_id:
                _die("NOTION_DATABASE_ID is required (set DRY_RUN=1 to skip Notion)")


def _die(msg: str) -> None:
    log.error(msg)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class Post:
    id: str
    title: str
    url: str
    score: int
    comment_count: int
    created_at: datetime
    community: str = ""

    @property
    def engagement(self) -> float:
        return self.score + 2 * self.comment_count

    @property
    def age_hours(self) -> float:
        delta = datetime.now(timezone.utc) - self.created_at
        return max(delta.total_seconds() / 3600, 0.1)


# ---------------------------------------------------------------------------
# Moltbook API client
# ---------------------------------------------------------------------------
class MoltbookAPI:
    BASE = "https://www.moltbook.com/api/v1"

    def __init__(self, api_key: str) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        )
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    # -- helpers --
    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self.BASE}{path}"
        resp = self.session.get(url, params=params, timeout=30)
        # respect rate-limit header
        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining is not None and int(remaining) < 5:
            reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
            wait = max(reset - int(time.time()), 1)
            log.info("Rate-limit near zero, sleeping %ds", wait)
            time.sleep(wait)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_post(raw: dict) -> Post:
        created = raw.get("created_at") or raw.get("createdAt") or ""
        if isinstance(created, str):
            # try ISO parse
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except ValueError:
                dt = datetime.now(timezone.utc)
        elif isinstance(created, (int, float)):
            dt = datetime.fromtimestamp(created, tz=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)

        post_id = str(raw.get("id") or raw.get("_id") or "")
        # Build permalink — API returns url=null for text posts
        url = raw.get("url") or raw.get("permalink") or raw.get("link") or ""
        if not url and post_id:
            url = f"https://www.moltbook.com/post/{post_id}"

        # submolt can be a nested object or a plain string
        submolt = raw.get("submolt") or raw.get("community") or ""
        if isinstance(submolt, dict):
            submolt = submolt.get("name") or submolt.get("display_name") or ""

        return Post(
            id=post_id,
            title=raw.get("title", "(untitled)"),
            url=url,
            score=int(raw.get("score") or raw.get("upvotes") or 0),
            comment_count=int(raw.get("comment_count") or raw.get("commentCount") or raw.get("comments", 0)),
            created_at=dt,
            community=str(submolt),
        )

    # -- public --
    def get_posts(self, sort: str = "hot", limit: int = 30) -> list[Post]:
        try:
            data = self._get("/posts", {"sort": sort, "limit": limit})
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                log.warning("sort=%s not supported by API, returning empty", sort)
                return []
            raise
        items = data if isinstance(data, list) else data.get("posts") or data.get("data") or data.get("results") or []
        return [self._parse_post(r) for r in items]

    def get_comments(self, post_id: str, limit: int = 3) -> list[dict]:
        try:
            data = self._get(f"/posts/{post_id}/comments", {"sort": "top"})
        except Exception:
            log.warning("Could not fetch comments for %s", post_id)
            return []
        items = data if isinstance(data, list) else data.get("comments") or data.get("data") or []
        out: list[dict] = []
        for c in items[:limit]:
            out.append(
                {
                    "author": c.get("author") or c.get("agentName", "anon"),
                    "text": (c.get("text") or c.get("content") or c.get("body", ""))[:300],
                    "score": int(c.get("score") or c.get("upvotes") or 0),
                }
            )
        return out


# ---------------------------------------------------------------------------
# Hacker News API client
# ---------------------------------------------------------------------------
class HackerNewsAPI:
    BASE = "https://hacker-news.firebaseio.com/v0"

    def __init__(self) -> None:
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def get_top_posts(
        self, limit: int = 30, min_score: int = 10, min_comments: int = 3
    ) -> list[Post]:
        resp = self.session.get(f"{self.BASE}/topstories.json", timeout=30)
        resp.raise_for_status()
        story_ids = resp.json()[: limit * 2]  # fetch extra to account for filtering

        posts: list[Post] = []
        for sid in story_ids:
            if len(posts) >= limit:
                break
            try:
                item_resp = self.session.get(
                    f"{self.BASE}/item/{sid}.json", timeout=15
                )
                item_resp.raise_for_status()
                item = item_resp.json()
                if not item or item.get("type") != "story":
                    continue
                score = item.get("score", 0)
                comments = item.get("descendants", 0)
                if score < min_score or comments < min_comments:
                    continue
                created = datetime.fromtimestamp(
                    item.get("time", 0), tz=timezone.utc
                )
                url = (
                    item.get("url")
                    or f"https://news.ycombinator.com/item?id={sid}"
                )
                posts.append(
                    Post(
                        id=f"hn_{sid}",
                        title=item.get("title", "(untitled)"),
                        url=url,
                        score=score,
                        comment_count=comments,
                        created_at=created,
                        community="HackerNews",
                    )
                )
            except Exception as exc:
                log.debug("Failed to fetch HN item %s: %s", sid, exc)
            time.sleep(0.05)

        log.info("Fetched %d HN posts (after filtering)", len(posts))
        return posts


# ---------------------------------------------------------------------------
# Cache manager
# ---------------------------------------------------------------------------
class CacheManager:
    def __init__(self, cache_dir: str = ".cache") -> None:
        self.dir = cache_dir
        os.makedirs(self.dir, exist_ok=True)
        self.snapshot_path = os.path.join(self.dir, "moltbook_snapshot.json")
        self.history_path = os.path.join(self.dir, "moltbook_history.json")
        self.state_path = os.path.join(self.dir, "state.json")
        self.hn_snapshot_path = os.path.join(self.dir, "hn_snapshot.json")
        self.hn_history_path = os.path.join(self.dir, "hn_history.json")

    # -- atomic write --
    def _write(self, path: str, data: Any) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _read(self, path: str) -> Any:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Cache read failed for %s: %s", path, exc)
            return {}

    # -- snapshot --
    def read_snapshot(self) -> dict:
        return self._read(self.snapshot_path)

    def write_snapshot(self, data: dict) -> None:
        self._write(self.snapshot_path, data)

    # -- history --
    def read_history(self) -> dict:
        return self._read(self.history_path)

    def write_history(self, data: dict) -> None:
        self._write(self.history_path, data)

    # -- state --
    def read_state(self) -> dict:
        return self._read(self.state_path)

    def write_state(self, data: dict) -> None:
        self._write(self.state_path, data)

    # -- HN snapshot --
    def read_hn_snapshot(self) -> dict:
        return self._read(self.hn_snapshot_path)

    def write_hn_snapshot(self, data: dict) -> None:
        self._write(self.hn_snapshot_path, data)

    # -- HN history --
    def read_hn_history(self) -> dict:
        return self._read(self.hn_history_path)

    def write_hn_history(self, data: dict) -> None:
        self._write(self.hn_history_path, data)


# ---------------------------------------------------------------------------
# Post scorer
# ---------------------------------------------------------------------------
class PostScorer:
    """Ranks posts by HOT engagement, RISING velocity, and comment surges."""

    def rank_hot(self, posts: list[Post], n: int) -> list[Post]:
        return sorted(posts, key=lambda p: p.engagement, reverse=True)[:n]

    def rank_rising(
        self, posts: list[Post], snapshot: dict, n: int
    ) -> list[tuple[Post, float]]:
        scored: list[tuple[Post, float]] = []
        now = datetime.now(timezone.utc)

        for p in posts:
            snap = snapshot.get(p.id)
            if snap:
                last_seen = datetime.fromisoformat(snap["last_seen"])
                delta_hours = max((now - last_seen).total_seconds() / 3600, 0.25)
                delta_score = p.score - snap.get("last_score", 0)
                delta_comments = p.comment_count - snap.get("last_comments", 0)
                velocity = (delta_score + 3 * delta_comments) / delta_hours
            else:
                # fallback: recency-weighted engagement per hour
                velocity = (p.score + 2 * p.comment_count) / max(p.age_hours, 0.5)

            # recency boost
            if p.age_hours <= 24:
                boost = 1.5
            elif p.age_hours <= 72:
                boost = 1.2
            else:
                boost = 1.0

            scored.append((p, velocity * boost))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def detect_surges(
        self, posts: list[Post], snapshot: dict, n: int
    ) -> list[tuple[Post, float, int]]:
        surges: list[tuple[Post, float, int]] = []
        now = datetime.now(timezone.utc)

        for p in posts:
            snap = snapshot.get(p.id)
            if not snap:
                continue
            last_seen = datetime.fromisoformat(snap["last_seen"])
            delta_hours = max((now - last_seen).total_seconds() / 3600, 0.25)
            delta_comments = p.comment_count - snap.get("last_comments", 0)
            if delta_comments < 2:
                continue
            velocity = delta_comments / delta_hours
            surges.append((p, velocity, delta_comments))

        surges.sort(key=lambda x: x[1], reverse=True)
        return surges[:n]


# ---------------------------------------------------------------------------
# Topic extractor
# ---------------------------------------------------------------------------
class TopicExtractor:
    KEYWORD_MAP: dict[str, list[str]] = {
        "AI/ML": ["ai", "machine learning", "llm", "gpt", "neural", "model", "transformer", "training", "inference", "openai", "anthropic", "claude", "gemini"],
        "Agents": ["agent", "agents", "autonomous", "agentic", "mcp", "tool use"],
        "Crypto/Web3": ["blockchain", "crypto", "nft", "web3", "defi", "token", "solana", "ethereum", "bitcoin"],
        "Dev Tools": ["developer", "devtools", "ide", "vscode", "github", "git", "api", "sdk", "framework"],
        "Open Source": ["open source", "oss", "foss", "mit license", "apache", "repo"],
        "Startups": ["startup", "founder", "vc", "funding", "launch", "yc", "product hunt"],
        "Security": ["security", "vulnerability", "exploit", "hack", "breach", "privacy"],
        "Gaming": ["game", "gaming", "steam", "unity", "unreal", "esports"],
        "Social": ["social media", "community", "viral", "meme", "content"],
        "Hardware": ["hardware", "chip", "gpu", "nvidia", "apple", "robot"],
        "Mobile": ["ios", "android", "mobile", "app store"],
        "Data": ["data", "database", "analytics", "sql", "pipeline"],
        "Design": ["design", "ui", "ux", "figma", "interface"],
        "Finance": ["finance", "stock", "trading", "market", "economy"],
        "Regulation": ["regulation", "policy", "government", "law", "ban", "gdpr"],
        "Culture": ["meme", "funny", "creative", "art", "music", "story"],
    }

    TOPIC_ZH: dict[str, str] = {
        "AI/ML": "人工智能/机器学习",
        "Agents": "智能体",
        "Crypto/Web3": "加密货币/Web3",
        "Dev Tools": "开发工具",
        "Open Source": "开源",
        "Startups": "创业",
        "Security": "安全",
        "Gaming": "游戏",
        "Social": "社交",
        "Hardware": "硬件",
        "Mobile": "移动端",
        "Data": "数据",
        "Design": "设计",
        "Finance": "金融",
        "Regulation": "监管",
        "Culture": "文化",
    }

    def translate(self, topic: str) -> str:
        """Return 'English (中文)' for known topics, or original for bigrams."""
        zh = self.TOPIC_ZH.get(topic)
        return f"{topic}（{zh}）" if zh else topic

    STOPWORDS = frozenset(
        "the a an is are was were be been being have has had do does did "
        "will would shall should may might can could to for in on at by "
        "of and or but not with from as it its this that these those "
        "i me my we our you your he she they them their what which who "
        "how all each every any some no more most very just about also "
        "so if than too up out off over into back then now here there".split()
    )

    def extract(self, posts: list[Post], max_tags: int = 6) -> list[str]:
        titles = " ".join(p.title.lower() for p in posts)
        scores: dict[str, int] = {}

        # keyword matching
        for topic, keywords in self.KEYWORD_MAP.items():
            total = sum(titles.count(kw) for kw in keywords)
            if total > 0:
                scores[topic] = total

        # bigram fallback
        tokens = [
            t for t in titles.split() if t.isalpha() and t not in self.STOPWORDS and len(t) > 2
        ]
        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
        covered = {kw for keywords in self.KEYWORD_MAP.values() for kw in keywords}
        for bg, cnt in Counter(bigrams).most_common(20):
            if cnt >= 2 and bg not in covered:
                # title-case for readability
                label = bg.title()
                if label not in scores:
                    scores[label] = cnt

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in ranked[:max_tags]]


# ---------------------------------------------------------------------------
# Streak detector
# ---------------------------------------------------------------------------
class StreakDetector:
    @staticmethod
    def detect_post_streaks(
        history: dict,
        current_hot_ids: list[str],
        current_rising_ids: list[str],
        snapshot: dict,
        n: int,
    ) -> list[dict]:
        dates = sorted(history.keys(), reverse=True)[:3]
        counts: dict[str, int] = defaultdict(int)

        for d in dates:
            day = history[d]
            for pid in set(day.get("hot_ids", []) + day.get("rising_ids", [])):
                counts[pid] += 1

        # add today
        for pid in set(current_hot_ids + current_rising_ids):
            counts[pid] += 1

        streaks = []
        for pid, cnt in counts.items():
            if cnt >= 2:
                title = snapshot.get(pid, {}).get("title", pid)
                category = "HOT" if pid in current_hot_ids else "RISING"
                streaks.append(
                    {"post_id": pid, "title": title, "streak_days": cnt, "category": category}
                )
        streaks.sort(key=lambda x: x["streak_days"], reverse=True)
        return streaks[:n]

    @staticmethod
    def detect_topic_streaks(
        history: dict, current_topics: list[str], n: int
    ) -> list[dict]:
        dates = sorted(history.keys(), reverse=True)[:5]
        counts: dict[str, int] = defaultdict(int)

        for d in dates:
            for tag in history[d].get("topic_tags", []):
                counts[tag] += 1

        # add today
        for tag in current_topics:
            counts[tag] += 1

        total_days = min(len(dates) + 1, 6)
        streaks = []
        for tag, cnt in counts.items():
            if cnt >= 3:
                streaks.append({"topic": tag, "frequency": f"{cnt}/{total_days} days"})
        streaks.sort(key=lambda x: int(x["frequency"].split("/")[0]), reverse=True)
        return streaks[:n]

    @staticmethod
    def update_history(
        history: dict,
        date_str: str,
        hot_ids: list[str],
        rising_ids: list[str],
        surge_ids: list[str],
        topics: list[str],
    ) -> dict:
        history[date_str] = {
            "hot_ids": hot_ids,
            "rising_ids": rising_ids,
            "surge_ids": surge_ids,
            "topic_tags": topics,
        }
        # prune older than 14 days
        dates = sorted(history.keys(), reverse=True)
        for d in dates[14:]:
            del history[d]
        return history


# ---------------------------------------------------------------------------
# Title translator (EN -> ZH via free MyMemory API)
# ---------------------------------------------------------------------------
class TitleTranslator:
    """Translates post titles to Chinese. Caches results per run."""

    API = "https://api.mymemory.translated.net/get"

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def translate(self, text: str) -> str:
        if not text:
            return ""
        if text in self._cache:
            return self._cache[text]
        try:
            resp = requests.get(
                self.API,
                params={"q": text[:500], "langpair": "en|zh-CN"},
                timeout=10,
            )
            resp.raise_for_status()
            translated = resp.json().get("responseData", {}).get("translatedText", "")
            # API returns the original text unchanged when it can't translate
            if translated and translated.lower() != text.lower():
                self._cache[text] = translated
                return translated
        except Exception as exc:
            log.debug("Translation failed for '%s': %s", text[:40], exc)
        self._cache[text] = ""
        return ""

    def bulk_translate(self, titles: list[str]) -> dict[str, str]:
        """Translate a list of titles, return {original: translated}."""
        result: dict[str, str] = {}
        for t in titles:
            zh = self.translate(t)
            result[t] = zh
        return result


# ---------------------------------------------------------------------------
# Digest builder
# ---------------------------------------------------------------------------
class DigestBuilder:
    def __init__(self, translator: TitleTranslator | None = None) -> None:
        self._tr = translator or TitleTranslator()

    def _zh(self, title: str) -> str:
        """Return Chinese translation line, or empty string if unavailable."""
        zh = self._tr.translate(title)
        return f"   {zh}" if zh else ""

    def build(self, data: dict) -> dict:
        topics: list[str] = data["topics"]
        hot: list[Post] = data["hot"]
        rising: list[tuple[Post, float]] = data["rising"]
        surges: list[tuple[Post, float, int]] = data["surges"]
        surge_comments: dict[str, list[dict]] = data.get("surge_comments", {})
        post_streaks: list[dict] = data["post_streaks"]
        topic_streaks: list[dict] = data["topic_streaks"]
        date_str: str = data["date"]

        # Pre-translate all titles in one pass
        all_titles = [p.title for p in hot]
        all_titles += [p.title for p, _ in rising]
        all_titles += [p.title for p, _, _ in surges]
        all_titles += [s["title"] for s in post_streaks]
        self._tr.bulk_translate(list(set(all_titles)))

        return {
            "title": f"Daily Brief — {date_str}",
            "date": date_str,
            "topics": topics,
            "tldr": self._tldr(topics, hot, rising, surges),
            "hot_section": self._hot_section(hot),
            "rising_section": self._rising_section(rising),
            "surge_section": self._surge_section(surges, surge_comments),
            "streak_section": self._streak_section(post_streaks, topic_streaks),
            "builder_notes": self._builder_notes(topics, rising, surges, post_streaks, topic_streaks),
        }

    # -- sections --
    def _tldr(
        self,
        topics: list[str],
        hot: list[Post],
        rising: list[tuple[Post, float]],
        surges: list[tuple[Post, float, int]],
    ) -> str:
        top = ", ".join(topics[:3]) if topics else "general chatter"
        parts = [f"Today's Moltbook is buzzing about {top}."]
        if hot:
            parts.append(f"{len(hot)} posts dominate HOT")
        if rising:
            parts.append(f"{len(rising)} are gaining velocity")
        if surges:
            parts.append(f"{len(surges)} threads saw comment surges")
        if len(parts) == 1:
            return parts[0]
        return parts[0] + " " + ", ".join(parts[1:-1]) + (", and " if len(parts) > 2 else " and ") + parts[-1] + "."

    def _hot_section(self, posts: list[Post]) -> str:
        lines = ["HOT Picks"]
        for i, p in enumerate(posts, 1):
            lines.append(
                f"{i}. {p.title}  (engagement {p.engagement:.0f})  {p.url}"
            )
            zh = self._zh(p.title)
            if zh:
                lines.append(zh)
        return "\n".join(lines)

    def _rising_section(self, rising: list[tuple[Post, float]]) -> str:
        lines = ["Rising Now"]
        for i, (p, vel) in enumerate(rising, 1):
            lines.append(
                f"{i}. {p.title}  (velocity {vel:.1f}/h)  {p.url}"
            )
            zh = self._zh(p.title)
            if zh:
                lines.append(zh)
        return "\n".join(lines)

    def _surge_section(
        self, surges: list[tuple[Post, float, int]], comments: dict[str, list[dict]]
    ) -> str:
        lines = ["Comment Surges"]
        for i, (p, vel, delta) in enumerate(surges, 1):
            lines.append(
                f"{i}. {p.title}  (+{delta} comments, {vel:.1f}/h)  {p.url}"
            )
            zh = self._zh(p.title)
            if zh:
                lines.append(zh)
            for c in comments.get(p.id, []):
                lines.append(f'   > {c["author"]}: "{c["text"]}"')
        if len(lines) == 1:
            lines.append("No significant comment surges detected.")
        return "\n".join(lines)

    def _streak_section(
        self, posts: list[dict], topics: list[dict]
    ) -> str:
        lines = ["持续升温（Multi-day Trends）"]
        if posts:
            lines.append("Posts:")
            for s in posts:
                lines.append(
                    f"  • {s['title']}  ({s['streak_days']}-day streak, {s['category']})"
                )
                zh = self._zh(s["title"])
                if zh:
                    lines.append(f"  {zh.strip()}")
        if topics:
            lines.append("Topics:")
            for s in topics:
                lines.append(f"  • {s['topic']}  ({s['frequency']})")
        if not posts and not topics:
            lines.append("No multi-day trends yet (need ≥2 days of data).")
        return "\n".join(lines)

    def _builder_notes(
        self,
        topics: list[str],
        rising: list[tuple[Post, float]],
        surges: list[tuple[Post, float, int]],
        post_streaks: list[dict],
        topic_streaks: list[dict],
    ) -> str:
        notes: list[str] = []

        # actionable signals
        if topic_streaks:
            ts = ", ".join(s["topic"] for s in topic_streaks[:3])
            notes.append(f"Actionable signals: {ts} showing sustained interest — consider building in this space.")

        if post_streaks:
            notes.append(
                f"Distribution plays: {len(post_streaks)} post(s) maintaining multi-day momentum. "
                "Study their format and timing for replication."
            )

        if surges:
            surge_titles = [s[0].title for s in surges[:2]]
            notes.append(
                f"What to copy/steal today: the discussion patterns in "
                f"\"{surge_titles[0]}\"" + (f" and \"{surge_titles[1]}\"" if len(surge_titles) > 1 else "")
                + " — high comment velocity = proven engagement hooks."
            )

        if rising:
            avg_vel = sum(v for _, v in rising) / len(rising)
            notes.append(f"Avg rising velocity: {avg_vel:.1f}/h across top {len(rising)} posts.")

        if not notes:
            notes.append("Steady activity across the board. Monitor tomorrow for emerging patterns.")

        return "\n".join(notes)

    def build_hn_section(
        self,
        hot: list[Post],
        rising: list[tuple[Post, float]],
        topics: list[str],
        post_streaks: list[dict],
    ) -> str:
        """Build the Hacker News section text."""
        all_titles = [p.title for p in hot] + [p.title for p, _ in rising]
        all_titles += [s["title"] for s in post_streaks]
        self._tr.bulk_translate(list(set(all_titles)))

        lines: list[str] = []

        if topics:
            lines.append("Top Themes: " + ", ".join(topics[:4]))
            lines.append("")

        lines.append("HOT Picks")
        for i, p in enumerate(hot, 1):
            lines.append(
                f"{i}. {p.title}  (engagement {p.engagement:.0f})  {p.url}"
            )
            zh = self._zh(p.title)
            if zh:
                lines.append(zh)

        if rising:
            lines.append("")
            lines.append("Rising Now")
            for i, (p, vel) in enumerate(rising, 1):
                lines.append(
                    f"{i}. {p.title}  (velocity {vel:.1f}/h)  {p.url}"
                )
                zh = self._zh(p.title)
                if zh:
                    lines.append(zh)

        if post_streaks:
            lines.append("")
            lines.append("持续升温")
            for s in post_streaks[:3]:
                lines.append(
                    f"  • {s['title']}  ({s['streak_days']}-day streak)"
                )
                zh = self._zh(s["title"])
                if zh:
                    lines.append(f"  {zh.strip()}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Notion client
# ---------------------------------------------------------------------------
class NotionClient:
    BASE = "https://api.notion.com/v1"

    def __init__(self, token: str, database_id: str) -> None:
        self.database_id = database_id
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Notion-Version": "2022-06-28",
            }
        )
        retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    @staticmethod
    def _rich_text(text: str, limit: int = 1900) -> list[dict]:
        """Notion rich_text blocks are limited to 2000 chars each.
        Use 1900 to leave headroom for multi-byte / surrogate chars.
        URLs are turned into clickable links."""
        import re
        url_re = re.compile(r'(https?://\S+)')
        parts = url_re.split(text)
        chunks: list[dict] = []
        for part in parts:
            if not part:
                continue
            if url_re.fullmatch(part):
                # Clickable link segment
                chunks.append({"type": "text", "text": {"content": part, "link": {"url": part}}})
            else:
                # Plain text, chunk if too long
                for i in range(0, len(part), limit):
                    chunks.append({"type": "text", "text": {"content": part[i : i + limit]}})
        return chunks

    def create_page(self, digest: dict) -> str:
        # -- build top_posts combined text --
        moltbook_text = "\n\n".join(
            filter(
                None,
                [
                    digest.get("hot_section", ""),
                    digest.get("rising_section", ""),
                    digest.get("surge_section", ""),
                    digest.get("streak_section", ""),
                ],
            )
        )
        if digest.get("hn_section"):
            top_posts_text = (
                "━━━ Moltbook ━━━\n\n"
                + moltbook_text
                + "\n\n━━━ Hacker News ━━━\n\n"
                + digest["hn_section"]
            )
        else:
            top_posts_text = moltbook_text

        properties: dict[str, Any] = {
            "Name": {"title": self._rich_text(digest["title"])},
            "Date": {"date": {"start": digest["date"]}},
            "Top Topics": {
                "multi_select": [{"name": t[:100]} for t in digest["topics"][:25]]
            },
            "TL;DR": {"rich_text": self._rich_text(digest["tldr"])},
            "Top Posts": {"rich_text": self._rich_text(top_posts_text)},
            "Builder Notes": {"rich_text": self._rich_text(digest["builder_notes"])},
        }

        body = {
            "parent": {"database_id": self.database_id},
            "properties": properties,
        }

        resp = self.session.post(f"{self.BASE}/pages", json=body, timeout=30)

        if resp.status_code == 400:
            detail = resp.json().get("message", resp.text)
            if "property" in detail.lower():
                _die(
                    f"Notion property mismatch: {detail}\n"
                    "Ensure the database has these properties:\n"
                    "  Name (Title), Date (Date), Top Topics (Multi-select),\n"
                    "  TL;DR (Text), Top Posts (Text), Builder Notes (Text)"
                )
            _die(f"Notion API error 400: {detail}")

        resp.raise_for_status()
        page_id = resp.json().get("id", "")
        log.info("Notion page created: %s", page_id)
        return page_id


# ---------------------------------------------------------------------------
# Schedule checker
# ---------------------------------------------------------------------------
class ScheduleChecker:
    def __init__(self) -> None:
        self._tz = ZoneInfo("America/Los_Angeles")

    def should_run(self, last_posted_date: str, *, force: bool = False) -> bool:
        if force:
            return True
        now = datetime.now(self._tz)
        if now.hour != 8 or now.minute >= 30:
            log.info("Not in 8:00-8:29 AM PT window (current: %s). Skipping.", now.strftime("%H:%M"))
            return False
        today = now.strftime("%Y-%m-%d")
        if last_posted_date == today:
            log.info("Already posted for %s. Skipping.", today)
            return False
        return True

    def today(self) -> str:
        return datetime.now(self._tz).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------
def build_snapshot_update(
    existing: dict, posts: list[Post]
) -> dict:
    """Merge current posts into snapshot, preserving first_seen."""
    now_iso = datetime.now(timezone.utc).isoformat()
    updated = dict(existing)
    for p in posts:
        prev = updated.get(p.id, {})
        updated[p.id] = {
            "last_seen": now_iso,
            "last_score": p.score,
            "last_comments": p.comment_count,
            "first_seen": prev.get("first_seen", now_iso),
            "title": p.title,
        }
    return updated


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = Config.from_env()
    cfg.validate()

    cache = CacheManager(cfg.cache_dir)
    scheduler = ScheduleChecker()

    # -- schedule gate --
    state = cache.read_state()
    if not scheduler.should_run(state.get("last_posted_date", ""), force=cfg.dry_run or cfg.force_run):
        sys.exit(0)

    today = scheduler.today()
    log.info("=== Moltbook Daily Digest for %s ===", today)

    # -- fetch posts --
    api = MoltbookAPI(cfg.moltbook_api_key)

    log.info("Fetching HOT (%d), NEW (%d), RISING (%d)…", cfg.hot_limit, cfg.new_limit, cfg.rising_limit)
    hot_posts = api.get_posts("hot", cfg.hot_limit)
    new_posts = api.get_posts("new", cfg.new_limit)
    rising_posts = api.get_posts("rising", cfg.rising_limit)

    log.info("Fetched %d hot, %d new, %d rising posts", len(hot_posts), len(new_posts), len(rising_posts))

    # deduplicate by id for combined pool
    seen_ids: set[str] = set()
    all_posts: list[Post] = []
    for p in hot_posts + rising_posts + new_posts:
        if p.id not in seen_ids:
            seen_ids.add(p.id)
            all_posts.append(p)

    # -- load snapshot --
    snapshot = cache.read_snapshot()

    # -- score & rank --
    scorer = PostScorer()
    top_hot = scorer.rank_hot(all_posts, cfg.top_hot_n)
    top_rising = scorer.rank_rising(all_posts, snapshot, cfg.top_rising_n)
    top_surges = scorer.detect_surges(all_posts, snapshot, cfg.top_surge_n)

    log.info("Ranked: %d hot, %d rising, %d surges", len(top_hot), len(top_rising), len(top_surges))

    # -- optional comment sampling --
    surge_comments: dict[str, list[dict]] = {}
    if cfg.enable_comment_sampling and top_surges:
        for p, _, _ in top_surges:
            comments = api.get_comments(p.id, limit=3)
            if comments:
                surge_comments[p.id] = comments
        log.info("Sampled comments for %d surge posts", len(surge_comments))

    # -- extract topics --
    extractor = TopicExtractor()
    # combine hot + rising posts for topic extraction
    rising_post_objs = [p for p, _ in top_rising]
    topics = extractor.extract(top_hot + rising_post_objs)
    log.info("Topics: %s", topics)

    # -- streak detection --
    history = cache.read_history()
    hot_ids = [p.id for p in top_hot]
    rising_ids = [p.id for p, _ in top_rising]
    surge_ids = [p.id for p, _, _ in top_surges]

    post_streaks = StreakDetector.detect_post_streaks(
        history, hot_ids, rising_ids, snapshot, cfg.top_streak_posts_n
    )
    topic_streaks = StreakDetector.detect_topic_streaks(
        history, topics, cfg.top_streak_topics_n
    )
    log.info("Streaks: %d posts, %d topics", len(post_streaks), len(topic_streaks))

    # -- HN integration --
    moltbook_topics = topics[:]  # preserve for Moltbook history
    hn_all_posts: list[Post] = []
    hn_top_hot: list[Post] = []
    hn_top_rising: list[tuple[Post, float]] = []
    hn_topics: list[str] = []
    hn_post_streaks: list[dict] = []
    hn_snapshot: dict = {}
    hn_history: dict = {}
    hn_hot_ids: list[str] = []
    hn_rising_ids: list[str] = []

    if cfg.enable_hn:
        log.info("Fetching Hacker News top stories…")
        hn_api = HackerNewsAPI()
        hn_all_posts = hn_api.get_top_posts(
            cfg.hn_limit, cfg.hn_min_score, cfg.hn_min_comments
        )

        if hn_all_posts:
            hn_snapshot = cache.read_hn_snapshot()

            hn_top_hot = scorer.rank_hot(hn_all_posts, cfg.top_hn_n)
            hn_top_rising = scorer.rank_rising(hn_all_posts, hn_snapshot, 5)

            hn_topics = extractor.extract(
                hn_top_hot + [p for p, _ in hn_top_rising], max_tags=4
            )

            # Combine topics: Moltbook + HN, deduplicated, max 6
            topics = list(dict.fromkeys(topics + hn_topics))[:6]

            hn_history = cache.read_hn_history()
            hn_hot_ids = [p.id for p in hn_top_hot]
            hn_rising_ids = [p.id for p, _ in hn_top_rising]

            hn_post_streaks = StreakDetector.detect_post_streaks(
                hn_history, hn_hot_ids, hn_rising_ids, hn_snapshot, 3
            )

            log.info(
                "HN ranked: %d hot, %d rising, %d streaks",
                len(hn_top_hot), len(hn_top_rising), len(hn_post_streaks),
            )

    # -- build digest --
    builder = DigestBuilder()
    digest = builder.build(
        {
            "date": today,
            "topics": topics,
            "hot": top_hot,
            "rising": top_rising,
            "surges": top_surges,
            "surge_comments": surge_comments,
            "post_streaks": post_streaks,
            "topic_streaks": topic_streaks,
        }
    )

    # Add HN section if available
    if hn_top_hot:
        digest["hn_section"] = builder.build_hn_section(
            hn_top_hot, hn_top_rising, hn_topics, hn_post_streaks
        )

    # -- dry-run output --
    if cfg.dry_run:
        import io, sys as _sys
        # Force UTF-8 output on Windows consoles
        if _sys.stdout.encoding and _sys.stdout.encoding.lower() != "utf-8":
            _sys.stdout = io.TextIOWrapper(
                _sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  {digest['title']}")
        print(f"{sep}\n")
        print(f"Top Themes: {', '.join(digest['topics'])}\n")
        print(f"TL;DR: {digest['tldr']}\n")
        has_hn = "hn_section" in digest
        if has_hn:
            print(f"{'━' * 20} Moltbook {'━' * 24}\n")
        print(digest["hot_section"])
        print()
        print(digest["rising_section"])
        print()
        print(digest["surge_section"])
        print()
        print(digest["streak_section"])
        if has_hn:
            print(f"\n{'━' * 20} Hacker News {'━' * 21}\n")
            print(digest["hn_section"])
        print()
        print("Builder Notes:")
        print(digest["builder_notes"])
        print(f"\n{sep}")
        log.info("DRY_RUN enabled -- skipping Notion post.")
    else:
        # -- post to Notion --
        notion = NotionClient(cfg.notion_token, cfg.notion_database_id)
        notion.create_page(digest)

    # -- update caches --
    updated_snapshot = build_snapshot_update(snapshot, all_posts)
    cache.write_snapshot(updated_snapshot)

    updated_history = StreakDetector.update_history(
        history, today, hot_ids, rising_ids, surge_ids, moltbook_topics
    )
    cache.write_history(updated_history)

    if hn_all_posts:
        cache.write_hn_snapshot(build_snapshot_update(hn_snapshot, hn_all_posts))
        cache.write_hn_history(
            StreakDetector.update_history(
                hn_history, today, hn_hot_ids, hn_rising_ids, [], hn_topics
            )
        )

    cache.write_state({"last_posted_date": today})

    log.info("Done. Caches updated.")


if __name__ == "__main__":
    main()
