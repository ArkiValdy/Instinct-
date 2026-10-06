"""Search-based collection from the YouTube Data API v3, one content type per run.

search.list (paged) -> video ids -> videos.list -> channels.list -> filters ->
data/raw/youtube_api_{content_type}.parquet, with columns from config/schema.py.

Quota costs that shape the design (daily allowance 10,000 units per project,
reset at midnight Pacific):

    search.list    100 units, at most 50 video ids per call
    videos.list      1 unit,  up to 50 ids per call
    channels.list    1 unit,  up to 50 ids per call

So search pages are the only expensive call. Every search page is followed by
one videos.list for its new ids and at most one channels.list for channels not
seen before, which means roughly 102 units per 50 candidate videos.

The publish window is passed to the API as publishedAfter / publishedBefore,
so quota is never spent on videos the window would discard. Subscriber count
cannot be filtered at search time, so that filter runs after channels.list.

Progress is written to disk after every search page. A rerun with the same
content type resumes from the saved query and pageToken instead of restarting.

Usage:
    python -m src.collection.youtube_api cooking --target 1500
    python -m src.collection.youtube_api cooking --target 1500 --budget 4000
    python -m src.collection.youtube_api cooking --fresh   # discard saved progress
    python -m src.collection.youtube_api cooking --target 1500 --durations medium,long
    python -m src.collection.youtube_api cooking --refilter  # re-apply filters to saved rows
    python -m src.collection.youtube_api cooking --thumbnails-only

Thumbnails go to data/thumbnails/ via simple_collect.download_thumbnails:
plain HTTPS against i.ytimg.com, no quota. Already-present files are skipped.
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.collection.simple_collect import download_thumbnails  # noqa: E402
from config.schema import (  # noqa: E402
    CHANNEL_ID,
    COLUMNS,
    CONTENT_TYPE,
    CONTENT_TYPES,
    MAX_DAYS_SINCE_PUBLISH,
    MIN_DAYS_SINCE_PUBLISH,
    MIN_SUBSCRIBERS,
    PUBLISHED_AT,
    SUBSCRIBER_COUNT,
    THUMBNAIL_FILE,
    THUMBNAIL_PATTERN,
    TITLE,
    VIDEO_ID,
    VIEW_COUNT,
)

API = "https://www.googleapis.com/youtube/v3"
COST = {"search": 100, "videos": 1, "channels": 1}
BATCH = 50  # the most ids videos.list and channels.list accept, and search's page size
DAILY_QUOTA = 10_000
QUOTA_BUDGET = 9_000  # headroom under the daily cap for manual exploration on the key
# Shorts can run up to three minutes, so a 60 second cutoff lets most through.
# This also drops genuine horizontal videos of three minutes or less, which is
# the price of keeping the Shorts view distribution out of the target.
SHORTS_MAX_SECONDS = 180
# relevanceLanguage="en" is a ranking hint, not a filter. Titles where fewer
# than this share of letters are Latin script are dropped. This catches Tamil,
# Hindi, Thai and so on, but not romanised non-English text, which stays.
MIN_LATIN_LETTER_SHARE = 0.8
SLICE_DAYS = 7  # width of each publish-date slice searched separately
PAGES_PER_TASK = 1  # search pages per (query, slice); more concentrates on slice ends

PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "data" / "raw"

# search.list stops paging at roughly 500 results per query however you page
# it, so one query cannot reach a 1,500 row target. Each content type gets
# several queries, worked through in order, with video ids deduplicated across
# them. Collection-owned, like the seed list, so schema.py stays untouched.
QUERIES = {
    "gaming": [
        "gaming", "gameplay", "let's play", "game review", "video game",
        "minecraft", "speedrun", "gaming news", "indie game", "game guide",
    ],
    "cooking": [
        "cooking", "recipe", "how to cook", "easy dinner recipe", "baking",
        "meal prep", "chef cooking", "home cooking", "dessert recipe",
        "cooking tips",
    ],
    "finance": [
        "personal finance", "investing", "stock market", "money tips",
        "budgeting", "financial advice", "real estate investing", "economy",
        "retirement planning", "passive income",
    ],
    "education": [
        "science explained", "educational", "math explained", "physics",
        "history documentary", "how it works", "engineering explained",
        "biology", "chemistry", "learn",
    ],
}

DURATION_RE = re.compile(
    r"^P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$"
)


class QuotaExceeded(RuntimeError):
    """Raised when the next call would exceed the run's unit budget."""


class Client:
    """API wrapper that charges each endpoint its real unit cost."""

    def __init__(self, key, budget=QUOTA_BUDGET):
        self.key = key
        self.budget = budget
        self.units = 0
        self.session = requests.Session()

    def get(self, endpoint, **params):
        cost = COST[endpoint]
        params["key"] = self.key
        url = f"{API}/{endpoint}"

        for attempt in range(3):
            # checked per attempt, because a failed request still spends quota
            if self.units + cost > self.budget:
                raise QuotaExceeded(
                    f"{endpoint} costs {cost}, which would pass the budget "
                    f"({self.units} of {self.budget} spent)"
                )
            self.units += cost
            try:
                response = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as error:  # reset, timeout: retry
                print(f"  ! {endpoint}: {type(error).__name__}, retrying",
                      file=sys.stderr)
                time.sleep(2 ** attempt)
                continue

            if response.status_code == 200:
                return response.json()

            if response.status_code == 403:
                body = response.json().get("error", {})
                reason = (body.get("errors") or [{}])[0].get("reason", "")
                if reason in ("quotaExceeded", "dailyLimitExceeded"):
                    raise QuotaExceeded("API returned quotaExceeded")
                raise SystemExit(
                    f"403 {reason}: the key is wrong or the YouTube Data API "
                    f"is not enabled on its project.\n{body.get('message', '')}"
                )

            if response.status_code in (401, 400):
                raise SystemExit(f"{response.status_code}: {response.text[:300]}")

            if attempt < 2:  # 5xx and anything else: back off and retry
                time.sleep(2 ** attempt)

        print(f"  ! giving up on {endpoint} after 3 attempts", file=sys.stderr)
        return None


def parse_duration(value):
    """ISO 8601 duration to seconds. Returns None if unparseable."""
    match = DURATION_RE.match(value or "")
    if not match:
        return None
    days, hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def latin_share(title):
    """Share of the title's letters that are Latin script. Emoji, digits and
    punctuation are not letters, so they do not count either way."""
    letters = [c for c in title if c.isalpha()]
    if not letters:
        return 1.0
    latin = sum(1 for c in letters if unicodedata.name(c, "").startswith("LATIN"))
    return latin / len(letters)


def to_rfc3339(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


# --- progress on disk -------------------------------------------------------

def paths_for(content_type):
    return (
        RAW / f"youtube_api_{content_type}.parquet",
        RAW / f"youtube_api_{content_type}_progress.json",
    )


def plan_tasks(queries, start, end, durations=None):
    """Split the window into weekly slices and pair each with a query.

    Results are ordered by date, newest first, and a query stops paging at
    roughly 500 results. Over the whole window, a broad query would return
    only the last few days before publishedBefore. Searching each week
    separately spreads the sample across the window.

    Ordered so the first len(slices) tasks visit every week once, with the
    query rotating, so a run that stops early is still spread across the
    window rather than concentrated at one end.

    With durations (search.list videoDuration values: "medium" is 4-20
    minutes, "long" over 20), each (query, week) is searched once per
    duration, so Shorts never cost a search page. Each week gets all its
    durations before the next week starts.
    """
    slices = []
    cursor = end
    while cursor > start:
        slices.append((max(start, cursor - timedelta(days=SLICE_DAYS)), cursor))
        cursor -= timedelta(days=SLICE_DAYS)

    tasks = []
    for k in range(len(queries) * len(slices)):
        after, before = slices[k % len(slices)]
        query = queries[(k // len(slices) + k) % len(queries)]
        for duration in durations or [None]:
            tasks.append({"query": query, "after": to_rfc3339(after),
                          "before": to_rfc3339(before), "duration": duration})
    return tasks


def replan(state, queries, durations, now):
    """Replace the remaining tasks with a plan for new durations. Rows, seen
    ids and the channel cache are kept, so nothing collected is fetched again.
    The window is recalculated so new rows are 30-365 days old today."""
    start = now - timedelta(days=MAX_DAYS_SINCE_PUBLISH)
    end = now - timedelta(days=MIN_DAYS_SINCE_PUBLISH)
    state.update({
        "published_after": to_rfc3339(start),
        "published_before": to_rfc3339(end),
        "tasks": plan_tasks(queries, start, end, durations),
        "durations": durations,
        "task_index": 0,
        "task_pages": 0,
        "page_token": None,
    })


def new_state(content_type, queries, now):
    """The tasks, and so the window, are fixed at the first run and reused on
    resume, because a pageToken is only valid for the exact query parameters
    that produced it."""
    start = now - timedelta(days=MAX_DAYS_SINCE_PUBLISH)
    end = now - timedelta(days=MIN_DAYS_SINCE_PUBLISH)
    return {
        "content_type": content_type,
        "published_after": to_rfc3339(start),
        "published_before": to_rfc3339(end),
        "tasks": plan_tasks(queries, start, end),
        "task_index": 0,
        "task_pages": 0,
        "page_token": None,
        "pages_done": 0,
        "units_total": 0,
        "seen_video_ids": [],
        "channels": {},  # channel id -> subscriber count, None if hidden or missing
        "dropped": {"low_subs": 0, "hidden_subs": 0, "no_views": 0, "short": 0,
                    "bad_duration": 0, "live_or_upcoming": 0, "missing": 0,
                    "non_latin": 0},
    }


def write_atomic(path, write):
    """Write to a temp file then rename, so an interrupt never leaves half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def save(state, rows, table_path, state_path):
    write_atomic(
        table_path,
        lambda p: pd.DataFrame(rows, columns=COLUMNS).to_parquet(p, index=False),
    )
    write_atomic(
        state_path,
        lambda p: p.write_text(json.dumps(state, indent=1), encoding="utf-8"),
    )


def load(content_type, queries, now, fresh):
    table_path, state_path = paths_for(content_type)
    if fresh or not state_path.exists():
        return new_state(content_type, queries, now), []
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["dropped"].setdefault("non_latin", 0)
    rows = []
    if table_path.exists():
        rows = pd.read_parquet(table_path).to_dict("records")
    return state, rows


# --- API stages -------------------------------------------------------------

def search_page(client, task, state):
    """One search.list page: up to 50 video ids and the next pageToken.

    order="date" rather than the default relevance ranking, which is itself a
    function of engagement and would bias the sample towards the target.
    """
    params = dict(
        part="id",
        q=task["query"],
        type="video",
        order="date",
        relevanceLanguage="en",
        publishedAfter=task["after"],
        publishedBefore=task["before"],
        maxResults=BATCH,
    )
    if task.get("duration"):
        params["videoDuration"] = task["duration"]
    if state["page_token"]:
        params["pageToken"] = state["page_token"]
    payload = client.get("search", **params) or {}
    ids = [item["id"]["videoId"] for item in payload.get("items", [])
           if item.get("id", {}).get("videoId")]
    return ids, payload.get("nextPageToken")


def fetch_videos(client, video_ids):
    """videos.list in batches of 50."""
    items = []
    for start in range(0, len(video_ids), BATCH):
        batch = video_ids[start:start + BATCH]
        payload = client.get(
            "videos", part="snippet,statistics,contentDetails", id=",".join(batch)
        )
        if payload:
            items.extend(payload.get("items", []))
    return items


def fetch_channels(client, channel_ids, known):
    """channels.list for ids not already in `known`, in batches of 50.

    Updates `known` in place. A channel the API does not return, or one that
    hides its subscriber count, is recorded as None so it is never re-requested.
    """
    wanted = sorted(set(channel_ids) - set(known))
    for start in range(0, len(wanted), BATCH):
        batch = wanted[start:start + BATCH]
        payload = client.get("channels", part="statistics", id=",".join(batch))
        if payload is None:
            continue  # transient failure: leave unknown so a later page retries
        found = {}
        for item in payload.get("items", []):
            stats = item.get("statistics", {})
            subs = stats.get("subscriberCount")
            hidden = stats.get("hiddenSubscriberCount", False)
            found[item["id"]] = None if hidden or subs is None else int(subs)
        for channel_id in batch:
            known[channel_id] = found.get(channel_id)


def build_rows(items, state):
    """Join videos to subscriber counts, apply the post-search filters, and
    project onto schema.COLUMNS. The publish window is already applied by the
    API, so it is not re-checked here."""
    rows, dropped, channels = [], state["dropped"], state["channels"]

    for item in items:
        snippet = item["snippet"]
        channel_id = snippet["channelId"]

        if channel_id not in channels:
            dropped["missing"] += 1
            continue
        subs = channels[channel_id]
        if subs is None:
            dropped["hidden_subs"] += 1
            continue
        if subs < MIN_SUBSCRIBERS:
            dropped["low_subs"] += 1
            continue

        # Live broadcasts and premieres report duration "P0D" and have view
        # dynamics unlike on-demand video.
        if snippet.get("liveBroadcastContent", "none") != "none":
            dropped["live_or_upcoming"] += 1
            continue

        views = item.get("statistics", {}).get("viewCount")
        if views is None:
            dropped["no_views"] += 1
            continue

        seconds = parse_duration(item.get("contentDetails", {}).get("duration"))
        if seconds is None:
            dropped["bad_duration"] += 1
            continue
        if seconds <= SHORTS_MAX_SECONDS:
            dropped["short"] += 1
            continue

        if latin_share(snippet["title"]) < MIN_LATIN_LETTER_SHARE:
            dropped["non_latin"] += 1
            continue

        video_id = item["id"]
        rows.append({
            VIDEO_ID: video_id,
            TITLE: snippet["title"],
            CHANNEL_ID: channel_id,
            SUBSCRIBER_COUNT: subs,
            VIEW_COUNT: int(views),
            PUBLISHED_AT: snippet["publishedAt"],
            CONTENT_TYPE: state["content_type"],
            THUMBNAIL_FILE: THUMBNAIL_PATTERN.format(video_id=video_id),
        })
    return rows


def collect(client, state, rows, target, pages_per_task, table_path, state_path):
    """Work through the tasks until `target` rows are kept, the tasks run out,
    or the budget is reached. Saves after every search page."""
    seen = set(state["seen_video_ids"])
    tasks = state["tasks"]

    while len(rows) < target and state["task_index"] < len(tasks):
        task = tasks[state["task_index"]]
        ids, next_token = search_page(client, task, state)

        new_ids = [i for i in dict.fromkeys(ids) if i not in seen]
        items = fetch_videos(client, new_ids)
        fetch_channels(client, [i["snippet"]["channelId"] for i in items],
                       state["channels"])
        kept = build_rows(items, state)

        # Ids that videos.list did not return (deleted, private) are still
        # marked seen: they would cost a detail call again for nothing.
        seen.update(new_ids)
        rows.extend(kept)

        state["pages_done"] += 1
        state["task_pages"] += 1
        if next_token and state["task_pages"] < pages_per_task:
            state["page_token"] = next_token
        else:
            state["task_index"] += 1
            state["task_pages"] = 0
            state["page_token"] = None
        state["seen_video_ids"] = sorted(seen)
        state["units_total"] = state["units_run_start"] + client.units
        save(state, rows, table_path, state_path)

        print(f"  [{task['query'][:18]:<18} wk {task['after'][:10]} "
              f"{(task.get('duration') or 'any'):<6}] "
              f"page {state['pages_done']:>3}: "
              f"{len(ids):>2} ids, {len(new_ids):>2} new, {len(kept):>2} kept | "
              f"rows {len(rows):>5}/{target} | "
              f"quota {client.units:>5} this run, {state['units_total']:>6} total")

    return rows


def refilter(client, state, table_path, state_path):
    """Re-apply the duration and language filters to rows already collected.

    Duration is not a schema column, so it is re-fetched: one videos.list call
    per 50 rows. Used once, when the Shorts cutoff rose from 60s to 180s.
    """
    rows = pd.read_parquet(table_path).to_dict("records")
    durations = {}
    ids = [r[VIDEO_ID] for r in rows]
    for start in range(0, len(ids), BATCH):
        payload = client.get("videos", part="contentDetails",
                             id=",".join(ids[start:start + BATCH])) or {}
        for item in payload.get("items", []):
            durations[item["id"]] = parse_duration(item["contentDetails"].get("duration"))

    kept, dropped = [], state["dropped"]
    for row in rows:
        seconds = durations.get(row[VIDEO_ID])
        if seconds is None:
            dropped["bad_duration"] += 1  # removed since collection, or unparseable
        elif seconds <= SHORTS_MAX_SECONDS:
            dropped["short"] += 1
        elif latin_share(row[TITLE]) < MIN_LATIN_LETTER_SHARE:
            dropped["non_latin"] += 1
        else:
            kept.append(row)

    state["units_total"] += client.units
    save(state, kept, table_path, state_path)
    print(f"Refiltered: {len(rows):,} -> {len(kept):,} rows, {client.units} units. "
          f"Dropped so far: " + ", ".join(f"{k} {v}" for k, v in dropped.items() if v))


def thumbnails(table_path):
    """Download thumbnails for every row. A failed download keeps its row with
    an empty thumbnail_file, since the title model can still use it."""
    rows = pd.read_parquet(table_path).to_dict("records")
    print(f"\nDownloading thumbnails for {len(rows):,} rows")
    got, failed, skipped = download_thumbnails(rows)
    print(f"  {got} downloaded, {skipped} already present, {failed} failed")
    write_atomic(
        table_path,
        lambda p: pd.DataFrame(rows, columns=COLUMNS).to_parquet(p, index=False),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("content_type", choices=sorted(QUERIES))
    parser.add_argument("--target", type=int, default=1500,
                        help="stop once this many rows survive the filters")
    parser.add_argument("--budget", type=int, default=QUOTA_BUDGET,
                        help="units this run may spend (daily cap is 10,000)")
    parser.add_argument("--durations", type=lambda v: v.split(","), default=None,
                        help="videoDuration values to search, e.g. medium,long. "
                             "Differing from the saved plan replans the "
                             "remaining tasks and keeps collected rows")
    parser.add_argument("--pages-per-task", type=int, default=PAGES_PER_TASK,
                        help="search pages per (query, week) before moving on")
    parser.add_argument("--no-thumbs", action="store_true",
                        help="skip downloading thumbnails after collection")
    parser.add_argument("--refilter", action="store_true",
                        help="re-apply duration and language filters to saved rows")
    parser.add_argument("--thumbnails-only", action="store_true",
                        help="download thumbnails for the saved table, no API calls")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore saved progress and start again")
    args = parser.parse_args()

    if args.content_type not in CONTENT_TYPES:
        print(f"  ! {args.content_type} is not in schema.CONTENT_TYPES")

    table_path, state_path = paths_for(args.content_type)
    if args.thumbnails_only:
        thumbnails(table_path)
        return

    load_dotenv(PROJECT / ".env")
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise SystemExit("YOUTUBE_API_KEY is not set. Copy .env.example to .env.")

    RAW.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    state, rows = load(args.content_type, QUERIES[args.content_type], now, args.fresh)
    state["units_run_start"] = state["units_total"]
    if args.durations and args.durations != state.get("durations"):
        replan(state, QUERIES[args.content_type], args.durations, now)
        print(f"Replanned for durations {args.durations}: "
              f"{len(state['tasks'])} tasks, keeping {len(rows)} rows")

    if args.refilter:
        refilter(Client(key, budget=args.budget), state, table_path, state_path)
        return

    resumed = state["pages_done"] > 0
    print(f"{'Resuming' if resumed else 'Starting'} {args.content_type}: "
          f"{len(rows)} rows, {state['pages_done']} pages done, "
          f"task {state['task_index'] + 1} of {len(state['tasks'])} (query x week)")
    print(f"Window (fixed at first run): {state['published_after']} to "
          f"{state['published_before']}")
    print(f"Budget this run: {args.budget} units. Each search page costs "
          f"{COST['search']} plus ~2 for details.\n")

    client = Client(key, budget=args.budget)
    stopped = None
    try:
        rows = collect(client, state, rows, args.target, args.pages_per_task,
                       table_path, state_path)
    except QuotaExceeded as error:
        stopped = str(error)

    subs = [r[SUBSCRIBER_COUNT] for r in rows]
    print(f"\n{len(rows):,} rows in {table_path.relative_to(PROJECT)}")
    if rows:
        print(f"  subscribers {min(subs):,} to {max(subs):,}, "
              f"{len(set(r[CHANNEL_ID] for r in rows)):,} channels")
    print("  dropped: " + ", ".join(f"{k} {v}" for k, v in state["dropped"].items() if v))
    print(f"Quota: {client.units} units this run, {state['units_total']} across all "
          f"runs of this content type (daily cap {DAILY_QUOTA:,})")

    if stopped:
        print(f"\nStopped: {stopped}. Progress is saved; rerun the same command "
              f"to resume. Quota resets at midnight Pacific.")
    elif len(rows) < args.target:
        print(f"\nTasks exhausted before the target. Add queries to "
              f"QUERIES[{args.content_type!r}] and rerun to continue.")

    if rows and not args.no_thumbs:
        thumbnails(table_path)


if __name__ == "__main__":
    main()
