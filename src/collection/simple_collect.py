"""Simple one-file dataset collection from the YouTube Data API v3.

This is the quick version: seed channels -> uploads playlists -> videos ->
filters -> table + thumbnails, with no checkpointing and no resume. It exists
so the team has a dataset to work with now.

The full pipeline described in docs/superpowers/specs/ supersedes this file.

Usage:
    python -m src.collection.simple_collect              # full run
    python -m src.collection.simple_collect --smoke      # 2 channels, cheap
    python -m src.collection.simple_collect --no-thumbs  # table only
"""

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.schema import (  # noqa: E402
    CHANNEL_ID,
    COLUMNS,
    CONTENT_TYPE,
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
THUMB_URL = "https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
MIN_DURATION_SECONDS = 60  # excludes Shorts
QUOTA_BUDGET = 9_000

PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
THUMBS = PROJECT / "data" / "thumbnails"

# Handles rather than channel IDs: the API resolves them, so there are no
# hand-copied IDs to get wrong, and a handle that fails to resolve is reported
# and skipped rather than silently collecting nothing.
#
# Seed list, verified against the API on 2026-09-16: every handle below
# resolved to a real channel above MIN_SUBSCRIBERS. Handle squatters exist
# (@Noclip resolves to a 1,790-sub impostor, not the documentary channel),
# so re-verify with scratchpad/verify_handles.py before adding to this list.
# Replace with the team's own curation; these are a working default.
SEED_HANDLES = {
    "gaming": [
        "@markiplier",                       # 38,900,000
        "@jacksepticeye",                    # 31,200,000
        "@DanTDM",                           # 29,100,000
        "@VanossGaming",                     # 26,000,000
        "@SSundee",                          # 25,500,000
        "@Aphmau",                           # 25,300,000
        "@LazarBeam",                        # 23,300,000
        "@penguinz0",                        # 18,300,000
        "@videogamedunkey",                  # 7,570,000
        "@CouRageJD",                        # 4,650,000
        "@CallMeKevin",                      # 3,660,000
        "@RTGame",                           # 2,960,000
        "@Sykkuno",                          # 2,550,000
        "@NakeyJakey",                       # 2,120,000
        "@GMTK",                             # 1,740,000
        "@SsethTzeentach",                   # 1,620,000
        "@MandaloreGaming",                  # 996,000
        "@Raycevick",                        # 651,000
        "@Whitelight",                       # 564,000
        "@Civvie11",                         # 493,000
        "@Matthewmatosis",                   # 245,000
        "@ThorHighHeels",                    # 161,000
    ],
    "cooking": [
        "@JoshuaWeissman",                   # 10,700,000
        "@bingingwithbabish",                # 10,500,000
        "@BonAppetit",                       # 7,400,000
        "@Epicurious",                       # 6,310,000
        "@JunsKitchen",                      # 5,170,000
        "@FutureCanoe",                      # 4,860,000
        "@FoodWishes",                       # 4,700,000
        "@MythicalKitchen",                  # 4,550,000
        "@TastingHistory",                   # 4,470,000
        "@SamTheCookingGuy",                 # 3,860,000
        "@AmericasTestKitchen",              # 3,020,000
        "@SortedFood",                       # 3,000,000
        "@aragusea",                         # 2,620,000
        "@ThatDudeCanCook",                  # 2,520,000
        "@ChefJeanPierre",                   # 2,470,000
        "@EthanChlebowski",                  # 2,430,000
        "@Frenchguycooking",                 # 2,070,000
        "@BrianLagerstrom",                  # 1,800,000
        "@JKenjiLopezAlt",                   # 1,760,000
        "@NotAnotherCookingShow",            # 1,410,000
        "@ChineseCookingDemystified",        # 1,010,000
        "@internetshaquille",                # 849,000
        "@CharlieAndersonCooking",           # 207,000
    ],
    "finance": [
        "@MarkTilbury",                      # 8,950,000
        "@GrahamStephan",                    # 5,180,000
        "@AndreiJikh",                       # 3,380,000
        "@EconomicsExplained",               # 2,880,000
        "@MinorityMindset",                  # 2,520,000
        "@MeetKevin",                        # 2,050,000
        "@GarysEconomics",                   # 1,660,000
        "@TheFinancialDiet",                 # 1,350,000
        "@NateOBrien",                       # 1,280,000
        "@ThePlainBagel",                    # 1,210,000
        "@WhiteBoardFinance",                # 1,030,000
        "@TheSwedishInvestor",               # 1,020,000
        "@TwoCentsPBS",                      # 826,000
        "@MoneyMacro",                       # 675,000
        "@BravosResearch",                   # 643,000
        "@BenFelixCSI",                      # 641,000
        "@JosephCarlsonShow",                # 511,000
        "@DamienTalksMoney",                 # 414,000
        "@Value-Investing",                  # 270,000
    ],
    "education": [
        "@markrober",                        # 82,600,000
        "@kurzgesagt",                       # 25,600,000
        "@Vsauce",                           # 25,100,000
        "@TEDEd",                            # 22,900,000
        "@veritasium",                       # 21,200,000
        "@crashcourse",                      # 17,200,000
        "@smartereveryday",                  # 11,900,000
        "@NileRed",                          # 10,900,000
        "@ElectroBOOM",                      # 8,820,000
        "@3blue1brown",                      # 8,620,000
        "@SciShow",                          # 8,410,000
        "@MinutePhysics",                    # 5,980,000
        "@TheActionLab",                     # 5,160,000
        "@PracticalEngineeringChannel",      # 4,830,000
        "@numberphile",                      # 4,770,000
        "@tierzoo",                          # 3,940,000
        "@SteveMould",                       # 3,610,000
        "@BranchEducation",                  # 2,750,000
        "@Computerphile",                    # 2,640,000
        "@RealScience",                      # 2,090,000
        "@standupmaths",                     # 1,350,000
        "@Mathologer",                       # 968,000
        "@AppliedScience",                   # 889,000
        "@AlphaPhoenixChannel",              # 723,000
        "@Artemkirsanov",                    # 374,000
        "@Reducible",                        # 337,000
    ],
}

DURATION_RE = re.compile(
    r"^P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$"
)


class QuotaExceeded(RuntimeError):
    """Raised when the run would exceed its unit budget."""


class Client:
    """Minimal API wrapper: counts quota units and retries transient errors."""

    def __init__(self, key, budget=QUOTA_BUDGET):
        self.key = key
        self.budget = budget
        self.units = 0
        self.session = requests.Session()

    def get(self, endpoint, **params):
        if self.units + 1 > self.budget:
            raise QuotaExceeded(f"budget of {self.budget} units reached")
        params["key"] = self.key
        url = f"{API}/{endpoint}"

        for attempt in range(3):
            response = self.session.get(url, params=params, timeout=30)
            self.units += 1

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


def resolve_channels(client, seeds):
    """Handles to channel records. Unresolvable handles are reported, not fatal."""
    channels = {}
    for content_type, handles in seeds.items():
        for handle in handles:
            payload = client.get(
                "channels", part="snippet,contentDetails,statistics", forHandle=handle
            )
            items = (payload or {}).get("items") or []
            if not items:
                print(f"  ! {handle} did not resolve, skipping")
                continue
            item = items[0]
            subs = item["statistics"].get("subscriberCount")
            channels[item["id"]] = {
                "handle": handle,
                "content_type": content_type,
                "uploads": item["contentDetails"]["relatedPlaylists"]["uploads"],
                "subscribers": int(subs) if subs else 0,
                "title": item["snippet"]["title"],
            }
            print(f"  {handle:<28} {item['snippet']['title'][:32]:<34} "
                  f"{int(subs) if subs else 0:>12,} subs")
    return channels


def discover_videos(client, channel, window_start, cap=None):
    """Page a uploads playlist newest-first, stopping once past the window."""
    video_ids, page_token = [], None
    while True:
        payload = client.get(
            "playlistItems",
            part="contentDetails",
            playlistId=channel["uploads"],
            maxResults=50,
            pageToken=page_token,
        )
        if not payload:
            break

        stop = False
        for item in payload.get("items", []):
            published = item["contentDetails"].get("videoPublishedAt")
            if not published:
                continue
            when = datetime.fromisoformat(published.replace("Z", "+00:00"))
            if when < window_start:
                stop = True  # newest-first, so everything after is older too
                break
            video_ids.append(item["contentDetails"]["videoId"])

        if cap and len(video_ids) >= cap:
            return video_ids[:cap]
        page_token = payload.get("nextPageToken")
        if stop or not page_token:
            break
    return video_ids


def fetch_videos(client, video_ids):
    """videos.list in batches of 50, the maximum the API accepts."""
    items = []
    for start in range(0, len(video_ids), 50):
        batch = video_ids[start:start + 50]
        payload = client.get(
            "videos", part="snippet,statistics,contentDetails", id=",".join(batch)
        )
        if payload:
            items.extend(payload.get("items", []))
    return items


def build_rows(items, channels, now):
    """Apply the filters and project onto schema.COLUMNS."""
    rows = []
    dropped = {"no_views": 0, "short": 0, "out_of_window": 0, "low_subs": 0,
               "unknown_channel": 0, "bad_duration": 0, "live_or_upcoming": 0}

    for item in items:
        channel_id = item["snippet"]["channelId"]
        channel = channels.get(channel_id)
        if not channel:
            dropped["unknown_channel"] += 1
            continue
        if channel["subscribers"] < MIN_SUBSCRIBERS:
            dropped["low_subs"] += 1
            continue

        # Live broadcasts and scheduled premieres report duration "P0D" and have
        # view dynamics unlike on-demand video. Excluded for the same reason as
        # Shorts. Checked before parsing so a genuine parse bug stays visible.
        if item["snippet"].get("liveBroadcastContent", "none") != "none":
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
        if seconds < MIN_DURATION_SECONDS:
            dropped["short"] += 1
            continue

        published = datetime.fromisoformat(
            item["snippet"]["publishedAt"].replace("Z", "+00:00")
        )
        age_days = (now - published).days
        if not (MIN_DAYS_SINCE_PUBLISH <= age_days <= MAX_DAYS_SINCE_PUBLISH):
            dropped["out_of_window"] += 1
            continue

        video_id = item["id"]
        rows.append({
            VIDEO_ID: video_id,
            TITLE: item["snippet"]["title"],
            CHANNEL_ID: channel_id,
            SUBSCRIBER_COUNT: channel["subscribers"],
            VIEW_COUNT: int(views),
            PUBLISHED_AT: item["snippet"]["publishedAt"],
            CONTENT_TYPE: channel["content_type"],
            THUMBNAIL_FILE: THUMBNAIL_PATTERN.format(video_id=video_id),
        })
    return rows, dropped


def download_thumbnails(rows):
    """hqdefault for every row. Plain HTTPS, costs no API quota."""
    THUMBS.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    got = failed = skipped = 0

    for index, row in enumerate(rows, 1):
        target = THUMBS / row[THUMBNAIL_FILE]
        if target.exists():
            skipped += 1
            continue
        try:
            response = session.get(
                THUMB_URL.format(video_id=row[VIDEO_ID]), timeout=30
            )
            if response.status_code == 200 and response.content:
                target.write_bytes(response.content)
                got += 1
            else:
                row[THUMBNAIL_FILE] = ""  # keep the row, title model still uses it
                failed += 1
        except requests.RequestException:
            row[THUMBNAIL_FILE] = ""
            failed += 1
        if index % 250 == 0:
            print(f"  {index}/{len(rows)} thumbnails")
    return got, failed, skipped


def write_table(rows):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    csv_path = PROCESSED / "videos.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    written = [csv_path]

    try:  # parquet preserves dtypes; CSV is the portable fallback
        import pandas as pd
        parquet_path = PROCESSED / "videos.parquet"
        pd.DataFrame(rows, columns=COLUMNS).to_parquet(parquet_path, index=False)
        written.append(parquet_path)
    except Exception as error:
        print(f"  (parquet skipped: {error})")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="2 channels, 50 videos each, to verify the key cheaply")
    parser.add_argument("--no-thumbs", action="store_true",
                        help="build the table without downloading images")
    parser.add_argument("--budget", type=int, default=QUOTA_BUDGET)
    args = parser.parse_args()

    load_dotenv(PROJECT / ".env")
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise SystemExit(
            "YOUTUBE_API_KEY is not set.\n"
            "  cp .env.example .env    then paste your key into it.\n"
            "  Get one at https://console.cloud.google.com -> enable\n"
            "  'YouTube Data API v3' -> Credentials -> Create API key."
        )

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=MAX_DAYS_SINCE_PUBLISH)
    window_end = now - timedelta(days=MIN_DAYS_SINCE_PUBLISH)

    seeds = SEED_HANDLES
    cap = None
    if args.smoke:
        first = next(iter(SEED_HANDLES))
        seeds = {first: SEED_HANDLES[first][:2]}
        cap = 50

    client = Client(key, budget=args.budget)
    RAW.mkdir(parents=True, exist_ok=True)

    print(f"Window: {window_start.date()} to {window_end.date()} "
          f"({MIN_DAYS_SINCE_PUBLISH}-{MAX_DAYS_SINCE_PUBLISH} days old)")
    print("\nResolving channels")
    channels = resolve_channels(client, seeds)
    if not channels:
        raise SystemExit("No channels resolved. Check the handles in SEED_HANDLES.")

    print(f"\nDiscovering videos from {len(channels)} channels")
    video_ids = []
    try:
        for channel in channels.values():
            found = discover_videos(client, channel, window_start, cap)
            video_ids.extend(found)
            print(f"  {channel['handle']:<28} {len(found):>5} in window")
    except QuotaExceeded as error:
        print(f"\n! {error} during discovery, continuing with what we have")

    video_ids = sorted(set(video_ids))
    print(f"\n{len(video_ids):,} unique video ids")

    print("Fetching video details")
    try:
        items = fetch_videos(client, video_ids)
    except QuotaExceeded as error:
        raise SystemExit(f"{error}. Quota resets at midnight Pacific.")

    (RAW / "videos_raw.jsonl").write_text(
        "\n".join(__import__("json").dumps(i) for i in items), encoding="utf-8"
    )

    rows, dropped = build_rows(items, channels, now)
    print(f"\n{len(rows):,} rows kept. Dropped: "
          + ", ".join(f"{k} {v}" for k, v in dropped.items() if v))

    if not rows:
        raise SystemExit("Nothing survived the filters. Check the window and seeds.")

    if not args.no_thumbs:
        print(f"\nDownloading {len(rows):,} thumbnails")
        got, failed, skipped = download_thumbnails(rows)
        print(f"  {got} downloaded, {skipped} already present, {failed} failed")

    written = write_table(rows)
    print(f"\nWrote {', '.join(str(p.relative_to(PROJECT)) for p in written)}")
    print(f"Quota spent: {client.units} units of {args.budget}")


if __name__ == "__main__":
    main()
