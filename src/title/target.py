"""Target variable for the title model.

views_per_sub      view count / subscriber count. Dividing by channel size is
                   meant to remove the channel-size confound, so the target
                   measures how far a video reached relative to its audience.
log_views_per_sub  ln(1 + views_per_sub). The raw ratio is heavily right-skewed:
                   an earlier ten-row sample spanned three orders of magnitude
                   with the top value eleven times the second, so under squared
                   error one viral video outweighs everything else combined.
log_views_vs_channel
                   ln(1 + views) minus its mean over the video's own channel:
                   how far this video did against that channel's typical
                   video. Subscribers are constant within a channel, so this
                   is the log of views per subscriber, re-centred per channel.
                   It is ln(1 + views) rather than log_views_per_sub because
                   the seed channels sit near 0.05 views per subscriber, where
                   ln(1 + ratio) is close to the ratio itself and leaves the
                   skew in. It only means something where a channel has many
                   videos, so it is for the seed source.

Two sources, with the same columns:

    search  data/raw/youtube_api_{content_type}.parquet, from search.list in
            date order. Many channels, about 1.4 videos each.
    seed    data/processed/videos.parquet, from the uploads playlists of the
            seed channels in all four content types. Few channels, dozens of
            videos each. Collected with a 60 second Shorts cutoff and no
            script filter, so the search source's filters are re-applied here
            on load, using the durations kept in data/raw/videos_raw.jsonl.

The analysis checks whether the ratio actually removes channel size: if either
target still correlates strongly with subscriber count, it has not.

Usage:
    python -m src.title.target cooking
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.schema import (  # noqa: E402
    CHANNEL_ID,
    CONTENT_TYPE,
    SUBSCRIBER_COUNT,
    TITLE,
    VIDEO_ID,
    VIEW_COUNT,
)
from src.collection.youtube_api import (  # noqa: E402
    MIN_LATIN_LETTER_SHARE,
    SHORTS_MAX_SECONDS,
    latin_share,
    parse_duration,
)

PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"

RAW_TARGET = "views_per_sub"
LOG_TARGET = "log_views_per_sub"
CHANNEL_TARGET = "log_views_vs_channel"
TARGETS = [RAW_TARGET, LOG_TARGET]

SOURCES = ["search", "seed"]
ALL = "all"  # content_type value that keeps every content type in the seed source

# not a schema column: only the seed source keeps it
DURATION = "duration_seconds"

# chart tokens: one series, so one hue; text in ink, not the series colour
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"


def load_dataset(content_type, source="search"):
    if source == "search":
        return pd.read_parquet(RAW / f"youtube_api_{content_type}.parquet")
    if source == "seed":
        return load_seed(content_type)
    raise ValueError(f"unknown source {source!r}, expected one of {SOURCES}")


def seed_durations():
    """video_id -> duration in seconds, from the raw API responses."""
    durations = {}
    with open(RAW / "videos_raw.jsonl", encoding="utf-8") as lines:
        for line in lines:
            item = json.loads(line)
            durations[item["id"]] = parse_duration(
                item.get("contentDetails", {}).get("duration"))
    return durations


def load_seed(content_type=ALL):
    """The seed channel dataset with the search source's filters applied:
    longer than SHORTS_MAX_SECONDS and mostly Latin script."""
    data = pd.read_parquet(PROCESSED / "videos.parquet")
    if content_type != ALL:
        data = data[data[CONTENT_TYPE] == content_type]
    data = data.assign(**{DURATION: data[VIDEO_ID].map(seed_durations())})
    keep = (data[DURATION] > SHORTS_MAX_SECONDS) & (
        data[TITLE].map(latin_share) >= MIN_LATIN_LETTER_SHARE)
    return data[keep].reset_index(drop=True)


def add_targets(dataset):
    """Return a copy of the dataset with the target columns added."""
    out = dataset.copy()
    out[RAW_TARGET] = out[VIEW_COUNT] / out[SUBSCRIBER_COUNT]
    out[LOG_TARGET] = np.log1p(out[RAW_TARGET])
    log_views = np.log1p(out[VIEW_COUNT])
    out[CHANNEL_TARGET] = log_views - log_views.groupby(out[CHANNEL_ID]).transform("mean")
    return out


def describe(data):
    """min, max, median, mean and skew for each target."""
    return pd.DataFrame({
        t: {
            "min": data[t].min(),
            "max": data[t].max(),
            "median": data[t].median(),
            "mean": data[t].mean(),
            "skew": data[t].skew(),
        }
        for t in TARGETS
    }).T


def subscriber_correlations(data):
    """Correlation of subscriber count with each target.

    Pearson on raw subscriber count is dominated by the few largest channels,
    so Pearson on log subscribers and Spearman (rank) are reported alongside.
    The Spearman figure is the one to trust: it is the same for the raw and
    log targets, because log1p preserves rank.
    """
    subs = data[SUBSCRIBER_COUNT]
    log_subs = np.log10(subs)
    return pd.DataFrame({
        t: {
            "pearson_subs": data[t].corr(subs),
            "pearson_log_subs": data[t].corr(log_subs),
            "spearman": data[t].corr(subs, method="spearman"),
        }
        for t in TARGETS
    }).T


def histogram(values, title, xlabel, path):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.hist(values, bins=50, color=SERIES, edgecolor=SURFACE, linewidth=1)
    median = np.median(values)
    ax.axvline(median, color=INK, linewidth=1, linestyle="--")
    ax.annotate(f"median {median:.2f}", xy=(median, ax.get_ylim()[1] * 0.92),
                xytext=(6, 0), textcoords="offset points", color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel("videos", color=INK_MUTED, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d0cfca")
    ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("content_type")
    args = parser.parse_args()

    data = add_targets(load_dataset(args.content_type))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    n = len(data)

    print(f"{args.content_type}: {n:,} videos\n")
    print("Distributions")
    print(describe(data).round(3).to_string())
    print("\nCorrelation with subscriber count")
    print(subscriber_correlations(data).round(3).to_string())

    paths = []
    for target, title, xlabel in [
        (RAW_TARGET, f"Views per subscriber, {args.content_type} (n={n:,})",
         "views / subscribers"),
        (LOG_TARGET, f"log(1 + views per subscriber), {args.content_type} (n={n:,})",
         "ln(1 + views / subscribers)"),
    ]:
        path = PROCESSED / f"target_{target}_{args.content_type}.png"
        histogram(data[target], title, xlabel, path)
        paths.append(path)
    print("\nHistograms: " + ", ".join(str(p.relative_to(PROJECT)) for p in paths))


if __name__ == "__main__":
    main()
