"""Context features: facts about a video that are not its title.

The title feature table holds features of the title string alone. These sit
in a separate table on purpose, because each one is a modelling decision the
title feature dictionary rules out by default:

    days_since_publish   age at collection. Views keep accumulating, so a
                         video collected 300 days after upload has had ten
                         times as long as one collected at 30. This is a
                         control, not something a creator chooses: when
                         scoring a proposed title it is fixed at one horizon.
    publish_hour_utc     hour of upload, 0-23, UTC.
    publish_weekday      day of upload, 0 Monday to 6 Sunday, UTC.
    log_duration         ln(duration in seconds). Seed source only, since the
                         search source did not keep duration.
    log_subscribers      log10 subscriber count. Subscribers are the
                         denominator of views_per_sub, so a model given this
                         can learn the target's arithmetic rather than
                         anything about the video. Kept out of every default
                         feature set and reported only as an ablation, until
                         all three pairs agree otherwise.

Collection time was not stored with the data. Every collector keeps videos at
least MIN_DAYS_SINCE_PUBLISH old, and the newest video in both sources lands
within a day of that floor, so collection time is taken as the newest
publish time plus MIN_DAYS_SINCE_PUBLISH.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.schema import (  # noqa: E402
    MIN_DAYS_SINCE_PUBLISH,
    PUBLISHED_AT,
    SUBSCRIBER_COUNT,
    VIDEO_ID,
)
from src.title.target import DURATION  # noqa: E402

TIMING_FEATURES = ["days_since_publish", "publish_hour_utc", "publish_weekday"]
DURATION_FEATURE = "log_duration"
SUBSCRIBER_FEATURE = "log_subscribers"
CONTEXT_FEATURES = TIMING_FEATURES + [DURATION_FEATURE, SUBSCRIBER_FEATURE]


def collected_at(published):
    """Approximate collection time; see the module docstring."""
    return published.max() + pd.Timedelta(days=MIN_DAYS_SINCE_PUBLISH)


def context_features(dataset):
    """One row per video, keyed by video_id. log_duration is NaN where the
    dataset has no duration column."""
    published = pd.to_datetime(dataset[PUBLISHED_AT], utc=True)
    durations = dataset[DURATION] if DURATION in dataset else np.nan
    table = pd.DataFrame({
        VIDEO_ID: dataset[VIDEO_ID].to_numpy(),
        "days_since_publish": (collected_at(published) - published).dt.total_seconds() / 86400,
        "publish_hour_utc": published.dt.hour,
        "publish_weekday": published.dt.weekday,
        DURATION_FEATURE: np.log(durations),
        SUBSCRIBER_FEATURE: np.log10(dataset[SUBSCRIBER_COUNT]),
    })
    return table.reset_index(drop=True)
