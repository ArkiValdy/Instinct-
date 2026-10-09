"""Context features and the channel-relative target."""

import numpy as np
import pandas as pd
import pytest

from config.schema import (
    CHANNEL_ID,
    MIN_DAYS_SINCE_PUBLISH,
    PUBLISHED_AT,
    SUBSCRIBER_COUNT,
    VIDEO_ID,
    VIEW_COUNT,
)
from src.title.context import DURATION_FEATURE, SUBSCRIBER_FEATURE, context_features
from src.title.target import CHANNEL_TARGET, DURATION, add_targets


@pytest.fixture
def dataset():
    return pd.DataFrame({
        VIDEO_ID: ["a", "b", "c", "d"],
        CHANNEL_ID: ["x", "x", "y", "y"],
        SUBSCRIBER_COUNT: [10_000, 10_000, 1_000_000, 1_000_000],
        VIEW_COUNT: [1_000, 9_999, 50_000, 50_000],
        # 2026-03-02 is a Monday
        PUBLISHED_AT: ["2026-03-02T15:30:00Z", "2026-03-07T08:00:00Z",
                       "2026-01-01T00:00:00Z", "2026-02-01T23:59:00Z"],
    })


def test_newest_video_sits_at_the_publish_floor(dataset):
    table = context_features(dataset).set_index(VIDEO_ID)
    assert table.loc["b", "days_since_publish"] == pytest.approx(MIN_DAYS_SINCE_PUBLISH)
    assert table.loc["a", "days_since_publish"] == pytest.approx(
        MIN_DAYS_SINCE_PUBLISH + 4 + 16.5 / 24)


def test_hour_and_weekday_are_utc(dataset):
    table = context_features(dataset).set_index(VIDEO_ID)
    assert table.loc["a", "publish_hour_utc"] == 15
    assert table.loc["a", "publish_weekday"] == 0
    assert table.loc["b", "publish_weekday"] == 5


def test_duration_is_nan_without_a_duration_column(dataset):
    assert context_features(dataset)[DURATION_FEATURE].isna().all()
    with_duration = dataset.assign(**{DURATION: [200, 600, 1200, 3600]})
    assert context_features(with_duration)[DURATION_FEATURE].iloc[0] == pytest.approx(np.log(200))


def test_subscribers_are_log10(dataset):
    assert context_features(dataset)[SUBSCRIBER_FEATURE].tolist() == [4, 4, 6, 6]


def test_channel_target_is_centred_within_each_channel(dataset):
    out = add_targets(dataset)
    assert out.groupby(CHANNEL_ID)[CHANNEL_TARGET].sum().abs().max() < 1e-12
    gap = np.log1p(9_999) - np.log1p(1_000)
    assert out[CHANNEL_TARGET].tolist()[:2] == pytest.approx([-gap / 2, gap / 2])
    assert out[CHANNEL_TARGET].tolist()[2:] == pytest.approx([0, 0])
