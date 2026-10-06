"""Title model: XGBoost regressor on the title features.

Baseline only. Two versions are trained with default hyperparameters, one on
views_per_sub and one on log_views_per_sub, and scored on held-out channels.

Two things the split and the feature set must get right:

    Split by channel. The same channel appears across many videos, and a
    random split would let the model memorise channel-level performance and
    report a score it cannot reproduce on a channel it has not seen. Every
    channel lands wholly in train or wholly in test.

    No leakage from the target. Subscriber count and view count are both
    inside views_per_sub, so neither is a feature. Only the columns listed in
    features.FEATURES go in.

The two targets live on different scales, so the comparison table reports
each model in its own space and again with the log model's predictions mapped
back to the ratio by expm1. The ratio-space rows are the like-for-like ones.
A row that predicts the training mean is the floor: R squared 0 by
construction on the training set, and a model that cannot beat it on the test
set has learned nothing about titles. The median row is the floor for MAE in
ratio space, where expm1 of a log prediction lands near the median and so
beats the mean on MAE without knowing anything about titles.

Usage:
    python -m src.title.model cooking
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.schema import CHANNEL_ID, VIDEO_ID  # noqa: E402
from src.title.features import FEATURES  # noqa: E402
from src.title.target import LOG_TARGET, RAW_TARGET, add_targets, load_dataset  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT / "data" / "processed"

TEST_SIZE = 0.2
SEED = 42


def load_training_table(content_type):
    """Features joined to targets and channel, one row per video."""
    dataset = add_targets(load_dataset(content_type))
    features = pd.read_parquet(PROCESSED / f"title_features_{content_type}.parquet")
    table = features.merge(
        dataset[[VIDEO_ID, CHANNEL_ID, RAW_TARGET, LOG_TARGET]],
        on=VIDEO_ID, how="inner", validate="one_to_one",
    )
    if len(table) != len(dataset):
        raise ValueError(f"{len(dataset) - len(table)} videos have no feature row; "
                         f"rerun python -m src.title.features {content_type}")
    return table


def channel_split(table):
    """Train and test row indices with no channel on both sides."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(table, groups=table[CHANNEL_ID]))
    shared = set(table[CHANNEL_ID].iloc[train_idx]) & set(table[CHANNEL_ID].iloc[test_idx])
    assert not shared, f"{len(shared)} channels on both sides of the split"
    return train_idx, test_idx


def new_model():
    """Library defaults: nothing is tuned yet."""
    return XGBRegressor(random_state=SEED)


def score(y_true, y_pred):
    return {"mae": mean_absolute_error(y_true, y_pred), "r2": r2_score(y_true, y_pred)}


def compare(table):
    """Fit both versions on the train channels and score them on the test
    channels. Returns the comparison table and the split sizes."""
    train_idx, test_idx = channel_split(table)
    train, test = table.iloc[train_idx], table.iloc[test_idx]
    X_train, X_test = train[FEATURES], test[FEATURES]

    raw_model = new_model().fit(X_train, train[RAW_TARGET])
    log_model = new_model().fit(X_train, train[LOG_TARGET])
    raw_pred = raw_model.predict(X_test)
    log_pred = log_model.predict(X_test)

    y_raw, y_log = test[RAW_TARGET], test[LOG_TARGET]
    rows = {
        ("ratio", "mean baseline"): score(y_raw, np.full(len(test), train[RAW_TARGET].mean())),
        ("ratio", "median baseline"): score(y_raw, np.full(len(test), train[RAW_TARGET].median())),
        ("ratio", "raw model"): score(y_raw, raw_pred),
        ("ratio", "log model, expm1"): score(y_raw, np.expm1(log_pred)),
        ("log", "mean baseline"): score(y_log, np.full(len(test), train[LOG_TARGET].mean())),
        ("log", "log model"): score(y_log, log_pred),
        ("log", "raw model, log1p"): score(y_log, np.log1p(np.clip(raw_pred, 0, None))),
    }
    results = pd.DataFrame(rows).T
    results.index.names = ["scored in", "predictions"]
    sizes = {
        "train videos": len(train), "train channels": train[CHANNEL_ID].nunique(),
        "test videos": len(test), "test channels": test[CHANNEL_ID].nunique(),
    }
    return results, sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("content_type")
    args = parser.parse_args()

    table = load_training_table(args.content_type)
    results, sizes = compare(table)

    print(f"{args.content_type}: {len(table):,} videos, {len(FEATURES)} features, "
          f"split by {CHANNEL_ID}")
    print(", ".join(f"{k} {v:,}" for k, v in sizes.items()))
    print("\nHeld-out channels, default hyperparameters")
    print(results.round(3).to_string())


if __name__ == "__main__":
    main()
