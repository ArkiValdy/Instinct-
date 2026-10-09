"""Title model: XGBoost regressor on the title features.

Two versions are trained with default hyperparameters, one on
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

Cross validation repeats the comparison over five folds grouped by channel
and reports the mean and standard deviation of each score. SHAP values for
the log model on the held-out channels show which features it leans on, and
the summary plot is saved to data/processed/.

Usage:
    python -m src.title.model cooking
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
)
from xgboost import XGBRegressor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.schema import CHANNEL_ID, CONTENT_TYPE, VIDEO_ID  # noqa: E402
from src.title.context import context_features  # noqa: E402
from src.title.features import FEATURES, feature_path  # noqa: E402
from src.title.target import (  # noqa: E402
    CHANNEL_TARGET,
    LOG_TARGET,
    RAW_TARGET,
    add_targets,
    load_dataset,
)

PROJECT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT / "data" / "processed"

TEST_SIZE = 0.2
N_FOLDS = 5
SEED = 42

# Defaults grow 100 trees of depth 6 with no minimum leaf weight, which fits
# noise on a dataset this size: on cooking they scored worse than predicting
# the mean in every fold. These are the opposite end: shallow trees, large
# leaves, shrinkage, row and column subsampling and an L2 penalty, set by hand
# before any tuning so they serve as a fixed reference point.
REGULARISED = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 2,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 10.0,
}

# Searched by tune(). Depth 1 is an additive model with no interactions, which
# is a plausible answer for weak title signals, so it is in range.
PARAM_SPACE = {
    "n_estimators": [100, 200, 400, 800],
    "learning_rate": [0.01, 0.03, 0.1],
    "max_depth": [1, 2, 3, 4],
    "min_child_weight": [1, 5, 20, 50],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.8, 1.0],
    "reg_lambda": [1.0, 10.0, 50.0],
    "objective": ["reg:squarederror", "reg:absoluteerror"],
}
N_ITER = 30
INNER_FOLDS = 3

# chart tokens, as in target.py. SEQUENTIAL is the blue ramp from step 250,
# the lightest step still visible on the surface, to step 700.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
SEQUENTIAL = ["#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]


def load_training_table(content_type, source="search"):
    """Title features, context features, targets, channel and content type,
    one row per video."""
    dataset = add_targets(load_dataset(content_type, source))
    features = pd.read_parquet(feature_path(content_type, source))
    table = features.merge(
        dataset[[VIDEO_ID, CHANNEL_ID, CONTENT_TYPE, RAW_TARGET, LOG_TARGET, CHANNEL_TARGET]],
        on=VIDEO_ID, how="inner", validate="one_to_one",
    )
    if len(table) != len(dataset):
        raise ValueError(f"{len(dataset) - len(table)} videos have no feature row; "
                         f"rerun python -m src.title.features {content_type} --source {source}")
    return table.merge(context_features(dataset), on=VIDEO_ID, validate="one_to_one")


def channel_split(table):
    """Train and test row indices with no channel on both sides."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(table, groups=table[CHANNEL_ID]))
    shared = set(table[CHANNEL_ID].iloc[train_idx]) & set(table[CHANNEL_ID].iloc[test_idx])
    assert not shared, f"{len(shared)} channels on both sides of the split"
    return train_idx, test_idx


def new_model(params=None):
    """Library defaults unless params are given."""
    return XGBRegressor(random_state=SEED, **(params or {}))


def tune(X, y, groups):
    """Hyperparameters chosen by random search, scored by MAE over grouped
    folds of the data given, so no channel is in both halves of an inner
    split either. Pass training rows only: tuning on the rows a model is
    later scored on reports a score the model cannot reproduce."""
    search = RandomizedSearchCV(
        new_model({"n_jobs": 1}), PARAM_SPACE, n_iter=N_ITER,
        scoring="neg_mean_absolute_error", cv=GroupKFold(n_splits=INNER_FOLDS),
        random_state=SEED, n_jobs=-1,
    )
    search.fit(X, y, groups=groups)
    return search.best_params_


def score(y_true, y_pred):
    return {"mae": mean_absolute_error(y_true, y_pred), "r2": r2_score(y_true, y_pred)}


def evaluate(train, test):
    """Fit both versions on train and score them on test, alongside the
    constant baselines. One row per (space scored in, predictions)."""
    X_train, X_test = train[FEATURES], test[FEATURES]
    raw_pred = new_model().fit(X_train, train[RAW_TARGET]).predict(X_test)
    log_pred = new_model().fit(X_train, train[LOG_TARGET]).predict(X_test)

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
    return results


def compare(table):
    """Score both versions on one held-out set of channels. Returns the
    comparison table and the split sizes."""
    train_idx, test_idx = channel_split(table)
    train, test = table.iloc[train_idx], table.iloc[test_idx]
    sizes = {
        "train videos": len(train), "train channels": train[CHANNEL_ID].nunique(),
        "test videos": len(test), "test channels": test[CHANNEL_ID].nunique(),
    }
    return evaluate(train, test), sizes


def cross_validate(table):
    """Grouped k-fold by channel: every channel is held out exactly once.

    Returns mean and standard deviation of each metric across folds, and the
    per-fold table. The spread matters as much as the mean, since a model
    that scores well on one fold and badly on another is not ready."""
    folds = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    per_fold = []
    for fold, (train_idx, test_idx) in enumerate(folds.split(table, groups=table[CHANNEL_ID])):
        results = evaluate(table.iloc[train_idx], table.iloc[test_idx])
        per_fold.append(results.assign(fold=fold + 1))
    per_fold = pd.concat(per_fold)
    summary = per_fold.groupby(level=[0, 1], sort=False)[["mae", "r2"]].agg(["mean", "std"])
    return summary, per_fold


def explain(table):
    """SHAP values for the log model on the held-out channels of the Step 5
    split. The log model is explained because it is the one that will be
    served; the raw model's attributions are dominated by a few viral rows."""
    train_idx, test_idx = channel_split(table)
    train, test = table.iloc[train_idx], table.iloc[test_idx]
    model = new_model().fit(train[FEATURES], train[LOG_TARGET])
    values = shap.TreeExplainer(model).shap_values(test[FEATURES])
    importance = pd.Series(np.abs(values).mean(axis=0), index=FEATURES)
    return values, test[FEATURES], importance.sort_values(ascending=False)


def shap_summary_plot(values, X, importance, title, path):
    """Beeswarm: one row per feature, ordered by mean |SHAP|. Colour is the
    feature's value ranked within that feature, on a one-hue sequential ramp
    that starts at a step still visible against the surface."""
    order = list(importance.index[::-1])
    ramp = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
    rng = np.random.default_rng(SEED)

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axvline(0, color="#b5b4af", linewidth=1, zorder=1)
    for row, feature in enumerate(order):
        x = values[:, FEATURES.index(feature)]
        # dense rank scaled to 0-1, so a binary feature's 0 and 1 sit at the
        # two ends of the ramp instead of at their average ranks
        dense = X[feature].rank(method="dense").to_numpy()
        rank = (dense - 1) / max(dense.max() - 1, 1)
        jitter = rng.uniform(-0.28, 0.28, size=len(x))
        ax.scatter(x, row + jitter, c=rank, cmap=ramp, vmin=0, vmax=1, s=10,
                   linewidths=0.4, edgecolors=SURFACE, zorder=2)
        ax.annotate(f"{importance[feature]:.3f}", xy=(1.0, row),
                    xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK_MUTED, fontsize=8)
    ax.annotate("mean |SHAP|", xy=(1.0, len(order) - 0.4),
                xycoords=("axes fraction", "data"), xytext=(6, 4),
                textcoords="offset points", color=INK_MUTED, fontsize=8)
    ax.set_yticks(range(len(order)), order, color=INK, fontsize=9)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("SHAP value: effect on predicted ln(1 + views / subscribers)",
                  color=INK_MUTED, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d0cfca")
    ax.grid(axis="x", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)

    bar = fig.colorbar(plt.cm.ScalarMappable(cmap=ramp), ax=ax, pad=0.12, fraction=0.03)
    bar.set_ticks([0, 1], labels=["low", "high"])
    bar.set_label("feature value (ranked within feature)", color=INK_MUTED, fontsize=8)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    bar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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

    summary, per_fold = cross_validate(table)
    print(f"\n{N_FOLDS}-fold cross validation grouped by {CHANNEL_ID}, "
          f"mean and std across folds")
    print(summary.round(3).to_string())
    per_fold = per_fold.sort_index()
    log_r2 = per_fold.loc[("log", "log model"), "r2"].to_numpy()
    log_mae = per_fold.loc[("log", "log model"), "mae"].to_numpy()
    base_mae = per_fold.loc[("log", "mean baseline"), "mae"].to_numpy()
    print(f"log model R squared by fold: {', '.join(f'{v:.3f}' for v in log_r2)}")
    print(f"folds where the log model beats the mean baseline on MAE: "
          f"{int((log_mae < base_mae).sum())} of {N_FOLDS}")

    values, X, importance = explain(table)
    path = PROCESSED / f"shap_summary_title_{args.content_type}.png"
    shap_summary_plot(values, X, importance,
                      f"Title model SHAP values, {args.content_type}, "
                      f"held-out channels (n={len(X):,})", path)
    length_share = importance[["char_length", "word_count"]].sum() / importance.sum()
    print("\nTop five features by mean |SHAP| (log model, held-out channels)")
    for feature, value in importance.head(5).items():
        print(f"  {feature:<22} {value:.4f}")
    print(f"sum of mean |SHAP| over all features {importance.sum():.4f}, "
          f"std of {LOG_TARGET} {table[LOG_TARGET].std():.4f}")
    print(f"share of attribution on char_length and word_count: {length_share:.0%}")
    print(f"Summary plot: {path.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
