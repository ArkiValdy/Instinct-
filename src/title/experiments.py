"""Does any version of the title model beat a constant?

The default model on cooking scored worse than predicting the mean in every
fold. This runs the candidate fixes side by side under one protocol
(docs/validation_protocol.md), so they are compared on equal terms:

    hyperparameters   library defaults, a hand-set regularised config, and a
                      config tuned by random search inside each training fold
    data              the search source (cooking, many channels, about 1.4
                      videos each) and the seed source (all four content
                      types, 85 channels, dozens of videos each)
    target            log_views_per_sub, and on the seed source
                      log_views_vs_channel, which takes the channel's own
                      typical video out of the target
    features          title alone, context alone, and both. The context-only
                      rows are the control: if title plus context does no
                      better than context alone, the title features add
                      nothing, however good the combined score looks

Every row is scored by five-fold cross validation grouped by channel, and the
tuned model is tuned on the training folds only (nested cross validation).
A model beats the baseline in a fold when its MAE is below the better of the
mean and median baselines in that fold.

Per-fold scores are written to data/processed/title_experiments.csv.

Usage:
    python -m src.title.experiments
    python -m src.title.experiments --only seed_channel_title_context
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.schema import CHANNEL_ID, CONTENT_TYPE, CONTENT_TYPES  # noqa: E402
from src.title.context import (  # noqa: E402
    DURATION_FEATURE,
    SUBSCRIBER_FEATURE,
    TIMING_FEATURES,
)
from src.title.features import FEATURES  # noqa: E402
from src.title.model import (  # noqa: E402
    N_FOLDS,
    PROCESSED,
    PROJECT,
    REGULARISED,
    SEED,
    load_training_table,
    new_model,
    tune,
)
from src.title.target import ALL, CHANNEL_TARGET, LOG_TARGET  # noqa: E402

# a channel needs this many videos for its mean to be a fair reference point
# for log_views_vs_channel
MIN_CHANNEL_VIDEOS = 5

CONTENT_DUMMIES = [f"is_{t}" for t in CONTENT_TYPES]
SEED_CONTEXT = TIMING_FEATURES + [DURATION_FEATURE]


@dataclass
class Experiment:
    name: str
    source: str
    content_type: str
    target: str
    features: list
    note: str = ""


EXPERIMENTS = [
    Experiment("search_title", "search", "cooking", LOG_TARGET, FEATURES,
               "the current model"),
    Experiment("search_timing", "search", "cooking", LOG_TARGET, TIMING_FEATURES,
               "control"),
    Experiment("search_title_timing", "search", "cooking", LOG_TARGET,
               FEATURES + TIMING_FEATURES),
    Experiment("search_title_timing_subs", "search", "cooking", LOG_TARGET,
               FEATURES + TIMING_FEATURES + [SUBSCRIBER_FEATURE],
               "ablation, not a candidate"),
    Experiment("seed_title", "seed", ALL, LOG_TARGET, FEATURES + CONTENT_DUMMIES),
    Experiment("seed_context", "seed", ALL, LOG_TARGET, SEED_CONTEXT + CONTENT_DUMMIES,
               "control"),
    Experiment("seed_title_context", "seed", ALL, LOG_TARGET,
               FEATURES + SEED_CONTEXT + CONTENT_DUMMIES),
    Experiment("seed_channel_title", "seed", ALL, CHANNEL_TARGET, FEATURES),
    Experiment("seed_channel_context", "seed", ALL, CHANNEL_TARGET, SEED_CONTEXT,
               "control"),
    Experiment("seed_channel_title_context", "seed", ALL, CHANNEL_TARGET,
               FEATURES + SEED_CONTEXT),
]

MODELS = ["default", "regularised", "tuned"]
BASELINES = ["mean baseline", "median baseline"]


def load(experiment, cache):
    """The training table for an experiment, with content type dummies and,
    for the channel target, small channels dropped."""
    key = (experiment.source, experiment.content_type)
    if key not in cache:
        table = load_training_table(experiment.content_type, experiment.source)
        for content_type, column in zip(CONTENT_TYPES, CONTENT_DUMMIES):
            table[column] = (table[CONTENT_TYPE] == content_type).astype(int)
        cache[key] = table
    table = cache[key]
    if experiment.target == CHANNEL_TARGET:
        sizes = table.groupby(CHANNEL_ID)[CHANNEL_ID].transform("size")
        table = table[sizes >= MIN_CHANNEL_VIDEOS]
    return table.reset_index(drop=True)


def metrics(y_true, y_pred):
    rho = spearmanr(y_true, y_pred).statistic if np.ptp(y_pred) > 0 else np.nan
    return {"mae": mean_absolute_error(y_true, y_pred), "r2": r2_score(y_true, y_pred),
            "spearman": rho}


def run(experiment, table):
    """Per-fold scores for every model and baseline, the tuned model's
    out-of-fold predictions, and the parameters tuning chose in each fold."""
    X, y, groups = table[experiment.features], table[experiment.target], table[CHANNEL_ID]
    folds = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows, chosen = [], []
    oof = pd.Series(np.nan, index=table.index)
    for fold, (train, test) in enumerate(folds.split(X, y, groups), start=1):
        X_train, y_train, y_test = X.iloc[train], y.iloc[train], y.iloc[test]
        params = tune(X_train, y_train, groups.iloc[train])
        chosen.append(params)
        predictions = {
            "mean baseline": np.full(len(test), y_train.mean()),
            "median baseline": np.full(len(test), y_train.median()),
            "default": new_model().fit(X_train, y_train).predict(X.iloc[test]),
            "regularised": new_model(REGULARISED).fit(X_train, y_train).predict(X.iloc[test]),
            "tuned": new_model(params).fit(X_train, y_train).predict(X.iloc[test]),
        }
        oof.iloc[test] = predictions["tuned"]
        for model, pred in predictions.items():
            rows.append({"experiment": experiment.name, "fold": fold, "model": model,
                         **metrics(y_test, pred)})
    return pd.DataFrame(rows), oof, chosen


def summarise(scores):
    """Mean and std of each metric per model, plus the number of folds in
    which the model's MAE is below the better baseline's."""
    summary = scores.groupby("model", sort=False)[["mae", "r2", "spearman"]].agg(["mean", "std"])
    fold_mae = scores.pivot(index="fold", columns="model", values="mae")
    best_baseline = fold_mae[BASELINES].min(axis=1)
    summary["folds beating baseline"] = [
        "" if model in BASELINES else f"{int((fold_mae[model] < best_baseline).sum())}/{N_FOLDS}"
        for model in summary.index
    ]
    return summary


def by_content_type(experiment, table, oof):
    """The tuned model's out-of-fold scores within each content type, against
    the baseline of predicting that content type's mean. Shows whether a
    pooled score is carried by one content type."""
    y = table[experiment.target]
    rows = {}
    for content_type, idx in table.groupby(CONTENT_TYPE).groups.items():
        base = np.full(len(idx), y[idx].mean())
        rows[content_type] = {
            "videos": len(idx), "channels": table.loc[idx, CHANNEL_ID].nunique(),
            "mae": mean_absolute_error(y[idx], oof[idx]),
            "baseline mae": mean_absolute_error(y[idx], base),
            "spearman": spearmanr(y[idx], oof[idx]).statistic,
        }
    return pd.DataFrame(rows).T


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=[e.name for e in EXPERIMENTS],
                        help="run these experiments only")
    args = parser.parse_args()

    experiments = [e for e in EXPERIMENTS if not args.only or e.name in args.only]
    cache, all_scores = {}, []
    for experiment in experiments:
        table = load(experiment, cache)
        scores, oof, chosen = run(experiment, table)
        all_scores.append(scores)

        note = f" ({experiment.note})" if experiment.note else ""
        print(f"\n=== {experiment.name}{note}")
        print(f"{experiment.source} source, {experiment.content_type}, target "
              f"{experiment.target}, {len(table):,} videos, "
              f"{table[CHANNEL_ID].nunique():,} channels, {len(experiment.features)} features")
        print(summarise(scores).round(3).to_string())
        depths = ", ".join(str(p["max_depth"]) for p in chosen)
        objectives = ", ".join(p["objective"].split(":")[1] for p in chosen)
        print(f"tuned max_depth by fold: {depths}; objective: {objectives}")
        if experiment.source == "seed":
            print("tuned model, out of fold, by content type")
            print(by_content_type(experiment, table, oof).round(3).to_string())

    path = PROCESSED / "title_experiments.csv"
    pd.concat(all_scores).to_csv(path, index=False)
    print(f"\nPer-fold scores: {path.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
