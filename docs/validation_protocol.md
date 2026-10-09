# Validation protocol

How the title model (and later the thumbnail model) is scored, what counts as
working, and what gets reported when nothing does. Agreed rules first, then the
title model results that were scored against them.

## Rules

**Split by channel, always.** Every channel lands wholly in training or wholly
in test. A channel's videos share an audience, a style and a typical view
count, so a random split lets a model memorise the channel and report a score
it cannot reproduce on a channel it has not seen.

**Five-fold cross validation, grouped by `channel_id`.** One held-out split is
reported alongside for continuity, but decisions rest on the five folds and
their spread. A model that scores well on one fold and badly on another is not
ready.

**Tune inside the training folds only.** Hyperparameters are chosen by random
search scored on grouped inner folds of each outer training fold (nested cross
validation). Tuning on the rows a model is later scored on reports a score it
cannot reproduce.

**Compare against two constant baselines.** The training-fold mean (the floor
for R squared) and the training-fold median (the floor for MAE). A model trained
on absolute error converges towards the median, so beating the mean on MAE alone
proves nothing.

**Every feature set gets a control.** When context features (age, upload time,
duration) go in alongside title features, the same model is scored on the
context features alone. Title features earn credit only for the gap between
the two.

**No feature built from the target.** Views and subscribers are both inside
`views_per_sub`. `log_subscribers` is scored only as a labelled ablation and is
not a candidate feature unless all three pairs agree to change this rule.

## What counts as working

A model is ready to serve through `score_title` when, under the rules above, it:

1. beats the better of the two baselines on MAE in at least 4 of 5 folds,
2. has mean R squared above 0 across folds, and
3. does better than its context-only control, so the gain comes from the title.

Spearman rank correlation between predictions and the target is reported too,
because the app ranks candidate titles against each other. A model that ranks
well while still missing condition 1 is worth noting, but is not ready.

## What gets reported when nothing works

A null result is a result. The write-up states the conditions that were not
met, the best score reached and where it came from, and what was tried, and it
does not move the bar after seeing the numbers. The app then shows that no
reliable title score is available, instead of serving a number that the
validation says is noise.

## Title model results, 2026-10-09

Run with `python -m src.title.experiments`. Per-fold scores are in
`data/processed/title_experiments.csv`. Two datasets were used:

- **search**: cooking only, from `search.list` in date order. 1,507 videos
  across 1,046 channels, about 1.4 videos per channel.
- **seed**: all four content types, from the uploads playlists of the seed
  channels, with the search source's filters re-applied (over 180 seconds,
  mostly Latin script). 5,716 videos across 85 channels. Collecting search data
  for the other three content types would take about 11,700 quota units each,
  more than a day's cap.

Seed experiments also use `log_views_vs_channel`, which is ln(1 + views) minus
the channel's mean, on channels with at least 5 videos. It asks the question a
title can actually answer: did this video do better or worse than the
channel's typical video?

Tuned model, five folds grouped by channel. "Best baseline" is the better MAE of
the two baselines.

| Experiment | Target | MAE | Best baseline MAE | Folds beating | R² | Spearman |
|---|---|---|---|---|---|---|
| search, title | log ratio | 0.468 | 0.458 | 0/5 | −0.082 | 0.070 |
| search, timing (control) | log ratio | 0.459 | 0.458 | 3/5 | −0.055 | 0.186 |
| search, title + timing | log ratio | 0.459 | 0.458 | 1/5 | −0.052 | 0.227 |
| search, + subscribers (ablation) | log ratio | 0.440 | 0.458 | 5/5 | 0.035 | 0.400 |
| seed, title | log ratio | 0.075 | 0.072 | 3/5 | −0.068 | 0.145 |
| seed, context (control) | log ratio | 0.077 | 0.072 | 1/5 | −0.088 | 0.131 |
| seed, title + context | log ratio | 0.076 | 0.072 | 1/5 | −0.075 | 0.133 |
| seed, title | vs channel | 0.528 | 0.526 | 1/5 | −0.002 | 0.006 |
| seed, context (control) | vs channel | 0.535 | 0.526 | 0/5 | −0.024 | 0.032 |
| seed, title + context | vs channel | 0.542 | 0.526 | 2/5 | −0.059 | 0.057 |

### Findings

**Regularisation fixed the overfitting but found no signal.** On search cooking
with title features, default hyperparameters scored R² −0.338. The hand-set
regularised configuration scored −0.008 and the tuned one −0.082, with MAE now
level with the median baseline instead of 0.18 above it. Tuning chose shallow
trees and an absolute-error objective in every fold, which are the settings
that sit closest to a constant.

**No configuration meets the bar.** No candidate beats the baseline in 4 of 5
folds or reaches positive mean R squared. The only row that does is the
subscriber ablation. Subscribers are the target's denominator, so that row shows
the rule against target-derived features doing its job, not a title effect.

**The cleanest test shows no title effect.** Against the channel's own typical
video, title features alone give R² −0.002 and Spearman 0.006, which is
indistinguishable from zero. Within a channel, these 12 features say nothing
about which videos did better.

**The faint ranking signal on search cooking comes mostly from timing.** Timing
alone reaches Spearman 0.186, and adding title features raises it to 0.227.
That gap is the only hint of title signal anywhere in the grid. It does not
come with an MAE gain, and it rests on one content type.

**Pooled seed scores are carried by content type.** On the seed log ratio
target, the tuned title model ranks videos within cooking, education and
finance negatively (Spearman −0.24 to −0.32). Whatever it learned was the
difference between content types, not between titles.

### Conclusion

The title model as specified, 12 hand-built features from the title string
under a gradient boosted tree, does not predict reach on held-out channels. It
should not be served as a reach score. `score_title` should not replace the
placeholder until a model meets the three conditions above.

What might change that, in order of cost:

1. **Richer title representations.** Sentence embeddings
   (`sentence-transformers` is already in `requirements.txt`) capture topic and
   phrasing, which the 12 surface features cannot.
2. **Score within channels.** The `log_views_vs_channel` setup is the right
   question for the product ("will this title beat my usual?"). A within-channel
   pairwise ranking objective fits it better than regression.
3. **More search data.** The search source covers cooking only. The other
   three content types need about 1.2 days of quota each.
