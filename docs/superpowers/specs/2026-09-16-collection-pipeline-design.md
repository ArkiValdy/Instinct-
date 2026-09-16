# Collection pipeline design

Date: 2026-09-16
Status: approved, not yet implemented
Owner: collection pair
Scope: `src/collection/`, plus a new `config/channels.py`

## Purpose

Build the Instinct+ training dataset from the YouTube Data API v3: one row per
public video, carrying the columns in `config/schema.py`, plus one thumbnail
image per row. The title pair and the thumbnail pair both train against the
output of this pipeline, so its correctness and its sampling bias set a ceiling
on every model built downstream.

## Decisions

Each of these was settled during design. The rationale matters more than the
choice, because it is what tells a future reader whether a change is safe.

### Discovery: seed channels, not search

Videos are discovered by walking the uploads playlist of a curated list of
channels, not by querying `search.list`.

`search.list` costs 100 units per call against a 10,000 unit daily quota, caps
out near 500 results per query however you page it, and orders results by
YouTube's own relevance ranking. That last point is disqualifying rather than
merely inconvenient: relevance ranking is itself a function of engagement, so a
search-discovered sample is biased by the very quantity the models predict.

Seed channels cost about 1,300 units for the whole dataset, are reproducible
across runs and machines, and let the team control the balance across content
types deliberately instead of accepting whatever search returns.

The cost is manual: someone assembles the channel list. This is real work and it
is the pipeline's main external dependency.

### Publish window: unchanged at 30-365 days

`MIN_DAYS_SINCE_PUBLISH = 30` and `MAX_DAYS_SINCE_PUBLISH = 365` in
`config/schema.py` stay as they are. Against a 2026-09-16 run this admits videos
published between 2025-09-16 and 2026-08-17.

The floor exists because view counts need time to settle before they represent a
video's reach. The ceiling exists because older videos accumulate views for
longer, which confounds the target variable.

Widening the window to cover January 2025 was considered and rejected for now.
It would add roughly 70% more rows, but a January 2025 video has had twenty
months to accumulate views against one month for the newest, making age the
dominant predictor unless views are age-normalised — a modelling decision that
both model pairs would inherit from a collection-stage choice.

Raw API responses retain full `publishedAt`, so a wider window can be re-cut
from `data/raw/` without spending quota again, should the team later agree to
the schema change.

### Thumbnails: hqdefault, 480x360

`https://i.ytimg.com/vi/{video_id}/hqdefault.jpg` exists for every video ever
uploaded. `maxresdefault` does not — it is absent whenever the uploader did not
supply a high-resolution image, which correlates with channel professionalism
and would put an availability bias into the thumbnail dataset that tracks the
target variable.

At roughly 15KB each, 32,000 thumbnails is about 500MB, which moves through a
Drive folder comfortably. `maxresdefault` would be nearer 2GB.

480x360 is ample for the planned features. Face detection, brightness, colour
histograms and text-area estimation all downscale below this resolution before
they do anything.

The cost is a 4:3 frame with letterboxing on 16:9 uploads. The thumbnail pair
crops it once at feature-extraction time.

Thumbnail downloads are plain HTTPS GETs against `i.ytimg.com`, not API calls,
and cost zero quota.

### Shorts: excluded

Videos shorter than 60 seconds are excluded. A Short and a twenty-minute video
with identical view counts do not represent the same reach — Shorts are served
through a different surface with different view-counting behaviour, and mixing
them puts two distinct distributions under one target variable.

Implementation: request `contentDetails` on `videos.list`, parse the ISO 8601
`duration`, drop anything under 60 seconds. The count of dropped Shorts appears
in the run summary.

This needs no schema change. `contentDetails` adds no quota cost, because
`videos.list` costs 1 unit per call regardless of which parts are requested.

### Seed list location: `config/channels.py`

The seed list lives in `config/channels.py` as `CHANNELS: dict[str, list[str]]`,
keyed by content type.

`config/` is shared across all three pairs per `README.md`, but `schema.py` is
the file that needs all-pair agreement to change. A separate `channels.py` keeps
the seed list collection-owned and leaves `schema.py` untouched.

The pipeline iterates whatever keys are present rather than hardcoding content
types, so the team can finalise `CONTENT_TYPES` later without touching
collection code. Keys not present in `schema.CONTENT_TYPES` produce a warning,
not an error, so the list can be developed incrementally.

## Architecture

Five stage modules under `src/collection/`, plus an entry point. Each stage
reads files and writes files, so any stage runs alone and re-runs without
redoing the stage before it.

| Module | Responsibility | Network |
|---|---|---|
| `client.py` | API wrapper: auth, retry, backoff, quota accounting | yes, the only one |
| `discover.py` | seed channels to video IDs | via client |
| `fetch.py` | video IDs to raw JSONL | via client |
| `build.py` | raw JSONL to processed table | no |
| `thumbnails.py` | processed table to image files | plain HTTPS, no quota |

Entry point `__main__.py` exposes
`python -m src.collection {discover,fetch,build,thumbnails,all}`.

`client.py` being the sole network boundary is what makes the rest testable:
every other module takes a client instance and is exercised in tests against a
fake.

## Data flow

```
config/channels.py
      |  discover:  channels.list -> uploads playlist id
      |             playlistItems.list -> video ids (paged, early-stopped)
      v
data/raw/video_ids.txt                       one id per line, deduped, sorted
      |  fetch:     videos.list   (50 ids per call)
      |             channels.list (50 ids per call)
      v
data/raw/videos_{run_id}.jsonl               one API item per line, verbatim
data/raw/channels_{run_id}.jsonl
      |  build:     join, filter, project to schema.COLUMNS
      v
data/processed/videos.parquet
      |  thumbnails: GET i.ytimg.com/vi/{id}/hqdefault.jpg
      v
data/thumbnails/{video_id}.jpg
```

`run_id` is a UTC timestamp, `YYYYMMDDTHHMMSSZ`. Fetch appends a new pair of
files per run rather than overwriting, so a partial run is never destructive and
`build` consumes every raw file it finds.

Raw JSONL holds API items exactly as returned, with no projection or renaming.
This is what makes the window re-cuttable and lets a feature-extraction bug be
fixed by re-running `build` rather than by spending quota.

`build` writes Parquet, which preserves dtypes and holds 32,000 rows in a few
megabytes. A `--csv` flag additionally writes CSV for anyone who wants to open
the table in a spreadsheet.

### Early stopping in discover

`playlistItems.list` returns a channel's uploads newest first and each item
carries `contentDetails.videoPublishedAt`. Paging stops as soon as an item falls
before the window's start, so a channel with ten years of uploads costs only the
pages that intersect the window.

## Filters

Applied in `build`, all read from `config/schema.py`, none hardcoded:

1. `subscriber_count >= MIN_SUBSCRIBERS` (10,000). Below this the API's rounding
   to three significant figures makes the channel-size feature too coarse to be
   useful.
2. `MIN_DAYS_SINCE_PUBLISH <= age_days <= MAX_DAYS_SINCE_PUBLISH`, age measured
   against the build's run time in UTC.
3. `duration >= 60s`, excluding Shorts.
4. `statistics.viewCount` present. Absent when the uploader hides counts.

Rows failing each filter are counted separately and reported, so a collapse in
dataset size is immediately attributable to one filter rather than needing a
bisect.

`build` is the authoritative filter. The early stopping in `discover` is a
quota optimisation using the same window, not a second source of truth: it
avoids paging into videos that `build` would discard anyway. A video that slips
past `discover` is still filtered here.

## Quota and resumption

`client.py` counts units against a budget defaulting to 9,000, leaving headroom
under the 10,000 daily cap for manual API exploration on the same key. Every
list call in use costs 1 unit.

Projected cost for 4 content types times 40 channels at roughly 200 in-window
videos each: 4 units to resolve uploads playlists in `discover`, about 640 for
playlist paging, about 640 for `videos.list` in `fetch`, and 4 more for the
`channels.list` call that supplies subscriber counts — totalling near 1,300
units for roughly 32,000 videos. The full dataset fits in a single day's quota
with room to spare.

On budget exhaustion or a `403 quotaExceeded`, the run writes
`data/raw/checkpoint.json` listing completed channel IDs and exits 0 with a
message naming the reset time, which is midnight Pacific. The next run reads the
checkpoint and skips those channels. Video IDs are deduped on load, so an
interrupted run never double-spends on work already done.

Quota is per Google Cloud project, so each team member can run with their own
key and their own 10,000 units against the same code.

## Error handling

| Condition | Behaviour |
|---|---|
| 5xx, timeout, connection reset | exponential backoff, 3 retries, then record and continue |
| 403 `quotaExceeded` | checkpoint and stop cleanly |
| 403 other, 401 | stop immediately, the key or its restrictions are wrong |
| 404 on a channel | record, continue, report at end |
| video missing `statistics` | dropped in build, counted |
| thumbnail download fails | row kept with empty `thumbnail_file`, counted |

A failed thumbnail keeps its row because the title model can still train on it.
Dropping the row would discard usable data over a missing image.

Every run ends with a summary: units spent, channels processed, videos
discovered, videos written, and a breakdown of every exclusion by cause.

## Testing

TDD. Tests use recorded JSON fixtures under `tests/fixtures/`; the suite makes no
network calls and needs no API key, so it runs in CI and on a machine whose quota
is exhausted.

Cases that earn their place:

- filter boundaries: exactly `MIN_SUBSCRIBERS`, exactly 30 and exactly 365 days,
  exactly 60 seconds
- batching: 137 ids becomes 3 calls of 50, 50, 37
- dedupe: an id present in two channels' playlists is fetched once
- quota accounting: budget exhaustion raises `QuotaExceeded` before the call that
  would exceed it, not after
- resumption: a checkpoint causes completed channels to be skipped
- ISO 8601 duration parsing, including `PT1H2M3S`, `PT60S` and `PT1M`
- thumbnail filenames match `schema.THUMBNAIL_PATTERN`
- `build` output columns equal `schema.COLUMNS` exactly, in order

## Known limitations

These are properties of the data source, not defects to fix. They are recorded
because the model pairs need them when interpreting results.

**Subscriber count is current, not historical.** The API reports a channel's
subscriber count at collection time, not at each video's publish time. For a
channel that grew substantially during the window, older videos are attributed a
subscriber count they did not have. This inflates the apparent underperformance
of older videos on fast-growing channels.

**Subscriber counts are rounded to three significant figures.** A channel
reported at 1,234,567 subscribers is returned as 1,230,000. `MIN_SUBSCRIBERS`
exists to keep the resulting granularity tolerable.

**View count is a snapshot.** Two videos in the dataset were measured at
different ages. This is what the 30-365 day window bounds rather than removes.

**The sample is the seed list.** Every bias in the channel list — language,
region, production budget, the team's own viewing habits — is a bias in the
dataset and therefore in the models. Assembling the list deliberately across
channel sizes matters more than assembling it quickly.

**Collected data is not redistributable.** The YouTube API terms restrict it,
which is why `data/` is gitignored and the dataset moves through Drive.

## Out of scope

Feature extraction, which belongs to the title and thumbnail pairs. Model
training. Any change to `config/schema.py` or `src/scoring/interface.py`, both of
which need all-pair agreement. Incremental top-up runs that refresh view counts
on already-collected videos, which is a plausible later addition but not needed
to build the first dataset.
