# Instinct+

Instinct+ predicts how far a YouTube video will reach from its title and
thumbnail, before it is published. A creator submits a proposed title and
thumbnail; the app returns a predicted reach score, an explanation of which
features drove that score, and suggested changes.

Two gradient boosted tree models are trained separately and combined: a title
model on text features, a thumbnail model on image features. Both are trained
on public data collected through the YouTube Data API v3.

## Setup

```bash
git clone <repo-url>
cd Instinct+

brew install libomp                # macOS only, before pip. see below

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

cp .env.example .env               # then fill in your API keys
```

`.env` is gitignored. Never commit it.

**macOS: xgboost needs libomp.** The wheel links against the OpenMP runtime
and does not bundle it, so without `brew install libomp` the install succeeds
and the import fails at load time:

```
Library not loaded: @rpath/libomp.dylib
  tried: '/opt/homebrew/opt/libomp/lib/libomp.dylib' (no such file)
```

Nothing in `requirements.txt` can fix this; it is a system library. Linux and
Windows wheels bundle their own, so this step is Mac only.

**matplotlib is a requirement, not an extra.** `src/title/target.py` and
`src/title/model.py` import it at module scope to draw target distributions
and the SHAP summary, so it is in `requirements.txt` rather than left to
whatever shap happens to pull in.

### The dataset

The dataset is shared through Google Drive, not through git. Thumbnails run to
gigabytes and the YouTube API terms restrict redistribution of collected data.

Drive folder: _<paste link here>_

Download it and unpack it into `data/` so that raw API responses land in
`data/raw/`, images in `data/thumbnails/` and feature tables in
`data/processed/`.

## Structure

| Path | Owner | Contents |
|---|---|---|
| `config/` | all three pairs | shared schema: column names, file patterns, collection filters |
| `data/raw/` | collection | API responses as collected. Not in git. |
| `data/thumbnails/` | collection | downloaded images, named `{video_id}.jpg`. Not in git. |
| `data/processed/` | all three pairs | feature tables. Not in git. |
| `src/collection/` | _<pair TBD>_ | YouTube Data API v3 pipeline |
| `src/title/` | _<name>_, _<name>_ | title features and title model |
| `src/thumbnail/` | _<name>_, _<name>_ | thumbnail features and thumbnail model |
| `src/scoring/` | all three pairs | the contract both models implement |
| `src/generation/` | _<name>_, _<name>_ | Claude API title generation |
| `app/` | _<name>_, _<name>_ | Streamlit interface, components and assets |
| `notebooks/` | one per person | exploration only |
| `models/` | title and thumbnail pairs | trained artefacts. Not in git. |
| `tests/` | all three pairs | contract and unit tests |
| `docs/` | all three pairs | feature dictionaries, schema notes, validation protocol |

## Rules

**Never commit data, trained models or `.env`.** The `.gitignore` covers this;
do not work around it.

**`config/schema.py` and `src/scoring/interface.py` need agreement from all
three pairs before changing.** They are what keeps the two models and the app
compatible. Everything else, each pair owns its own directory.

**Notebooks are for exploration only.** One per person, named after you. Do not
import between notebooks. When code works, move it into `src/` as a function
with a test.

`src/scoring/placeholder.py` returns fake scores so the app can be built before
the models exist. Delete it once both real models are working.

## Branching

`main` plus one branch per pair:

- `title`
- `thumbnail`
- `app`

Work on your pair's branch. Merge into `main` at milestone weeks, so `main`
always runs. Pull `main` into your branch before you start a merge.

## Tests

```bash
pytest
```

The contract tests assert that both scoring functions return a `ScoreResult`
with a score between 0 and 100 and a dict of drivers. They must keep passing
when the placeholder is replaced by the real models.
