# ClickCraft

Predicts how a YouTube title and thumbnail will perform before publication.

## Setup

```bash
git clone <repo url>
cd clickcraft
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env             # then fill in your keys
```

## Structure

| Path | Owner | Contents |
|---|---|---|
| `src/collection/` | [name] | YouTube API pipeline |
| `src/title/` | [names] | Title features and model |
| `src/thumbnail/` | [names] | Thumbnail features and model |
| `src/scoring/` | shared | The contract both models implement |
| `src/generation/` | [names] | Claude API title generation |
| `app/` | [names] | Streamlit application |
| `docs/` | everyone | Feature dictionaries, schema, validation protocol |

## Rules

**Never commit data, models, or `.env`.** The dataset lives in [shared Drive link].

**`config/schema.py` and `src/scoring/interface.py` need all three pairs to agree
before changing.** Everything else, each pair owns.

**Notebooks are for exploration only.** One per person, no imports between them.
Working code moves into `src/` as a function.

## Branches

`main` plus one branch per pair. Merge to `main` at milestone weeks so `main`
always runs.
