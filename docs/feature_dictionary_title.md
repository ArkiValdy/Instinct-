# Title feature dictionary

Every feature the title model uses, how it is calculated, and why it might
matter. Implemented in `src/title/features.py`; the column names below are the
exact names in the feature table.

All features are computed from the title string alone. Nothing about the
channel, the video's age or its view count enters the feature table, because
those are either in the target (views, subscribers) or not known when a
creator is choosing a title.

## Summary

| Column | Type | Source | Cooking mean (n=595) |
|---|---|---|---|
| `char_length` | int | string | 66.6 |
| `word_count` | int | string | 10.7 |
| `uppercase_ratio` | float 0–1 | string | 0.23 |
| `has_all_caps_word` | 0/1 | string | 0.28 |
| `has_number` | 0/1 | spaCy lexical | 0.30 |
| `ends_with_question` | 0/1 | string | 0.008 |
| `has_question` | 0/1 | string | 0.018 |
| `has_brackets_or_pipe` | 0/1 | string | 0.50 |
| `emoji_count` | int | string (Unicode) | 0.21 |
| `exclamation_count` | int | string | 0.29 |
| `sentiment_compound` | float −1–1 | VADER | 0.23 |
| `has_superlative` | 0/1 | spaCy statistical | 0.10 |

"Source" says how far each feature can be trusted. String features are exact.
`has_number` uses a spaCy lexical attribute, which is a deterministic lookup on
the token text. VADER is a fixed lexicon with rules. `has_superlative` is the
only feature that rests on a statistical model's prediction, and it is the one
expected to degrade on titles, which are short, Title Cased and often
ungrammatical.

## Features

### `char_length`

**Calculation:** number of characters in the title, including spaces,
punctuation and emoji (each code point counts once).

**Why it might matter:** YouTube truncates titles at roughly 60–70 characters
in most surfaces, so length decides whether the whole title is seen.

### `word_count`

**Calculation:** whitespace-separated tokens that contain at least one letter
or digit. A lone `|` or `-` is not a word.

**Why it might matter:** separates dense, keyword-stuffed titles from short
punchy ones at the same character length.

### `uppercase_ratio`

**Calculation:** uppercase letters divided by all letters. Spaces, digits,
punctuation and emoji are excluded from both sides, so the ratio is not
diluted by long titles with many spaces. 0 for a title with no letters.

**Why it might matter:** heavy capitalisation is a common attention tactic, and
a continuous measure captures degree, which the flag below does not.

### `has_all_caps_word`

**Calculation:** 1 if any word has two or more letters and all of its letters
are uppercase. The two-letter minimum keeps "I" and "A" out.

**Why it might matter:** a single shouted word ("The BEST Pasta") is a distinct
tactic from an all-caps title. **Limitation:** acronyms such as BBQ, USA or
KFC also trigger it.

### `has_number`

**Calculation:** 1 if any token has spaCy's `like_num` attribute. This catches
digits ("5", "1,000", "1/2"), number words ("five", "Twenty") and ordinals
("first"). The title is lowercased first, which does not affect `like_num`.

**Why it might matter:** list and quantity titles ("5 Easy Dinners", "in 15
Minutes") set a concrete expectation and are a long-standing title formula.
**Limitation:** tokens that fuse a number with letters, such as "10k" or "2x",
are not caught.

### `ends_with_question`

**Calculation:** 1 if the title, after trailing whitespace is removed, ends
in "?".

**Why it might matter:** a question as the whole framing of the title invites
the viewer to click for the answer. **Data note:** only 5 of 595 cooking titles
end in "?", so the model has very little to learn from.

### `has_question`

**Calculation:** 1 if "?" appears anywhere in the title.

**Why it might matter:** catches questions followed by a qualifier ("Is This
Worth It? | Taste Test"), which the feature above misses. Rare in cooking:
11 of 595.

### `has_brackets_or_pipe`

**Calculation:** 1 if the title contains any of `( ) [ ] { } |`, or the CJK
brackets `【 】 「 」 『 』 〔 〕`, which serve the same purpose ("【ENG DUB】").

**Why it might matter:** brackets and pipes mark the title as a main hook plus
qualifiers or keywords ("Chicken Curry | Easy Weeknight Recipe"), a structure
aimed at search rather than browsing. Present in about half of cooking titles.

### `emoji_count`

**Calculation:** number of emoji as a reader sees them. The title is split into
grapheme clusters, and a cluster counts once if it contains a character drawn
as emoji by default or carries the emoji variation selector (U+FE0F). So 👍🏽
(with skin tone) and 👨‍👩‍👧 (joined family) each count once, and ™ and ©,
which are drawn as text, do not count.

**Why it might matter:** emoji add colour in a text-only list of titles and
signal tone. They are also a marker of short-form-style packaging.

### `exclamation_count`

**Calculation:** number of "!" characters.

**Why it might matter:** a measure of excitement and urgency. A count rather
than a flag, since "!!!" is a stronger signal than "!".

### `sentiment_compound`

**Calculation:** VADER's compound score, from −1 (most negative) to +1 (most
positive). VADER accounts for capitalisation, "!" and some emoji, which suits
titles better than a model trained on long text.

The title is cleaned for VADER first, because VADER matches words by splitting
on whitespace and stripping ASCII punctuation only. The emoji variation
selector (U+FE0F) is removed, since it hid the heart in "❤️" (scored 0, against
0.64 for "❤"). Curly quotes become straight so "don’t" still negates. Other
non-ASCII punctuation becomes a space, since "Better【" otherwise never
matched "better". This changed the score of 8 of 595 cooking titles.

**Why it might matter:** whether emotionally charged or positive titles reach
further than neutral ones. **Limitation:** VADER translates emoji to words
before scoring, so 🔥 becomes "fire" and scores negative, although in titles
it means the opposite. Most cooking titles are neutral (median 0).

### `has_superlative`

**Calculation:** 1 if any token in the lowercased title has the part of speech
tag `JJS` (superlative adjective: best, easiest, crispiest) or `RBS`
(superlative adverb: most). Tags come from `en_core_web_sm`'s tagger.

**Why it might matter:** superlatives make the strongest possible claim ("The
Best Burger") and are a classic clickbait marker.

**Statistical, so expect noise.** On the original casing, the tagger reads
Title Case and ALL CAPS words as proper nouns and missed "Best", "BEST",
"EASIEST" and "Most". Tagging a lowercased copy recovered 10 of 10
superlatives in a hand check, against 4 of 10, with no false positives on
look-alikes such as "West" or "Forest". On the cooking set, 61 of 62 tags were
real superlatives; one was a 🍳 emoji tagged `JJS`. "Most" in "Most People Get
This Wrong" means "the majority" but is still tagged as a superlative.

## Not included, and why

- **Hashtags.** Considered because a third of the titles had them before
  Shorts were removed. Afterwards only 17 of 595 cooking titles contain "#".
- **Language.** Titles with under 80% Latin-script letters are removed at
  collection. Romanised non-English text, such as Hinglish, remains and its
  features are computed with English tools.
