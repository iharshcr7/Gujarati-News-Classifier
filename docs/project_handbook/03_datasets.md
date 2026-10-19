# 03 — The dataset

## Where the data lives

```
data/raw/
  Gujarati_SHC_LDC_LPC_Train.csv
  Gujarati_SHC_LDC_LPC_Valid.csv
  Gujarati_SHC_LDC_LPC_Test.csv
```

The filename encodes the format: **SHC** (the active format here) and LDC/LPC
(alternate formats that the loader recognises but that are **not** present).
`config.DATA_FORMAT = "SHC"`.

## What the files look like

Each CSV is **two columns**:

| Column | Meaning | Config constant |
| --- | --- | --- |
| `text` | The news headline (Gujarati) | `config.TEXT_COLUMN` |
| `labels` | The category name | `config.LABEL_COLUMN` |

- **Encoding:** read with `utf-8-sig` (`config.TRAIN_ENCODING`), which strips a
  possible byte-order mark at the start of the file. Gujarati text is Unicode, so
  correct encoding matters.
- A headline can be **long** (a full multi-sentence article body in some rows) or
  a **short** title.

### Illustrative shape of a row

```text
text  : "ભારતે નવો રેકોર્ડ બનાવ્યો"          (illustrative)
labels: "Nation"
```

## Sizes: raw vs. cleaned

Two sets of counts matter, and they differ because preprocessing **drops empty
or invalid rows**.

| Split | Rows in raw CSV | Rows after cleaning |
| --- | --- | --- |
| Train | 79,416 | **78,806** |
| Validation | 10,251 | **10,175** |
| Test | 10,041 | **9,972** |

The cleaned counts (78,806 / 10,175 / 9,972) are the numbers every later stage
uses and asserts. They are the project's canonical row counts.

## Missing values and data quirks

The raw files contain some missing values. The inspection report found:

| Split | Missing `text` | Missing `labels` |
| --- | --- | --- |
| Train | 605 | 0 |
| Validation | 74 | 3 |
| Test | 69 | 3 |

Rows with an empty/invalid text (or unusable label) are removed during
preprocessing, which is why cleaned counts are smaller. The 3 missing validation
labels and 3 missing test labels explain part of the reduction there.

## Class distribution (raw CSVs)

The dataset is **imbalanced**: State and International are roughly 5× larger than
Auto. This directly shapes the metrics used later — **macro-F1** weights every
class equally and is the project's selection metric precisely because plain
accuracy would flatter the big classes.

**Train (raw, 79,416 rows):**

| Category | Count |
| --- | --- |
| State | 12,552 |
| International | 12,195 |
| Entertainment | 8,895 |
| Business | 8,871 |
| Nation | 8,742 |
| Technology | 7,881 |
| Sports | 7,116 |
| Astro | 6,897 |
| Crime | 3,720 |
| Auto | 2,547 |

**Validation (raw, 10,251 rows):**

| Category | Count |
| --- | --- |
| State | 1,629 |
| International | 1,494 |
| Nation | 1,182 |
| Business | 1,179 |
| Entertainment | 1,134 |
| Sports | 1,044 |
| Technology | 927 |
| Astro | 840 |
| Crime | 486 |
| Auto | 333 |
| *(missing)* | 3 |

## The cardinal rule: split integrity

The three splits are **fixed and never reshuffled**:

- The model is **trained** only on train.
- **Everything that involves a choice** (TF-IDF vocabulary, model selection) uses
  train + validation.
- The **test split is touched exactly once**, after the configuration is frozen,
  to produce the final numbers.

This is the standard way to avoid **data leakage** (accidentally letting the
model "see" the answers). The plans and reports repeatedly assert the exact row
counts and positional label alignment to guarantee no row ever crosses splits.

## Known quirks and their consequences

1. **Morphological richness.** Gujarati packs grammar into word endings, so the
   same idea appears as many surface forms; `min_count=2` in Word2Vec keeps only
   handles the rare tail.
2. **Mixed script.** Many headlines contain ASCII (English) fragments — brand
   names, "By:", `#hashtags`, URLs, "Updated at : ... (IST)". The error report
   found **6,259 validation headlines** contain ASCII letters. Cleaning removes
   URLs/HTML/emails but keeps word characters, so some English survives.
3. **Very long "headlines".** Some rows are full article bodies (hundreds of
   tokens), while others are a few words. Length affects behaviour: the error
   analysis showed an error rate of ~0.13 for 8–15 token rows vs. ~0.099 for
   16+ token rows.
4. **Genuinely overlapping categories.** State vs. Nation vs. International, and
   Business vs. Nation vs. Technology, share vocabulary and topic space. These
   are the dominant confusions (Chapter 08).

## Why this dataset is a good fit for the approach

It is big enough to train Word2Vec embeddings from scratch (tens of thousands of
headlines) but small enough that classical models converge quickly and results
stay reproducible on a laptop. That is exactly the regime where
**Word2Vec + gradient boosting** shines without needing GPUs or transformers.

Next: [Chapter 04 — Text preprocessing](04_preprocessing.md).
