# English–Hindi emotion benchmark: splits, translations and code

This repository releases the exact train, validation and test splits used in Chapter 5,
"Cross-Lingual Emotion Detection in English and Hindi: Benchmark and Transfer", of the PhD thesis

> Rakesh Sahani, *A Study and Design of a Deep Learning Technique for Identification of Emotion
> in Text Based on Semantic and Syntactic Relations*, Department of Computer Science and
> Engineering, Rajiv Gandhi University, 2026.

The splits are released as sentence identifiers, so they can be rebuilt from the public corpora
without redistributing them. The Hindi surprise sentences are machine translations made for this
work and exist nowhere else, so they are released in full.

## Contents

| Path | What it holds |
|---|---|
| `splits/en_train.csv`, `en_val.csv`, `en_test.csv` | One row per English comment: GoEmotions comment ID, label, SHA-1 identifier |
| `splits/hi_train.csv`, `hi_val.csv`, `hi_test.csv` | One row per Hindi sentence: source (`bhaav` or `xed_hindi`), row number in the source file, label, SHA-1 identifier |
| `data/xed_hindi_surprise.csv` | The 1,334 machine-translated Hindi surprise sentences, with their split |
| `rebuild_splits.py` | Rebuilds every split from the public corpora and checks it |
| `code/` | The experiment code of Chapter 5 |
| `results/` | Saved per-seed results and predictions behind the tables of Chapter 5 |
| `SHA256SUMS.txt` | Checksums of every file in this release |

Rows in each split file are in the order used in the experiments. Labels are `joy`, `sadness`,
`anger`, `fear` and `surprise`. The predictions in `results/predictions_en_3seed.json` and
`results/predictions_hi_3seed.json` are in the row order of `splits/en_test.csv` and
`splits/hi_test.csv`.

## Sentences per class and split (Table 5.2 of the thesis)

| Class | EN train | EN val | EN test | HI train | HI val | HI test |
|---|---:|---:|---:|---:|---:|---:|
| Joy | 16,521 | 2,119 | 1,994 | 1,962 | 245 | 245 |
| Sadness | 2,420 | 277 | 291 | 2,510 | 314 | 314 |
| Anger | 5,341 | 685 | 703 | 1,167 | 146 | 146 |
| Fear | 630 | 88 | 93 | 1,203 | 150 | 151 |
| Surprise | 4,516 | 527 | 579 | 1,066 | 134 | 134 |
| Total | 29,428 | 3,696 | 3,660 | 7,908 | 989 | 990 |

## Rebuilding the splits

1. Download the official BHAAV release from Zenodo (DOI
   [10.5281/zenodo.3457467](https://doi.org/10.5281/zenodo.3457467)) and unzip it.
2. Install the requirements: `pip install pandas openpyxl datasets`
3. Run, giving the path to `Datasets/Bhaav/Bhaav-Dataset.xlsx` from the unzipped release:

   ```
   python rebuild_splits.py --bhaav "path/to/Datasets/Bhaav/Bhaav-Dataset.xlsx"
   ```

The script downloads GoEmotions from Hugging Face, writes the six splits (columns `text`,
`label`) to `rebuilt/`, checks every sentence against its SHA-1 identifier and checks the
counts against the table above.

## How the data were built

**English.** GoEmotions, simplified configuration (`google-research-datasets/go_emotions`), with
its official train, validation and test splits. The 27 fine-grained labels are mapped to five
classes using the Ekman grouping published with GoEmotions, except that desire and embarrassment
are dropped, along with disgust and neutral. Comments with no remaining label are removed. A
comment whose remaining labels map to more than one class keeps the first of them in the order
in which the dataset lists its labels. The text is cleaned as in `clean_english` in
`rebuild_splits.py`.

**Hindi, four classes.** BHAAV, official release (codes 0 anger, 1 joy, 2 sad, 3 suspense,
4 neutral). Neutral is dropped and suspense is used as fear. Repeated copies of a sentence are
removed, keeping the first, and sentences that occur with conflicting labels are removed
entirely, leaving 8,553 sentences, split 80/10/10 with stratification by class and seed 42.
`row` in `splits/hi_*.csv` is the 0-based row of the sentence in `Bhaav-Dataset.xlsx`
(spreadsheet row = `row` + 2).

**Hindi surprise.** BHAAV has no surprise label. The 1,385 English sentences of XED whose labels
include surprise (label 7) were machine-translated into Hindi with the MarianMT model
`Helsinki-NLP/opus-mt-en-hi`, giving 1,334 distinct sentences, split 80/10/10 with seed 42.
Their labels come from annotation of the English originals, not of the Hindi text.

**Identifiers.** The SHA-1 identifier of a sentence is the SHA-1 hash of its UTF-8 text with
leading and trailing spaces removed; for English, of the cleaned text.

## Licences and sources

The files in `splits/`, `data/` and `results/` are released under the Creative Commons
Attribution 4.0 International licence (CC BY 4.0); the code is released under the MIT licence
(see `LICENSE`). The source corpora keep their own licences:

- GoEmotions: Demszky et al., "GoEmotions: A dataset of fine-grained emotions", ACL 2020
  (Apache 2.0).
- BHAAV: Kumar et al., "BHAAV: A text corpus for emotion analysis from Hindi stories", 2019,
  Zenodo DOI 10.5281/zenodo.3457467 (CC BY 4.0).
- XED: Öhman et al., "XED: A multilingual dataset for sentiment analysis and emotion detection",
  COLING 2020 (CC BY 4.0). The translations in `data/xed_hindi_surprise.csv` are derived from it.
- OPUS-MT: Tiedemann and Thottingal, "OPUS-MT – Building open translation services for the
  World", EAMT 2020.

If you use this release, please cite the thesis and the corpora above.
