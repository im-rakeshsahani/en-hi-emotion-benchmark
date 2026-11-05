"""
ch5_counts_and_split_ids.py
---------------------------
Put this file in the emotion_pipeline project root (next to run_experiments.py) and run:

    python ch5_counts_and_split_ids.py

It does three things, using only your existing loaders:

  1. Writes 10 random Hindi training sentences per pipeline label to
     results/hindi_label_samples.txt (UTF-8; open it in Notepad) so you can
     read what each label really contains.
  2. Checks the Hindi class totals against BHAAV's published distribution
     (Kumar et al. 2019, arXiv:1910.04073, Table 3) and reports which BHAAV
     class each pipeline label matches.
  3. Prints the exact per-class x split counts for English and Hindi (Table 5.2)
     and writes split-ID files to results/split_ids/ for public release.
     IDs are SHA-1 hashes of the sentence text, so the split can be rebuilt
     without redistributing BHAAV text (research-use licence).

Run it once now (on the current loader) and again after correcting the BHAAV
label mapping; the second run gives the numbers for the revised Table 5.2.
"""

import os
import sys
import hashlib

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pandas as pd
from src.data_utils import build_english_dataset, build_hindi_dataset

CLASSES = ["joy", "sadness", "anger", "fear", "surprise"]
OUT_DIR = os.path.join(ROOT, "results")
ID_DIR = os.path.join(OUT_DIR, "split_ids")
os.makedirs(ID_DIR, exist_ok=True)

# BHAAV published totals (whole corpus, 20,304 sentences)
BHAAV_PUBLISHED = {"joy": 2463, "anger": 1464, "suspense": 1512, "sad": 3168, "neutral": 11697}
# What a correct harmonisation should give for the BHAAV part
EXPECTED_BHAAV_MAPPED = {"joy": 2463, "sadness": 3168, "anger": 1464, "fear": 1512}


def count_table(name, splits):
    rows = {}
    for split_name, df in splits.items():
        vc = df["label"].value_counts()
        rows[split_name] = {c: int(vc.get(c, 0)) for c in CLASSES}
    t = pd.DataFrame(rows)
    t["total"] = t.sum(axis=1)
    t.loc["TOTAL"] = t.sum()
    t["share_%"] = (t["total"] / t.loc["TOTAL", "total"] * 100).round(1)
    print(f"\n=== {name}: exact counts per class and split ===")
    print(t.to_string())
    csv_path = os.path.join(OUT_DIR, f"table5_2_counts_{name.lower()}.csv")
    t.to_csv(csv_path)
    print(f"  -> saved {csv_path}")
    return t


def write_split_ids(lang, splits):
    for split_name, df in splits.items():
        ids = df["text"].astype(str).str.strip().map(
            lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())
        out = pd.DataFrame({"split": split_name, "label": df["label"].values, "sha1": ids.values})
        path = os.path.join(ID_DIR, f"{lang}_{split_name}.csv")
        out.to_csv(path, index=False)
        dup = int(ids.duplicated().sum())
        print(f"  -> {path}  ({len(out)} rows, {dup} duplicate hashes within split)")


def main():
    # ---------------- Hindi ----------------
    hi_tr, hi_va, hi_te = build_hindi_dataset()
    hi = {"train": hi_tr, "val": hi_va, "test": hi_te}

    sample_path = os.path.join(OUT_DIR, "hindi_label_samples.txt")
    with open(sample_path, "w", encoding="utf-8") as f:
        for c in CLASSES:
            sub = hi_tr[hi_tr["label"] == c]
            f.write(f"\n===== pipeline label: {c.upper()}  (n_train = {len(sub)}) =====\n")
            for s in sub.sample(min(10, len(sub)), random_state=0)["text"]:
                f.write(f"  - {s}\n")
    print(f"\nSample sentences per Hindi label written to: {sample_path}")
    print("Open it and read the FEAR and ANGER blocks: do they read as fear and anger?")

    t_hi = count_table("Hindi", hi)

    print("\n=== Hindi label check against BHAAV's published distribution ===")
    ok = True
    for c in ["joy", "sadness", "anger", "fear"]:
        n = int(t_hi.loc[c, "total"])
        match = [k for k, v in BHAAV_PUBLISHED.items() if v == n]
        exp = EXPECTED_BHAAV_MAPPED[c]
        flag = "OK" if n == exp else "MISMATCH"
        if n != exp:
            ok = False
        print(f"  {c:9} total={n:6}  expected={exp:6}  equals BHAAV class: {match or '-'}  [{flag}]")
    if ok:
        print("  Hindi labels match a correct BHAAV harmonisation (neutral dropped, suspense -> fear).")
    else:
        print("  The BHAAV code->label mapping in the loader is wrong. Fix it before any Hindi rerun,")
        print("  and delete the saved hi_*_seed42.pt checkpoints so the skip-train logic cannot reuse them.")

    write_split_ids("hi", hi)

    # ---------------- English ----------------
    en_tr, en_va, en_te = build_english_dataset()
    en = {"train": en_tr, "val": en_va, "test": en_te}
    count_table("English", en)
    write_split_ids("en", en)

    print("\nDone. Use the two table5_2_counts_*.csv files for Table 5.2 and release results/split_ids/.")


if __name__ == "__main__":
    main()
