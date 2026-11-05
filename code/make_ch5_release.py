"""
make_ch5_release.py
-------------------
Builds the public release of the Chapter 5 splits (the "split IDs" asked for in the
examiner's report) in a new folder, ch5_release/, ready to upload to GitHub.

Put this file and the release_kit folder in the emotion_pipeline project root (next to
run_experiments.py), then run:

    python make_ch5_release.py

It reads the splits through your own loaders (src/data_utils.py), so what it releases is
exactly what your experiments used. It needs the files those loaders read:

    data/bhaav_train.csv, data/bhaav_val.csv, data/bhaav_test.csv
    data/bhaav_official/Datasets/Bhaav/Bhaav-Dataset.xlsx   (the unzipped official release)
    data/xed_hindi_surprise.csv                               (the 1,334 translated sentences)
    GoEmotions from Hugging Face (cached by your earlier runs)

Every check must pass before anything is written; otherwise it stops and says what to fix.
"""

import glob
import hashlib
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
try:                                   # Hindi text in messages on a Windows console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd

OUT = os.path.join(ROOT, "ch5_release")
KIT = os.path.join(ROOT, "release_kit")
BHAAV_XLSX = os.path.join("data", "bhaav_official", "Datasets", "Bhaav", "Bhaav-Dataset.xlsx")
XED_CSV = os.path.join("data", "xed_hindi_surprise.csv")

SPLITS = ("train", "val", "test")
CLASSES = ("joy", "sadness", "anger", "fear", "surprise")

# Table 5.2 of the thesis: sentences per class (joy, sadness, anger, fear, surprise) and split
TABLE_5_2 = {
    "en": {"train": (16521, 2420, 5341, 630, 4516),
           "val":   (2119, 277, 685, 88, 527),
           "test":  (1994, 291, 703, 93, 579)},
    "hi": {"train": (1962, 2510, 1167, 1203, 1066),
           "val":   (245, 314, 146, 150, 134),
           "test":  (245, 314, 146, 151, 134)},
}

# Official BHAAV codes kept in Chapter 5 (3 = suspense, used as fear; 4 = neutral, dropped)
BHAAV_CODE = {0: "anger", 1: "joy", 2: "sadness", 3: "fear"}

# Code published with the splits (copied only if present in your project)
CODE_FILES = ["config.py", "run_experiments.py", "requirements.txt",
              "src/__init__.py", "src/data_utils.py", "src/models.py", "src/trainer.py",
              "rebuild_bhaav_from_zenodo.py", "ch5_counts_and_split_ids.py",
              "build_ch5_tables.py", "make_ch5_release.py"]
# Saved results behind the Chapter 5 tables (copied only if present)
RESULT_GLOBS = ["results/*_3seed.json", "results/table5_2_counts_*.csv",
                "branch_training_curves.json"]
# A file is not published if it seems to contain a key, token or password
SECRET = re.compile(r"(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{8,}"
                    r"|\bsk-[A-Za-z0-9_-]{16,}|\bhf_[A-Za-z0-9]{20,}", re.IGNORECASE)


def fail(msg):
    print("\nSTOPPED: " + msg)
    print("Nothing was written to ch5_release.")
    sys.exit(1)


def sha1(text):
    """Split identifier: SHA-1 of the UTF-8 sentence text, leading and trailing spaces removed."""
    return hashlib.sha1(str(text).strip().encode("utf-8")).hexdigest()


def english_splits():
    try:
        from datasets import load_dataset
    except ImportError:
        fail("the Hugging Face 'datasets' package is not installed (pip install datasets).")
    from config import GOEMOTIONS_TO_5CLASS
    from src.data_utils import build_english_dataset, clean_english

    print("English: reading GoEmotions with its comment IDs ...")
    raw = load_dataset("google-research-datasets/go_emotions", "simplified")
    names = raw["train"].features["labels"].feature.names
    used = dict(zip(SPLITS, build_english_dataset()))
    out = {}
    for split, hf_split in zip(SPLITS, ("train", "validation", "test")):
        rows = []
        for ex in raw[hf_split]:
            mapped = [GOEMOTIONS_TO_5CLASS[names[i]] for i in ex["labels"]
                      if names[i] in GOEMOTIONS_TO_5CLASS]
            if mapped:                       # same rule as load_goemotions: first surviving label
                text = clean_english(ex["text"])
                rows.append((ex["id"], mapped[0], sha1(text), text))
        df = pd.DataFrame(rows, columns=["goemotions_id", "label", "sha1", "text"])
        exp = used[split]
        same = (len(df) == len(exp)
                and df["text"].tolist() == exp["text"].astype(str).tolist()
                and df["label"].tolist() == exp["label"].tolist())
        if not same:
            fail(f"English {split}: the rows rebuilt with comment IDs differ from the data your "
                 "experiments loaded (build_english_dataset). Has src/data_utils.py changed?")
        out[split] = df[["goemotions_id", "label", "sha1"]]
    return out


def hindi_splits():
    from src.data_utils import build_hindi_dataset

    xlsx, xcsv = os.path.join(ROOT, BHAAV_XLSX), os.path.join(ROOT, XED_CSV)
    if not os.path.isfile(xlsx):
        fail(f"{BHAAV_XLSX} not found. Run rebuild_bhaav_from_zenodo.py once; it unzips the "
             "official release there.")
    if not os.path.isfile(xcsv):
        fail(f"{XED_CSV} not found; it holds the 1,334 translated surprise sentences.")

    print("Hindi: reading the official BHAAV file and the translated XED sentences ...")
    off = pd.read_excel(xlsx)
    if not {"Sentences", "Annotation"}.issubset(off.columns):
        fail(f"{BHAAV_XLSX} has columns {list(off.columns)}; expected 'Sentences' and 'Annotation'.")
    off = off[["Sentences", "Annotation"]].copy()
    off["Sentences"] = off["Sentences"].astype(str).str.strip()
    off["Annotation"] = pd.to_numeric(off["Annotation"], errors="coerce")
    first_row = {}                            # first occurrence, as in rebuild_bhaav_from_zenodo.py
    for idx, (s, code) in zip(off.index, zip(off["Sentences"], off["Annotation"])):
        if code in BHAAV_CODE:
            first_row.setdefault(s, int(idx))

    xed = pd.read_csv(xcsv)
    xed_text = xed["text"].astype(str).str.strip().tolist()
    xed_row = {t: i for i, t in enumerate(xed_text)}
    if len(xed_row) != len(xed_text):
        fail(f"{XED_CSV} contains repeated sentences; expected 1,334 distinct ones.")

    used = dict(zip(SPLITS, build_hindi_dataset()))
    out, xed_split, seen = {}, {}, set()
    for split in SPLITS:
        rows = []
        for text, label in zip(used[split]["text"].astype(str), used[split]["label"]):
            t = text.strip()
            if label == "surprise":
                if t not in xed_row:
                    fail(f"a Hindi {split} surprise sentence is not in {XED_CSV}: {t[:60]}")
                key = ("xed_hindi", xed_row[t])
                xed_split[xed_row[t]] = split
            else:
                if t not in first_row:
                    fail(f"a Hindi {split} sentence is not in the official BHAAV file: {t[:60]}")
                r = first_row[t]
                if BHAAV_CODE[int(off.at[r, "Annotation"])] != label:
                    fail(f"BHAAV row {r} has a different label in the official file than in "
                         f"your {split} split ({label}).")
                key = ("bhaav", r)
            if key in seen:
                fail(f"{key[0]} row {key[1]} occurs twice in the Hindi splits.")
            seen.add(key)
            rows.append((key[0], key[1], label, sha1(t)))
        out[split] = pd.DataFrame(rows, columns=["source", "row", "label", "sha1"])

    if len(xed_split) != len(xed_text):
        fail(f"{len(xed_text) - len(xed_split)} sentences in {XED_CSV} are in no split.")
    translations = pd.DataFrame({"row": range(len(xed_text)),
                                 "split": [xed_split[i] for i in range(len(xed_text))],
                                 "label": "surprise", "text": xed_text})
    for col in xed.columns:                  # keep any extra column, e.g. the English source
        if col not in ("text", "label"):
            translations[col] = xed[col].values
    return out, translations


def check_counts(lang, splits):
    name = "English" if lang == "en" else "Hindi"
    print(f"\n{name}: sentences per class, checked against Table 5.2")
    print("  " + f"{'split':7}" + "".join(f"{c:>10}" for c in CLASSES) + f"{'total':>9}")
    ok = True
    for s in SPLITS:
        vc = splits[s]["label"].value_counts()
        got = tuple(int(vc.get(c, 0)) for c in CLASSES)
        exp = TABLE_5_2[lang][s]
        ok &= got == exp
        flag = "ok" if got == exp else "MISMATCH (Table 5.2: " + ", ".join(f"{n:,}" for n in exp) + ")"
        print("  " + f"{s:7}" + "".join(f"{n:>10,}" for n in got) + f"{sum(got):>9,}   {flag}")
    return ok


def copy_listed(rel_paths, dst_root):
    copied, withheld = [], []
    for rel in rel_paths:
        src = os.path.join(ROOT, rel)
        if not os.path.isfile(src):
            continue
        with open(src, encoding="utf-8", errors="replace") as f:
            hit = SECRET.search(f.read())
        if hit:
            withheld.append((rel, hit.group(0)[:40]))
            continue
        dst = os.path.join(dst_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
    return copied, withheld


def main():
    if os.path.exists(OUT):
        fail("ch5_release already exists. Delete or rename it, then run this again.")
    if not os.path.isdir(KIT):
        fail("the release_kit folder was not found next to this script.")

    hi, translations = hindi_splits()
    en = english_splits()
    ok = check_counts("en", en)
    ok = check_counts("hi", hi) and ok
    if not ok:
        fail("the counts above do not match Table 5.2.")

    os.makedirs(os.path.join(OUT, "splits"))
    os.makedirs(os.path.join(OUT, "data"))
    for lang, parts in (("en", en), ("hi", hi)):
        for s in SPLITS:
            parts[s].to_csv(os.path.join(OUT, "splits", f"{lang}_{s}.csv"), index=False,
                            encoding="utf-8")
    translations.to_csv(os.path.join(OUT, "data", "xed_hindi_surprise.csv"), index=False,
                        encoding="utf-8")
    for name in os.listdir(KIT):
        shutil.copy2(os.path.join(KIT, name), os.path.join(OUT, name))

    code, held1 = copy_listed(CODE_FILES, os.path.join(OUT, "code"))
    results = sorted({os.path.relpath(p, ROOT).replace(os.sep, "/")
                      for g in RESULT_GLOBS for p in glob.glob(os.path.join(ROOT, g))})
    res, held2 = copy_listed(results, os.path.join(OUT, "results"))

    sums = []
    for dirpath, _, files in os.walk(OUT):
        for fn in sorted(files):
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, OUT).replace(os.sep, "/")
            if rel != "SHA256SUMS.txt":
                with open(p, "rb") as f:
                    sums.append(f"{hashlib.sha256(f.read()).hexdigest()}  {rel}")
    with open(os.path.join(OUT, "SHA256SUMS.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(sums, key=lambda x: x.split("  ", 1)[1])) + "\n")

    print(f"\nRelease written to {OUT}")
    print(f"  splits/            6 files ({sum(len(d) for d in en.values()):,} English and "
          f"{sum(len(d) for d in hi.values()):,} Hindi sentences)")
    print(f"  data/              xed_hindi_surprise.csv ({len(translations):,} translated sentences)")
    print(f"  code/              {len(code)} files: {', '.join(code)}")
    print(f"  results/           {len(res)} files")
    for rel, what in held1 + held2:
        print(f"  NOT PUBLISHED: {rel} seems to contain a key or password ({what}...). "
              "Remove it from the file, then copy the file in by hand.")
    print("\nNext: test it with rebuild_splits.py (step 3 of the instructions).")


if __name__ == "__main__":
    main()
