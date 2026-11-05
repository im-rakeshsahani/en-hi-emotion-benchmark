"""
rebuild_bhaav_from_zenodo.py
----------------------------
Replaces the mislabelled BHAAV mirror files with splits built from the authors'
official release (Zenodo DOI 10.5281/zenodo.3457467, CC BY 4.0).

Put this file in the emotion_pipeline project root. With the official zip saved as
data/bhaav_official.zip, run:

    python rebuild_bhaav_from_zenodo.py

What it does
  1. Verifies the zip (MD5 published on Zenodo) and unzips it into data/bhaav_official/.
  2. Reads Datasets/Bhaav/Bhaav-Dataset.xlsx (columns 'Sentences', 'Annotation') and
     checks that every other Bhaav-Dataset*.xlsx copy in the release agrees with it.
  3. Checks the annotation codes against BHAAV's published class sizes
     (Kumar et al. 2019, Table 3): 0 anger 1,464 | 1 joy 2,463 | 2 sad 3,168 |
     3 suspense 1,512 | 4 neutral 11,697. Stops without writing if they differ.
  4. Drops neutral, empty rows and duplicate sentences (a sentence that appears
     with conflicting labels is removed entirely).
  5. Makes a stratified 80/10/10 split with seed 42.
  6. Backs up the current data/bhaav_*.csv files (*.mirror_backup.csv) and writes
     new ones with the columns your loader expects: ['text', 'emotion'],
     emotion in {joy, sadness, anger, suspense}. The loader maps suspense -> fear.

Afterwards: delete saved_models/hi_*.pt and any cached Hindi data, then run
ch5_counts_and_split_ids.py again for the final Hindi half of Table 5.2.
"""

import os
import sys
import glob
import shutil
import zipfile
import hashlib

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import pandas as pd
from sklearn.model_selection import train_test_split
from config import BHAAV_TRAIN_CSV, BHAAV_VAL_CSV, BHAAV_TEST_CSV, DATA_DIR

ZIP_PATH = os.path.join(DATA_DIR, "bhaav_official.zip")
ZENODO_MD5 = "cdcaf1997a422f18ee12d713a17ffdea"     # from the Zenodo record page
OFFICIAL_DIR = os.path.join(DATA_DIR, "bhaav_official")
MAIN_FILE = os.path.join("Datasets", "Bhaav", "Bhaav-Dataset.xlsx")

CODE_TO_CLASS = {0: "anger", 1: "joy", 2: "sad", 3: "suspense", 4: "neutral"}
PUBLISHED = {"anger": 1464, "joy": 2463, "sad": 3168, "suspense": 1512, "neutral": 11697}
TO_PIPELINE = {"joy": "joy", "sad": "sadness", "anger": "anger", "suspense": "suspense"}
SEED = 42


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract():
    if os.path.isfile(os.path.join(OFFICIAL_DIR, MAIN_FILE)):
        print(f"Using extracted release in {OFFICIAL_DIR}")
        return
    if not os.path.isfile(ZIP_PATH):
        sys.exit("Official zip not found. Download Datasets-20190922T151602Z-001.zip from\n"
                 "  https://zenodo.org/records/3457467\n"
                 f"and save it as\n  {ZIP_PATH}")
    got = md5_of(ZIP_PATH)
    if got != ZENODO_MD5:
        sys.exit(f"{ZIP_PATH} does not match Zenodo's MD5 ({got} != {ZENODO_MD5}). Download it again.")
    print(f"Official zip verified (md5 {got}). Extracting ...")
    os.makedirs(OFFICIAL_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(OFFICIAL_DIR)


def load_release(path):
    df = pd.read_excel(path)
    if not {"Sentences", "Annotation"}.issubset(df.columns):
        sys.exit(f"{path} has columns {list(df.columns)}; expected 'Sentences' and 'Annotation'.")
    df = df[["Sentences", "Annotation"]].copy()
    df["Sentences"] = df["Sentences"].astype(str).str.strip()
    return df


def main():
    extract()
    main_path = os.path.join(OFFICIAL_DIR, MAIN_FILE)
    df = load_release(main_path)
    print(f"Read {MAIN_FILE}: {len(df)} rows")

    # every other copy of the dataset in the release should agree
    for other in glob.glob(os.path.join(OFFICIAL_DIR, "**", "Bhaav-Dataset*.xlsx"), recursive=True):
        if os.path.abspath(other) == os.path.abspath(main_path):
            continue
        o = load_release(other)
        same_len = len(o) == len(df)
        lab_diff = int((o["Annotation"].values != df["Annotation"].values).sum()) if same_len else None
        txt_diff = int((o["Sentences"].values != df["Sentences"].values).sum()) if same_len else None
        print(f"  copy {os.path.relpath(other, OFFICIAL_DIR)}: rows={len(o)}, "
              f"label differences={lab_diff}, text differences={txt_diff}")

    counts = df["Annotation"].value_counts().to_dict()
    print("\nAnnotation code counts:", dict(sorted(counts.items())))
    for code, cls in CODE_TO_CLASS.items():
        n = int(counts.get(code, 0))
        ok = n == PUBLISHED[cls]
        print(f"  code {code} -> {cls:9} n={n:6}  published={PUBLISHED[cls]:6}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            sys.exit("Code counts do not match BHAAV's published distribution. Nothing was written.")

    df["cls"] = df["Annotation"].map(CODE_TO_CLASS)
    df = df[(df["cls"] != "neutral") & (df["Sentences"].str.len() > 0) & (df["Sentences"] != "nan")]
    n0 = len(df)
    per_text = df.groupby("Sentences")["cls"].nunique()
    conflicts = set(per_text[per_text > 1].index)
    df = df[~df["Sentences"].isin(conflicts)].drop_duplicates(subset="Sentences", keep="first")
    print(f"\nEmotion rows (neutral dropped): {n0}. Sentences with conflicting labels removed: "
          f"{len(conflicts)}. Rows removed in total (conflicts + duplicates): {n0 - len(df)}. "
          f"Kept: {len(df)}.")

    df = df.rename(columns={"Sentences": "text"})
    df["emotion"] = df["cls"].map(TO_PIPELINE)
    train, rest = train_test_split(df, test_size=0.20, stratify=df["emotion"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["emotion"], random_state=SEED)

    for p in (BHAAV_TRAIN_CSV, BHAAV_VAL_CSV, BHAAV_TEST_CSV):
        if os.path.exists(p):
            bak = p.replace(".csv", ".mirror_backup.csv")
            if not os.path.exists(bak):
                shutil.copy2(p, bak)
                print(f"Backed up {os.path.basename(p)} -> {os.path.basename(bak)}")
    for part, p in ((train, BHAAV_TRAIN_CSV), (val, BHAAV_VAL_CSV), (test, BHAAV_TEST_CSV)):
        part[["text", "emotion"]].to_csv(p, index=False, encoding="utf-8")

    summary = pd.DataFrame({s: d["emotion"].value_counts() for s, d in
                            (("train", train), ("val", val), ("test", test))}).fillna(0).astype(int)
    summary["total"] = summary.sum(axis=1)
    summary.loc["TOTAL"] = summary.sum()
    print("\nNew BHAAV splits written (XED surprise is added later by the loader):")
    print(summary.to_string())
    print("\nNext: delete saved_models\\hi_*.pt (already done) and run python ch5_counts_and_split_ids.py")


if __name__ == "__main__":
    main()
