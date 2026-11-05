"""
rebuild_splits.py
-----------------
Rebuilds the train, validation and test splits of Chapter 5 of the thesis from the public
corpora, checks every sentence against its SHA-1 identifier, and checks the class counts
against Table 5.2 of the thesis.

    pip install pandas openpyxl datasets
    python rebuild_splits.py --bhaav "path/to/Bhaav-Dataset.xlsx"

Bhaav-Dataset.xlsx is in the official BHAAV release (Zenodo, DOI 10.5281/zenodo.3457467),
in the folder Datasets/Bhaav/. GoEmotions is downloaded from Hugging Face. The translated
Hindi surprise sentences are in data/xed_hindi_surprise.csv of this release.

Output: rebuilt/en_train.csv, en_val.csv, en_test.csv, hi_train.csv, hi_val.csv, hi_test.csv
(columns text, label), in the order used in the experiments.
Use --only en or --only hi to rebuild one language.
"""

import argparse
import hashlib
import os
import re
import sys

import pandas as pd

try:                                   # Hindi text in messages on a Windows console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = ("train", "val", "test")
CLASSES = ("joy", "sadness", "anger", "fear", "surprise")
TABLE_5_2 = {
    "en": {"train": (16521, 2420, 5341, 630, 4516),
           "val":   (2119, 277, 685, 88, 527),
           "test":  (1994, 291, 703, 93, 579)},
    "hi": {"train": (1962, 2510, 1167, 1203, 1066),
           "val":   (245, 314, 146, 150, 134),
           "test":  (245, 314, 146, 151, 134)},
}


def clean_english(text):
    """The cleaning applied to GoEmotions comments in the experiments."""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)      # URLs
    text = re.sub(r"@\w+", " ", text)                 # user mentions
    text = re.sub(r"#", "", text)                     # hash characters
    text = re.sub(r"<[^>]+>", " ", text)              # HTML tags
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)        # runs of 3+ identical characters
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sha1(text):
    return hashlib.sha1(str(text).strip().encode("utf-8")).hexdigest()


def report(lang, split, df, bad):
    vc = df["label"].value_counts()
    got = tuple(int(vc.get(c, 0)) for c in CLASSES)
    ok = bad == 0 and got == TABLE_5_2[lang][split]
    note = []
    if bad:
        note.append(f"{bad} sentences do not match their identifier")
    if got != TABLE_5_2[lang][split]:
        note.append("counts differ from Table 5.2 " + str(TABLE_5_2[lang][split]))
    print(f"  {lang}_{split:6}" + "".join(f"{n:>9,}" for n in got) + f"{len(df):>9,}   "
          + ("ok" if ok else "PROBLEM: " + "; ".join(note)))
    return ok


def english(out_dir):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install the Hugging Face datasets package first: pip install datasets")
    raw = load_dataset("google-research-datasets/go_emotions", "simplified")
    ok = True
    for split, hf_split in zip(SPLITS, ("train", "validation", "test")):
        text_by_id = {ex["id"]: ex["text"] for ex in raw[hf_split]}
        ids = pd.read_csv(os.path.join(HERE, "splits", f"en_{split}.csv"))
        missing = [i for i in ids["goemotions_id"] if i not in text_by_id]
        if missing:
            print(f"  en_{split}: {len(missing)} comment IDs not found in GoEmotions, e.g. {missing[:3]}")
            ok = False
            continue
        texts = [clean_english(text_by_id[i]) for i in ids["goemotions_id"]]
        bad = sum(sha1(t) != h for t, h in zip(texts, ids["sha1"]))
        df = pd.DataFrame({"text": texts, "label": ids["label"]})
        df.to_csv(os.path.join(out_dir, f"en_{split}.csv"), index=False, encoding="utf-8")
        ok = report("en", split, df, bad) and ok
    return ok


def hindi(bhaav_path, out_dir):
    off = pd.read_excel(bhaav_path)
    sentences = off["Sentences"].astype(str).str.strip().tolist()
    xed = pd.read_csv(os.path.join(HERE, "data", "xed_hindi_surprise.csv"))
    xed_text = dict(zip(xed["row"].astype(int), xed["text"].astype(str)))
    ok = True
    for split in SPLITS:
        ids = pd.read_csv(os.path.join(HERE, "splits", f"hi_{split}.csv"))
        texts = [sentences[int(r)] if src == "bhaav" else xed_text[int(r)]
                 for src, r in zip(ids["source"], ids["row"])]
        bad = sum(sha1(t) != h for t, h in zip(texts, ids["sha1"]))
        df = pd.DataFrame({"text": texts, "label": ids["label"]})
        df.to_csv(os.path.join(out_dir, f"hi_{split}.csv"), index=False, encoding="utf-8")
        ok = report("hi", split, df, bad) and ok
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bhaav", help="path to Bhaav-Dataset.xlsx from the official BHAAV release")
    ap.add_argument("--only", choices=["en", "hi"], help="rebuild one language only")
    ap.add_argument("--out", default=os.path.join(HERE, "rebuilt"), help="output folder")
    args = ap.parse_args()
    if args.only != "en" and not args.bhaav:
        ap.error("--bhaav is needed for the Hindi splits (or use --only en)")
    os.makedirs(args.out, exist_ok=True)

    print("  split   " + "".join(f"{c:>9}" for c in CLASSES) + f"{'total':>9}")
    ok = True
    if args.only != "hi":
        ok = english(args.out) and ok
    if args.only != "en":
        ok = hindi(args.bhaav, args.out) and ok
    if ok:
        print(f"\nAll sentences match their identifiers and the counts match Table 5.2.\n"
              f"Rebuilt splits: {args.out}")
    else:
        sys.exit("\nSome checks failed; see PROBLEM lines above.")


if __name__ == "__main__":
    main()
