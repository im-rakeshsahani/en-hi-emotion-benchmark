"""
src/data_utils.py
─────────────────
Downloads and preprocesses GoEmotions, BHAAV, XED-Hindi.
Applies 5-class label harmonisation defined in config.py.
Produces PyTorch Dataset objects for all branches.
"""

import os, re, csv, logging, requests
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 1.  TEXT CLEANING
# ═══════════════════════════════════════════════════════════

def clean_english(text: str) -> str:
    """Light cleaning for GoEmotions Reddit text."""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)           # URLs
    text = re.sub(r"@\w+", " ", text)                      # mentions
    text = re.sub(r"#", "", text)                           # hashtag marker
    text = re.sub(r"<[^>]+>", " ", text)                   # HTML tags
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)             # repeated chars
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_hindi(text: str) -> str:
    """Light cleaning for BHAAV / XED Hindi text."""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ═══════════════════════════════════════════════════════════
# 2.  GoEmotions (English)  — via HuggingFace datasets
# ═══════════════════════════════════════════════════════════

def load_bhaav():
    """
    Loads BHAAV train/validation/test CSV files.
    Expected CSV columns: text, emotion
    Returns train / val / test DataFrames with columns [text, label].
    Maps BHAAV suspense -> fear and drops neutral.
    """
    import pandas as pd
    from pathlib import Path

    data_dir = Path("data")
    files = {
        "train": data_dir / "bhaav_train.csv",
        "validation": data_dir / "bhaav_val.csv",
        "test": data_dir / "bhaav_test.csv",
    }

    label_map = {
        "joy": "joy",
        "sadness": "sadness",
        "anger": "anger",
        "suspense": "fear",
        "fear": "fear",
        "neutral": None,
    }

    splits = {}

    for split_name, file_path in files.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing BHAAV file: {file_path}")

        df = pd.read_csv(file_path)

        if "emotion" in df.columns:
            label_col = "emotion"
        elif "label" in df.columns:
            label_col = "label"
        elif "Annotation" in df.columns:
            # Official BHAAV codes (Bhaav-Dataset.xlsx, Zenodo 10.5281/zenodo.3457467),
            # identified by the published class sizes:
            #   0 anger 1,464 | 1 joy 2,463 | 2 sad 3,168 | 3 suspense 1,512 | 4 neutral 11,697
            # The earlier map {1: joy, 2: sadness, 3: anger, 4: suspense, 5: neutral} was wrong:
            # it labelled suspense as anger and neutral as suspense (-> fear) and dropped anger.
            num_map = {0: "anger", 1: "joy", 2: "sadness", 3: "suspense", 4: "neutral"}
            df["emotion"] = df["Annotation"].map(num_map)
            label_col = "emotion"
        else:
            raise ValueError(f"No label column found in {file_path}. Columns: {df.columns.tolist()}")

        if "text" not in df.columns:
            if "Sentences" in df.columns:
                df = df.rename(columns={"Sentences": "text"})
            else:
                raise ValueError(f"No text column found in {file_path}. Columns: {df.columns.tolist()}")

        df["label"] = df[label_col].astype(str).str.lower().map(label_map)
        df = df.dropna(subset=["text", "label"])
        df = df[["text", "label"]].reset_index(drop=True)

        splits[split_name] = df
        log.info(f"BHAAV {split_name}: {len(df)} samples")

    return splits["train"], splits["validation"], splits["test"]


def load_goemotions():
    """
    Returns train / validation / test DataFrames with columns [text, label].
    Uses HuggingFace GoEmotions simplified split because it provides official
    train/validation/test splits and a labels field.
    Applies GOEMOTIONS_TO_5CLASS mapping.
    """
    from datasets import load_dataset
    log.info("Loading GoEmotions simplified split from HuggingFace ...")

    raw = load_dataset("google-research-datasets/go_emotions", "simplified")

    label_names = raw["train"].features["labels"].feature.names

    splits = {}
    for split_name in ["train", "validation", "test"]:
        rows = []

        for ex in raw[split_name]:
            text = clean_english(ex["text"])

            labels = [label_names[i] for i in ex["labels"]]

            mapped = [
                GOEMOTIONS_TO_5CLASS[label]
                for label in labels
                if label in GOEMOTIONS_TO_5CLASS
            ]

            if not mapped:
                continue

            rows.append({"text": text, "label": mapped[0]})

        splits[split_name] = pd.DataFrame(rows)
        log.info(f"GoEmotions {split_name}: {len(splits[split_name])} samples")

    return splits["train"], splits["validation"], splits["test"]


def _download_xed_hindi():
    """Downloads XED Hindi surprise data. Falls back gracefully if GitHub API is rate-limited."""
    from pathlib import Path

    direct_urls = [
        "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/Projections/hi-projections.tsv",
        "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/Projections/hin-projections.tsv",
    ]

    for url in direct_urls:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                lines = r.text.splitlines()
                log.info(f"XED Hindi downloaded from {url}: {len(lines)} lines")
                return lines
        except Exception:
            continue

    try:
        tree_url = "https://api.github.com/repos/Helsinki-NLP/XED/git/trees/master?recursive=1"
        r = requests.get(tree_url, timeout=60)
        if r.status_code != 200:
            log.warning(f"GitHub API returned {r.status_code} — likely rate limited.")
            return []
        files = r.json().get("tree", [])
        candidates = [
            item["path"] for item in files
            if item["path"].lower().startswith("projections/")
            and any(x in item["path"].lower() for x in ["/hi.", "/hin.", "hindi"])
        ]
        if not candidates:
            log.warning("No Hindi file found in XED repo tree.")
            return []
        raw_url = "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/" + candidates[0]
        rr = requests.get(raw_url, timeout=120)
        rr.raise_for_status()
        return rr.text.splitlines()
    except Exception as e:
        log.warning(f"XED Hindi download failed: {e}. Returning empty set.")
        return []

    files = r.json().get("tree", [])

    candidates = []
    for item in files:
        p = item.get("path", "")
        low = p.lower()

        if not low.startswith("projections/"):
            continue

        if not (low.endswith(".txt") or low.endswith(".tsv") or low.endswith(".csv")):
            continue

        name = Path(low).name

        # Try likely Hindi identifiers: hi, hin, hindi
        if (
            name in ["hi.txt", "hi.tsv", "hi.csv", "hin.txt", "hin.tsv", "hin.csv"]
            or "hindi" in name
            or "/hindi" in low
            or "/hi." in low
            or "/hin." in low
        ):
            candidates.append(p)

    if not candidates:
        log.warning("No Hindi projection file found in XED repository. Returning empty XED Hindi surprise set.")
        return []

    chosen = candidates[0]
    raw_url = "https://raw.githubusercontent.com/Helsinki-NLP/XED/master/" + chosen
    log.info(f"Downloading XED Hindi candidate: {chosen}")

    rr = requests.get(raw_url, timeout=120)
    rr.raise_for_status()

    lines = rr.text.splitlines()
    log.info(f"Downloaded XED Hindi lines: {len(lines)}")
    return lines


def load_xed_hindi_surprise():
    """
    Loads XED Hindi surprise-only samples.
    XED format is usually: sentence<TAB>label1,label2
    In XED, surprise is label 7.
    Returns DataFrame with columns [text, label].
    """
    import pandas as pd
    from pathlib import Path

    cache_path = Path("data") / "xed_hindi_surprise.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path)
        if "label" not in df.columns:
            df["label"] = "surprise"
        return df[["text", "label"]]

    rows = _download_xed_hindi()
    parsed = []

    for line in rows:
        line = line.strip()
        if not line or "\t" not in line:
            continue

        sent, labels = line.split("\t", 1)
        label_set = {x.strip().lower() for x in labels.replace(";", ",").split(",") if x.strip()}

        # Strict surprise-only selection
        if "7" in label_set or "surprise" in label_set:
            parsed.append({"text": clean_hindi(sent), "label": "surprise"})

    df = pd.DataFrame(parsed, columns=["text", "label"])
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)

    log.info(f"XED Hindi surprise samples: {len(df)}")
    return df


def build_english_dataset():
    """Returns (train_df, val_df, test_df) for English GoEmotions 5-class."""
    return load_goemotions()


def build_hindi_dataset(val_size=0.10, test_size=0.10, random_state=42):
    """
    Combines BHAAV (joy/sadness/anger/fear) + XED-Hindi (surprise).
    Returns (train_df, val_df, test_df).
    """
    from sklearn.model_selection import train_test_split

    bhaav_train, bhaav_val, bhaav_test = load_bhaav()
    xed_surprise = load_xed_hindi_surprise()

    if len(xed_surprise) > 0:
        xed_trainval, xed_test = train_test_split(
            xed_surprise, test_size=test_size, random_state=random_state)
        xed_train, xed_val = train_test_split(
            xed_trainval, test_size=val_size / (1 - test_size), random_state=random_state)
        train_df = pd.concat([bhaav_train, xed_train], ignore_index=True).sample(
            frac=1, random_state=random_state).reset_index(drop=True)
        val_df   = pd.concat([bhaav_val,   xed_val],   ignore_index=True)
        test_df  = pd.concat([bhaav_test,  xed_test],  ignore_index=True)
    else:
        log.warning("XED Hindi is empty — using BHAAV only (4 classes: joy/sadness/anger/fear).")
        train_df = bhaav_train.sample(frac=1, random_state=random_state).reset_index(drop=True)
        val_df   = bhaav_val.reset_index(drop=True)
        test_df  = bhaav_test.reset_index(drop=True)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        log.info(f"  Hindi {name}: {len(df)} | "
                 + " | ".join(f"{k}:{v}"
                               for k, v in sorted(Counter(df["label"]).items())))

    return train_df, val_df, test_df


# ═══════════════════════════════════════════════════════════
# 6.  COMPUTE CLASS WEIGHTS  (for weighted cross-entropy)
# ═══════════════════════════════════════════════════════════

def compute_class_weights(labels: list) -> torch.Tensor:
    """Inverse-frequency class weights as a FloatTensor of shape (NUM_CLASSES,)."""
    counts = Counter(labels)
    total  = sum(counts.values())
    weights = torch.zeros(NUM_CLASSES)
    for label, cnt in counts.items():
        if label in CLASS_TO_IDX:
            weights[CLASS_TO_IDX[label]] = total / (NUM_CLASSES * cnt)
    return weights


# ═══════════════════════════════════════════════════════════
# 7.  PYTORCH DATASET CLASSES
# ═══════════════════════════════════════════════════════════

class TransformerDataset(Dataset):
    """Tokenised dataset for HuggingFace Transformer branch."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = TRANSFORMER_MAX_LEN):
        self.texts  = df["text"].tolist()
        self.labels = [CLASS_TO_IDX[l] for l in df["label"]]
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids",
                                      torch.zeros(self.max_len, dtype=torch.long)).squeeze(0),
            "labels"        : torch.tensor(self.labels[idx], dtype=torch.long),
        }


class EmbeddingDataset(Dataset):
    """
    Pre-tokenised dataset for RNN / CNN models.
    Returns padded/truncated token-index sequences.
    """

    def __init__(self, df: pd.DataFrame, vocab: dict, max_len: int = RNN_MAX_LEN,
                 lang: str = "en"):
        self.max_len = max_len
        self.vocab   = vocab
        self.labels  = [CLASS_TO_IDX[l] for l in df["label"]]
        self.sequences = [self._encode(t, lang) for t in df["text"]]

    def _tokenise(self, text: str, lang: str) -> list:
        if lang == "hi":
            return text.split()   # simple whitespace tokenisation for Hindi
        return text.lower().split()

    def _encode(self, text: str, lang: str) -> list:
        tokens = self._tokenise(text, lang)[:self.max_len]
        ids    = [self.vocab.get(t, 1) for t in tokens]   # 1 = <UNK>
        # Pad to max_len
        ids   += [0] * (self.max_len - len(ids))          # 0 = <PAD>
        return ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.sequences[idx], dtype=torch.long),
            "labels"   : torch.tensor(self.labels[idx],    dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════
# 8.  EMBEDDING MATRIX BUILDER
# ═══════════════════════════════════════════════════════════

def build_vocab_and_matrix(texts: list, vec_path: str, lang: str = "en"):
    """
    Build vocab dict and embedding matrix from a .vec file.
    vocab: token → index  (0=PAD, 1=UNK, 2..N = real tokens)
    matrix: np.ndarray of shape (|vocab|, EMBED_DIM)
    """
    log.info(f"  Building vocab from {len(texts)} sentences …")
    from gensim.models import KeyedVectors

    # Simple word-level tokenisation
    all_tokens = set()
    for t in texts:
        all_tokens.update(t.lower().split() if lang == "en" else t.split())

    log.info(f"  Unique tokens: {len(all_tokens)}")
    log.info(f"  Loading vectors from {vec_path} (this may take a few minutes) …")

    # Load only the vectors we need (limit=None loads all; subset saves RAM)
    kv = KeyedVectors.load_word2vec_format(vec_path, binary=False, no_header=False)

    vocab  = {"<PAD>": 0, "<UNK>": 1}
    matrix = [np.zeros(EMBED_DIM), np.random.normal(0, 0.1, EMBED_DIM)]  # PAD, UNK

    for tok in sorted(all_tokens):
        if tok in kv:
            vocab[tok] = len(vocab)
            matrix.append(kv[tok])

    matrix = np.array(matrix, dtype=np.float32)
    log.info(f"  Vocab size: {len(vocab)} | Matrix shape: {matrix.shape}")
    return vocab, matrix
