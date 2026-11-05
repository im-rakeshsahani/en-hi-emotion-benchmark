"""
config.py  —  Central configuration for all experiments.
Edit ONLY this file to change hyperparameters, paths, or model choices.
"""

import os

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEEDS = [42, 1337, 2024]  # 3 seeds → mean ± std reported

# ─────────────────────────────────────────────
# Emotion scheme  (5-class Ekman-aligned)
# ─────────────────────────────────────────────
EMOTION_CLASSES  = ['joy', 'sadness', 'anger', 'fear', 'surprise']
CLASS_TO_IDX     = {e: i for i, e in enumerate(EMOTION_CLASSES)}
IDX_TO_CLASS     = {i: e for e, i in CLASS_TO_IDX.items()}
NUM_CLASSES      = len(EMOTION_CLASSES)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODELS_DIR      = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
PLOTS_DIR       = os.path.join(BASE_DIR, "plots")
EMBED_DIR       = os.path.join(BASE_DIR, "embeddings")

for _d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, EMBED_DIR]:
    os.makedirs(_d, exist_ok=True)

# BHAAV splits, written by rebuild_bhaav_from_zenodo.py from the official release
# (Zenodo 10.5281/zenodo.3457467, CC BY 4.0): neutral dropped, conflicting and
# duplicate sentences removed, stratified 80/10/10 with seed 42.
BHAAV_TRAIN_CSV = os.path.join(DATA_DIR, "bhaav_train.csv")
BHAAV_TEST_CSV  = os.path.join(DATA_DIR, "bhaav_test.csv")
BHAAV_VAL_CSV   = os.path.join(DATA_DIR, "bhaav_val.csv")

# XED surprise sentences (English XED, label 7) machine-translated to Hindi
# with Helsinki-NLP/opus-mt-en-hi; split 80/10/10 with seed 42 in data_utils.py
XED_HI_FILE     = os.path.join(DATA_DIR, "xed_hindi_surprise.csv")

# Pre-trained 300-d word vectors for the recurrent models (user must download — see README)
EN_FASTTEXT_PATH = os.path.join(EMBED_DIR, "wiki.en.vec")   # fastText Wikipedia, 2,519,370 words
HI_INDICFT_PATH  = os.path.join(EMBED_DIR, "hi.vec")        # 1,876,653 words in the file used

# ─────────────────────────────────────────────
# HuggingFace Transformer model IDs
# ─────────────────────────────────────────────
TRANSFORMER_MODELS = {
    "mBERT"     : "bert-base-multilingual-cased",
    "XLM-R"     : "xlm-roberta-base",
    "MuRIL"     : "google/muril-base-cased",
    "IndicBERT" : "ai4bharat/indic-bert",
}

# ─────────────────────────────────────────────
# Transformer branch hyperparameters
# ─────────────────────────────────────────────
TRANSFORMER_LR           = 2e-5
TRANSFORMER_BATCH_SIZE   = 16          # fits the 6 GB GPU; every Chapter 5 run uses 16
TRANSFORMER_EPOCHS       = 4
TRANSFORMER_MAX_LEN      = 128
WARMUP_RATIO             = 0.10
WEIGHT_DECAY             = 0.01
GRAD_CLIP                = 1.0
EARLY_STOP_PATIENCE      = 2       # epochs without val F1 improvement

# ─────────────────────────────────────────────
# Recurrent branch hyperparameters
# ─────────────────────────────────────────────
EMBED_DIM        = 300
RNN_HIDDEN       = 128    # BiLSTM hidden size (→ 256 bidirectional)
GRU_HIDDEN       = 64     # GRU hidden size
RNN_MAX_LEN      = 64
RNN_BATCH_SIZE   = 64
RNN_LR           = 1e-3
RNN_EPOCHS       = 15
DROPOUT          = 0.30

# CNN baseline specific
CNN_FILTERS      = 128
CNN_KERNELS      = [3, 4, 5]

# ─────────────────────────────────────────────
# Ensemble fusion
# ─────────────────────────────────────────────
ALPHA_GRID = [round(x * 0.1, 1) for x in range(1, 10)]   # [0.1 … 0.9]
# α is selected separately for each seed: the grid value with the highest
# validation macro-F1 (ties go to the smaller α).

# ─────────────────────────────────────────────
# GoEmotions label → 5-class mapping
# (applied to the 28 fine-grained GoEmotions labels)
# Follows the GoEmotions authors' Ekman grouping, except that desire and
# embarrassment are dropped (the official grouping puts them under joy and
# sadness). A multi-label comment takes the first label that survives the mapping.
# ─────────────────────────────────────────────
GOEMOTIONS_TO_5CLASS = {
    # Joy
    "joy": "joy", "amusement": "joy", "excitement": "joy",
    "love": "joy", "gratitude": "joy", "pride": "joy",
    "admiration": "joy", "approval": "joy", "caring": "joy",
    "optimism": "joy", "relief": "joy",
    # Sadness
    "sadness": "sadness", "disappointment": "sadness",
    "grief": "sadness", "remorse": "sadness",
    # Anger
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    # Fear
    "fear": "fear", "nervousness": "fear",
    # Surprise
    "surprise": "surprise", "curiosity": "surprise",
    "realization": "surprise", "confusion": "surprise",
}
# Labels to DROP (neutral, disgust, embarrassment, desire)
GOEMOTIONS_DROP = {"neutral", "disgust", "embarrassment", "desire"}

# BHAAV label → 5-class mapping. For reference: the mapping actually applied is the
# label_map in src/data_utils.load_bhaav, which is identical.
# Official BHAAV codes: 0 anger, 1 joy, 2 sad, 3 suspense, 4 neutral.
BHAAV_TO_5CLASS = {
    "joy"      : "joy",
    "sad"      : "sadness",
    "sadness"  : "sadness",
    "anger"    : "anger",
    "suspense" : "fear",
}
# neutral (code 4) is dropped

# XED Plutchik emotion IDs → 5-class (only label 7, surprise, is used)
XED_ID_TO_NAME = {
    1: "anger", 2: "anticipation", 3: "disgust",
    4: "fear",  5: "joy",          6: "sadness",
    7: "surprise", 8: "trust",     0: "neutral"
}
XED_TO_5CLASS = {
    "joy": "joy", "trust": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "fear": "fear",
    "surprise": "surprise",
}
