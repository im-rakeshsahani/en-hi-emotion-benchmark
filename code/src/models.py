"""
src/models.py
─────────────
All model architectures used in the paper:
  Classical : NaiveBayes, LinearSVM
  Deep RNN  : CNN, BiLSTM, GRU, BiLSTM-Attention, BiLSTM-GRU (ensemble branch)
  Transformer: mBERT, XLM-R, MuRIL, IndicBERT (via HuggingFace)
  Ensemble  : MuRIL + BiLSTM-GRU with weighted log-prob fusion
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import *


# ═══════════════════════════════════════════════════════════
# 1.  CLASSICAL ML  (sklearn pipelines)
# ═══════════════════════════════════════════════════════════

# scikit-learn's default token pattern r"(?u)\b\w\w+\b" breaks Hindi words apart: Python's \w
# does not match Devanagari vowel signs (matras), virama or anusvara, so "मेरी देह में आग लग गयी"
# is tokenised as ['आग', 'लग', 'गय'] and most words vanish. This pattern adds the Devanagari block
# (except the danda punctuation) to the word characters. For text without Devanagari it yields
# exactly the default tokens, so English results are unchanged.
TFIDF_TOKEN_PATTERN = r"(?u)[\wऀ-ॣ०-९ॱ-ॿ]{2,}"


def build_naive_bayes():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import MaxAbsScaler
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                                  sublinear_tf=True, min_df=2,
                                  token_pattern=TFIDF_TOKEN_PATTERN)),
        ("scaler", MaxAbsScaler()),
        ("clf",   MultinomialNB(alpha=0.1)),
    ])


def build_linear_svm():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                                  sublinear_tf=True, min_df=2,
                                  token_pattern=TFIDF_TOKEN_PATTERN)),
        ("clf",   CalibratedClassifierCV(
                      LinearSVC(C=1.0, max_iter=5000), cv=3)),
    ])


# ═══════════════════════════════════════════════════════════
# 2.  EMBEDDING-BASED DEEP MODELS  (PyTorch)
# ═══════════════════════════════════════════════════════════

class _EmbeddingBase(nn.Module):
    """Shared embedding + dropout setup."""
    def __init__(self, embed_matrix: np.ndarray, freeze_epochs: int = 2):
        super().__init__()
        vocab_size, embed_dim = embed_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embed_matrix, dtype=torch.float32))
        self.freeze_epochs = freeze_epochs   # freeze embeddings for first N epochs
        self.drop = nn.Dropout(DROPOUT)

    def freeze_embed(self):
        self.embedding.weight.requires_grad_(False)

    def unfreeze_embed(self):
        self.embedding.weight.requires_grad_(True)


class CNNClassifier(_EmbeddingBase):
    """Multi-kernel 1-D CNN text classifier."""
    def __init__(self, embed_matrix: np.ndarray):
        super().__init__(embed_matrix)
        in_ch = embed_matrix.shape[1]
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch, CNN_FILTERS, k) for k in CNN_KERNELS
        ])
        self.fc = nn.Linear(CNN_FILTERS * len(CNN_KERNELS), NUM_CLASSES)

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)          # (B, L, D)
        x = self.drop(x).permute(0, 2, 1)     # (B, D, L)
        pooled = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]
        out = self.drop(torch.cat(pooled, dim=1))
        return F.log_softmax(self.fc(out), dim=-1)


class BiLSTMClassifier(_EmbeddingBase):
    """Standard BiLSTM with mean pooling."""
    def __init__(self, embed_matrix: np.ndarray):
        super().__init__(embed_matrix)
        self.lstm = nn.LSTM(embed_matrix.shape[1], RNN_HIDDEN,
                            batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(RNN_HIDDEN * 2, NUM_CLASSES)

    def forward(self, input_ids, **kwargs):
        x    = self.drop(self.embedding(input_ids))
        out, _ = self.lstm(x)
        out  = out.mean(dim=1)
        return F.log_softmax(self.fc(self.drop(out)), dim=-1)


class GRUClassifier(_EmbeddingBase):
    """Bidirectional GRU with mean pooling — matches BiLSTM design for fair comparison."""
    def __init__(self, embed_matrix: np.ndarray):
        super().__init__(embed_matrix)
        self.gru = nn.GRU(embed_matrix.shape[1], RNN_HIDDEN,
                          batch_first=True, bidirectional=True)
        self.fc  = nn.Linear(RNN_HIDDEN * 2, NUM_CLASSES)

    def forward(self, input_ids, **kwargs):
        x      = self.drop(self.embedding(input_ids))
        out, _ = self.gru(x)
        out    = out.mean(dim=1)
        return F.log_softmax(self.fc(self.drop(out)), dim=-1)

class BiLSTMAttentionClassifier(_EmbeddingBase):
    """BiLSTM with additive self-attention pooling."""
    def __init__(self, embed_matrix: np.ndarray):
        super().__init__(embed_matrix)
        self.lstm    = nn.LSTM(embed_matrix.shape[1], RNN_HIDDEN,
                               batch_first=True, bidirectional=True)
        self.attn_w  = nn.Linear(RNN_HIDDEN * 2, 1)
        self.fc      = nn.Linear(RNN_HIDDEN * 2, NUM_CLASSES)

    def forward(self, input_ids, **kwargs):
        x, _ = self.lstm(self.drop(self.embedding(input_ids)))
        w     = torch.softmax(self.attn_w(x), dim=1)   # (B, L, 1)
        ctx   = (w * x).sum(dim=1)                      # (B, 2H)
        return F.log_softmax(self.fc(self.drop(ctx)), dim=-1)


class BiLSTMGRUBranch(_EmbeddingBase):
    """
    Recurrent branch of the ensemble:
      Embedding → BiLSTM(128) → GRU(64) → final GRU state → Linear → LogSoftmax
    Also used standalone to pre-train the recurrent branch.

    Sequences arrive right-padded to RNN_MAX_LEN with index 0 (<PAD>). They are
    packed by true length, so the BiLSTM's backward direction starts at the last
    real token and the GRU's final state is its state at the last real token.

    Revision note (Chapter 5): the earlier version ran both layers over the
    padding and read the GRU state at position RNN_MAX_LEN. For a 10-15 token
    sentence that state comes after ~50 padding steps and is almost independent
    of the input, so the branch produced a near-uniform distribution (training
    loss fixed at ln 5 = 1.609, the collapse reported in Chapter 5). Old
    checkpoints have identical shapes but encode the collapsed model; they are
    not reused (run_experiments.py saves this version under a new file name).
    """
    def __init__(self, embed_matrix: np.ndarray):
        super().__init__(embed_matrix)
        self.bilstm = nn.LSTM(embed_matrix.shape[1], RNN_HIDDEN,
                              batch_first=True, bidirectional=True)
        self.gru    = nn.GRU(RNN_HIDDEN * 2, GRU_HIDDEN, batch_first=True)
        self.fc     = nn.Linear(GRU_HIDDEN, NUM_CLASSES)

    def forward(self, input_ids, return_logprobs=False, **kwargs):
        pack, unpack = nn.utils.rnn.pack_padded_sequence, nn.utils.rnn.pad_packed_sequence
        lengths = (input_ids != 0).sum(dim=1).clamp(min=1).cpu()     # true lengths (0 = <PAD>)
        x = self.drop(self.embedding(input_ids))
        x, _ = self.bilstm(pack(x, lengths, batch_first=True, enforce_sorted=False))
        x, _ = unpack(x, batch_first=True)
        x = self.drop(x)
        _, h = self.gru(pack(x, lengths, batch_first=True, enforce_sorted=False))
        h = self.drop(h[-1])                                          # state at last real token
        return F.log_softmax(self.fc(h), dim=-1)


# ═══════════════════════════════════════════════════════════
# 3.  TRANSFORMER CLASSIFIER  (HuggingFace)
# ═══════════════════════════════════════════════════════════

class TransformerClassifier(nn.Module):
    """
    Wraps any HuggingFace encoder.
    CLS embedding → Dropout → Linear → LogSoftmax
    """

    def __init__(self, model_name: str):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(DROPOUT)
        self.fc   = nn.Linear(hidden, NUM_CLASSES)

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                return_logprobs=False, **kwargs):
        # Some models (XLM-R) don't accept token_type_ids
        enc_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            try:
                out = self.encoder(**enc_kwargs, token_type_ids=token_type_ids)
            except TypeError:
                out = self.encoder(**enc_kwargs)
        else:
            out = self.encoder(**enc_kwargs)

        cls = out.last_hidden_state[:, 0, :]   # [CLS] token
        lp  = F.log_softmax(self.fc(self.drop(cls)), dim=-1)
        return lp


# ═══════════════════════════════════════════════════════════
# 4.  ENSEMBLE  (MuRIL + BiLSTM-GRU, weighted log-prob fusion)
# ═══════════════════════════════════════════════════════════

class EnsembleModel(nn.Module):
    """
    Inference-only ensemble.
    P_ens = alpha * P_T + (1 - alpha) * P_R
    alpha is tuned on the validation set.
    """

    def __init__(self, transformer_model: TransformerClassifier,
                 recurrent_model: BiLSTMGRUBranch, alpha: float = 0.65):
        super().__init__()
        self.transformer = transformer_model
        self.recurrent   = recurrent_model
        self.alpha       = alpha

    def forward(self, transformer_batch: dict, recurrent_batch: dict):
        with torch.no_grad():
            pt = self.transformer(**{k: v for k, v in transformer_batch.items()
                                     if k != "labels"})
            pr = self.recurrent(**{k: v for k, v in recurrent_batch.items()
                                   if k != "labels"})
        return self.alpha * pt + (1 - self.alpha) * pr

    @torch.no_grad()
    def predict(self, transformer_batch: dict, recurrent_batch: dict):
        logits = self.forward(transformer_batch, recurrent_batch)
        return logits.argmax(dim=-1).cpu().numpy()


# ═══════════════════════════════════════════════════════════
# 5.  MODEL REGISTRY  — convenience factory
# ═══════════════════════════════════════════════════════════

CLASSICAL_MODELS = {
    "Naive Bayes (TF-IDF)": build_naive_bayes,
    "Linear SVM (TF-IDF)" : build_linear_svm,
}

RNN_MODELS = {
    "CNN"                      : CNNClassifier,
    "BiLSTM"                   : BiLSTMClassifier,
    "GRU"                      : GRUClassifier,
    "BiLSTM-Attention"         : BiLSTMAttentionClassifier,
}

def get_rnn_suffix(lang: str) -> str:
    return "+ FastText" if lang == "en" else "+ IndicFT"

def get_transformer_model(name: str) -> TransformerClassifier:
    model_id = TRANSFORMER_MODELS[name]
    return TransformerClassifier(model_id)

def get_rnn_model(name: str, embed_matrix: np.ndarray):
    cls = RNN_MODELS[name]
    return cls(embed_matrix)
