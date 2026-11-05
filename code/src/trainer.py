"""
src/trainer.py
──────────────
Training loops for:
  • Classical ML  (sklearn fit/predict)
  • RNN models    (PyTorch, with embedding freeze/unfreeze)
  • Transformer   (AdamW + linear warmup + early stopping)
  • Ensemble alpha search (grid search on validation set)
"""

import os, sys, logging, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import *

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Seed helper
# ═══════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ═══════════════════════════════════════════════════════════
# 1.  CLASSICAL ML TRAINER
# ═══════════════════════════════════════════════════════════

def train_classical(model, train_df, val_df=None):
    """Fit an sklearn pipeline; returns the fitted model."""
    X_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()
    model.fit(X_train, y_train)
    if val_df is not None:
        from sklearn.metrics import f1_score
        preds = model.predict(val_df["text"].tolist())
        f1 = f1_score(val_df["label"].tolist(), preds, average="macro",
                      labels=EMOTION_CLASSES, zero_division=0)
        log.info(f"    Val macro-F1: {f1:.4f}")
    return model


# ═══════════════════════════════════════════════════════════
# 2.  RNN TRAINER
# ═══════════════════════════════════════════════════════════

def train_rnn(model, train_dataset, val_dataset,
              class_weights: torch.Tensor, device: torch.device,
              model_save_path: str = None) -> nn.Module:
    """
    Train a CNN / BiLSTM / GRU / BiLSTM-Attention / BiLSTMGRUBranch model.
    Freezes embeddings for first `freeze_epochs` epochs.
    Returns the best model (by val macro-F1) as state_dict.
    """
    from sklearn.metrics import f1_score as sklearn_f1

    # num_workers=0: the sequences are already encoded in memory, and on Windows every
    # worker is a fresh process that re-imports the project (~3 s each, 4 per epoch).
    # Batches, shuffling and results are identical to num_workers=2; only faster.
    train_loader = DataLoader(train_dataset, batch_size=RNN_BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=RNN_BATCH_SIZE * 2,
                              shuffle=False, num_workers=0, pin_memory=True)

    model.to(device)
    model.freeze_embed()

    criterion = nn.NLLLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=RNN_LR)

    best_f1, best_state = -1.0, None   # -1 so the first epoch is always kept

    for epoch in range(1, RNN_EPOCHS + 1):
        if epoch == model.freeze_epochs + 1:
            model.unfreeze_embed()
            log.info("    Embeddings unfrozen.")

        # ── train ──
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"  Epoch {epoch}/{RNN_EPOCHS}", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()

        # ── validate ──
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                preds = model(batch["input_ids"].to(device)).argmax(dim=-1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(batch["labels"].tolist())

        val_f1 = sklearn_f1(all_labels, all_preds, average="macro",
                            labels=list(range(NUM_CLASSES)), zero_division=0)
        avg_loss = total_loss / len(train_loader)
        log.info(f"    Epoch {epoch}: loss={avg_loss:.4f}  val_macro_F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if model_save_path:
                torch.save(best_state, model_save_path)

    model.load_state_dict(best_state)
    log.info(f"    Best val macro-F1: {best_f1:.4f}")
    return model


# ═══════════════════════════════════════════════════════════
# 3.  TRANSFORMER TRAINER
# ═══════════════════════════════════════════════════════════

def train_transformer(model, train_dataset, val_dataset,
                      class_weights: torch.Tensor, device: torch.device,
                      model_save_path: str = None) -> nn.Module:
    """
    Fine-tune a TransformerClassifier with AdamW + linear warmup.
    Early stopping on val macro-F1.
    """
    from sklearn.metrics import f1_score as sklearn_f1

    train_loader = DataLoader(train_dataset, batch_size=TRANSFORMER_BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=TRANSFORMER_BATCH_SIZE * 2,
                              shuffle=False, num_workers=0, pin_memory=True)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=TRANSFORMER_LR, weight_decay=WEIGHT_DECAY)
    total_steps  = TRANSFORMER_EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
                       optimizer, warmup_steps, total_steps)

    criterion = nn.NLLLoss(weight=class_weights.to(device))

    best_f1, best_state, patience_count = -1.0, None, 0   # -1 so the first epoch is always kept

    for epoch in range(1, TRANSFORMER_EPOCHS + 1):
        # ── train ──
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader,
                          desc=f"  [Transformer] Epoch {epoch}/{TRANSFORMER_EPOCHS}",
                          leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            optimizer.zero_grad()
            logits = model(**batch)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # ── validate ──
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                preds  = model(**batch).argmax(dim=-1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().tolist())

        val_f1   = sklearn_f1(all_labels, all_preds, average="macro",
                              labels=list(range(NUM_CLASSES)), zero_division=0)
        avg_loss = total_loss / len(train_loader)
        log.info(f"    Epoch {epoch}: loss={avg_loss:.4f}  val_macro_F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1       = val_f1
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            if model_save_path:
                torch.save(best_state, model_save_path)
        else:
            patience_count += 1
            if patience_count >= EARLY_STOP_PATIENCE:
                log.info(f"    Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    log.info(f"    Best val macro-F1: {best_f1:.4f}")
    return model


# ═══════════════════════════════════════════════════════════
# 4.  ENSEMBLE ALPHA SEARCH
# ═══════════════════════════════════════════════════════════

def find_best_alpha(transformer_model, recurrent_model,
                    val_transformer_dataset, val_rnn_dataset,
                    device: torch.device, return_grid: bool = False):
    """
    Grid search over ALPHA_GRID using validation macro-F1.
    Both models must already be trained and in eval mode.

    Selection rule: the alpha with the highest validation macro-F1; on an exact
    tie the smallest alpha in the grid wins (first maximum). With
    return_grid=True also returns {alpha: val_macro_F1} for every grid value,
    so the full grid can be reported.

    The end points alpha = 0 (recurrent branch alone) and alpha = 1 (Transformer
    alone) are evaluated and recorded as well, so the whole curve from one branch
    to the other can be reported. They are not candidates for selection: the
    selected alpha always comes from ALPHA_GRID.
    """
    from sklearn.metrics import f1_score as sklearn_f1

    transformer_model.eval()
    recurrent_model.eval()

    t_loader = DataLoader(val_transformer_dataset, batch_size=128,
                          shuffle=False, num_workers=0)
    r_loader = DataLoader(val_rnn_dataset,         batch_size=128,
                          shuffle=False, num_workers=0)

    all_labels = []
    all_pt, all_pr = [], []

    with torch.no_grad():
        for tb, rb in zip(t_loader, r_loader):
            tb = {k: v.to(device) for k, v in tb.items()}
            rb = {k: v.to(device) for k, v in rb.items()}
            all_labels.extend(tb["labels"].cpu().tolist())
            pt = transformer_model(**{k: v for k, v in tb.items() if k != "labels"})
            pr = recurrent_model(**{k: v for k, v in rb.items() if k != "labels"})
            all_pt.append(pt.cpu())
            all_pr.append(pr.cpu())

    all_pt = torch.cat(all_pt, dim=0)
    all_pr = torch.cat(all_pr, dim=0)

    selectable = sorted(set(float(a) for a in ALPHA_GRID))
    end_points = {0.0: "  (recurrent branch alone)", 1.0: "  (Transformer alone)"}
    best_alpha, best_f1 = None, -1.0
    grid = {}
    for alpha in sorted(set(selectable) | set(end_points)):
        logits = alpha * all_pt + (1 - alpha) * all_pr
        preds  = logits.argmax(dim=-1).tolist()
        f1     = sklearn_f1(all_labels, preds, average="macro",
                            labels=list(range(NUM_CLASSES)), zero_division=0)
        grid[round(float(alpha), 2)] = round(float(f1), 4)
        note = end_points.get(alpha, "") if alpha not in selectable else ""
        log.info(f"    alpha={alpha:.1f}  val_macro_F1={f1:.4f}{note}")
        if alpha in selectable and f1 > best_f1:
            best_f1, best_alpha = f1, alpha

    in_grid = [grid[round(a, 2)] for a in selectable]
    spread = max(in_grid) - min(in_grid)
    log.info(f"  Best alpha: {best_alpha}  (val macro-F1={best_f1:.4f}; "
             f"grid spread {spread:.4f}; branch alone {grid[0.0]:.4f}, "
             f"Transformer alone {grid[1.0]:.4f})")
    return (best_alpha, grid) if return_grid else best_alpha
