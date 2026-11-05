"""
run_experiments.py
──────────────────
Main entry point.  Runs ALL experiments and saves results to results/.

Usage:
    python run_experiments.py --seeds 42 1337 2024      # full run, three seeds
    python run_experiments.py --lang en                 # English only
    python run_experiments.py --lang hi                 # Hindi only
    python run_experiments.py --skip-classical          # skip NB and SVM

Chapter 5 revision (three seeds):
  • Results are written with the suffix "_3seed" (results/main_en_3seed.json,
    main_hi_3seed.json, crosslingual_3seed.json, crosslingual_control_3seed.json,
    ablation_3seed.json, recurrent_branch_3seed.json). Files from earlier runs
    are left untouched. Re-running a subset of stages MERGES into the _3seed
    files instead of overwriting rows produced by other stages.
  • Every model's test predictions are saved per seed in
    results/predictions_{lang}_3seed.json (with the gold labels), so every table,
    confusion matrix and McNemar / seed-paired test can be recomputed in one pass.
  • The ensemble's recurrent branch uses the corrected BiLSTMGRUBranch (packed
    sequences) and is saved as {lang}_bilstmgru_packed_seed{seed}.pt, so the old
    collapsed checkpoints ({lang}_bilstmgru_seed{seed}.pt) are never reused.
  • The selected fusion weight alpha and the full validation grid are recorded
    for every seed (no hard-coded alpha label anywhere).
  • Cross-lingual transfer adds a Hindi-only control: pretrained MuRIL fine-tuned
    on exactly the same k% Hindi subset with the same recipe. Subsets are drawn
    from one fixed permutation per seed, so they are nested (5% ⊂ 10% ⊂ 20% ⊂ 50%).
  • Datasets and fastText/IndicFT vectors are loaded once per language and
    reused by the cross-lingual and ablation stages.
  • A saved checkpoint is reused only if it is newer than the data files it was
    trained on. A Hindi checkpoint older than data/bhaav_*.csv or
    data/xed_hindi_surprise.csv (such as the seed-42 models from the May runs on the
    mislabelled BHAAV files) is renamed to *.pt.stale and retrained, never loaded.
  • Training writes to <name>.pt.partial and renames it to <name>.pt only when the
    model has finished training, so an interrupted run never leaves a half-trained
    checkpoint that a later run would load as complete.
  • --generate-tables-only prints the tables from the saved _3seed results and exits
    without training anything.
"""

import os, sys, json, argparse, logging, copy
import numpy as np
import pandas as pd
import torch

# ── project imports ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from src.data_utils  import (build_english_dataset, build_hindi_dataset,
                              build_vocab_and_matrix, compute_class_weights,
                              TransformerDataset, EmbeddingDataset)
from src.models      import (build_naive_bayes, build_linear_svm,
                              get_rnn_model, get_transformer_model,
                              BiLSTMGRUBranch, TransformerClassifier,
                              CLASSICAL_MODELS, RNN_MODELS, get_rnn_suffix)
from src.trainer     import (set_seed, train_classical, train_rnn,
                              train_transformer, find_best_alpha)
from src.evaluator   import (compute_metrics, aggregate_seed_metrics,
                              predict_classical, predict_rnn,
                              predict_transformer, predict_ensemble,
                              mcnemar_test, build_main_table, build_ablation_table,
                              build_crosslingual_table, build_perclass_table,
                              plot_confusion_matrices, save_results, print_table)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(os.path.join(BASE_DIR, "experiment.log"))])
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

RESULTS_TAG = "3seed"
ENSEMBLE_NAME = "Proposed Ensemble (MuRIL + BiLSTM-GRU)"


# ═══════════════════════════════════════════════════════════
# RESULT FILES (tagged, merged across invocations)
# ═══════════════════════════════════════════════════════════

def _tagged(name: str) -> str:
    return f"{name}_{RESULTS_TAG}"


def load_tagged(name: str) -> dict:
    path = os.path.join(RESULTS_DIR, f"{_tagged(name)}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_merged(data: dict, name: str) -> dict:
    """Update results/<name>_3seed.json with `data` (new keys replace old ones)."""
    merged = load_tagged(name)
    merged.update(data)
    save_results(merged, _tagged(name))
    return merged


def save_overwrite(data: dict, name: str) -> dict:
    save_results(data, _tagged(name))
    return data


# Per-seed test predictions: lang -> {"y_true": [...], "models": {name: {seed: [...]}}}
PRED_STORE = {}


def store_preds(lang: str, cache, name: str, seed: int, preds):
    d = PRED_STORE.setdefault(lang, {"y_true": [int(v) for v in cache.get_labels("test")],
                                     "models": {}})
    d["models"].setdefault(name, {})[str(seed)] = [int(p) for p in preds]


def flush_preds():
    for lang, d in PRED_STORE.items():
        existing = load_tagged(f"predictions_{lang}")
        models = {}
        if existing:
            if existing.get("y_true") == d["y_true"]:
                models = existing.get("models", {})
            else:
                log.warning(f"  Stored {lang} test labels differ from the current test set; "
                            f"previous predictions in predictions_{lang}_{RESULTS_TAG}.json are replaced.")
        for name, per_seed in d["models"].items():
            models.setdefault(name, {}).update(per_seed)
        save_results({"y_true": d["y_true"], "models": models}, _tagged(f"predictions_{lang}"))


def _empty_cache():
    """Release cached GPU memory. Callers `del` their own model references first."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lang",           nargs="+", default=["en", "hi"],
                   choices=["en", "hi"])
    p.add_argument("--models",         nargs="+", default=None,
                   help="Subset of models to run. If None, run all.")
    p.add_argument("--seeds",          nargs="+", type=int, default=SEEDS)
    p.add_argument("--skip-classical", action="store_true")
    p.add_argument("--skip-rnn",       action="store_true")
    p.add_argument("--skip-transformers", action="store_true")
    p.add_argument("--skip-ensemble",  action="store_true")
    p.add_argument("--skip-crosslingual", action="store_true")
    p.add_argument("--skip-ablation",  action="store_true")
    p.add_argument("--generate-tables-only", action="store_true",
                   help="Skip training; just regenerate tables from saved results.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════
# DATA LOADING (cached across seeds)
# ═══════════════════════════════════════════════════════════

class DataCache:
    """Loads datasets once and serves them to all experiments."""

    def __init__(self, lang: str):
        self.lang = lang
        log.info(f"\n{'='*60}\nLoading {lang.upper()} dataset …\n{'='*60}")

        if lang == "en":
            self.train_df, self.val_df, self.test_df = build_english_dataset()
            self.vec_path = EN_FASTTEXT_PATH
        else:
            self.train_df, self.val_df, self.test_df = build_hindi_dataset()
            self.vec_path = HI_INDICFT_PATH

        self.class_weights = compute_class_weights(self.train_df["label"].tolist())
        log.info(f"  Class weights: {self.class_weights.tolist()}")

        # Build vocabulary and embedding matrix once
        if os.path.exists(self.vec_path):
            all_texts = (self.train_df["text"].tolist() +
                         self.val_df["text"].tolist() +
                         self.test_df["text"].tolist())
            self.vocab, self.embed_matrix = build_vocab_and_matrix(
                all_texts, self.vec_path, lang=lang)
        else:
            log.warning(
                f"  FastText vectors not found at {self.vec_path}.\n"
                f"  RNN models require this file.  Download instructions:\n"
                f"  English: https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec\n"
                f"  Hindi  : https://indicnlp.ai4bharat.org/indicft/   (hi.vec)\n"
                f"  Place files in: {EMBED_DIR}/")
            self.vocab, self.embed_matrix = None, None

    def get_tokenizer(self, model_name_hf: str):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name_hf)

    def get_transformer_datasets(self, tokenizer):
        return (TransformerDataset(self.train_df, tokenizer),
                TransformerDataset(self.val_df,   tokenizer),
                TransformerDataset(self.test_df,  tokenizer))

    def get_rnn_datasets(self):
        if self.vocab is None:
            return None, None, None
        return (EmbeddingDataset(self.train_df, self.vocab, lang=self.lang),
                EmbeddingDataset(self.val_df,   self.vocab, lang=self.lang),
                EmbeddingDataset(self.test_df,  self.vocab, lang=self.lang))

    def get_labels(self, split: str) -> list:
        df = {"train": self.train_df, "val": self.val_df, "test": self.test_df}[split]
        return [CLASS_TO_IDX[l] for l in df["label"]]


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

# Data files each language's models are trained on. GoEmotions is read from the
# Hugging Face cache and has not changed since the May runs, so English needs no check.
DATA_FILES = {
    "en": [],
    "hi": [BHAAV_TRAIN_CSV, BHAAV_VAL_CSV, BHAAV_TEST_CSV, XED_HI_FILE],
}


def _replace(src: str, dst: str, tries: int = 10):
    """os.replace with retries: on Windows a file that was just written can be held
    open for a moment by antivirus or the search indexer (PermissionError)."""
    import time
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(3)


def _usable_checkpoint(ckpt: str, lang: str) -> bool:
    """
    True if `ckpt` exists and is newer than every data file of `lang`.

    A checkpoint older than the data was trained on an earlier version of it: it is
    renamed to <name>.pt.stale (kept, never loaded) and the model is retrained. This
    catches the seed-42 Hindi checkpoints from the May runs, which were trained on the
    mislabelled BHAAV files and carry exactly the names used here.
    """
    if not os.path.exists(ckpt):
        return False
    stamps = [os.path.getmtime(f) for f in DATA_FILES.get(lang, []) if os.path.exists(f)]
    if stamps and os.path.getmtime(ckpt) < max(stamps):
        _replace(ckpt, ckpt + ".stale")
        log.warning(f"    {os.path.basename(ckpt)} is older than the current {lang.upper()} "
                    f"data files: kept as {os.path.basename(ckpt)}.stale and retrained.")
        return False
    return True


def _train_and_publish(train_fn, model, train_d, val_d, class_weights, ckpt):
    """
    Train with best-epoch saves going to <ckpt>.partial, and rename that file to <ckpt>
    only when training has finished. An interrupted run therefore never leaves a
    half-trained checkpoint under the final name.
    """
    partial = ckpt + ".partial"
    model = train_fn(model, train_d, val_d, class_weights, DEVICE, partial)
    if os.path.exists(partial):
        _replace(partial, ckpt)
    else:
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt)
    return model


def load_or_train_transformer(key, lang, seed, train_d, val_d, class_weights):
    """Load {lang}_{key}_seed{seed}.pt if present and current, else train it and save it there."""
    model = get_transformer_model(key)
    ckpt = os.path.join(MODELS_DIR, f"{lang}_{key.lower()}_seed{seed}.pt")
    if _usable_checkpoint(ckpt, lang):
        log.info(f"    [SKIP TRAIN] Loading {lang.upper()} {key}: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
    else:
        model = _train_and_publish(train_transformer, model, train_d, val_d, class_weights, ckpt)
    return model


def load_or_train_branch(cache, lang, seed, train_d_r, val_d_r):
    """Corrected recurrent branch; saved under a new name so collapsed checkpoints are never reused."""
    model = BiLSTMGRUBranch(cache.embed_matrix)
    ckpt = os.path.join(MODELS_DIR, f"{lang}_bilstmgru_packed_seed{seed}.pt")
    if _usable_checkpoint(ckpt, lang):
        log.info(f"    [SKIP TRAIN] Loading BiLSTM-GRU branch: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
    else:
        model = _train_and_publish(train_rnn, model, train_d_r, val_d_r, cache.class_weights, ckpt)
    return model


# ═══════════════════════════════════════════════════════════
# EXPERIMENT RUNNERS
# ═══════════════════════════════════════════════════════════

def run_classical(cache: DataCache, seeds: list, lang: str) -> dict:
    """Run Naive Bayes and Linear SVM over multiple seeds."""
    results = {}
    for name, builder in CLASSICAL_MODELS.items():
        log.info(f"\n  Running {name} …")
        seed_metrics = []
        for seed in seeds:
            set_seed(seed)
            model = builder()
            model = train_classical(model, cache.train_df, cache.val_df)
            preds_str = predict_classical(model, cache.test_df)
            # Convert string labels → integer indices to match y_true
            preds = [CLASS_TO_IDX[p] for p in preds_str]
            y_true = cache.get_labels("test")
            seed_metrics.append(compute_metrics(y_true, preds))
            store_preds(lang, cache, name, seed, preds)
        results[name] = aggregate_seed_metrics(seed_metrics)
        log.info(f"    → Macro F1: {results[name]['macro_f1']:.4f} "
                 f"± {results[name]['macro_f1_std']:.4f}")
    return results


def run_rnn_models(cache: DataCache, seeds: list, lang: str) -> dict:
    """Run CNN, BiLSTM, GRU, BiLSTM-Attention baselines."""
    results = {}
    suffix  = get_rnn_suffix(lang)
    train_d, val_d, test_d = cache.get_rnn_datasets()
    if train_d is None:
        log.warning("  Skipping RNN models — embedding file missing.")
        return results

    for name in RNN_MODELS:
        full_name = f"{name} {suffix}"
        log.info(f"\n  Running {full_name} …")
        seed_metrics = []
        for seed in seeds:
            set_seed(seed)
            model = get_rnn_model(name, cache.embed_matrix)
            save_path = os.path.join(MODELS_DIR, f"{lang}_{name.lower()}_seed{seed}.pt")
            model = train_rnn(model, train_d, val_d,
                              cache.class_weights, DEVICE, save_path)
            preds  = predict_rnn(model, test_d, DEVICE)
            y_true = cache.get_labels("test")
            seed_metrics.append(compute_metrics(y_true, preds))
            store_preds(lang, cache, full_name, seed, preds)
        results[full_name] = aggregate_seed_metrics(seed_metrics)
        log.info(f"    → Macro F1: {results[full_name]['macro_f1']:.4f} "
                 f"± {results[full_name]['macro_f1_std']:.4f}")
    return results


def run_transformer_models(cache: DataCache, seeds: list, lang: str) -> dict:
    """Run mBERT, XLM-R, MuRIL, IndicBERT.  Skips models with existing checkpoints."""
    results = {}
    for model_key, model_id in TRANSFORMER_MODELS.items():
        log.info(f"\n  Running {model_key} ({model_id}) …")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        train_d, val_d, test_d = cache.get_transformer_datasets(tokenizer)

        seed_metrics = []
        for seed in seeds:
            set_seed(seed)
            model = load_or_train_transformer(model_key, lang, seed, train_d, val_d,
                                              cache.class_weights)
            preds  = predict_transformer(model, test_d, DEVICE)
            y_true = cache.get_labels("test")
            seed_metrics.append(compute_metrics(y_true, preds))
            store_preds(lang, cache, f"{model_key} (fine-tuned)", seed, preds)
            del model
            _empty_cache()
        results[f"{model_key} (fine-tuned)"] = aggregate_seed_metrics(seed_metrics)
        log.info(f"    → Macro F1: "
                 f"{results[f'{model_key} (fine-tuned)']['macro_f1']:.4f} "
                 f"± {results[f'{model_key} (fine-tuned)']['macro_f1_std']:.4f}")
    return results


def run_ensemble(cache: DataCache, seeds: list, lang: str) -> tuple:
    """
    Train and evaluate the proposed MuRIL + BiLSTM-GRU ensemble.
    Returns (results_dict, alphas_per_seed, final_seed_preds, branch_results)
    """
    log.info("\n  Running Proposed Ensemble (MuRIL + BiLSTM-GRU) …")
    muril_id   = TRANSFORMER_MODELS["MuRIL"]
    from transformers import AutoTokenizer
    tokenizer  = AutoTokenizer.from_pretrained(muril_id)

    train_d_t, val_d_t, test_d_t = cache.get_transformer_datasets(tokenizer)
    train_d_r, val_d_r, test_d_r = cache.get_rnn_datasets()
    if train_d_r is None:
        log.warning("  Skipping ensemble — embedding file missing.")
        return {}, [], [], {}

    seed_metrics, branch_metrics = [], []
    ensemble_preds = []   # last seed predictions
    alphas, grids  = {}, {}
    y_true = cache.get_labels("test")

    for seed in seeds:
        set_seed(seed)

        # ── 1. Load already-trained MuRIL (from the main transformer run) ──
        t_model = load_or_train_transformer("MuRIL", lang, seed, train_d_t, val_d_t,
                                            cache.class_weights)

        # ── 2. Train (or load) the corrected BiLSTM-GRU branch ──
        r_model = load_or_train_branch(cache, lang, seed, train_d_r, val_d_r)

        # branch alone on test (documents that the branch now learns)
        r_preds = predict_rnn(r_model, test_d_r, DEVICE)
        branch_metrics.append(compute_metrics(y_true, r_preds))
        store_preds(lang, cache, "BiLSTM-GRU branch alone", seed, r_preds)

        # ── 3. Find best alpha on validation set (full grid recorded) ──
        alpha, grid = find_best_alpha(t_model, r_model, val_d_t, val_d_r, DEVICE,
                                      return_grid=True)
        alphas[str(seed)] = alpha
        grids[str(seed)]  = grid

        # ── 4. Evaluate ensemble on test set ──
        preds  = predict_ensemble(t_model, r_model, test_d_t, test_d_r,
                                  alpha, DEVICE)
        seed_metrics.append(compute_metrics(y_true, preds))
        store_preds(lang, cache, ENSEMBLE_NAME, seed, preds)
        if seed == seeds[-1]:
            ensemble_preds = preds
        del t_model, r_model
        _empty_cache()
    results = {ENSEMBLE_NAME: aggregate_seed_metrics(seed_metrics)}
    results[ENSEMBLE_NAME]["alpha_per_seed"] = alphas
    results[ENSEMBLE_NAME]["alpha_grid_per_seed"] = grids
    branch_results = aggregate_seed_metrics(branch_metrics)
    log.info(f"    → Macro F1: {results[ENSEMBLE_NAME]['macro_f1']:.4f} "
             f"± {results[ENSEMBLE_NAME]['macro_f1_std']:.4f}  alpha per seed={alphas}")
    log.info(f"    → BiLSTM-GRU branch alone: {branch_results['macro_f1']:.4f} "
             f"± {branch_results['macro_f1_std']:.4f}")
    return results, list(alphas.values()), ensemble_preds, branch_results


def run_crosslingual(cache_en: DataCache, cache_hi: DataCache, seeds: list) -> tuple:
    """
    English→Hindi cross-lingual transfer: zero-shot, few-shot at 5/10/20/50 %, and
    100 % Hindi; plus a Hindi-only control at every non-zero fraction.

      transfer : English-fine-tuned MuRIL, then fine-tuned on k % of Hindi train
      control  : pretrained MuRIL fine-tuned on the SAME k % subset, same recipe
                 (at 100 % this is the Table 5.4 MuRIL, so its checkpoint is reused)

    Returns (transfer_results, control_results, per_seed_macro_f1).
    """
    log.info("\n  Running English→Hindi cross-lingual transfer with Hindi-only control …")
    import torch.utils.data as tud

    muril_id  = TRANSFORMER_MODELS["MuRIL"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(muril_id)

    train_d_en, val_d_en, _ = cache_en.get_transformer_datasets(tokenizer)
    train_d_hi, val_d_hi, test_d_hi = cache_hi.get_transformer_datasets(tokenizer)
    y_true = cache_hi.get_labels("test")
    n_train = len(train_d_hi)

    regimes = {
        "Zero-shot (0% HI)"         : 0.0,
        "Few-shot 5% HI"            : 0.05,
        "Few-shot 10% HI"           : 0.10,
        "Few-shot 20% HI"           : 0.20,
        "Few-shot 50% HI"           : 0.50,
        "Full supervised (100% HI)" : 1.0,
    }
    transfer, control, per_seed = {}, {}, {}

    for regime_name, frac in regimes.items():
        log.info(f"\n    Regime: {regime_name}")
        t_metrics, c_metrics = [], []
        per_seed[regime_name] = {"transfer": {}, "control": {}}

        for seed in seeds:
            # One fixed permutation per seed → identical subsets for transfer and
            # control, nested across fractions.
            perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(seed)).tolist()
            if 0.0 < frac < 1.0:
                n = max(1, int(n_train * frac))
                train_subset = tud.Subset(train_d_hi, perm[:n])
            elif frac >= 1.0:
                train_subset = train_d_hi
            else:
                train_subset = None

            # (a) transfer
            set_seed(seed)
            model = load_or_train_transformer("MuRIL", "en", seed, train_d_en, val_d_en,
                                              cache_en.class_weights)
            if train_subset is not None:
                log.info(f"      transfer: fine-tuning EN MuRIL on {len(train_subset)} Hindi sentences")
                model = train_transformer(model, train_subset, val_d_hi,
                                          cache_hi.class_weights, DEVICE)
            preds = predict_transformer(model, test_d_hi, DEVICE)
            m = compute_metrics(y_true, preds)
            t_metrics.append(m)
            per_seed[regime_name]["transfer"][str(seed)] = m["macro_f1"]
            store_preds("hi", cache_hi, f"transfer | {regime_name}", seed, preds)
            del model
            _empty_cache()
            # (b) Hindi-only control (not defined for zero-shot)
            if train_subset is not None:
                set_seed(seed)
                if frac >= 1.0:
                    model = load_or_train_transformer("MuRIL", "hi", seed, train_d_hi, val_d_hi,
                                                      cache_hi.class_weights)
                else:
                    log.info(f"      control : fine-tuning pretrained MuRIL on the same "
                             f"{len(train_subset)} Hindi sentences")
                    model = get_transformer_model("MuRIL")
                    model = train_transformer(model, train_subset, val_d_hi,
                                              cache_hi.class_weights, DEVICE)
                preds = predict_transformer(model, test_d_hi, DEVICE)
                m = compute_metrics(y_true, preds)
                c_metrics.append(m)
                per_seed[regime_name]["control"][str(seed)] = m["macro_f1"]
                store_preds("hi", cache_hi, f"control | {regime_name}", seed, preds)
                del model
                _empty_cache()
        transfer[regime_name] = aggregate_seed_metrics(t_metrics)
        msg = f"      → transfer {transfer[regime_name]['macro_f1']:.4f}"
        if c_metrics:
            control[regime_name] = aggregate_seed_metrics(c_metrics)
            msg += f" | Hindi-only control {control[regime_name]['macro_f1']:.4f}"
        log.info(msg)

    return transfer, control, per_seed


def build_control_table(transfer: dict, control: dict, per_seed: dict) -> pd.DataFrame:
    rows = []
    for regime, t in transfer.items():
        row = {"Training regime": regime,
               "EN→HI transfer": f"{t['macro_f1']:.3f} ± {t['macro_f1_std']:.3f}",
               "Hindi-only control": "—",
               "Δ (transfer − control)": "—",
               "Transfer better (seeds)": "—"}
        c = control.get(regime)
        if c:
            ts, cs = per_seed[regime]["transfer"], per_seed[regime]["control"]
            diffs = [ts[s] - cs[s] for s in ts if s in cs]
            row["Hindi-only control"] = f"{c['macro_f1']:.3f} ± {c['macro_f1_std']:.3f}"
            row["Δ (transfer − control)"] = f"{np.mean(diffs):+.3f}"
            row["Transfer better (seeds)"] = f"{sum(d > 0 for d in diffs)}/{len(diffs)}"
        rows.append(row)
    return pd.DataFrame(rows)


def run_ablation(cache_hi: DataCache, seeds: list) -> dict:
    """Table 5: ablation on Hindi test set."""
    log.info("\n  Running Ablation Study (Hindi) …")
    muril_id  = TRANSFORMER_MODELS["MuRIL"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(muril_id)
    train_d_t, val_d_t, test_d_t = cache_hi.get_transformer_datasets(tokenizer)
    train_d_r, val_d_r, test_d_r = cache_hi.get_rnn_datasets()
    y_true = cache_hi.get_labels("test")

    configs = {}

    # 1. MuRIL only (no recurrent branch)
    seed_metrics = []
    for seed in seeds:
        set_seed(seed)
        model = load_or_train_transformer("MuRIL", "hi", seed, train_d_t, val_d_t,
                                          cache_hi.class_weights)
        preds = predict_transformer(model, test_d_t, DEVICE)
        seed_metrics.append(compute_metrics(y_true, preds))
        del model
        _empty_cache()
    configs["MuRIL only (no recurrent branch)"] = aggregate_seed_metrics(seed_metrics)

    if train_d_r is None:
        return configs

    # 2. BiLSTM-GRU only (no Transformer) — the same corrected branch the ensemble uses
    seed_metrics = []
    for seed in seeds:
        set_seed(seed)
        model = load_or_train_branch(cache_hi, "hi", seed, train_d_r, val_d_r)
        preds = predict_rnn(model, test_d_r, DEVICE)
        seed_metrics.append(compute_metrics(y_true, preds))
        del model
        _empty_cache()
    configs["BiLSTM-GRU only (no Transformer)"] = aggregate_seed_metrics(seed_metrics)

    # 3. Equal-weight fusion, and the full proposed model with validation-tuned alpha
    for alpha_label, alpha_val in [("equal-weight fusion (α=0.5)", 0.5),
                                   ("validation-tuned scalar", None)]:
        seed_metrics, alphas, grids = [], {}, {}
        for seed in seeds:
            set_seed(seed)
            t_model = load_or_train_transformer("MuRIL", "hi", seed, train_d_t, val_d_t,
                                                cache_hi.class_weights)
            r_model = load_or_train_branch(cache_hi, "hi", seed, train_d_r, val_d_r)
            if alpha_val is None:
                a, grid = find_best_alpha(t_model, r_model, val_d_t, val_d_r, DEVICE,
                                          return_grid=True)
                alphas[str(seed)], grids[str(seed)] = a, grid
            else:
                a = alpha_val
            preds = predict_ensemble(t_model, r_model, test_d_t, test_d_r, a, DEVICE)
            seed_metrics.append(compute_metrics(y_true, preds))
            del t_model, r_model
            _empty_cache()
        agg = aggregate_seed_metrics(seed_metrics)
        if alpha_val is None:
            alpha_txt = ", ".join(f"{alphas[str(s)]:.1f}" for s in seeds)
            key = f"Full proposed (validation-tuned α per seed: {alpha_txt})"
            agg["alpha_per_seed"] = alphas
            agg["alpha_grid_per_seed"] = grids
        else:
            key = f"Ensemble, {alpha_label}"
        configs[key] = agg

    return configs


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

MAIN_TITLES = {"en": "Table 3: English GoEmotions (5-class)",
               "hi": "Table 4: Hindi BHAAV+XED (5-class)"}


def print_saved_tables(langs: list):
    """Print Tables 3/4, 6, 6 (control), 5 and 7 from the saved _3seed results. No training."""
    main_res = {}
    for lang in langs:
        main_res[lang] = load_tagged(f"main_{lang}")
        if main_res[lang]:
            print_table(build_main_table(main_res[lang], lang), MAIN_TITLES[lang])
        else:
            log.warning(f"  results/main_{lang}_{RESULTS_TAG}.json not found.")
    transfer = load_tagged("crosslingual")
    control, per_seed = load_tagged("crosslingual_control"), load_tagged("crosslingual_per_seed")
    if transfer:
        print_table(build_crosslingual_table(transfer), "Table 6: Cross-lingual Transfer")
        if control and per_seed:
            print_table(build_control_table(transfer, control, per_seed),
                        "Table 6 (control): EN→HI transfer vs Hindi-only training on the same subset")
    abl = load_tagged("ablation")
    if abl:
        print_table(build_ablation_table(abl), "Table 5: Ablation Study")
    en, hi = main_res.get("en", {}), main_res.get("hi", {})
    if ENSEMBLE_NAME in en and ENSEMBLE_NAME in hi:
        print_table(build_perclass_table(en[ENSEMBLE_NAME], hi[ENSEMBLE_NAME]),
                    "Table 7: Per-class F1")


def main():
    args  = parse_args()
    seeds = args.seeds
    if args.generate_tables_only:
        log.info(f"Printing tables from results/*_{RESULTS_TAG}.json (no training).")
        print_saved_tables(args.lang)
        return
    log.info(f"Seeds: {seeds} | Device: {DEVICE} | results suffix: _{RESULTS_TAG}")

    caches = {}

    def get_cache(lang):
        if lang not in caches:
            caches[lang] = DataCache(lang)
        return caches[lang]

    all_results = {}

    for lang in args.lang:
        cache = get_cache(lang)
        lang_results = {}

        # ── Classical ML ──────────────────────────────────
        if not args.skip_classical:
            lang_results.update(run_classical(cache, seeds, lang))
            flush_preds()

        # ── RNN baselines ─────────────────────────────────
        if not args.skip_rnn:
            lang_results.update(run_rnn_models(cache, seeds, lang))
            flush_preds()

        # ── Transformer baselines ─────────────────────────
        if not args.skip_transformers:
            lang_results.update(run_transformer_models(cache, seeds, lang))
            flush_preds()

        # ── Ensemble ──────────────────────────────────────
        if not args.skip_ensemble:
            r, alphas, ens_preds, branch = run_ensemble(cache, seeds, lang)
            lang_results.update(r)
            if branch:
                save_merged({lang: branch}, "recurrent_branch")
            flush_preds()

        merged = save_merged(lang_results, f"main_{lang}")
        all_results[lang] = merged

        # ── Print Table 3 or 4 ────────────────────────────
        if merged:
            print_table(build_main_table(merged, lang), MAIN_TITLES[lang])

    # ── Cross-lingual transfer (requires both en and hi) ──
    if (not args.skip_crosslingual
            and "en" in args.lang and "hi" in args.lang):
        transfer, control, per_seed = run_crosslingual(get_cache("en"), get_cache("hi"), seeds)
        save_merged(transfer, "crosslingual")
        save_merged(control, "crosslingual_control")
        save_merged(per_seed, "crosslingual_per_seed")
        flush_preds()
        print_table(build_crosslingual_table(transfer), "Table 6: Cross-lingual Transfer")
        print_table(build_control_table(transfer, control, per_seed),
                    "Table 6 (control): EN→HI transfer vs Hindi-only training on the same subset")

    # ── Ablation (Hindi only) ─────────────────────────────
    if not args.skip_ablation and "hi" in args.lang:
        abl = run_ablation(get_cache("hi"), seeds)
        save_overwrite(abl, "ablation")
        print_table(build_ablation_table(abl), "Table 5: Ablation Study")

    # ── Table 7: Per-class F1 ─────────────────────────────
    if "en" in all_results and "hi" in all_results:
        if ENSEMBLE_NAME in all_results["en"] and ENSEMBLE_NAME in all_results["hi"]:
            print_table(
                build_perclass_table(all_results["en"][ENSEMBLE_NAME],
                                     all_results["hi"][ENSEMBLE_NAME]),
                "Table 7: Per-class F1")

    log.info(f"\n✅  All experiments complete.  Results saved to results/*_{RESULTS_TAG}.json")
    log.info(f"    Per-seed test predictions: results/predictions_<lang>_{RESULTS_TAG}.json")


if __name__ == "__main__":
    main()
