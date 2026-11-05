"""
build_ch5_tables.py
───────────────────
Builds every Chapter 5 table and figure from the three-seed outputs of run_experiments.py.

    python build_ch5_tables.py
    python build_ch5_tables.py --results results --out results/ch5 --plots plots/ch5

Reads   results/predictions_{en,hi}_3seed.json    per-seed test predictions and gold labels
        results/main_{en,hi}_3seed.json           selected alpha and validation grid per seed
        results/ablation_3seed.json               equal-weight fusion row (aggregate only)
        results/table5_2_counts_{english,hindi}.csv
        branch_training_curves.json               (optional) per-epoch loss of the recurrent branch
Writes  results/ch5/Chapter5_tables.docx          every table with caption and note, and the figures
                                                  (needs:  pip install python-docx)
        results/ch5/*.csv                         the same tables plus per-seed details
        results/ch5/ch5_numbers.json              every number quoted in the revised text
        plots/ch5/*.png                           figures at 300 dpi

Statistics follow Section 3.11 of the thesis and the examiner's advice on Chapter 4:
  • mean ± sample standard deviation (n − 1) over seeds 42, 1337 and 2024
    (run_experiments.py prints the population SD; n − 1 matches Chapter 7)
  • seed-paired t-test on macro-F1 differences (Eq. 3.25, df = 2), with the number of seeds agreeing
  • McNemar's test with continuity correction (Eq. 3.24) run separately on each seed's predictions
    (the seeds share test items, so their predictions are not pooled), Holm-adjusted within each
    family of comparisons for each seed
  • seed-averaged item bootstrap (B = 10,000) for 95 % intervals on the key differences
"""

import os, sys, csv, json, math, argparse
import numpy as np

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise"]
EMO_TITLE = [e.capitalize() for e in EMOTIONS]
SEEDS = ["42", "1337", "2024"]
ENS = "Proposed Ensemble (MuRIL + BiLSTM-GRU)"
BRANCH = "BiLSTM-GRU branch alone"
MURIL, XLMR = "MuRIL (fine-tuned)", "XLM-R (fine-tuned)"
REGIMES = [("Zero-shot (0% HI)", 0.0), ("Few-shot 5% HI", 0.05), ("Few-shot 10% HI", 0.10),
           ("Few-shot 20% HI", 0.20), ("Few-shot 50% HI", 0.50), ("Full supervised (100% HI)", 1.0)]
ALPHA_REPORTED_IN_19 = 0.65

FAMILIES = {"classical": "Classical (TF-IDF)", "wordvec": "Neural on word vectors",
            "transformer": "Fine-tuned Transformer", "ensemble": "Transformer–recurrent ensemble"}

# Chart colours: categorical slots 1-4 and the single-hue blue ramp of the reference palette
# (validated on a white surface: adjacent and all-pairs CVD separation pass).
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FAMILY_COLOR = {"classical": SLOT[0], "wordvec": SLOT[1], "transformer": SLOT[2], "ensemble": SLOT[3]}


def model_order(lang):
    emb = "FastText" if lang == "en" else "IndicFT"
    return [("Naive Bayes (TF-IDF)", "classical"), ("Linear SVM (TF-IDF)", "classical"),
            (f"CNN + {emb}", "wordvec"), (f"BiLSTM + {emb}", "wordvec"),
            (f"GRU + {emb}", "wordvec"), (f"BiLSTM-Attention + {emb}", "wordvec"),
            ("mBERT (fine-tuned)", "transformer"), (XLMR, "transformer"),
            ("IndicBERT (fine-tuned)", "transformer"), (MURIL, "transformer"), (ENS, "ensemble")]


def display(name):
    return {"IndicBERT (fine-tuned)": "IndicBERT v1 (fine-tuned)",
            ENS: "Transformer–recurrent ensemble (MuRIL + BiLSTM–GRU)",
            BRANCH: "BiLSTM–GRU branch alone"}.get(name, name)


# ═══════════════════════════════════════════════════════════
# Metrics and statistics
# ═══════════════════════════════════════════════════════════

def seed_metrics(y, p):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    P, R, F, _ = precision_recall_fscore_support(y, p, labels=range(5), average="macro", zero_division=0)
    _, _, Fc, _ = precision_recall_fscore_support(y, p, labels=range(5), average=None, zero_division=0)
    _, _, F4, _ = precision_recall_fscore_support(y, p, labels=range(4), average="macro", zero_division=0)
    y, p = np.asarray(y), np.asarray(p)
    return {"precision": float(P), "recall": float(R), "macro_f1": float(F),
            "accuracy": float(accuracy_score(y, p)), "per_class_f1": [float(v) for v in Fc],
            "macro_f1_without_surprise": float(F4),
            "surprise_vs_rest_accuracy": float(np.mean((p == 4) == (y == 4)))}


def msd(vals):
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def fmt(m, s, d=4):
    return f"{m:.{d}f} ± {s:.{d}f}"


def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def fmt_p3(ps):
    """Three per-seed p-values in seed order, compact: 0.383 / 0.683 / 0.013."""
    return " / ".join("<.001" if p < 0.001 else f"{p:.3f}" for p in ps)


def holm_per_seed(comps):
    """Holm-adjust the per-seed McNemar p-values across a family of comparisons, separately for
    each seed. Adds c['p_mcnemar_seed_holm'] (list in SEEDS order) and c['mcnemar_sig_seeds']."""
    for k in range(len(SEEDS)):
        adj = holm([c["p_mcnemar_per_seed"][k] for c in comps])
        for c, a in zip(comps, adj):
            c.setdefault("p_mcnemar_seed_holm", [None] * len(SEEDS))[k] = float(a)
    for c in comps:
        c["mcnemar_sig_seeds"] = int(sum(p < 0.05 for p in c["p_mcnemar_seed_holm"]))
        c["mcnemar_worst_p"] = float(max(c["p_mcnemar_seed_holm"]))
    return comps


def stars(p):
    if p is None or math.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def mcnemar(y, preds_a, preds_b):
    """Eq. 3.24. b: A right and B wrong; c: the converse. Called with one seed's predictions."""
    from scipy.stats import chi2
    y = np.asarray(y)
    ca = np.concatenate([np.asarray(p) == y for p in preds_a])
    cb = np.concatenate([np.asarray(p) == y for p in preds_b])
    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    x = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "chi2": float(x), "p": float(chi2.sf(x, 1))}


def paired_t(fa, fb):
    """Eq. 3.25: seed-paired t-test on per-seed macro-F1, df = S - 1."""
    from scipy.stats import t as student_t
    d = np.asarray(fa, dtype=float) - np.asarray(fb, dtype=float)
    sd = d.std(ddof=1)
    if sd == 0:
        return float(d.mean()), float("nan"), float("nan")
    t = d.mean() / (sd / math.sqrt(len(d)))
    return float(d.mean()), float(t), float(2 * student_t.sf(abs(t), len(d) - 1))


def holm(pvals):
    p = np.asarray([np.nan if v is None else v for v in pvals], dtype=float)
    out = np.full(p.shape, np.nan)
    idx = np.where(~np.isnan(p))[0]
    order = idx[np.argsort(p[idx], kind="stable")]
    running, m = 0.0, len(order)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def boot_ci(y, preds_a, preds_b, B=10000, seed=2026, chunk=500):
    """Seed-averaged item bootstrap: resample test items (same resample for every seed and both
    models), average macro-F1 over seeds, take A - B; returns the 95 % percentile interval."""
    y = np.asarray(y)
    n, S = len(y), len(preds_a)
    cols = []
    for preds in (preds_a, preds_b):
        for p in preds:
            p = np.asarray(p)
            for c in range(5):
                cols += [(y == c) & (p == c), p == c, y == c]
    M = np.stack(cols, axis=1).astype(np.float64)
    rng = np.random.default_rng(seed)
    probs = np.full(n, 1.0 / n)
    diffs = []
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        W = rng.multinomial(n, probs, size=b).astype(np.float64)
        C = (W @ M).reshape(b, 2, S, 5, 3)
        tp, pred, true = C[..., 0], C[..., 1], C[..., 2]
        den = pred + true
        f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
        macro = f1.mean(axis=-1).mean(axis=-1)
        diffs.append(macro[:, 0] - macro[:, 1])
    lo, hi = np.percentile(np.concatenate(diffs), [2.5, 97.5])
    return float(lo), float(hi)


def pooled_confusion(y, preds):
    from sklearn.metrics import confusion_matrix
    return sum(confusion_matrix(y, p, labels=range(5)) for p in preds)


def chance_macro_f1(y):
    """Expected macro-F1 of a classifier that guesses uniformly at random."""
    pi = np.bincount(np.asarray(y), minlength=5) / len(y)
    q = 1 / 5
    return float(np.mean(2 * pi * q / (pi + q)))


# ═══════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_counts(path):
    rows = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            key = r.get("") or r.get("class") or list(r.values())[0]
            rows[key.strip()] = r
    return rows


class Lang:
    def __init__(self, results_dir, lang):
        self.lang = lang
        P = load_json(os.path.join(results_dir, f"predictions_{lang}_3seed.json"))
        self.y = np.asarray(P["y_true"])
        self.preds = P["models"]
        self.main = load_json(os.path.join(results_dir, f"main_{lang}_3seed.json"))
        self.m = {}
        for name, per_seed in self.preds.items():
            missing = [s for s in SEEDS if s not in per_seed]
            if missing:
                print(f"  [warn] {lang}: {name} has no predictions for seeds {missing}")
            self.m[name] = {s: seed_metrics(self.y, per_seed[s]) for s in SEEDS if s in per_seed}

    def plist(self, name):
        return [self.preds[name][s] for s in SEEDS if s in self.preds[name]]

    def vals(self, name, key):
        return [self.m[name][s][key] for s in SEEDS if s in self.m[name]]

    def summary(self, name, key):
        return msd(self.vals(name, key))


def compare(L, a, b, B):
    fa, fb = L.vals(a, "macro_f1"), L.vals(b, "macro_f1")
    d = np.asarray(fa) - np.asarray(fb)
    mean_d, t, p_t = paired_t(fa, fb)
    per_seed = [mcnemar(L.y, [L.preds[a][s]], [L.preds[b][s]]) for s in SEEDS]
    lo, hi = boot_ci(L.y, L.plist(a), L.plist(b), B=B) if B else (float("nan"), float("nan"))
    return {"a": a, "b": b, "delta_mean": float(d.mean()), "delta_sd": float(d.std(ddof=1)),
            "per_seed_delta": [float(v) for v in d], "seeds_a_better": int(np.sum(d > 0)),
            "t": t, "p_t": p_t,
            "mcnemar_per_seed": [{"seed": s, "b": m["b"], "c": m["c"], "chi2": m["chi2"], "p": m["p"]}
                                 for s, m in zip(SEEDS, per_seed)],
            "p_mcnemar_per_seed": [m["p"] for m in per_seed],
            "boot_lo": lo, "boot_hi": hi}


# ═══════════════════════════════════════════════════════════
# Tables
# ═══════════════════════════════════════════════════════════

def table(number, caption, columns, rows, note="", bold_cells=(), widths=None, key=None, font=9.5):
    return {"number": number, "caption": caption, "columns": columns, "rows": rows, "note": note,
            "bold": set(bold_cells), "widths": widths, "key": key or number, "font": font}


def t_counts(results_dir):
    en = read_counts(os.path.join(results_dir, "table5_2_counts_english.csv"))
    hi = read_counts(os.path.join(results_dir, "table5_2_counts_hindi.csv"))
    rows = []
    for e in EMOTIONS + ["TOTAL"]:
        r_en, r_hi = en[e], hi[e]
        label = "Total" if e == "TOTAL" else e.capitalize()
        cell = lambda r: f"{int(r['total']):,} ({float(r['share_%']):.1f} %)"
        rows.append([label] + [f"{int(r_en[k]):,}" for k in ("train", "val", "test")] + [cell(r_en)]
                    + [f"{int(r_hi[k]):,}" for k in ("train", "val", "test")] + [cell(r_hi)])
    n_hi_train = int(hi["TOTAL"]["train"])
    note = ("English: GoEmotions (Hugging Face “simplified” configuration) with the authors’ splits. "
            "Hindi: the official BHAAV release (Zenodo 10.5281/zenodo.3457467, CC BY 4.0), neutral "
            "dropped, conflicting and duplicate sentences removed, stratified 80/10/10 split with seed 42; "
            "Hindi fear is BHAAV suspense, and Hindi surprise is English XED surprise sentences "
            "machine-translated with opus-mt-en-hi. Sentence IDs (SHA-1 of the text) for every split "
            "are released in results/split_ids/.")
    cols = ["Emotion", "EN train", "EN val", "EN test", "EN total", "HI train", "HI val", "HI test", "HI total"]
    return table("5.2", "Class distribution of the harmonised five-class data by split.", cols, rows, note,
                 bold_cells=[(5, j) for j in range(9)], widths=[0.7, 0.6, 0.5, 0.52, 1.05, 0.6, 0.45, 0.5, 1.0],
                 key="t5_2_counts", font=9), n_hi_train


def t_main(L, number, caption, B):
    order = model_order(L.lang)
    comps = holm_per_seed([compare(L, ENS, name, B=0) for name, _ in order if name != ENS])
    star = {c["b"]: stars(c["mcnemar_worst_p"]) for c in comps}      # level reached on every seed
    rows, best = [], max(order, key=lambda o: L.summary(o[0], "macro_f1")[0])[0]
    bold = []
    for i, (name, _) in enumerate(order):
        label = display(name) + (f" {star[name]}" if star.get(name) else "")
        rows.append([label] + [fmt(*L.summary(name, k)) for k in ("accuracy", "precision", "recall", "macro_f1")])
        if name == best:
            bold.append((i, 4))
    lang_name = "English" if L.lang == "en" else "Hindi"
    note = (f"Mean ± sample standard deviation over seeds 42, 1337 and 2024 on the {lang_name} test set "
            f"(n = {len(L.y):,}). Stars mark models that differ from the ensemble under McNemar’s test "
            "(Eq. 3.24) on every one of the three seeds, run separately per seed and Holm-adjusted over the "
            "ten comparisons: * p < 0.05, ** p < 0.01, *** p < 0.001 on each seed. The classical models are "
            "deterministic, so their standard deviation is zero. Bold: highest macro-F1.")
    return table(number, caption, ["Model", "Accuracy", "Macro-P", "Macro-R", "Macro-F1"], rows, note,
                 bold_cells=bold, widths=[2.06, 0.985, 0.985, 0.985, 0.985],
                 key=f"t{number.replace('.', '_')}_{L.lang}_main", font=9), comps


def t_key(langs, B):
    spec = [("en", ENS, MURIL), ("en", ENS, XLMR), ("en", XLMR, MURIL),
            ("hi", ENS, MURIL), ("hi", ENS, XLMR), ("hi", MURIL, XLMR)]
    comps = holm_per_seed([dict(compare(langs[l], a, b, B=B), lang=l) for l, a, b in spec])
    pt_adj = holm([c["p_t"] for c in comps])
    rows, bold, current = [], [], None
    short = {ENS: "Ensemble", MURIL: "MuRIL", XLMR: "XLM-R"}
    for c, pt in zip(comps, pt_adj):
        c["p_t_holm"] = float(pt)
        if c["lang"] != current:
            current = c["lang"]
            bold.append((len(rows), 0))
            rows.append(["English test set" if current == "en" else "Hindi test set", "", "", "", "", ""])
        rows.append([f"   {short[c['a']]} − {short[c['b']]}",
                     f"{c['delta_mean']:+.4f} ± {c['delta_sd']:.4f}", f"{c['seeds_a_better']}/3",
                     fmt_p(c["p_t_holm"]), fmt_p3(c["p_mcnemar_seed_holm"]),
                     f"[{c['boot_lo']:+.4f}, {c['boot_hi']:+.4f}]"])
    note = ("Δ is the difference in macro-F1 (first minus second model), mean ± SD over the three seeds; "
            "‘Seeds’ counts the seeds on which the first model scores higher. p (paired t): seed-paired "
            "t-test on macro-F1 (Eq. 3.25, df = 2). McNemar: Eq. 3.24 run separately on each seed’s "
            "predictions, listed for seeds 42 / 1337 / 2024. Both are Holm-adjusted over the six comparisons "
            f"(McNemar within each seed). The interval is a seed-averaged item bootstrap (B = {B:,}) of the "
            "macro-F1 difference.")
    return table("5.5", "Paired comparisons of the ensemble and the two strongest encoders.",
                 ["Comparison", "Δ macro-F1", "Seeds", "p (paired t)", "McNemar p per seed", "95 % CI of Δ"],
                 rows, note, bold_cells=bold, widths=[1.42, 1.08, 0.48, 0.66, 1.3, 1.06],
                 key="t5_5_key_comparisons", font=9), comps


def t_ablation(H, abl_json):
    full = H.vals(ENS, "macro_f1")
    rows, stats = [], {}
    cfgs = [("MuRIL only (no recurrent branch)", MURIL), ("BiLSTM–GRU only (no Transformer)", BRANCH)]
    comps = holm_per_seed([compare(H, ENS, name, B=0) for _, name in cfgs])
    m_full, s_full = msd(full)
    for (label, name), c in zip(cfgs, comps):
        m, s = H.summary(name, "macro_f1")
        rows.append([label, fmt(m, s), f"{m - m_full:+.4f}", f"{c['seeds_a_better']}/3",
                     fmt_p3(c["p_mcnemar_seed_holm"])])
        stats[name] = c
    eq = next((v for k, v in abl_json.items() if k.startswith("Ensemble, equal-weight")), None)
    if eq is not None:
        sd = eq["macro_f1_std"] * math.sqrt(3 / 2)          # population SD (pipeline) -> sample SD
        rows.append(["Ensemble, equal-weight fusion (α = 0.5)", fmt(eq["macro_f1"], sd),
                     f"{eq['macro_f1'] - m_full:+.4f}", "—", "—"])
        stats["equal_weight"] = {"macro_f1": eq["macro_f1"], "macro_f1_sd": sd}
    alphas = ", ".join(f"{H.main[ENS]['alpha_per_seed'][s]:.1f}" for s in SEEDS)
    rows.append([f"Full ensemble (validation-tuned α per seed: {alphas})", fmt(m_full, s_full), "—", "—", "—"])
    note = ("Hindi test set, mean ± sample SD over three seeds. Δ is relative to the full ensemble; "
            "‘Full better’ counts the seeds on which the full ensemble scores higher. McNemar: the full "
            "ensemble against each configuration, run separately on each seed (seeds 42 / 1337 / 2024) and "
            "Holm-adjusted over the two comparisons within each seed. The equal-weight row was evaluated "
            "without saving per-seed predictions, so it has no paired test; its SD is converted from the "
            "population SD saved by the pipeline.")
    return table("5.6", "Ablation of the ensemble on the Hindi test set.",
                 ["Configuration", "Macro-F1", "Δ vs full", "Full better", "McNemar p per seed"], rows, note,
                 widths=[2.2, 1.1, 0.7, 0.62, 1.38], key="t5_6_ablation", font=9), stats


def t_alpha(langs):
    rows, out = [], {}
    for lang in ("en", "hi"):
        L = langs[lang]
        ens = L.main[ENS]
        for s in SEEDS:
            g = ens["alpha_grid_per_seed"][s]
            a = ens["alpha_per_seed"][s]
            te, tm = L.m[ENS][s]["macro_f1"], L.m[MURIL][s]["macro_f1"]
            rows.append(["English" if lang == "en" else "Hindi", s, f"{a:.1f}", f"{g['0.0']:.4f}",
                         f"{g['0.5']:.4f}", f"{g[f'{a:.1f}']:.4f}", f"{g['1.0']:.4f}", f"{te:.4f}", f"{tm:.4f}"])
            out[f"{lang}_{s}"] = {"alpha": a, "grid": g, "test_ensemble": te, "test_muril": tm}
        for a in ("0.6", "0.7"):
            out[f"{lang}_val_mean_at_{a}"] = float(np.mean([ens["alpha_grid_per_seed"][s][a] for s in SEEDS]))
        out[f"{lang}_val_mean_best"] = float(np.mean([ens["alpha_grid_per_seed"][s][f"{ens['alpha_per_seed'][s]:.1f}"] for s in SEEDS]))
        out[f"{lang}_val_mean_muril"] = float(np.mean([ens["alpha_grid_per_seed"][s]["1.0"] for s in SEEDS]))
    note = ("α is selected per seed from {0.1, …, 0.9} on the validation set (highest macro-F1, ties to the "
            "smaller α). α = 0 (recurrent branch alone) and α = 1 (MuRIL alone) are evaluated for reference "
            f"only. The value α = {ALPHA_REPORTED_IN_19} reported in [19] is not on this grid. Mean validation "
            f"macro-F1 at α = 0.6 and 0.7: English {out['en_val_mean_at_0.6']:.4f} and {out['en_val_mean_at_0.7']:.4f} "
            f"(selected α {out['en_val_mean_best']:.4f}); Hindi {out['hi_val_mean_at_0.6']:.4f} and "
            f"{out['hi_val_mean_at_0.7']:.4f} (selected α {out['hi_val_mean_best']:.4f}).")
    return table("5.7", "Fusion weight α selected on validation, with validation and test macro-F1 per seed.",
                 ["Language", "Seed", "α", "Val α = 0", "Val α = 0.5", "Val at α", "Val α = 1",
                  "Test: ensemble", "Test: MuRIL"], rows, note,
                 widths=[0.72, 0.45, 0.35, 0.7, 0.78, 0.7, 0.7, 0.8, 0.8], key="t5_7_alpha", font=9), out


def t_transfer(H, n_train, B):
    rows, out, comps = [], {}, []
    for regime, frac in REGIMES:
        tname, cname = f"transfer | {regime}", f"control | {regime}"
        tm, ts = H.summary(tname, "macro_f1")
        n = 0 if frac == 0 else max(1, int(n_train * frac))
        entry = {"n": n, "transfer": tm, "transfer_sd": ts, "transfer_per_seed": H.vals(tname, "macro_f1")}
        if cname in H.preds:
            c = compare(H, tname, cname, B=B)
            cm, cs = H.summary(cname, "macro_f1")
            entry.update(control=cm, control_sd=cs, control_per_seed=H.vals(cname, "macro_f1"), test=c)
            comps.append(entry)
        out[regime] = entry
    holm_per_seed([e["test"] for e in comps])
    labels = {"Zero-shot (0% HI)": "Zero-shot (0 % HI)", "Few-shot 5% HI": "Few-shot 5 % HI",
              "Few-shot 10% HI": "Few-shot 10 % HI", "Few-shot 20% HI": "Few-shot 20 % HI",
              "Few-shot 50% HI": "Few-shot 50 % HI", "Full supervised (100% HI)": "Full Hindi (100 %)"}
    for regime, _ in REGIMES:
        e = out[regime]
        if "test" in e:
            c = e["test"]
            rows.append([labels[regime], f"{e['n']:,}", fmt(e["transfer"], e["transfer_sd"], 3),
                         fmt(e["control"], e["control_sd"], 3), f"{c['delta_mean']:+.3f}",
                         f"{c['seeds_a_better']}/3", f"{c['mcnemar_sig_seeds']}/3",
                         f"[{c['boot_lo']:+.3f}, {c['boot_hi']:+.3f}]"])
        else:
            rows.append([labels[regime], "0", fmt(e["transfer"], e["transfer_sd"], 3), "—", "—", "—", "—", "—"])
    chance = chance_macro_f1(H.y)
    note = ("Hindi test set, macro-F1, mean ± sample SD over three seeds. Transfer: MuRIL fine-tuned on English "
            "GoEmotions, then fine-tuned on k % of the Hindi training set. Control: pretrained MuRIL fine-tuned "
            "on the same k % subset with the same recipe (four epochs, early stopping on Hindi validation). "
            "Within a seed both arms see identical sentences, and the subsets are nested across k. At 100 % "
            "the control is the MuRIL of Table 5.4. ‘Seeds’ counts seeds on which transfer scores higher; "
            "‘Sig.’ counts seeds on which McNemar’s test, run separately per seed and Holm-adjusted over the "
            "five regimes, gives p < 0.05; the interval is a seed-averaged item bootstrap "
            f"(B = {B:,}). A classifier guessing uniformly at random scores {chance:.2f} macro-F1 on this "
            "test set.")
    return table("5.8", "English→Hindi transfer against a Hindi-only control trained on the same subset.",
                 ["Training regime", "Hindi train", "EN→HI transfer", "Hindi-only control", "Δ", "Seeds",
                  "Sig.", "95 % CI of Δ"], rows, note,
                 widths=[1.1, 0.52, 0.88, 0.88, 0.58, 0.5, 0.59, 0.95], key="t5_8_transfer_control",
                 font=9), out, chance


def t_perclass(langs):
    rows, out = [], {}
    conf = {}
    for lang in ("en", "hi"):
        L = langs[lang]
        cm = pooled_confusion(L.y, L.plist(ENS))
        conf[lang] = cm
        out[lang] = {"confusion_counts_pooled": cm.tolist()}
    for i, emo in enumerate(EMO_TITLE):
        cells = [emo]
        for lang in ("en", "hi"):
            L = langs[lang]
            m, s = msd([L.m[ENS][sd]["per_class_f1"][i] for sd in SEEDS])
            cells.append(fmt(m, s, 3))
        for lang in ("en", "hi"):
            cm = conf[lang].astype(float)
            row = cm[i] / cm[i].sum()
            off = [(row[j], j) for j in range(5) if j != i]
            share, j = max(off)
            cells.append(f"→ {EMO_TITLE[j]} ({100 * share:.0f} %)")
            out[lang][f"main_confusion_{EMOTIONS[i]}"] = {"to": EMOTIONS[j], "share": float(share),
                                                         "recall": float(row[i])}
        rows.append(cells)
    macro = ["Macro average"] + [fmt(*langs[l].summary(ENS, "macro_f1"), 3) for l in ("en", "hi")] + ["—", "—"]
    rows.append(macro)
    four = {l: langs[l].summary(ENS, "macro_f1_without_surprise") for l in ("en", "hi")}
    out["macro_f1_without_surprise"] = {l: {"mean": four[l][0], "sd": four[l][1]} for l in four}
    note = ("Per-class F1 of the ensemble, mean ± sample SD over three seeds. The main confusion is the largest "
            "off-diagonal share of each true class in the confusion matrix summed over the three seeds "
            f"(Figure 5.8). Macro-F1 over the four classes other than surprise: English {four['en'][0]:.3f} ± "
            f"{four['en'][1]:.3f}, Hindi {four['hi'][0]:.3f} ± {four['hi'][1]:.3f}.")
    return table("5.9", "Per-class F1 of the Transformer–recurrent ensemble and the main confusion of each class.",
                 ["Emotion", "English F1", "Hindi F1", "Main confusion (EN)", "Main confusion (HI)"], rows,
                 note, widths=[1.0, 1.05, 1.05, 1.45, 1.45], key="t5_9_perclass"), out, conf


# ═══════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════

def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Arial", "Segoe UI", "Helvetica", "Liberation Sans", "DejaVu Sans") if f in have),
                  "DejaVu Sans")
    plt.rcParams.update({
        "font.family": family, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "savefig.dpi": 300,
    })
    return plt


def fig_classdist(plt, results_dir, path):
    """Figure 5.1: share of each class in the harmonised data, English against Hindi."""
    en = read_counts(os.path.join(results_dir, "table5_2_counts_english.csv"))
    hi = read_counts(os.path.join(results_dir, "table5_2_counts_hindi.csv"))
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    y = np.arange(len(EMOTIONS))[::-1]
    h = 0.34
    for k, (rows, label, color) in enumerate(
            ((en, f"English (GoEmotions, n = {int(en['TOTAL']['total']):,})", SLOT[0]),
             (hi, f"Hindi (BHAAV + XED-Hindi, n = {int(hi['TOTAL']['total']):,})", SLOT[1]))):
        shares = [float(rows[e]["share_%"]) for e in EMOTIONS]
        pos = y + (h / 2 + 0.02 if k == 0 else -h / 2 - 0.02)
        ax.barh(pos, shares, height=h, color=color, label=label, zorder=2)
        for p, s in zip(pos, shares):
            ax.text(s + 0.8, p, f"{s:.1f} %", va="center", ha="left", fontsize=7.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(EMO_TITLE)
    ax.set_xlim(0, 66)
    ax.set_xlabel("Share of the harmonised data (%)")
    ax.xaxis.grid(True, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_models(plt, L, path):
    order = model_order(L.lang)
    names = [n for n, _ in order][::-1]
    fams = [f for _, f in order][::-1]
    means = [L.summary(n, "macro_f1")[0] for n in names]
    sds = [L.summary(n, "macro_f1")[1] for n in names]
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    y = np.arange(len(names))
    ax.barh(y, means, height=0.56, color=[FAMILY_COLOR[f] for f in fams], zorder=2)
    ax.errorbar(means, y, xerr=sds, fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2, zorder=3)
    for yi, m, s in zip(y, means, sds):
        ax.text(m + s + 0.008, yi, f"{m:.3f}", va="center", ha="left", fontsize=7.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([display(n).replace(" (MuRIL + BiLSTM–GRU)", "") for n in names])
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("Macro-F1 (mean ± SD over three seeds)")
    ax.xaxis.grid(True, zorder=0)
    ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch
    handles = [Patch(color=FAMILY_COLOR[k], label=v) for k, v in FAMILIES.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=7.2,
               handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_branch(plt, curves, path):
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.7))
    style = {("en", False): (MUTED, "English, original branch (May run)"),
             ("en", True): (SLOT[0], "English, corrected branch"),
             ("hi", True): (SLOT[1], "Hindi, corrected branch")}
    seen = set()
    for run in curves["runs"]:
        key = (run["lang"], run["fixed"])
        if key not in style:
            continue                          # the May Hindi run used the mislabelled data
        color, label = style[key]
        lab = label if key not in seen else None
        seen.add(key)
        kw = dict(color=color, lw=1.5, zorder=3 if run["fixed"] else 4)
        if not run["fixed"]:
            kw.update(marker="o", ms=3.5, mec="white", mew=0.8)
        axes[0].plot(run["epochs"], run["loss"], label=lab, **kw)
        axes[1].plot(run["epochs"], run["val_macro_f1"], **kw)
    ln5 = curves.get("uniform_loss_ln5", math.log(5))
    axes[0].axhline(ln5, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=2)
    axes[0].text(15.4, ln5 + 0.05, f"ln 5 = {ln5:.3f} (uniform prediction)", va="bottom", ha="right",
                 fontsize=7, color=INK2)
    axes[0].set_ylabel("Training loss (weighted NLL)")
    axes[1].set_ylabel("Validation macro-F1")
    axes[0].set_ylim(0, 1.85)
    axes[1].set_ylim(0, 0.7)
    for ax, t in zip(axes, ["(a) Training loss", "(b) Validation macro-F1"]):
        ax.set_xlabel("Epoch")
        ax.set_xticks([1, 5, 10, 15])
        ax.set_xlim(0.5, 15.5)
        ax.yaxis.grid(True, zorder=0)
        ax.set_title(t, loc="left", color=INK)
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_alpha(plt, langs, path):
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.8))
    alphas = [round(0.1 * i, 1) for i in range(11)]
    markers = ["o", "s", "D"]
    for ax, lang, title in zip(axes, ("en", "hi"), ("(a) English validation", "(b) Hindi validation")):
        ens = langs[lang].main[ENS]
        lo = []
        for k, s in enumerate(SEEDS):
            g = ens["alpha_grid_per_seed"][s]
            vals = [g[f"{a:.1f}"] for a in alphas]
            lo.append(min(vals))
            ax.plot(alphas, vals, color=SLOT[k], lw=1.5, marker=markers[k], ms=3.5, mec="white", mew=0.7,
                    label=f"seed {s}", zorder=3)
            a = ens["alpha_per_seed"][s]
            ax.plot([a], [g[f"{a:.1f}"]], marker=markers[k], ms=6.2, color=SLOT[k], mec="white", mew=1.0,
                    zorder=4 + k)
        ax.axvline(ALPHA_REPORTED_IN_19, color=INK2, lw=0.9, ls=(0, (1.5, 2)), zorder=2)
        ymin, ymax = ax.get_ylim()
        ax.text(ALPHA_REPORTED_IN_19 - 0.02, ymin + 0.04 * (ymax - ymin), "α = 0.65\nin [19]",
                ha="right", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(["0\nbranch", "0.2", "0.4", "0.6", "0.8", "1\nMuRIL"])
        ax.set_xlabel("Fusion weight α on MuRIL")
        ax.yaxis.grid(True, zorder=0)
        ax.set_title(title, loc="left", color=INK)
    axes[0].set_ylabel("Validation macro-F1")
    axes[1].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_transfer(plt, tr, chance, path):
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    pts = [(e["n"], e) for r, e in tr.items() if e["n"] > 0]
    n = [p[0] for p in pts]
    for key, color, label, mk in (("transfer", SLOT[0], "EN→HI transfer (English-fine-tuned MuRIL)", "o"),
                                  ("control", SLOT[1], "Hindi-only control (pretrained MuRIL)", "s")):
        m = [e[key] for _, e in pts]
        s = [e[f"{key}_sd"] for _, e in pts]
        ax.errorbar(n, m, yerr=s, color=color, lw=1.5, marker=mk, ms=5, mec="white", mew=0.9,
                    elinewidth=0.9, capsize=2, label=label, zorder=3)
    zs = tr["Zero-shot (0% HI)"]["transfer"]
    ax.axhline(zs, color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text(n[-1] * 1.2, zs + 0.012, f"zero-shot transfer, no Hindi data ({zs:.3f})", va="bottom", ha="right",
            fontsize=7, color=INK2)
    ax.axhline(chance, color=MUTED, lw=0.9, ls=(0, (1.5, 2)), zorder=2)
    ax.text(n[-1] * 1.2, chance + 0.012, f"uniform random guessing ({chance:.2f})", va="bottom", ha="right",
            fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([f"{v:,}\n({100 * f:g} %)" for v, f in zip(n, [0.05, 0.10, 0.20, 0.50, 1.0])])
    ax.minorticks_off()
    ax.set_xlim(n[0] / 1.25, n[-1] * 1.25)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Hindi training sentences (share of the Hindi training set)")
    ax.set_ylabel("Hindi test macro-F1")
    ax.yaxis.grid(True, zorder=0)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_confusion(plt, conf, path):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_ramp", ["#ffffff"] + BLUE_RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0))
    for ax, lang, title in zip(axes, ("en", "hi"), ("(a) English (GoEmotions)", "(b) Hindi (BHAAV + XED-Hindi)")):
        cm = conf[lang].astype(float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.pcolormesh(norm, cmap=cmap, vmin=0, vmax=1, edgecolors="white", linewidth=1.5)
        for i in range(5):
            for j in range(5):
                v = norm[i, j]
                ax.text(j + 0.5, i + 0.5, f"{100 * v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.5 else INK)
        ax.set_xticks(np.arange(5) + 0.5)
        ax.set_yticks(np.arange(5) + 0.5)
        ax.set_xticklabels(EMO_TITLE, rotation=35, ha="right")
        ax.set_yticklabels(EMO_TITLE)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("Predicted")
        ax.set_title(title, loc="left", color=INK)
    axes[0].set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# Word document
# ═══════════════════════════════════════════════════════════

def build_docx(path, tables, figures, numbering, key_numbers):
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    import re
    minus = lambda s: re.sub(r"(?<![\w.])-(?=\d)", "−", str(s))     # typographic minus in numbers

    doc = Document()
    zoom = doc.settings.element.find(qn("w:zoom"))
    if zoom is not None and zoom.get(qn("w:percent")) is None:
        zoom.set(qn("w:percent"), "100")
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    sec.left_margin, sec.right_margin = Inches(1.25), Inches(1.0)
    sec.top_margin = sec.bottom_margin = Inches(1.0)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Times New Roman")
    normal.paragraph_format.space_after = Pt(6)
    for name in ("Heading 1", "Heading 2"):
        st = doc.styles[name]
        st.font.name, st.font.color.rgb = "Times New Roman", RGBColor(0, 0, 0)
        st.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:ascii"), "Times New Roman")
        st.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:hAnsi"), "Times New Roman")

    def border(cell, **edges):
        tcPr = cell._tc.get_or_add_tcPr()
        b = tcPr.find(qn("w:tcBorders"))
        if b is None:
            b = OxmlElement("w:tcBorders")
            tcPr.append(b)
        for edge, sz in edges.items():
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(sz))
            el.set(qn("w:color"), "000000")
            b.append(el)

    def caption(label, text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(12)
        r = p.add_run(label + " ")
        r.bold = True
        p.add_run(text)
        return p

    def note(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(14)
        r = p.add_run("Note. ")
        r.italic = True
        r.font.size = Pt(9.5)
        r2 = p.add_run(minus(text))
        r2.font.size = Pt(9.5)

    def fix_widths(tb, widths, lr_margin=58):
        tb.autofit = False                               # fixed layout
        for j, w in enumerate(widths):
            tb.columns[j].width = Inches(w)              # grid column (used by Word and LibreOffice)
            for cell in tb.columns[j].cells:
                cell.width = Inches(w)
        tblPr = tb._tbl.tblPr                            # narrower left/right cell padding (twips)
        mar = OxmlElement("w:tblCellMar")
        for side in ("left", "right"):
            el = OxmlElement(f"w:{side}")
            el.set(qn("w:w"), str(lr_margin))
            el.set(qn("w:type"), "dxa")
            mar.append(el)
        look = tblPr.find(qn("w:tblLook"))
        if look is not None:
            look.addprevious(mar)
        else:
            tblPr.append(mar)

    def add_table(t):
        caption(f"Table {t['number']}.", t["caption"])
        cols, rows = t["columns"], t["rows"]
        tb = doc.add_table(rows=len(rows) + 1, cols=len(cols))
        tb.alignment = WD_TABLE_ALIGNMENT.CENTER
        widths = t["widths"] or [6.0 / len(cols)] * len(cols)
        scale = min(1.0, 6.0 / sum(widths))
        fix_widths(tb, [w * scale for w in widths])
        for i, row in enumerate([cols] + rows):
            tr = tb.rows[i]
            trPr = tr._tr.get_or_add_trPr()
            cant = OxmlElement("w:cantSplit")
            trPr.append(cant)
            if i == 0:
                hdr = OxmlElement("w:tblHeader")
                trPr.append(hdr)
            for j, val in enumerate(row):
                cell = tr.cells[j]
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.space_before = Pt(1)
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(minus(val))
                run.font.size = Pt(t["font"])
                run.bold = (i == 0) or ((i - 1, j) in t["bold"])
                if i == 0:
                    border(cell, top=12, bottom=6)
                if i == len(rows):
                    border(cell, bottom=12)
        if t["note"]:
            note(t["note"])

    def add_figure(f):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(10)
        p.add_run().add_picture(f["path"], width=Inches(6.0))
        cp = caption(f"Figure {f['number']}.", f["caption"])
        cp.paragraph_format.keep_with_next = False
        cp.paragraph_format.space_after = Pt(14)

    doc.add_heading("Chapter 5: revised tables and figures (three seeds)", level=1)
    doc.add_paragraph(
        "Built by build_ch5_tables.py from the three-seed run (seeds 42, 1337, 2024) with the corrected BHAAV "
        "labels and the corrected recurrent branch. Every value is recomputed from the saved per-seed test "
        "predictions. Values are mean ± sample standard deviation over the three seeds; tests follow "
        "Section 3.11. The suggested numbering assumes the two new tables and two new figures are placed "
        "where they are first discussed.")
    doc.add_heading("Numbering", level=2)
    nt = doc.add_table(rows=len(numbering) + 1, cols=3)
    nt.alignment = WD_TABLE_ALIGNMENT.CENTER
    fix_widths(nt, [1.1, 3.6, 1.3])
    for i, row in enumerate([("Item", "Content", "Status")] + numbering):
        for j, val in enumerate(row):
            c = nt.rows[i].cells[j]
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            r = p.add_run(val)
            r.font.size = Pt(9.5)
            r.bold = i == 0
            if i == 0:
                border(c, top=12, bottom=6)
            if i == len(numbering):
                border(c, bottom=12)
    doc.add_heading("Tables", level=2)
    for t in tables:
        add_table(t)
    doc.add_page_break()
    doc.add_heading("Figures", level=2)
    for f in figures:
        add_figure(f)
    doc.add_heading("Numbers for the text", level=2)
    for line in key_numbers:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(minus(line)).font.size = Pt(10.5)
    doc.save(path)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def write_csv(path, t):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(t["columns"])
        w.writerows(t["rows"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=os.path.join("results", "ch5"))
    ap.add_argument("--plots", default=os.path.join("plots", "ch5"))
    ap.add_argument("--curves", default="branch_training_curves.json")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples (0 to skip)")
    ap.add_argument("--no-docx", action="store_true")
    args = ap.parse_args()
    try:                                     # α, −, → print safely even when output is redirected on Windows
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.plots, exist_ok=True)

    print("Loading predictions …")
    langs = {l: Lang(args.results, l) for l in ("en", "hi")}
    H = langs["hi"]
    abl = load_json(os.path.join(args.results, "ablation_3seed.json"))

    print("Building tables (bootstrap B = %d) …" % args.boot)
    t52, n_hi_train = t_counts(args.results)
    t53, comps_en = t_main(langs["en"], "5.3", "English emotion classification on harmonised GoEmotions.", args.boot)
    t54, comps_hi = t_main(H, "5.4", "Hindi emotion classification on harmonised BHAAV + XED-Hindi.", args.boot)
    t55, key = t_key(langs, args.boot)
    t56, abl_stats = t_ablation(H, abl)
    t57, alpha = t_alpha(langs)
    t58, tr, chance = t_transfer(H, n_hi_train, args.boot)
    t59, perclass, conf = t_perclass(langs)
    tables = [t52, t53, t54, t55, t56, t57, t58, t59]
    for t in tables:
        write_csv(os.path.join(args.out, f"{t['key']}.csv"), t)

    # per-seed macro-F1 of every model (appendix / checking)
    with open(os.path.join(args.out, "per_seed_macro_f1.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["language", "model"] + [f"seed {s}" for s in SEEDS] + ["mean", "sample SD"])
        for l, L in langs.items():
            for name in L.preds:
                v = L.vals(name, "macro_f1")
                w.writerow([l, name] + [f"{x:.4f}" for x in v] + [f"{np.mean(v):.4f}", f"{np.std(v, ddof=1):.4f}"])
    with open(os.path.join(args.out, "pairwise_vs_ensemble.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        head = ["language", "model", "Δ F1 (ensemble − model)", "seeds ensemble better", "p paired t"]
        for s in SEEDS:
            head += [f"seed {s}: b (ens right, model wrong)", f"seed {s}: c", f"seed {s}: chi2",
                     f"seed {s}: p", f"seed {s}: p Holm"]
        w.writerow(head + ["seeds significant (Holm p < 0.05)"])
        for l, comps in (("en", comps_en), ("hi", comps_hi)):
            for c in comps:
                row = [l, c["b"], f"{c['delta_mean']:+.4f}", c["seeds_a_better"], fmt_p(c["p_t"])]
                for m, ph in zip(c["mcnemar_per_seed"], c["p_mcnemar_seed_holm"]):
                    row += [m["b"], m["c"], f"{m['chi2']:.3f}", fmt_p(m["p"]), fmt_p(ph)]
                w.writerow(row + [c["mcnemar_sig_seeds"]])

    print("Drawing figures …")
    plt = setup_matplotlib()
    figs = []
    p = os.path.join(args.plots, "fig5_1_class_distribution.png")
    fig_classdist(plt, args.results, p)
    figs.append({"number": "5.1", "path": p, "caption": "Share of each emotion class in the harmonised English "
                 "(GoEmotions) and Hindi (BHAAV + XED-Hindi) data, all splits combined."})
    p = os.path.join(args.plots, "fig5_3_english_models.png")
    fig_models(plt, langs["en"], p)
    figs.append({"number": "5.3", "path": p, "caption": "English emotion classification on harmonised GoEmotions: "
                 "macro-F1 of the eleven configurations, mean ± SD over three seeds."})
    p = os.path.join(args.plots, "fig5_4_hindi_models.png")
    fig_models(plt, H, p)
    figs.append({"number": "5.4", "path": p, "caption": "Hindi emotion classification on harmonised BHAAV + "
                 "XED-Hindi: macro-F1 of the eleven configurations, mean ± SD over three seeds."})
    if os.path.exists(args.curves):
        p = os.path.join(args.plots, "fig5_5_branch_before_after.png")
        fig_branch(plt, load_json(args.curves), p)
        figs.append({"number": "5.5", "path": p, "caption": "Training loss and validation macro-F1 of the recurrent "
                     "branch by epoch. The original branch read the GRU state after the padding and stayed at the "
                     "uniform-prediction loss ln 5; the corrected branch (packed sequences) learns in both "
                     "languages (three seeds each)."})
    else:
        print(f"  [info] {args.curves} not found: skipping the before/after branch figure")
    p = os.path.join(args.plots, "fig5_6_alpha_curves.png")
    fig_alpha(plt, langs, p)
    figs.append({"number": "5.6", "path": p, "caption": "Validation macro-F1 of the fused prediction as a function "
                 "of α for each seed. Large markers show the selected α (chosen from 0.1 to 0.9); α = 0 is the "
                 "recurrent branch alone and α = 1 is MuRIL alone. The dotted line marks the α = 0.65 reported in [19]."})
    p = os.path.join(args.plots, "fig5_7_transfer_vs_control.png")
    fig_transfer(plt, tr, chance, p)
    figs.append({"number": "5.7", "path": p, "caption": "English→Hindi transfer against a Hindi-only control trained on "
                 "the same Hindi sentences: Hindi test macro-F1 (mean ± SD over three seeds) by amount of Hindi "
                 "training data, log scale."})
    p = os.path.join(args.plots, "fig5_8_confusion.png")
    fig_confusion(plt, conf, p)
    figs.append({"number": "5.8", "path": p, "caption": "Row-normalised confusion matrices (per cent) of the "
                 "Transformer–recurrent ensemble, summed over the three seeds; diagonals are per-class recall."})

    # numbers for the text
    E = langs["en"]
    ens = {l: langs[l].summary(ENS, "macro_f1") for l in ("en", "hi")}
    mur = {l: langs[l].summary(MURIL, "macro_f1") for l in ("en", "hi")}
    br = {l: langs[l].summary(BRANCH, "macro_f1") for l in ("en", "hi")}
    sur = {name: H.summary(name, "surprise_vs_rest_accuracy") for name in (ENS, MURIL, "Naive Bayes (TF-IDF)")}
    k_en = {(c["a"], c["b"]): c for c in key if c["lang"] == "en"}
    k_hi = {(c["a"], c["b"]): c for c in key if c["lang"] == "hi"}
    full_t = tr["Full supervised (100% HI)"]["transfer"]
    a_en = ", ".join("%.1f" % alpha["en_" + s]["alpha"] for s in SEEDS)
    a_hi = ", ".join("%.1f" % alpha["hi_" + s]["alpha"] for s in SEEDS)
    key_numbers = [
        f"English: XLM-R {fmt(*E.summary(XLMR, 'macro_f1'))}, MuRIL {fmt(*mur['en'])}, ensemble {fmt(*ens['en'])}; "
        f"ensemble − MuRIL {k_en[(ENS, MURIL)]['delta_mean']:+.4f} (higher on {k_en[(ENS, MURIL)]['seeds_a_better']}/3 "
        f"seeds; McNemar per seed {fmt_p3(k_en[(ENS, MURIL)]['p_mcnemar_seed_holm'])}).",
        f"Hindi: MuRIL {fmt(*mur['hi'])}, XLM-R {fmt(*H.summary(XLMR, 'macro_f1'))}, ensemble {fmt(*ens['hi'])}; "
        f"ensemble − MuRIL {k_hi[(ENS, MURIL)]['delta_mean']:+.4f} (higher on {k_hi[(ENS, MURIL)]['seeds_a_better']}/3 "
        f"seeds; McNemar per seed {fmt_p3(k_hi[(ENS, MURIL)]['p_mcnemar_seed_holm'])}).",
        f"Recurrent branch alone (corrected): English {fmt(*br['en'])}, Hindi {fmt(*br['hi'])}.",
        f"Selected α per seed: English {a_en}; Hindi {a_hi}.",
        f"Zero-shot transfer {fmt(tr['Zero-shot (0% HI)']['transfer'], tr['Zero-shot (0% HI)']['transfer_sd'], 3)} "
        f"against chance {chance:.2f}; transfer reaches {100 * tr['Few-shot 20% HI']['transfer'] / full_t:.0f} % of its "
        f"full-data score with 20 % of the Hindi data.",
        f"Hindi surprise-vs-rest accuracy (source separability): ensemble {100 * sur[ENS][0]:.1f} %, MuRIL "
        f"{100 * sur[MURIL][0]:.1f} %.",
        f"Macro-F1 without surprise: English {perclass['macro_f1_without_surprise']['en']['mean']:.3f}, Hindi "
        f"{perclass['macro_f1_without_surprise']['hi']['mean']:.3f}.",
    ]
    numbers = {
        "main": {l: {name: {k: dict(zip(("mean", "sd"), langs[l].summary(name, k)))
                            for k in ("accuracy", "precision", "recall", "macro_f1", "macro_f1_without_surprise",
                                      "surprise_vs_rest_accuracy")}
                     for name in langs[l].preds} for l in langs},
        "pairwise_vs_ensemble": {"en": comps_en, "hi": comps_hi},
        "key_comparisons": key, "ablation": abl_stats, "alpha": alpha, "transfer_control": tr,
        "chance_macro_f1_hi": chance, "per_class": perclass,
        "per_class_f1": {l: {EMOTIONS[i]: dict(zip(("mean", "sd"), msd([langs[l].m[ENS][s]["per_class_f1"][i]
                                                                       for s in SEEDS])))
                             for i in range(5)} for l in langs},
        "key_numbers": key_numbers,
    }
    with open(os.path.join(args.out, "ch5_numbers.json"), "w", encoding="utf-8") as f:
        json.dump(numbers, f, indent=1, ensure_ascii=False)

    numbering = [("Table 5.2", "Class distribution by split (replaces the old Table 5.2)", "Revised"),
                 ("Table 5.3", "English results, three seeds", "Revised"),
                 ("Table 5.4", "Hindi results, three seeds", "Revised"),
                 ("Table 5.5", "Paired comparisons of the ensemble and the strongest encoders", "New"),
                 ("Table 5.6", "Hindi ablation (was Table 5.5)", "Revised"),
                 ("Table 5.7", "Fusion weight α per seed", "New"),
                 ("Table 5.8", "Transfer with Hindi-only control (was Table 5.6)", "Revised"),
                 ("Table 5.9", "Per-class F1 and main confusion (was Table 5.7)", "Revised"),
                 ("Figure 5.1", "Class distribution (replaces the pie chart)", "Revised"),
                 ("Figure 5.3", "English model comparison", "Revised"),
                 ("Figure 5.4", "Hindi model comparison", "Revised"),
                 ("Figure 5.5", "Recurrent branch before and after the fix", "New"),
                 ("Figure 5.6", "Validation macro-F1 against α", "New"),
                 ("Figure 5.7", "Transfer against control (was Figure 5.5)", "Revised"),
                 ("Figure 5.8", "Confusion matrices (was Figure 5.6)", "Revised")]
    if not args.no_docx:
        try:
            build_docx(os.path.join(args.out, "Chapter5_tables.docx"), tables, figs, numbering, key_numbers)
            print(f"  Word file: {os.path.join(args.out, 'Chapter5_tables.docx')}")
        except ImportError:
            print("  [info] python-docx is not installed (pip install python-docx); CSV files and figures were written.")

    print("\nKey numbers")
    for line in key_numbers:
        print("  •", line)
    print(f"\nTables: {args.out}   Figures: {args.plots}")


if __name__ == "__main__":
    main()
