#!/usr/bin/env python3
"""
Diagnosis-Stratified AUC Analysis
===================================
Runs reference_combined and demo_only logreg CV AUC within CN, MCI, and
Dementia strata for each cohort. Tests the severity-entanglement hypothesis:
if CDR-SB is uniformly zero in CN subjects, the severity-feature block should
lose discriminative power there while demographics retain whatever signal exists.

Diagnosis column mapping:
  ADNI:   DX_DIAGNOSIS  1=CN, 2=MCI, 3=Dementia
  OASIS3: dx_harmonized CN / MCI / Dementia / Other/Unknown

Outputs:
  dx_stratified_auc/
    dx_stratified_results.csv
    dx_stratified_plot.png / .pdf
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

OUTDIR = Path("/project/aereditato/abhat/adni-mri-classification/dx_stratified_auc")
OUTDIR.mkdir(parents=True, exist_ok=True)

COHORTS = {
    "adni_tau90": {
        "label": "ADNI tau90",
        "path": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau90/subject_level_input_table.csv",
        "dx_col": "DX_DIAGNOSIS",
        "dx_map": {"1.0": "CN", "2.0": "MCI", "3.0": "Dementia"},
    },
    "adni_tau180": {
        "label": "ADNI tau180",
        "path": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau180/subject_level_input_table.csv",
        "dx_col": "DX_DIAGNOSIS",
        "dx_map": {"1.0": "CN", "2.0": "MCI", "3.0": "Dementia"},
    },
    "oasis3_tau180": {
        "label": "OASIS3 tau180",
        "path": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau180/subject_level_input_table.csv",
        "dx_col": "dx_harmonized",
        "dx_map": {"CN": "CN", "MCI": "MCI", "Dementia": "Dementia"},
    },
    "oasis3_tau90": {
        "label": "OASIS3 tau90",
        "path": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau90/subject_level_input_table.csv",
        "dx_col": "dx_harmonized",
        "dx_map": {"CN": "CN", "MCI": "MCI", "Dementia": "Dementia"},
    },
}

DEMO_NUM = ["age_h", "apoe_e4_carrier_h", "apoe_e4_count_h", "education_years_h"]
DEMO_CAT = ["sex_h"]
SEV_NUM  = ["cdr_sumboxes_h", "faq_total_h"]

EXPERIMENTS = {
    "reference_combined": (DEMO_NUM + SEV_NUM, DEMO_CAT),
    "demo_only":          (DEMO_NUM,            DEMO_CAT),
}

DX_ORDER   = ["CN", "MCI", "Dementia"]
MIN_POS    = 8    # below this, flag as unreliable
N_SPLITS   = 5
RANDOM_STATE = 42


def make_pipe(num_cols, cat_cols):
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]),
            cat_cols,
        ))
    return Pipeline([
        ("preproc", ColumnTransformer(transformers, remainder="drop")),
        ("model",   LogisticRegression(
            max_iter=5000, class_weight="balanced",
            solver="liblinear", random_state=RANDOM_STATE,
        )),
    ])


def cv_auc(df, num_cols, cat_cols):
    use_num = [c for c in num_cols if c in df.columns]
    use_cat = [c for c in cat_cols if c in df.columns]
    if not (use_num + use_cat):
        return np.nan
    X = df[use_num + use_cat].copy()
    y = df["y_target"].astype(int).to_numpy()
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos < 2 or n_neg < 2:
        return np.nan
    n_splits = min(N_SPLITS, n_pos, n_neg)
    if n_splits < 2:
        return np.nan
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pipe = make_pipe(use_num, use_cat)
    fold_aucs = []
    for tr, te in cv.split(X, y):
        pipe.fit(X.iloc[tr], y[tr])
        if len(np.unique(y[te])) < 2:
            continue
        prob = pipe.predict_proba(X.iloc[te])[:, 1]
        fold_aucs.append(roc_auc_score(y[te], prob))
    return float(np.mean(fold_aucs)) if fold_aucs else np.nan


rows = []

for cohort_key, cfg in COHORTS.items():
    print(f"\n{'='*65}")
    print(f"{cfg['label']}")
    print(f"{'='*65}")

    df = pd.read_csv(cfg["path"], low_memory=False)
    df["y_target"] = pd.to_numeric(df["y_target"], errors="coerce")
    df = df[df["y_target"].notna()].copy()

    # Map diagnosis to CN / MCI / Dementia
    dx_raw = df.get(cfg["dx_col"], pd.Series(dtype=str)).astype(str).str.strip()
    df["_dx"] = dx_raw.map(cfg["dx_map"])

    print(f"{'DX':<12} {'N':>5} {'N+':>5} {'tau+ %':>8}")
    print("-" * 35)
    for dx in DX_ORDER:
        sub_all = df[df["_dx"] == dx]
        n_total = len(sub_all)
        n_pos   = int(sub_all["y_target"].sum()) if n_total > 0 else 0
        pct     = 100 * n_pos / n_total if n_total > 0 else np.nan
        print(f"{dx:<12} {n_total:>5} {n_pos:>5} {pct:>7.1f}%")

        for exp_name, (num_cols, cat_cols) in EXPERIMENTS.items():
            use_num = [c for c in num_cols if c in sub_all.columns]
            use_cat = [c for c in cat_cols if c in sub_all.columns]
            n_pos_sub = int(sub_all["y_target"].sum()) if n_total > 0 else 0
            reliable  = n_pos_sub >= MIN_POS and (n_total - n_pos_sub) >= MIN_POS

            auc = cv_auc(sub_all, use_num, use_cat) if reliable else np.nan

            rows.append({
                "cohort_key":   cohort_key,
                "cohort_label": cfg["label"],
                "dx_class":     dx,
                "experiment":   exp_name,
                "n_total":      n_total,
                "n_pos":        n_pos_sub,
                "n_neg":        n_total - n_pos_sub,
                "tau_pos_pct":  round(pct, 1) if not np.isnan(pct) else np.nan,
                "auc":          round(auc, 4) if not np.isnan(auc) else np.nan,
                "reliable":     reliable,
            })

results_df = pd.DataFrame(rows)
results_df.to_csv(OUTDIR / "dx_stratified_results.csv", index=False)
print(f"\nSaved: {OUTDIR / 'dx_stratified_results.csv'}")


# ── Print summary table ────────────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY — AUC by diagnosis class (reference_combined | demo_only)")
print("="*80)
pivot = results_df.pivot_table(
    index=["cohort_label", "dx_class"],
    columns="experiment",
    values=["n_pos", "auc", "reliable"],
    aggfunc="first",
)
for (cohort_label, dx), grp in results_df.groupby(["cohort_label", "dx_class"]):
    ref_row  = grp[grp["experiment"] == "reference_combined"].iloc[0] if not grp[grp["experiment"] == "reference_combined"].empty else None
    demo_row = grp[grp["experiment"] == "demo_only"].iloc[0]          if not grp[grp["experiment"] == "demo_only"].empty          else None
    if ref_row is None or demo_row is None:
        continue
    ref_auc  = f"{ref_row['auc']:.3f}" if not pd.isna(ref_row['auc'])  else "NA"
    demo_auc = f"{demo_row['auc']:.3f}" if not pd.isna(demo_row['auc']) else "NA"
    flag_ref  = "" if ref_row["reliable"]  else " [UNRELIABLE]"
    flag_demo = "" if demo_row["reliable"] else " [UNRELIABLE]"
    n_pos = ref_row["n_pos"]
    n_tot = ref_row["n_total"]
    print(f"{cohort_label:18s}  {dx:<10}  N+={n_pos:>3}/{n_tot:<4}  "
          f"ref={ref_auc}{flag_ref}  demo={demo_auc}{flag_demo}")


# ── Plot ───────────────────────────────────────────────────────────────────────
COHORT_COLORS = {
    "adni_tau90":   "#2166AC",
    "adni_tau180":  "#6BAED6",
    "oasis3_tau180":"#4DAC26",
    "oasis3_tau90": "#B8E186",
}
EXP_STYLES = {
    "reference_combined": "-o",
    "demo_only":          "--s",
}
EXP_LABELS = {
    "reference_combined": "Reference combined",
    "demo_only":          "Demo only",
}
DX_X = {dx: i for i, dx in enumerate(DX_ORDER)}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig.subplots_adjust(wspace=0.15)

for ax, exp_name in zip(axes, ["reference_combined", "demo_only"]):
    sub = results_df[(results_df["experiment"] == exp_name)]
    for cohort_key, cfg in COHORTS.items():
        csub = sub[sub["cohort_key"] == cohort_key].copy()
        csub = csub[csub["dx_class"].isin(DX_ORDER)].copy()
        csub["dx_x"] = csub["dx_class"].map(DX_X)
        csub = csub.sort_values("dx_x")

        reliable   = csub[csub["reliable"]]
        unreliable = csub[~csub["reliable"]]

        color = COHORT_COLORS[cohort_key]
        if not reliable.empty:
            ax.plot(reliable["dx_x"], reliable["auc"],
                    EXP_STYLES[exp_name], color=color, label=cfg["label"],
                    markersize=7, linewidth=1.8)
        if not unreliable.empty:
            ax.scatter(unreliable["dx_x"], [0.5] * len(unreliable),
                       marker="x", color=color, s=50, zorder=3)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(list(DX_X.values()))
    ax.set_xticklabels(DX_ORDER, fontsize=11)
    ax.set_xlabel("Diagnosis class", fontsize=11)
    ax.set_ylim(0.3, 1.05)
    ax.set_title(EXP_LABELS[exp_name], fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    if ax is axes[0]:
        ax.set_ylabel("5-fold CV AUC (logreg balanced)", fontsize=11)
        ax.legend(fontsize=9, framealpha=0.8, loc="upper left")

fig.suptitle(
    "AUC stratified by diagnosis class\n"
    r"($\times$ = unreliable: N$_+$ < 5)",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()

for ext in ["png", "pdf"]:
    fig.savefig(OUTDIR / f"dx_stratified_plot.{ext}",
                dpi=300 if ext == "png" else None, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUTDIR / 'dx_stratified_plot.png'}")
print(f"Saved: {OUTDIR / 'dx_stratified_plot.pdf'}")
