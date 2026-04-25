#!/usr/bin/env python3
"""
CDR-SB = 0 Sensitivity Analysis
=================================
Restricts each cohort to subjects with CDR-SB = 0 (no detectable functional
impairment) and reports:
  1. Subgroup size and tau-positivity rate
  2. Feature distributions (age, APOE, education) by tau status
  3. Cross-validated AUC for reference_combined and demo_only experiments
     (reported with explicit caveat about extreme class imbalance)

The very low tau+ rate within CDR-SB=0 (1–5 %) directly supports the
stage-composition interpretation: functionally intact subjects are
predominantly tau-negative / preclinical.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

OUTDIR = Path("/project/aereditato/abhat/adni-mri-classification/sensitivity_cdrsb0")
OUTDIR.mkdir(parents=True, exist_ok=True)

COHORTS = {
    "adni_tau90": {
        "label": "ADNI tau90",
        "path": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau90/subject_level_input_table.csv",
        "cdrsb_col": "cdr_sumboxes_h",
    },
    "adni_tau180": {
        "label": "ADNI tau180",
        "path": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau180/subject_level_input_table.csv",
        "cdrsb_col": "cdr_sumboxes_h",
    },
    "oasis3_tau180": {
        "label": "OASIS3 tau180",
        "path": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau180/subject_level_input_table.csv",
        "cdrsb_col": "cdr_sumboxes_h",
    },
    "oasis3_tau90": {
        "label": "OASIS3 tau90",
        "path": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau90/subject_level_input_table.csv",
        "cdrsb_col": "cdr_sumboxes_h",
    },
}

# Matched shared features (same as primary analysis)
NUM_FEATURES = ["age_h", "apoe_e4_carrier_h", "apoe_e4_count_h", "education_years_h"]
CAT_FEATURES = ["sex_h"]
MIN_POS_FOR_AUC = 10  # below this, AUC is flagged as unreliable


def make_logreg_pipe(num_cols, cat_cols):
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
    preproc = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([
        ("preproc", preproc),
        ("model", LogisticRegression(
            max_iter=5000, class_weight="balanced",
            solver="liblinear", random_state=42,
        )),
    ])


def cv_auc(df, num_cols, cat_cols, n_splits=5):
    feat_cols = num_cols + cat_cols
    use_num = [c for c in num_cols if c in df.columns]
    use_cat = [c for c in cat_cols if c in df.columns]
    if not (use_num + use_cat):
        return np.nan, 0
    X = df[use_num + use_cat].copy()
    y = df["y_target"].astype(int).to_numpy()
    n_pos = int(y.sum())
    if n_pos < 2 or (len(y) - n_pos) < 2:
        return np.nan, n_pos
    n_splits_actual = min(n_splits, n_pos)
    cv = StratifiedKFold(n_splits=n_splits_actual, shuffle=True, random_state=42)
    pipe = make_logreg_pipe(use_num, use_cat)
    fold_aucs = []
    for tr, te in cv.split(X, y):
        pipe.fit(X.iloc[tr], y[tr])
        if len(np.unique(y[te])) < 2:
            continue
        prob = pipe.predict_proba(X.iloc[te])[:, 1]
        fold_aucs.append(roc_auc_score(y[te], prob))
    return (float(np.mean(fold_aucs)) if fold_aucs else np.nan), n_pos


def compare_feature(col, sub, label_pos="tau+", label_neg="tau-"):
    pos = pd.to_numeric(sub.loc[sub["y_target"] == 1, col], errors="coerce").dropna()
    neg = pd.to_numeric(sub.loc[sub["y_target"] == 0, col], errors="coerce").dropna()
    if len(pos) < 2 or len(neg) < 2:
        return {"feature": col, "tau+_mean": pos.mean() if len(pos) else np.nan,
                "tau-_mean": neg.mean() if len(neg) else np.nan, "p_value": np.nan,
                "note": "too few tau+ for test"}
    _, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    return {
        "feature": col,
        f"{label_pos}_n": len(pos), f"{label_pos}_mean": float(pos.mean()), f"{label_pos}_sd": float(pos.std()),
        f"{label_neg}_n": len(neg), f"{label_neg}_mean": float(neg.mean()), f"{label_neg}_sd": float(neg.std()),
        "p_value_mwu": float(p),
    }


summary_rows = []
feature_rows = []
auc_rows = []

for cohort_key, cfg in COHORTS.items():
    print(f"\n{'='*70}")
    print(f"{cfg['label']}")
    print(f"{'='*70}")

    df = pd.read_csv(cfg["path"], low_memory=False)
    df["y_target"] = pd.to_numeric(df["y_target"], errors="coerce")
    df = df[df["y_target"].notna()].copy()
    df["y_target"] = df["y_target"].astype(int)

    cdrsb = pd.to_numeric(df.get(cfg["cdrsb_col"], pd.Series(dtype=float)), errors="coerce")
    df["_cdrsb"] = cdrsb

    n_full = len(df)
    n_pos_full = int(df["y_target"].sum())

    sub = df[df["_cdrsb"] == 0].copy()
    n_sub = len(sub)
    n_pos_sub = int(sub["y_target"].sum())
    n_neg_sub = n_sub - n_pos_sub

    print(f"Full cohort:        N={n_full}, tau+={n_pos_full} ({100*n_pos_full/n_full:.1f}%)")
    print(f"CDR-SB=0 subgroup:  N={n_sub}, tau+={n_pos_sub} ({100*n_pos_sub/n_sub:.1f}% if n_sub>0 else '—')")

    summary_rows.append({
        "cohort_key": cohort_key,
        "cohort_label": cfg["label"],
        "n_full": n_full,
        "n_pos_full": n_pos_full,
        "pos_rate_full_pct": round(100 * n_pos_full / n_full, 1),
        "n_cdrsb0": n_sub,
        "n_pos_cdrsb0": n_pos_sub,
        "n_neg_cdrsb0": n_neg_sub,
        "pos_rate_cdrsb0_pct": round(100 * n_pos_sub / n_sub, 1) if n_sub > 0 else np.nan,
    })

    # Feature comparison within CDR-SB=0
    for col in ["age_h", "education_years_h", "apoe_e4_count_h"]:
        row = compare_feature(col, sub)
        row["cohort_key"] = cohort_key
        row["cohort_label"] = cfg["label"]
        feature_rows.append(row)
        if "p_value_mwu" in row:
            print(f"  {col}: tau+ mean={row.get('tau+_mean', np.nan):.2f} vs tau- mean={row.get('tau-_mean', np.nan):.2f}, p={row['p_value_mwu']:.3f}")

    # AUC within CDR-SB=0 for reference_combined and demo_only
    for exp_name, num_cols, cat_cols in [
        ("reference_combined", NUM_FEATURES, CAT_FEATURES),
        ("demo_only",          NUM_FEATURES, CAT_FEATURES),
    ]:
        auc, n_pos_used = cv_auc(sub, num_cols, cat_cols)
        reliable = n_pos_used >= MIN_POS_FOR_AUC
        flag = "" if reliable else f" [UNRELIABLE: N_pos={n_pos_used} < {MIN_POS_FOR_AUC}]"
        print(f"  AUC {exp_name}: {auc:.3f}{flag}" if not np.isnan(auc) else f"  AUC {exp_name}: NA{flag}")
        auc_rows.append({
            "cohort_key": cohort_key,
            "cohort_label": cfg["label"],
            "experiment": exp_name,
            "n_cdrsb0": n_sub,
            "n_pos_cdrsb0": n_pos_used,
            "auc": round(auc, 4) if not np.isnan(auc) else np.nan,
            "reliable": reliable,
            "note": "N_pos below reliable threshold" if not reliable else "",
        })

summary_df = pd.DataFrame(summary_rows)
feature_df = pd.DataFrame(feature_rows)
auc_df     = pd.DataFrame(auc_rows)

summary_df.to_csv(OUTDIR / "cdrsb0_subgroup_summary.csv", index=False)
feature_df.to_csv(OUTDIR / "cdrsb0_feature_comparison.csv", index=False)
auc_df.to_csv(OUTDIR / "cdrsb0_auc_results.csv", index=False)

print("\n" + "="*70)
print("SUBGROUP SUMMARY")
print("="*70)
print(summary_df[["cohort_label","n_full","pos_rate_full_pct",
                   "n_cdrsb0","n_pos_cdrsb0","pos_rate_cdrsb0_pct"]].to_string(index=False))

print("\nSaved:")
for f in ["cdrsb0_subgroup_summary.csv", "cdrsb0_feature_comparison.csv", "cdrsb0_auc_results.csv"]:
    print(" ", OUTDIR / f)
