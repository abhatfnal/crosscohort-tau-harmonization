#!/usr/bin/env python3
"""
Logistic Regression Feature Importance
=======================================
Fits the primary balanced logistic regression pipeline on the full dataset
for each cohort × experiment and extracts standardized coefficients.
Features are already scaled by StandardScaler in the pipeline, so
coefficients are directly comparable within an experiment.

Outputs:
  logreg_feature_importance/
    feature_coefficients.csv   — long-form table
    feature_coefficients.tex   — LaTeX supplementary table
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

OUTDIR = Path("/project/aereditato/abhat/adni-mri-classification/logreg_feature_importance")
OUTDIR.mkdir(parents=True, exist_ok=True)

COHORTS = {
    "adni_tau90": {
        "label": "ADNI tau90",
        "path": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau90/subject_level_input_table.csv",
    },
    "adni_tau180": {
        "label": "ADNI tau180",
        "path": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau180/subject_level_input_table.csv",
    },
    "oasis3_tau180": {
        "label": "OASIS3 tau180",
        "path": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau180/subject_level_input_table.csv",
    },
    "oasis3_tau90": {
        "label": "OASIS3 tau90",
        "path": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau90/subject_level_input_table.csv",
    },
}

DEMO_NUM = ["age_h", "apoe_e4_carrier_h", "apoe_e4_count_h", "education_years_h"]
DEMO_CAT = ["sex_h"]
SEV_NUM  = ["cdr_sumboxes_h", "faq_total_h"]

EXPERIMENTS = {
    "reference_combined": (DEMO_NUM + SEV_NUM, DEMO_CAT),
    "demo_only":          (DEMO_NUM,            DEMO_CAT),
    "severity_only":      (SEV_NUM,             []),
}

FEATURE_LABELS = {
    "age_h":              "Age",
    "apoe_e4_carrier_h":  "APOE ε4 carrier",
    "apoe_e4_count_h":    "APOE ε4 count",
    "education_years_h":  "Education (years)",
    "cdr_sumboxes_h":     "CDR Sum of Boxes",
    "faq_total_h":        "FAQ total",
    "sex_h_0.0":          "Sex (female)",
    "sex_h_1.0":          "Sex (male)",
    "sex_h_F":            "Sex (female)",
    "sex_h_M":            "Sex (male)",
}


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
                      ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
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


def get_feature_names(pipe, num_cols, cat_cols):
    names = list(num_cols)
    if cat_cols:
        ohe = pipe.named_steps["preproc"].named_transformers_["cat"].named_steps["ohe"]
        for feat, cats in zip(cat_cols, ohe.categories_):
            for c in cats:
                names.append(f"{feat}_{c}")
    return names


rows = []

for cohort_key, cfg in COHORTS.items():
    print(f"\n{cfg['label']}")
    df = pd.read_csv(cfg["path"], low_memory=False)
    df["y_target"] = pd.to_numeric(df["y_target"], errors="coerce")
    df = df[df["y_target"].notna()].copy()
    y = df["y_target"].astype(int).to_numpy()

    for exp_name, (num_cols, cat_cols) in EXPERIMENTS.items():
        use_num = [c for c in num_cols if c in df.columns]
        use_cat = [c for c in cat_cols if c in df.columns]
        if not (use_num + use_cat):
            continue
        X = df[use_num + use_cat].copy()
        pipe = make_pipe(use_num, use_cat)
        pipe.fit(X, y)

        feat_names = get_feature_names(pipe, use_num, use_cat)
        coefs = pipe.named_steps["model"].coef_[0]

        for fname, coef in zip(feat_names, coefs):
            label = FEATURE_LABELS.get(fname, fname)
            rows.append({
                "cohort_key":   cohort_key,
                "cohort_label": cfg["label"],
                "experiment":   exp_name,
                "feature":      fname,
                "feature_label": label,
                "coefficient":  round(float(coef), 4),
            })
            print(f"  {exp_name:22s}  {label:25s}  {coef:+.4f}")


coef_df = pd.DataFrame(rows)
coef_df.to_csv(OUTDIR / "feature_coefficients.csv", index=False)
print(f"\nSaved: {OUTDIR / 'feature_coefficients.csv'}")


# ── LaTeX table ───────────────────────────────────────────────────────────────
COHORT_ORDER  = ["adni_tau90", "adni_tau180", "oasis3_tau180", "oasis3_tau90"]
COHORT_LABELS = {k: v["label"] for k, v in COHORTS.items()}
EXP_ORDER     = ["reference_combined", "demo_only", "severity_only"]
EXP_LABELS    = {
    "reference_combined": "Reference",
    "demo_only":          "Demo only",
    "severity_only":      "Severity only",
}

# Pivot: rows = feature × experiment, columns = cohort
pivot = coef_df.pivot_table(
    index=["experiment", "feature", "feature_label"],
    columns="cohort_key",
    values="coefficient",
    aggfunc="first",
).reset_index()

lines = []
lines.append(r"\begin{table}[ht]")
lines.append(r"\centering")
lines.append(r"\small")
col_spec = "ll" + "r" * len(COHORT_ORDER)
lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
lines.append(r"\toprule")
header = "Experiment & Feature"
for ck in COHORT_ORDER:
    header += f" & {COHORT_LABELS[ck]}"
lines.append(header + r" \\")
lines.append(r"\midrule")

for exp in EXP_ORDER:
    sub = pivot[pivot["experiment"] == exp].copy()
    if sub.empty:
        continue
    lines.append(rf"\multicolumn{{{2 + len(COHORT_ORDER)}}}{{l}}{{\textit{{{EXP_LABELS[exp]}}}}} \\")
    for _, row in sub.iterrows():
        feat_label = row["feature_label"]
        line = f"  & {feat_label}"
        for ck in COHORT_ORDER:
            val = row.get(ck, np.nan)
            line += f" & {val:+.3f}" if not pd.isna(val) else " & —"
        lines.append(line + r" \\")
    lines.append(r"\midrule")

lines[-1] = r"\bottomrule"
lines.append(r"\end{tabular}")
lines.append(r"\caption{Standardized logistic regression coefficients fitted on the full dataset")
lines.append(r"for each cohort and feature-block experiment. Positive values indicate association")
lines.append(r"with tau positivity. Coefficients are on the z-scored scale (numeric features")
lines.append(r"standardized by StandardScaler; categorical sex encoded by one-hot encoding).}")
lines.append(r"\label{tab:logreg_coefficients}")
lines.append(r"\end{table}")

tex = "\n".join(lines)
with open(OUTDIR / "feature_coefficients.tex", "w") as f:
    f.write(tex)
print(f"Saved: {OUTDIR / 'feature_coefficients.tex'}")
