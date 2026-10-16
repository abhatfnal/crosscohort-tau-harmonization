#!/usr/bin/env python3
"""
Table 2: demo-only AUC before and after feature harmonization
=============================================================
ADNI tau180 vs. OASIS3 tau180, demographic/genetic block only.

  pre-harmonization   the feature availability before the OASIS3 audit:
                      ADNI   age, APOE e4 count, education, sex
                      OASIS3 education, sex (age and APOE not yet recovered)
  post-harmonization  the shared block used in the matched-feature analysis:
                      age, APOE e4 carrier, APOE e4 count, education, sex

Same pipeline as the primary analysis (balanced logistic regression, five-fold
stratified CV, seed 42) on the final subject-level tables. The post-
harmonization AUCs must equal the Table 3 demo-only values, which is checked
against the bootstrap summary.

Output:
  paper_tables_and_figures/table2_harmonization_gap.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import bootstrap_crosscohort_auc_ci as primary

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "paper_tables_and_figures"
BOOTSTRAP_SUMMARY = ROOT / "crosscohort_bootstrap_ci" / "bootstrap_auc_summary.csv"

COHORTS = ["adni_tau180", "oasis3_tau180"]
POST_NUM = ["age_h", "apoe_e4_carrier_h", "apoe_e4_count_h", "education_years_h"]
SETTINGS = {
    "pre_harmonization": {
        "adni_tau180": (["age_h", "apoe_e4_count_h", "education_years_h"], ["sex_h"]),
        "oasis3_tau180": (["education_years_h"], ["sex_h"]),
    },
    "post_harmonization": {k: (POST_NUM, ["sex_h"]) for k in COHORTS},
}


def cv_auc(df, num_cols, cat_cols):
    X = df[num_cols + cat_cols]
    y = df["y_target"].to_numpy()
    cv = StratifiedKFold(n_splits=primary.N_SPLITS, shuffle=True, random_state=primary.RANDOM_STATE)
    pipe = Pipeline([
        ("preproc", primary.make_preprocessor(num_cols, cat_cols)),
        ("model", primary.make_model(primary.PRIMARY_MODEL)),
    ])
    aucs = []
    for tr, te in cv.split(X, y):
        fitted = clone(pipe).fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], fitted.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs))


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = {k: primary.read_subject_table(primary.DATASETS[k]["subject_table"]) for k in COHORTS}

    rows = []
    for setting, specs in SETTINGS.items():
        row = {"setting": setting}
        for k in COHORTS:
            num_cols, cat_cols = specs[k]
            row[f"{k}_auc"] = cv_auc(data[k], num_cols, cat_cols)
            row[f"{k}_features"] = "|".join(num_cols + cat_cols)
        row["gap_adni_minus_oasis3"] = row["adni_tau180_auc"] - row["oasis3_tau180_auc"]
        row["gap_absolute"] = abs(row["gap_adni_minus_oasis3"])
        rows.append(row)
    table = pd.DataFrame(rows)

    # Post-harmonization must reproduce the primary demo-only point estimates.
    summary = pd.read_csv(BOOTSTRAP_SUMMARY)
    post = table.set_index("setting").loc["post_harmonization"]
    for k in COHORTS:
        expected = summary.query("cohort_key == @k and experiment == 'demo_only'")["auc_point"].iloc[0]
        if not np.isclose(post[f"{k}_auc"], expected, atol=1e-9):
            raise RuntimeError(f"{k}: post-harmonization AUC {post[f'{k}_auc']:.6f} != Table 3 {expected:.6f}")

    out = OUTDIR / "table2_harmonization_gap.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
