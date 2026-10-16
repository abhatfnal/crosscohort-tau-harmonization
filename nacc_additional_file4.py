#!/usr/bin/env python3
"""
Additional file 4: NACC strict A/T sensitivity results
======================================================
Feature-block AUCs for the strict NACC endpoint (A+/T+ vs A-/T-), balanced
logistic regression with five-fold stratified CV (seed 42), the same pipeline
as the primary ADNI/OASIS3 analysis.

Cohorts:
  main          all strict A/T subjects; tau PET by flortaucipir or MK6240, so the
                tracer (NACC SCAN TRACER 6 = flortaucipir, 7 = MK6240) is a feature
  flortaucipir  flortaucipir-only subset (TRACER = 6); no tracer feature

Feature blocks follow nacc_at_strict_ablation_v1.py: the demographic block adds
race, Hispanic ethnicity and handedness to the harmonized variables, and the
severity block adds category fluency (animals, vegetables) to CDR, FAQ and MoCA.

Outputs (aggregate only):
  paper_tables_and_figures/additional_file4_nacc_sensitivity.csv
  paper_tables_and_figures/additional_file4_nacc_sensitivity.tex
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
NACC_DIR = Path("/project/aereditato/abhat/NACC_17259/phase2_tau_amyloid_90d/model_ready_v3_from_master")
TARGET = "y_AT_strict"

COHORTS = {
    "main": {
        "label": "NACC main",
        "csv": NACC_DIR / "nacc_AT_strict_model_table_nodx_withtracer_v3.csv",
        "tracer": ["tau_tracer"],
    },
    "flortaucipir": {
        "label": "NACC flortaucipir only",
        "csv": NACC_DIR / "nacc_AT_strict_tracer6_model_table_nodx_v3.csv",
        "tracer": [],
    },
}
# Logreg AUCs already written by nacc_at_strict_ablation_v1.py, used as a consistency check
ABLATION_REFERENCE = {
    ("main", "reference"): ("ablation_v1/all_ablation_metrics.csv", "main_combined_withtracer"),
    ("flortaucipir", "reference"): ("ablation_v1/all_ablation_metrics.csv", "tracer6_combined_notracer"),
    ("main", "demo_only"): ("ablation_v1/all_ablation_metrics.csv", "main_demo_apoe_withtracer"),
    ("main", "demo_only_no_tracer"): ("ablation_v1/all_ablation_metrics.csv", "main_demo_apoe_notracer"),
    ("main", "severity_only"): ("ablation_v1/all_ablation_metrics.csv", "main_severity_only"),
}

DEMO_NUM = ["age", "education_years", "apoe_e4_count", "apoe_e4_carrier"]
DEMO_CAT = ["sex", "race", "hispanic", "handedness"]
SEV = ["cdr_global", "cdr_sumboxes", "faq_total", "moca", "animal_fluency", "vegetable_fluency"]


def experiments(tracer):
    def drop(cols):
        return [c for c in SEV if c not in cols]
    return {
        "reference":            (DEMO_NUM + SEV, DEMO_CAT + tracer, "all features"),
        "severity_only":        (SEV, [], "severity block only"),
        "demo_only":            (DEMO_NUM, DEMO_CAT + tracer, "demographic block (as in Table 4)"),
        "demo_only_no_tracer":  (DEMO_NUM, DEMO_CAT, "demographic block without tracer"),
        "demo_only_harmonized": (DEMO_NUM, ["sex"], "five harmonized ADNI/OASIS3 variables"),
        "minus_all_cdr":        (DEMO_NUM + drop({"cdr_global", "cdr_sumboxes"}), DEMO_CAT + tracer, "reference minus CDR global and CDR-SB"),
        "minus_faq":            (DEMO_NUM + drop({"faq_total"}), DEMO_CAT + tracer, "reference minus FAQ"),
        "minus_moca":           (DEMO_NUM + drop({"moca"}), DEMO_CAT + tracer, "reference minus MoCA"),
        "minus_cdr_faq_moca":   (DEMO_NUM + drop({"cdr_global", "cdr_sumboxes", "faq_total", "moca"}),
                                 DEMO_CAT + tracer, "reference minus CDR, FAQ and MoCA (fluency kept)"),
    }


def cv_auc(df, num_cols, cat_cols):
    X = df[num_cols + cat_cols]
    y = df[TARGET].astype(int).to_numpy()
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
    rows = []
    for key, cfg in COHORTS.items():
        df = pd.read_csv(cfg["csv"], low_memory=False)
        n, n_pos = len(df), int(df[TARGET].astype(int).sum())
        for exp, (num_cols, cat_cols, desc) in experiments(cfg["tracer"]).items():
            if exp == "demo_only_no_tracer" and not cfg["tracer"]:
                continue  # identical to demo_only in a single-tracer cohort
            rows.append({"cohort": cfg["label"], "cohort_key": key, "n": n, "n_pos": n_pos,
                         "experiment": exp, "description": desc,
                         "features": "|".join(num_cols + cat_cols),
                         "auc": cv_auc(df, num_cols, cat_cols)})
    long = pd.DataFrame(rows)

    for (key, exp), (rel, ablation_exp) in ABLATION_REFERENCE.items():
        ab = pd.read_csv(NACC_DIR / rel)
        expected = ab[(ab["experiment"] == ablation_exp) & (ab["model"] == primary.PRIMARY_MODEL)]["roc_auc_mean"].iloc[0]
        got = long[(long["cohort_key"] == key) & (long["experiment"] == exp)]["auc"].iloc[0]
        if not np.isclose(got, expected, atol=1e-9):
            raise RuntimeError(f"{key}/{exp}: {got:.6f} != ablation output {expected:.6f}")

    wide = long.pivot_table(index=["experiment", "description"], columns="cohort", values="auc", aggfunc="first")
    wide = wide.reindex(list(experiments(["tau_tracer"]).keys()), level=0).reset_index()
    for key, cfg in COHORTS.items():
        sub = long[long["cohort_key"] == key].set_index("experiment")["auc"]
        ref = sub["reference"]
        print(f"{cfg['label']}: n={long[long.cohort_key == key].n.iloc[0]}, A+T+={long[long.cohort_key == key].n_pos.iloc[0]}; "
              f"severity retention {sub['severity_only'] / ref:.3f}, demo retention {sub['demo_only'] / ref:.3f}, "
              f"CDR drop {ref - sub['minus_all_cdr']:.3f}, full drop (ref - demo) {ref - sub['demo_only']:.3f}")

    out_csv = OUTDIR / "additional_file4_nacc_sensitivity.csv"
    long.drop(columns=["cohort_key"]).to_csv(out_csv, index=False)

    labels = [COHORTS[k]["label"] for k in COHORTS]
    counts = {COHORTS[k]["label"]: long[long.cohort_key == k][["n", "n_pos"]].iloc[0] for k in COHORTS}
    tex = [r"\begin{tabular}{lrr}", r"\toprule",
           "Experiment & " + " & ".join(f"{l} ($n={counts[l].n}$; {counts[l].n_pos} A+/T+)" for l in labels) + r" \\",
           r"\midrule"]
    for _, r in wide.iterrows():
        cells = [f"{r[l]:.3f}" if pd.notna(r.get(l)) else "---" for l in labels]
        tex.append(r["description"].replace("&", r"\&") + " & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    out_tex = OUTDIR / "additional_file4_nacc_sensitivity.tex"
    out_tex.write_text("\n".join(tex) + "\n")

    print(wide.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved: {out_csv}\nSaved: {out_tex}")


if __name__ == "__main__":
    main()
