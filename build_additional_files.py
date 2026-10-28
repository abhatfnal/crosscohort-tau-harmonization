#!/usr/bin/env python3
"""
Build Additional files 1-8 for the Alzheimer's Research & Therapy submission
===========================================================================
One .xlsx per Additional file. Sheet 1 is a README (title, description, sources,
sheet list and a definition for every column); the remaining sheets hold
aggregate, cohort-level results only. Balanced logistic regression is reported
everywhere except Additional file 1, which also carries the random-forest and
extra-trees robustness models in an explicit Model column.

Inputs are the outputs of the other scripts in this repository (plus row and
non-missing counts read from the controlled-access subject-level tables, which
are never written out). The script refuses to write a workbook if a column has
no definition or a table has one row per participant.

Usage:
    python build_additional_files.py --outdir <folder>
"""
from pathlib import Path
import argparse
import json
import re
import zipfile

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import bootstrap_crosscohort_auc_ci as primary

ROOT = Path(__file__).resolve().parent
ADNI_STRIP = {
    "adni_tau90": Path("/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau90/all_severity_strip_metrics.csv"),
    "adni_tau180": Path("/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau180/all_severity_strip_metrics.csv"),
}
OASIS_BUILD = {
    "oasis3_tau90": Path("/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau90/build_metadata.json"),
    "oasis3_tau180": Path("/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau180/build_metadata.json"),
}
OASIS_RAW = {
    "oasis3_tau90_raw": Path("/project/aereditato/abhat/oasis/phase0/oasis3_clinical_master_v7_tau90_amy90_fs90_fixed.csv"),
    "oasis3_tau180_raw": Path("/project/aereditato/abhat/oasis/phase0/oasis3_clinical_master_v7_tau180_amy180_fs180_fixed.csv"),
}
NACC_MAIN = Path("/project/aereditato/abhat/NACC_17259/phase2_tau_amyloid_90d/model_ready_v3_from_master/nacc_AT_strict_model_table_nodx_withtracer_v3.csv")

# Titles exactly as in the manuscript's "Supplementary information" paragraph
TITLES = {
    1: "Full matched-feature rerun metrics across all experiments and model families",
    2: "OASIS feature-audit tables and recovered column mappings",
    3: "Full-feature (unmatched) ADNI feature-block ablation results",
    4: "NACC sensitivity results (balanced logistic regression), including a flortaucipir-only subset and demographic-block variants without tau tracer",
    5: "OASIS3 GMM threshold-sensitivity results, including Supplementary Fig. S1 and Supplementary Table S5",
    6: "Standardized logistic regression feature importance coefficients across cohorts and experiments (Supplementary Table S6)",
    7: "Diagnosis-stratified feature-block AUC results across CN, MCI, and Dementia strata (Supplementary Fig. S2)",
    8: "Within-OASIS3 continuous tauopathy SUVR sensitivity analysis, including Supplementary Table S7 and Supplementary Fig. S3",
}

COHORT_LABEL = {"adni_tau90": "ADNI tau90", "adni_tau180": "ADNI tau180",
                "oasis3_tau180": "OASIS3 tau180", "oasis3_tau90": "OASIS3 tau90"}
COHORT_ROLE = {"adni_tau90": "primary", "adni_tau180": "primary",
               "oasis3_tau180": "primary", "oasis3_tau90": "supplementary"}
COHORT_ORDER = ["adni_tau90", "adni_tau180", "oasis3_tau180", "oasis3_tau90"]
PRIMARY_COHORTS = COHORT_ORDER[:3]
EXP_LABEL = {
    "reference_combined": "Reference-combined", "severity_only": "Severity-only",
    "demo_only": "Demo-only", "minus_all_cdr": "Minus all CDR", "minus_faq": "Minus FAQ",
    "minus_cdr_global": "Minus CDR global", "minus_cdr_sumboxes": "Minus CDR-SB",
    "minus_moca": "Minus MoCA", "minus_cdr_faq": "Minus CDR and FAQ",
    "minus_cdr_moca": "Minus CDR and MoCA", "minus_faq_moca": "Minus FAQ and MoCA",
    "minus_cdr_faq_moca": "Minus CDR, FAQ and MoCA (full strip)",
}
EXP_ORDER = list(EXP_LABEL)
MODEL_LABEL = {"logreg_balanced": "Balanced logistic regression",
               "random_forest_balanced": "Random forest (balanced)",
               "extra_trees_balanced": "Extra-trees (balanced)",
               "ridge": "Ridge regression"}
FEATURE_LABEL = {
    "age_h": "age", "education_years_h": "education", "apoe_e4_count_h": "APOE e4 count",
    "apoe_e4_carrier_h": "APOE e4 carrier", "sex_h": "sex", "cdr_global_h": "CDR global",
    "cdr_sumboxes_h": "CDR-SB", "faq_total_h": "FAQ total", "moca_h": "MoCA",
    # NACC model-table names
    "age": "age", "education_years": "education", "apoe_e4_count": "APOE e4 count",
    "apoe_e4_carrier": "APOE e4 carrier", "sex": "sex", "race": "race", "hispanic": "Hispanic ethnicity",
    "handedness": "handedness", "tau_tracer": "tau tracer", "cdr_global": "CDR global", "cdr_sumboxes": "CDR-SB",
    "faq_total": "FAQ total", "moca": "MoCA", "animal_fluency": "animal fluency", "vegetable_fluency": "vegetable fluency",
}
PARTICIPANT_COUNTS = {438, 547, 368, 142, 413, 296}   # cohort sizes: a table this long is suspect

# Shared column definitions; sheets may add their own
COLDEF = {
    "Cohort": "Analytic cohort (tau90/tau180 = 90-day/180-day modality-matching window).",
    "Role": "Primary or supplementary cohort, as in the manuscript.",
    "Experiment": "Feature-block experiment (see manuscript Section 2.6).",
    "Model": "Classifier family.",
    "N": "Number of participants in the analysis (aggregate count).",
    "N tau+": "Number of tau-positive participants.",
    "N tau-": "Number of tau-negative participants.",
    "N features": "Number of predictors entering the model.",
    "Features": "Predictors used (harmonized variable names).",
    "AUC": "Area under the ROC curve, mean of five stratified cross-validation folds (seed 42).",
    "AUC SD": "Standard deviation of AUC across the five folds.",
    "AP": "Average precision, mean of five folds.",
    "AP SD": "Standard deviation of average precision across folds.",
    "Balanced accuracy": "Balanced accuracy at the default 0.5 probability threshold, mean of five folds.",
    "Balanced accuracy SD": "Standard deviation of balanced accuracy across folds.",
    "F1": "F1 score at the default 0.5 probability threshold, mean of five folds.",
    "F1 SD": "Standard deviation of F1 across folds.",
}


def features_text(num, cat=""):
    cols = [c for c in str(num).split("|") + str(cat).split("|") if c and c != "nan"]
    return "; ".join(FEATURE_LABEL.get(c, c) for c in cols)


def order_frame(df, key_col="cohort_key", exp_col="experiment"):
    df = df.copy()
    df["_c"] = df[key_col].map({k: i for i, k in enumerate(COHORT_ORDER)})
    df["_e"] = df[exp_col].map({k: i for i, k in enumerate(EXP_ORDER)})
    sort = ["_c", "_e"] + (["_m"] if "_m" in df else [])
    return df.sort_values(sort).drop(columns=[c for c in ("_c", "_e", "_m") if c in df])


# ----------------------------------------------------------------------------- workbook writer
ARIAL = "Arial"
HEADER_FILL = PatternFill("solid", fgColor="D9D9D9")


def number_format(col):
    if col.startswith(("N", "n ")) or col in ("Threshold rank",):
        return "0"
    if "%" in col:
        return "0.0"
    if col.startswith(("Coefficient",)) or col in ("ADNI tau90", "ADNI tau180", "OASIS3 tau180"):
        return "0.0000"
    return "0.000"


def check_aggregate(name, df):
    bad_header = [c for c in df.columns
                  if re.search(r"(?i)\b(subject|subj|id|ids|ptid|rid|naccid|oasisid|date|dates|visit|visits|session|sessions)\b",
                               re.sub(r"[_\-]", " ", str(c)))]
    if bad_header:
        raise RuntimeError(f"{name}: identifier-like column(s) {bad_header}")
    if len(df) > 100 or len(df) in PARTICIPANT_COUNTS:
        raise RuntimeError(f"{name}: {len(df)} rows looks like a participant-level table")


def write_workbook(path, number, description, sources, sheets, notes=(), figures=()):
    """sheets: list of (sheet name, dataframe, sheet description, {column: definition})."""
    wb = Workbook()
    readme = wb.active
    readme.title = "README"
    rows = [(f"Additional file {number}", None), (TITLES[number], None), (None, None),
            ("Description", description), ("Source", sources),
            ("Content", "Aggregate, cohort-level results only. No participant-level data.")]
    for note in notes:
        rows.append(("Note", note))
    rows += [(None, None), ("Sheets", None)]
    for name, df, desc, _ in sheets:
        rows.append((name, desc))
    for name, (caption, _) in figures:
        rows.append((name, caption))
    rows += [(None, None), ("Column definitions", None)]
    for name, df, desc, extra in sheets:
        check_aggregate(f"Additional file {number} / {name}", df)
        for col in df.columns:
            definition = extra.get(col, COLDEF.get(col))
            if not definition:
                raise RuntimeError(f"Additional file {number} / {name}: no definition for column {col!r}")
            rows.append((f"{name}: {col}", definition))
    for r, (a, b) in enumerate(rows, start=1):
        ca = readme.cell(row=r, column=1, value=a)
        cb = readme.cell(row=r, column=2, value=b)
        bold = r in (1, 2) or a in ("Sheets", "Column definitions", "Description", "Source", "Content", "Note")
        ca.font = Font(name=ARIAL, size=12 if r <= 2 else 10, bold=bold)
        cb.font = Font(name=ARIAL, size=10)
        ca.alignment = Alignment(vertical="top", wrap_text=True)
        cb.alignment = Alignment(vertical="top", wrap_text=True)
    readme.column_dimensions["A"].width = 42
    readme.column_dimensions["B"].width = 110

    for name, df, desc, extra in sheets:
        ws = wb.create_sheet(name)
        for j, col in enumerate(df.columns, start=1):
            c = ws.cell(row=1, column=j, value=col)
            c.font = Font(name=ARIAL, size=10, bold=True)
            c.fill = HEADER_FILL
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for i, row in enumerate(df.itertuples(index=False), start=2):
            for j, (col, val) in enumerate(zip(df.columns, row), start=1):
                if isinstance(val, (float, np.floating)) and np.isnan(val):
                    val = None
                elif isinstance(val, np.integer):
                    val = int(val)
                elif isinstance(val, np.floating):
                    val = float(val)
                elif isinstance(val, np.bool_):
                    val = bool(val)
                c = ws.cell(row=i, column=j, value=val)
                c.font = Font(name=ARIAL, size=10)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    c.number_format = number_format(col)
        for j, col in enumerate(df.columns, start=1):
            longest = max([len(str(col))] + [len(str(v)) for v in df[col].tolist()])
            ws.column_dimensions[get_column_letter(j)].width = min(max(10, longest + 2), 60)
        ws.freeze_panes = "A2"

    for name, (caption, png) in figures:
        ws = wb.create_sheet(name)
        ws.cell(row=1, column=1, value=caption).font = Font(name=ARIAL, size=10, bold=True)
        img = XLImage(str(png))
        scale = 960 / img.width
        img.width, img.height = int(img.width * scale), int(img.height * scale)
        ws.add_image(img, "A3")

    wb.properties.creator = "Bhat et al."
    wb.properties.title = f"Additional file {number}"
    wb.save(path)
    strip_core_timestamps(path)


def strip_core_timestamps(path):
    """Drop the created/modified timestamps openpyxl writes into docProps/core.xml."""
    with zipfile.ZipFile(path) as zin:
        items = [(i, zin.read(i.filename)) for i in zin.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for info, data in items:
            if info.filename == "docProps/core.xml":
                data = re.sub(rb"<dcterms:(created|modified)[^>]*>[^<]*</dcterms:\1>", b"", data)
            zout.writestr(info, data)


# ----------------------------------------------------------------------------- Additional files
def af1(outdir):
    am = pd.read_csv(ROOT / "crosscohort_matched_rerun" / "all_metrics.csv")
    am["_m"] = am["model"].map({"logreg_balanced": 0, "random_forest_balanced": 1, "extra_trees_balanced": 2})
    am = order_frame(am)
    metrics = pd.DataFrame({
        "Cohort": am["cohort_key"].map(COHORT_LABEL), "Role": am["cohort_key"].map(COHORT_ROLE),
        "Experiment": am["experiment"].map(EXP_LABEL), "Model": am["model"].map(MODEL_LABEL),
        "N": am["n_subjects"], "N tau+": am["n_pos"], "N tau-": am["n_neg"],
        "N features": am["n_features_used"],
        "Features": [features_text(a, b) for a, b in zip(am["numeric_features_used"], am["categorical_features_used"])],
        "AUC": am["roc_auc_mean"], "AUC SD": am["roc_auc_std"], "AP": am["ap_mean"], "AP SD": am["ap_std"],
        "Balanced accuracy": am["bal_acc_mean"], "Balanced accuracy SD": am["bal_acc_std"],
        "F1": am["f1_mean"], "F1 SD": am["f1_std"],
    })

    lr = am[am["model"] == "logreg_balanced"].pivot(index="cohort_key", columns="experiment", values="roc_auc_mean")
    counts = am.drop_duplicates("cohort_key").set_index("cohort_key")
    summary = pd.DataFrame([{
        "Cohort": COHORT_LABEL[k], "Role": COHORT_ROLE[k], "N": counts.loc[k, "n_subjects"], "N tau+": counts.loc[k, "n_pos"],
        "AUC reference": lr.loc[k, "reference_combined"], "AUC severity-only": lr.loc[k, "severity_only"],
        "AUC demo-only": lr.loc[k, "demo_only"], "AUC minus all CDR": lr.loc[k, "minus_all_cdr"],
        "AUC minus FAQ": lr.loc[k, "minus_faq"],
        "Severity retention": lr.loc[k, "severity_only"] / lr.loc[k, "reference_combined"],
        "Demo retention": lr.loc[k, "demo_only"] / lr.loc[k, "reference_combined"],
        "CDR drop": lr.loc[k, "reference_combined"] - lr.loc[k, "minus_all_cdr"],
        "Full drop": lr.loc[k, "reference_combined"] - lr.loc[k, "demo_only"],
        "FAQ drop": lr.loc[k, "reference_combined"] - lr.loc[k, "minus_faq"],
    } for k in COHORT_ORDER])

    bs = order_frame(pd.read_csv(ROOT / "crosscohort_bootstrap_ci" / "bootstrap_auc_summary.csv"))
    boot = pd.DataFrame({
        "Cohort": bs["cohort_key"].map(COHORT_LABEL), "Experiment": bs["experiment"].map(EXP_LABEL),
        "Model": bs["model"].map(MODEL_LABEL), "N": bs["n_subjects"], "N tau+": bs["n_pos"], "N tau-": bs["n_neg"],
        "AUC": bs["auc_point"], "Bootstrap mean AUC": bs["auc_boot_mean"],
        "95% CI lower": bs["auc_ci_low"], "95% CI upper": bs["auc_ci_high"], "N bootstrap resamples": bs["n_bootstrap"],
    })
    bd = pd.read_csv(ROOT / "crosscohort_bootstrap_ci" / "bootstrap_auc_differences.csv")
    contrast_label = {"diff_ref_minus_severity": "Reference minus severity-only",
                      "diff_ref_minus_demo": "Reference minus demo-only",
                      "diff_ref_minus_minusallcdr": "Reference minus minus-all-CDR"}
    bd["_c"] = bd["cohort_key"].map({k: i for i, k in enumerate(COHORT_ORDER)})
    bd["_d"] = bd["difference"].map({k: i for i, k in enumerate(contrast_label)})
    bd = bd.sort_values(["_c", "_d"])
    contrasts = pd.DataFrame({
        "Cohort": bd["cohort_key"].map(COHORT_LABEL), "Contrast": bd["difference"].map(contrast_label),
        "Model": MODEL_LABEL["logreg_balanced"], "AUC difference": bd["auc_diff_point"],
        "Bootstrap mean difference": bd["auc_diff_boot_mean"],
        "95% CI lower": bd["auc_diff_ci_low"], "95% CI upper": bd["auc_diff_ci_high"],
        "N bootstrap resamples": bd["n_bootstrap"],
    })

    boot_defs = {
        "AUC": "Five-fold cross-validated AUC on the full cohort (the Table 3 value and the Fig. 2 point).",
        "Bootstrap mean AUC": "Mean AUC over the bootstrap replicates (each replicate reruns five-fold CV on a resample).",
        "95% CI lower": "2.5th percentile of the bootstrap distribution.",
        "95% CI upper": "97.5th percentile of the bootstrap distribution.",
        "N bootstrap resamples": "Number of subject-level bootstrap resamples, stratified by tau status (seed 42).",
    }
    sum_defs = {
        "AUC reference": "AUC of the reference-combined model (balanced logistic regression).",
        "AUC severity-only": "AUC of the severity-only model.", "AUC demo-only": "AUC of the demo-only model.",
        "AUC minus all CDR": "AUC of the reference model with CDR variables removed.",
        "AUC minus FAQ": "AUC of the reference model with FAQ removed.",
        "Severity retention": "AUC severity-only / AUC reference.", "Demo retention": "AUC demo-only / AUC reference.",
        "CDR drop": "AUC reference - AUC minus all CDR.", "Full drop": "AUC reference - AUC demo-only.",
        "FAQ drop": "AUC reference - AUC minus FAQ.",
    }
    write_workbook(
        outdir / "Additional_file_1.xlsx", 1,
        "Matched-feature rerun across ADNI and OASIS3 using only the shared harmonized variables (age, APOE e4 count, "
        "APOE e4 carrier, education, sex; CDR-SB, FAQ). Every cohort x experiment x model family, with five-fold "
        "cross-validated AUC, average precision, balanced accuracy and F1. Balanced logistic regression is the primary "
        "classifier; random forest and extra-trees are robustness checks and appear only in this file. Also includes the "
        "primary-model summary behind Tables 3 and 4 and the bootstrap confidence intervals behind Fig. 2 and the text.",
        "crosscohort_tau_matched_rerun.py (crosscohort_matched_rerun/all_metrics.csv); bootstrap_crosscohort_auc_ci.py "
        "(crosscohort_bootstrap_ci/bootstrap_auc_summary.csv, bootstrap_auc_differences.csv).",
        [("Matched_rerun_metrics", metrics, "All cohorts x experiments x model families.", {}),
         ("Primary_model_summary", summary, "Balanced logistic regression; Table 3 (primary cohorts) and the OASIS3 tau90 row of Table 4.", sum_defs),
         ("Bootstrap_AUC_CI", boot, "Bootstrap 95% CIs for the primary cohorts (balanced logistic regression); Fig. 2.", boot_defs),
         ("Bootstrap_contrasts", contrasts, "Bootstrap 95% CIs for paired AUC contrasts quoted in the text.",
          {**boot_defs, "Contrast": "Paired AUC difference computed within each bootstrap replicate.",
           "AUC difference": "Difference of the full-cohort cross-validated AUCs.",
           "Bootstrap mean difference": "Mean of the paired differences over bootstrap replicates."})],
    )


def af2(outdir):
    audit = pd.read_csv(ROOT / "crosscohort_matched_rerun" / "oasis_feature_audit" / "oasis_feature_audit.csv")
    total = {k: len(pd.read_csv(p, usecols=[0], low_memory=False)) for k, p in OASIS_RAW.items()}
    for k in ("oasis3_tau90", "oasis3_tau180"):
        total[f"{k}_subject"] = len(pd.read_csv(primary.DATASETS[k]["subject_table"], usecols=["y_target"]))
    table_label = {"oasis3_tau90_raw": "OASIS3 clinical master, tau90 window (visit-level)",
                   "oasis3_tau180_raw": "OASIS3 clinical master, tau180 window (visit-level)",
                   "oasis3_tau90_subject": "OASIS3 tau90 subject-level analysis table",
                   "oasis3_tau180_subject": "OASIS3 tau180 subject-level analysis table"}
    note = {("raw", "age"): "Audit matched the visit-offset column by name; age was recovered from 'age at visit' "
                            "(see Recovered_mappings)."}
    audit_out = pd.DataFrame({
        "Table": audit["dataset"].map(table_label),
        "Feature family": audit["feature_family"],
        "Matched column": audit["matched_col"].fillna("none found"),
        "Candidate columns": audit["candidate_columns"].fillna("none").str.replace("|", "; ", regex=False),
        "N rows": audit["dataset"].map(total),
        "N non-missing": audit["nonmissing_n"],
        "N distinct values": audit["unique_nonnull"],
        "% missing": 100 * (1 - audit["nonmissing_n"] / audit["dataset"].map(total)),
        "Comment": [note.get(("raw" if d.endswith("_raw") else "subject", f), "") for d, f in zip(audit["dataset"], audit["feature_family"])],
    })

    maps = []
    for k, p in OASIS_BUILD.items():
        meta = json.loads(p.read_text())["feature_column_map"]["source_columns"]
        subj = pd.read_csv(primary.DATASETS[k]["subject_table"], low_memory=False)
        for harm, src in meta.items():
            nn = int(subj[harm].notna().sum()) if harm in subj else 0
            maps.append({"Cohort": COHORT_LABEL[k], "Harmonized variable": FEATURE_LABEL.get(harm, harm),
                         "OASIS3 source column": src if src else "not available in OASIS3 extract",
                         "N": len(subj), "N non-missing": nn, "% missing": 100 * (1 - nn / len(subj))})
    maps = pd.DataFrame(maps)

    av = pd.read_csv(ROOT / "crosscohort_matched_rerun" / "feature_availability_by_cohort.csv")
    shared = json.loads((ROOT / "crosscohort_matched_rerun" / "shared_feature_manifest.json").read_text())
    shared_set = set(shared["shared_demo_num"] + shared["shared_demo_cat"] + shared["shared_severity_num"])
    n_by = {k: len(pd.read_csv(primary.DATASETS[k]["subject_table"], usecols=["y_target"])) for k in COHORT_ORDER}
    av = order_frame(av.assign(experiment="reference_combined"))
    avail = pd.DataFrame({
        "Cohort": av["cohort_key"].map(COHORT_LABEL), "Harmonized variable": av["feature"].map(FEATURE_LABEL),
        "Block": av["feature_family"].map({"demo_num": "demographic/genetic", "demo_cat": "demographic/genetic",
                                           "severity": "severity"}),
        "Present": av["present"].map({1: "yes", 0: "no"}), "N": av["cohort_key"].map(n_by),
        "N non-missing": av["nonmissing_n"],
        "% missing": 100 * (1 - av["nonmissing_n"] / av["cohort_key"].map(n_by)),
        "In matched feature set": av["feature"].map(lambda f: "yes" if f in shared_set else "no"),
    })
    defs = {
        "Table": "Table audited (visit-level clinical master or subject-level analysis table).",
        "Feature family": "Variable family searched for.",
        "Matched column": "First column whose name matched the family's search list.",
        "Candidate columns": "All columns whose names matched the family's search list.",
        "N rows": "Rows in the audited table (visits for the clinical master, participants for subject-level tables).",
        "N non-missing": "Rows with a non-missing value.",
        "N distinct values": "Number of distinct non-missing values in the matched column.",
        "% missing": "Percentage of rows missing the variable.",
        "Comment": "Interpretation note.",
        "Harmonized variable": "Harmonized analysis variable.",
        "OASIS3 source column": "OASIS3 column the harmonized variable was derived from.",
        "Block": "Feature block the variable belongs to.",
        "Present": "Whether the harmonized variable exists in the cohort's analysis table.",
        "In matched feature set": "Whether the variable is one of the shared variables used in the matched-feature analysis.",
    }
    write_workbook(
        outdir / "Additional_file_2.xlsx", 2,
        "Feature audit of the OASIS3 visit-level and subject-level tables, showing which demographic, genetic and "
        "severity variables were recoverable (age and APOE) and which were absent (MoCA, CDR global); the column "
        "mappings used to build the harmonized OASIS3 variables; and non-missing counts of every harmonized variable "
        "in all four ADNI/OASIS3 analytic cohorts. Variable names, counts and missingness only.",
        "audit_oasis_feature_availability.py (crosscohort_matched_rerun/oasis_feature_audit/); "
        "crosscohort_tau_severity_strip.py (build metadata); crosscohort_tau_matched_rerun.py "
        "(crosscohort_matched_rerun/feature_availability_by_cohort.csv).",
        [("Feature_audit", audit_out, "Search for each variable family in the OASIS3 tables.", defs),
         ("Recovered_mappings", maps, "Harmonized variable to OASIS3 source column, with non-missing counts.", defs),
         ("Feature_availability", avail, "Harmonized variables in each ADNI/OASIS3 analytic cohort.", defs)],
    )


def af3(outdir):
    rows, derived = [], []
    for k, path in ADNI_STRIP.items():
        m = pd.read_csv(path)
        m = m[m["model"] == "logreg_balanced"].copy()          # primary model only, never best-of-family
        y = pd.read_csv(primary.DATASETS[k]["subject_table"], usecols=["y_target"])["y_target"]
        for _, r in order_frame(m.assign(cohort_key=k)).iterrows():
            dropped = []
            if isinstance(r["dropped_features"], str):
                for item in r["dropped_features"].split(";"):
                    col, pct, _ = item.split(":")
                    dropped.append(f"{FEATURE_LABEL.get(col, col)} ({float(pct):.1f}% missing)")
            rows.append({
                "Cohort": COHORT_LABEL[k], "Experiment": EXP_LABEL[r["experiment"]], "Model": MODEL_LABEL["logreg_balanced"],
                "N": int(r["n_subjects"]), "N tau+": int(y.sum()), "N features": int(r["n_features_used"]),
                "Features": features_text(r["numeric_features_used"], r["categorical_features_used"]),
                "Excluded features": "; ".join(dropped) if dropped else "none",
                "AUC": r["roc_auc_mean"], "AUC SD": r["roc_auc_std"], "AP": r["ap_mean"], "AP SD": r["ap_std"],
                "Balanced accuracy": r["bal_acc_mean"], "Balanced accuracy SD": r["bal_acc_std"],
                "F1": r["f1_mean"], "F1 SD": r["f1_std"],
            })
        a = m.set_index("experiment")["roc_auc_mean"]
        derived.append({
            "Cohort": COHORT_LABEL[k], "AUC reference": a["reference_combined"], "AUC severity-only": a["severity_only"],
            "AUC demo-only": a["demo_only"], "AUC minus CDR global": a["minus_cdr_global"],
            "AUC minus CDR-SB": a["minus_cdr_sumboxes"], "AUC minus all CDR": a["minus_all_cdr"],
            "Severity retention": a["severity_only"] / a["reference_combined"],
            "Demo retention": a["demo_only"] / a["reference_combined"],
            "CDR drop": a["reference_combined"] - a["minus_all_cdr"],
            "Full drop": a["reference_combined"] - a["minus_cdr_faq_moca"],
        })
    defs = {
        "Excluded features": "Severity variables excluded before fitting because more than 35% of values were missing.",
        "AUC reference": "AUC of the reference-combined model.", "AUC severity-only": "AUC of the severity-only model.",
        "AUC demo-only": "AUC of the demo-only model.", "AUC minus CDR global": "AUC with CDR global removed.",
        "AUC minus CDR-SB": "AUC with CDR-SB removed.", "AUC minus all CDR": "AUC with CDR global and CDR-SB removed.",
        "Severity retention": "AUC severity-only / AUC reference.", "Demo retention": "AUC demo-only / AUC reference.",
        "CDR drop": "AUC reference - AUC minus all CDR.",
        "Full drop": "AUC reference - AUC with every severity variable removed (equal to demo-only here).",
    }
    write_workbook(
        outdir / "Additional_file_3.xlsx", 3,
        "Within-cohort feature-block ablation in ADNI using all available harmonized variables (before feature "
        "matching): age, education, APOE e4 count and carrier status, sex, and the severity variables CDR global, "
        "CDR-SB, FAQ and MoCA. Variables with more than 35% missing values were excluded before fitting, which removed "
        "FAQ and MoCA in both ADNI cohorts, so experiments that differ only in FAQ or MoCA give identical AUCs. "
        "Balanced logistic regression only.",
        "crosscohort_tau_severity_strip.py (all_severity_strip_metrics.csv for ADNI tau90 and tau180), "
        "rows for balanced logistic regression.",
        [("ADNI_ablation_metrics", pd.DataFrame(rows), "Every feature-block experiment, both ADNI cohorts.", defs),
         ("ADNI_ablation_summary", pd.DataFrame(derived), "Retention fractions and drops derived from the AUCs.", defs)],
        notes=["Earlier tables selected the best of three classifier families per experiment; this file reports balanced "
               "logistic regression throughout (for example, ADNI tau180 minus CDR global AUC 0.858)."],
    )


def af4(outdir):
    nacc = pd.read_csv(ROOT / "paper_tables_and_figures" / "additional_file4_nacc_sensitivity.csv")
    exp_label = {"reference": "Reference-combined", "severity_only": "Severity-only", "demo_only": "Demo-only",
                 "demo_only_no_tracer": "Demo-only without tau tracer",
                 "demo_only_harmonized": "Demo-only, five harmonized ADNI/OASIS3 variables",
                 "minus_all_cdr": "Minus all CDR", "minus_faq": "Minus FAQ", "minus_moca": "Minus MoCA",
                 "minus_cdr_faq_moca": "Minus CDR, FAQ and MoCA (category fluency kept)"}
    blocks = pd.DataFrame({
        "Cohort": nacc["cohort"], "N": nacc["n"], "N A+/T+": nacc["n_pos"],
        "Experiment": nacc["experiment"].map(exp_label), "Model": MODEL_LABEL["logreg_balanced"],
        "Features": nacc["features"].map(features_text), "AUC": nacc["auc"],
    })
    summ = []
    for cohort, g in nacc.groupby("cohort", sort=False):
        a = g.set_index("experiment")["auc"]
        summ.append({"Cohort": cohort, "N": int(g["n"].iloc[0]), "N A+/T+": int(g["n_pos"].iloc[0]),
                     "AUC reference": a["reference"], "AUC severity-only": a["severity_only"], "AUC demo-only": a["demo_only"],
                     "Severity retention": a["severity_only"] / a["reference"], "Demo retention": a["demo_only"] / a["reference"],
                     "CDR drop": a["reference"] - a["minus_all_cdr"], "Full drop": a["reference"] - a["demo_only"],
                     "Drop minus CDR, FAQ and MoCA": a["reference"] - a["minus_cdr_faq_moca"]})
    d = pd.read_csv(NACC_MAIN, usecols=["tau_tracer", "y_AT_strict"])
    tracer = (d.groupby("tau_tracer")["y_AT_strict"].agg(["size", "sum"]).reset_index())
    tracer = pd.DataFrame({
        "Tau tracer": tracer["tau_tracer"].map({6: "Flortaucipir (NACC SCAN TRACER code 6)", 7: "MK-6240 (NACC SCAN TRACER code 7)"}),
        "N": tracer["size"], "N A+/T+": tracer["sum"], "% A+/T+": 100 * tracer["sum"] / tracer["size"],
    })
    defs = {
        "N A+/T+": "Number of participants in the strict A+/T+ group (the remainder are A-/T-).",
        "% A+/T+": "Percentage of participants in the strict A+/T+ group.",
        "Tau tracer": "Tau PET tracer (NACC SCAN & CLARiTI PET Researcher's Data Dictionary codes).",
        "Features": "Predictors used. The demographic block adds race, Hispanic ethnicity and handedness (and the tau "
                    "tracer where listed); the severity block adds animal and vegetable category fluency.",
        "AUC reference": "AUC of the reference-combined model.", "AUC severity-only": "AUC of the severity-only model.",
        "AUC demo-only": "AUC of the demo-only model (including tau tracer in the main cohort).",
        "Severity retention": "AUC severity-only / AUC reference.", "Demo retention": "AUC demo-only / AUC reference.",
        "CDR drop": "AUC reference - AUC minus all CDR.", "Full drop": "AUC reference - AUC demo-only (Table 4 definition).",
        "Drop minus CDR, FAQ and MoCA": "AUC reference - AUC with CDR, FAQ and MoCA removed (category fluency kept).",
    }
    write_workbook(
        outdir / "Additional_file_4.xlsx", 4,
        "NACC strict A+/T+ versus A-/T- endpoint, balanced logistic regression with five-fold cross-validation. Main "
        "cohort (flortaucipir and MK-6240 tau PET, tracer included as a feature) and the flortaucipir-only subset (no "
        "tracer feature), with demographic-block variants without the tracer term and restricted to the five harmonized "
        "ADNI/OASIS3 variables, plus A+/T+ prevalence by tracer.",
        "nacc_at_strict_ablation_v1.py and nacc_additional_file4.py "
        "(paper_tables_and_figures/additional_file4_nacc_sensitivity.csv).",
        [("NACC_feature_blocks", blocks, "Feature-block AUCs for both NACC cohorts.", defs),
         ("NACC_summary", pd.DataFrame(summ), "Retention fractions and drops (Table 4 NACC row for the main cohort).", defs),
         ("Tau_tracer_prevalence", tracer, "Strict A+/T+ prevalence by tau tracer in the main cohort.", defs)],
    )


def af5(outdir):
    sw = pd.read_csv(ROOT / "oasis3_threshold_sensitivity" / "threshold_sweep_results.csv")
    sw = sw.assign(_c=sw["cohort_key"].map({"oasis3_tau180": 0, "oasis3_tau90": 1})).sort_values(["_c", "threshold"])
    kind = np.where(sw["is_published_cutoff"], "published OASIS-3 cutoff",
                    np.where(sw["is_gmm_threshold"], "GMM midpoint", "sweep"))
    sweep = pd.DataFrame({
        "Cohort": sw["cohort_key"].map(COHORT_LABEL), "Threshold type": kind, "Threshold (SUVR)": sw["threshold"],
        "GMM midpoint (SUVR)": sw["gmm_midpoint"], "% from GMM midpoint": sw["pct_from_gmm"],
        "N": sw["n_total"], "N tau+": sw["n_pos"], "N tau-": sw["n_neg"],
        "AUC demo-only": sw["auc_full"], "Reliable": sw["reliable"].map({True: "yes", False: "no"}),
    })
    sel = sw[(sw["is_published_cutoff"]) |
             ((sw["cohort_key"] == "oasis3_tau180") & sw["pct_from_gmm"].between(-9, 6)) |
             ((sw["cohort_key"] == "oasis3_tau90") & sw["pct_from_gmm"].between(-9, 9))]
    s5 = pd.DataFrame({
        "Cohort": sel["cohort_key"].map(COHORT_LABEL),
        "% from GMM midpoint": sel["pct_from_gmm"],
        "Threshold type": np.where(sel["is_published_cutoff"], "published", np.where(sel["is_gmm_threshold"], "GMM midpoint", "sweep")),
        "Threshold (SUVR)": sel["threshold"], "N": sel["n_total"], "N tau+": sel["n_pos"],
        "AUC demo-only": sel["auc_full"].where(sel["reliable"]), "Reliable": sel["reliable"].map({True: "yes", False: "no"}),
    })
    pc = pd.read_csv(ROOT / "oasis3_threshold_sensitivity" / "published_cutoff_comparison.csv")
    comp = pd.DataFrame({
        "Cohort": pc["cohort_key"].map(COHORT_LABEL),
        "Definition": pc["definition"].map({"gmm_midpoint": "GMM midpoint", "published_cutoff": "Published OASIS-3 cutoff"}),
        "Threshold (SUVR)": pc["threshold"], "N": pc["n_total"], "N tau+": pc["n_pos"], "% tau+": pc["pct_pos"],
        "N with amyloid PET": pc["n_with_amyloid"], "% amyloid+ (all with amyloid PET)": pc["pct_amyloid_pos"],
        "N tau+ with amyloid PET": pc["n_pos_with_amyloid"], "N tau+ amyloid-": pc["n_pos_amyloid_neg"],
        "% amyloid- among tau+": pc["pct_pos_amyloid_neg"], "N tau+ with CDR-SB = 0": pc["n_pos_cdrsb0"],
        "% CDR-SB = 0 among tau+": pc["pct_pos_cdrsb0"],
        "AUC reference": pc["auc_reference"], "AUC severity-only": pc["auc_severity"], "AUC demo-only": pc["auc_demo"],
    })
    defs = {
        "Threshold type": "Sweep point, the GMM midpoint used for the primary OASIS3 definition, or the published cutoff.",
        "Threshold (SUVR)": "Tau-positivity threshold on the OASIS-3 tauopathy summary (partial-volume-corrected SUVR).",
        "GMM midpoint (SUVR)": "Midpoint of the two Gaussian-mixture component means for the cohort.",
        "% from GMM midpoint": "Threshold relative to the GMM midpoint, in percent.",
        "AUC demo-only": "Five-fold cross-validated AUC of the demographic/genetic-only model (age, APOE e4 count and "
                         "carrier status, education, sex), balanced logistic regression; blank when not reliable.",
        "Reliable": "yes when at least 10 tau-positive and 10 tau-negative participants.",
        "Definition": "Tau-positivity definition compared.",
        "% tau+": "Percentage of the cohort classified tau-positive.",
        "N with amyloid PET": "Participants with a matched amyloid PET scan.",
        "% amyloid+ (all with amyloid PET)": "Amyloid-positive percentage among all participants with amyloid PET "
                                             "(OASIS-3 cutoffs: PiB 1.42, florbetapir 1.19, partial-volume-corrected).",
        "N tau+ with amyloid PET": "Tau-positive participants with a matched amyloid PET scan.",
        "N tau+ amyloid-": "Tau-positive participants who were amyloid-negative.",
        "% amyloid- among tau+": "Amyloid-negative percentage among tau-positive participants with amyloid PET.",
        "N tau+ with CDR-SB = 0": "Tau-positive participants with CDR-SB = 0.",
        "% CDR-SB = 0 among tau+": "Percentage of tau-positive participants with CDR-SB = 0.",
        "AUC reference": "AUC of the reference-combined model at this definition.",
        "AUC severity-only": "AUC of the severity-only model at this definition.",
    }
    write_workbook(
        outdir / "Additional_file_5.xlsx", 5,
        "OASIS3 tau-threshold sensitivity: demographic/genetic-only balanced logistic regression across a +/-30% sweep "
        "around each cohort's GMM midpoint and at the published OASIS-3 tauopathy cutoff (1.22), the selected rows shown "
        "as Supplementary Table S5, and a comparison of the GMM midpoint with the published cutoff.",
        "oasis3_threshold_sensitivity.py (oasis3_threshold_sensitivity/threshold_sweep_results.csv, "
        "published_cutoff_comparison.csv, threshold_sweep_plot.png).",
        [("Threshold_sweep", sweep, "Every threshold evaluated, both OASIS3 cohorts.", defs),
         ("Table_S5", s5, "Rows shown in Supplementary Table S5.", defs),
         ("Published_cutoff_comparison", comp, "GMM midpoint versus the published cutoff.", defs)],
        figures=[("Fig_S1", ("Supplementary Fig. S1. OASIS3 tau-threshold sensitivity analysis.",
                             ROOT / "oasis3_threshold_sensitivity" / "threshold_sweep_plot.png"))],
    )


def af6(outdir):
    co = pd.read_csv(ROOT / "logreg_feature_importance" / "feature_coefficients.csv")
    co = co[co["cohort_key"].isin(PRIMARY_COHORTS)]
    term = {"cdr_sumboxes_h": "CDR-SB", "age_h": "Age", "apoe_e4_carrier_h": "APOE e4 carrier",
            "apoe_e4_count_h": "APOE e4 count", "education_years_h": "Education", "faq_total_h": "FAQ",
            "sex_h_F": "Sex: female (one-hot term)", "sex_h_M": "Sex: male (one-hot term)",
            "sex_h_F_minus_M": "Sex: female vs. male (difference of one-hot terms)"}
    order = list(term)
    ref = co[co["experiment"] == "reference_combined"].pivot(index="feature", columns="cohort_key", values="coefficient")
    s6 = pd.DataFrame({"Term": [term[f] for f in order],
                       **{COHORT_LABEL[k]: [ref.loc[f, k] for f in order] for k in PRIMARY_COHORTS}})
    other = co[co["experiment"] != "reference_combined"].copy()
    other["_e"] = other["experiment"].map({k: i for i, k in enumerate(EXP_ORDER)})
    other["_c"] = other["cohort_key"].map({k: i for i, k in enumerate(COHORT_ORDER)})
    other["_t"] = other["feature"].map({f: i for i, f in enumerate(order)})
    other = other.sort_values(["_e", "_c", "_t"])
    others = pd.DataFrame({"Cohort": other["cohort_key"].map(COHORT_LABEL), "Experiment": other["experiment"].map(EXP_LABEL),
                           "Term": other["feature"].map(term), "Coefficient": other["coefficient"]})
    defs = {
        "Term": "Model term. Numeric features (including the binary APOE e4 carrier indicator) were median-imputed and "
                "z-scored, so their coefficients are log-odds per one-SD increase. Sex enters as two one-hot terms; "
                "their difference (female minus male) is the interpretable sex effect, in log-odds (not per SD).",
        "ADNI tau90": "Coefficient in the reference-combined model, ADNI tau90.",
        "ADNI tau180": "Coefficient in the reference-combined model, ADNI tau180.",
        "OASIS3 tau180": "Coefficient in the reference-combined model, OASIS3 tau180.",
        "Coefficient": "Coefficient of the term in the listed experiment's model.",
    }
    write_workbook(
        outdir / "Additional_file_6.xlsx", 6,
        "Coefficients of the balanced logistic regression fitted on the full cohort for the three primary cohorts. "
        "Table_S6 gives the reference-combined model (Supplementary Table S6), including both sex one-hot terms and the "
        "female-vs-male contrast; Other_experiments gives the demo-only and severity-only models.",
        "logreg_feature_importance.py (logreg_feature_importance/feature_coefficients.csv).",
        [("Table_S6", s6, "Reference-combined model, primary cohorts.", defs),
         ("Other_experiments", others, "Demo-only and severity-only models, primary cohorts.", defs)],
        notes=["APOE e4 carrier status and allele count are highly collinear (r = 0.92-0.93), so their individual "
               "coefficients should not be interpreted separately."],
    )


def af7(outdir):
    dx = pd.read_csv(ROOT / "dx_stratified_auc" / "dx_stratified_results.csv")
    dx = dx[dx["cohort_key"].isin(PRIMARY_COHORTS)]
    wide = dx.pivot_table(index=["cohort_key", "dx_class"], columns="experiment", values="auc", aggfunc="first")
    base = dx.drop_duplicates(["cohort_key", "dx_class"]).set_index(["cohort_key", "dx_class"])
    rows = []
    for k in PRIMARY_COHORTS:
        for s in ["CN", "MCI", "Dementia"]:
            b = base.loc[(k, s)]
            est = bool(b["reliable"])
            rows.append({"Cohort": COHORT_LABEL[k], "Diagnosis stratum": s, "N": int(b["n_total"]),
                         "N tau+": int(b["n_pos"]), "N tau-": int(b["n_neg"]), "% tau+": float(b["tau_pos_pct"]),
                         "AUC reference": wide.loc[(k, s), "reference_combined"] if est else np.nan,
                         "AUC demo-only": wide.loc[(k, s), "demo_only"] if est else np.nan,
                         "Not estimated": "yes" if not est else "no"})
    defs = {
        "Diagnosis stratum": "Baseline clinical diagnosis (ADNI DX_DIAGNOSIS; OASIS3 harmonized diagnosis).",
        "% tau+": "Percentage tau-positive in the stratum.",
        "AUC reference": "Five-fold cross-validated AUC of the reference-combined model within the stratum "
                         "(balanced logistic regression); blank when not estimated.",
        "AUC demo-only": "Five-fold cross-validated AUC of the demo-only model within the stratum; blank when not estimated.",
        "Not estimated": "yes when the stratum had fewer than 10 tau-positive or fewer than 10 tau-negative participants.",
    }
    write_workbook(
        outdir / "Additional_file_7.xlsx", 7,
        "Reference-combined and demo-only AUCs within baseline diagnosis strata (CN, MCI, Dementia) for the three "
        "primary cohorts, with stratum sizes. Strata with fewer than 10 participants in either class were not estimated.",
        "dx_stratified_auc.py (dx_stratified_auc/dx_stratified_results.csv, dx_stratified_plot.png).",
        [("Stratified_AUC", pd.DataFrame(rows), "One row per cohort and diagnosis stratum.", defs)],
        notes=["OASIS3 tau90 is not shown: none of its strata met the estimation rule, as in Supplementary Fig. S2. "
               "OASIS3 tau180 has four participants with another or missing diagnosis, excluded from stratification."],
        figures=[("Fig_S2", ("Supplementary Fig. S2. Diagnosis-stratified feature-block AUC analysis.",
                             ROOT / "dx_stratified_auc" / "dx_stratified_plot.png"))],
    )


def af8(outdir):
    cm = pd.read_csv(ROOT / "oasis3_continuous_suvr_sensitivity" / "continuous_suvr_metrics.csv")
    cm = cm[cm["model"] == "ridge"].copy()
    cm["cohort_key"] = pd.Categorical(cm["cohort_key"], ["oasis3_tau180", "oasis3_tau90"], ordered=True)
    cm["_e"] = cm["experiment"].map({k: i for i, k in enumerate(["reference_combined", "demo_only", "severity_only", "minus_all_cdr", "minus_faq"])})
    cm = cm.sort_values(["cohort_key", "_e"])
    r2 = cm.pivot(index="cohort_key", columns="experiment", values="oof_r2")
    s7 = pd.DataFrame({"Cohort": [COHORT_LABEL[k] for k in r2.index],
                       "R2 reference": r2["reference_combined"].values, "R2 demo-only": r2["demo_only"].values,
                       "R2 severity-only": r2["severity_only"].values, "R2 minus all CDR": r2["minus_all_cdr"].values,
                       "R2 minus FAQ": r2["minus_faq"].values})
    metrics = pd.DataFrame({
        "Cohort": cm["cohort_key"].astype(str).map(COHORT_LABEL), "Experiment": cm["experiment"].map(EXP_LABEL),
        "Model": MODEL_LABEL["ridge"], "N": cm["n_subjects"], "N features": cm["n_features_used"],
        "Features": [features_text(a, b) for a, b in zip(cm["numeric_features_used"], cm["categorical_features_used"])],
        "R2 (out-of-fold)": cm["oof_r2"], "Pearson r": cm["oof_pearson_r"], "Spearman rho": cm["oof_spearman_r"],
        "MAE": cm["oof_mae"], "RMSE": cm["oof_rmse"], "R2 fold mean": cm["fold_r2_mean"], "R2 fold SD": cm["fold_r2_std"],
    })
    target = cm.drop_duplicates("cohort_key")
    target = pd.DataFrame({"Cohort": target["cohort_key"].astype(str).map(COHORT_LABEL), "N": target["n_subjects"],
                           "Mean tauopathy SUVR": target["target_mean"], "SD tauopathy SUVR": target["target_sd"]})
    defs = {
        "R2 reference": "Out-of-fold R2, reference-combined model.", "R2 demo-only": "Out-of-fold R2, demo-only model.",
        "R2 severity-only": "Out-of-fold R2, severity-only model.",
        "R2 minus all CDR": "Out-of-fold R2 with CDR-SB removed (the only CDR variable in the matched OASIS3 set).",
        "R2 minus FAQ": "Out-of-fold R2 with FAQ removed.",
        "R2 (out-of-fold)": "R2 of out-of-fold predictions pooled over five target-stratified folds (seed 42).",
        "Pearson r": "Pearson correlation of out-of-fold predictions with observed SUVR.",
        "Spearman rho": "Spearman rank correlation of out-of-fold predictions with observed SUVR.",
        "MAE": "Mean absolute error of out-of-fold predictions (SUVR units).",
        "RMSE": "Root-mean-square error of out-of-fold predictions (SUVR units).",
        "R2 fold mean": "Mean of per-fold R2.", "R2 fold SD": "Standard deviation of per-fold R2.",
        "Mean tauopathy SUVR": "Cohort mean of the continuous outcome.",
        "SD tauopathy SUVR": "Cohort standard deviation of the continuous outcome.",
    }
    write_workbook(
        outdir / "Additional_file_8.xlsx", 8,
        "Ridge regression (alpha = 1, exact solver) predicting the continuous OASIS-3 tauopathy SUVR from the matched "
        "feature blocks within OASIS3 tau180 and tau90, with five-fold target-stratified cross-validation. This analysis "
        "tests whether the OASIS3 findings depend on the binary GMM threshold; it is not a cross-cohort comparison.",
        "oasis3_continuous_suvr_sensitivity.py (oasis3_continuous_suvr_sensitivity/continuous_suvr_metrics.csv, "
        "continuous_suvr_r2_by_feature_block.png).",
        [("Table_S7", s7, "Out-of-fold R2 by feature block (Supplementary Table S7).", defs),
         ("Ridge_metrics", metrics, "All metrics for each cohort and feature block.", defs),
         ("Outcome_summary", target, "Mean and SD of the outcome in each cohort.", defs)],
        figures=[("Fig_S3", ("Supplementary Fig. S3. Within-OASIS3 continuous tauopathy SUVR sensitivity analysis.",
                             ROOT / "oasis3_continuous_suvr_sensitivity" / "continuous_suvr_r2_by_feature_block.png"))],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for build in (af1, af2, af3, af4, af5, af6, af7, af8):
        build(outdir)
        print(f"built {build.__name__}")


if __name__ == "__main__":
    main()
