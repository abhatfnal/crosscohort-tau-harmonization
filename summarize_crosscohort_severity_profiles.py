from pathlib import Path
import numpy as np
import pandas as pd

OUTDIR = Path("/project/aereditato/abhat/adni-mri-classification/crosscohort_severity_summary")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

STANDARD_COHORTS = [
    {
        "cohort_key": "adni_tau90",
        "cohort_label": "ADNI tau90",
        "cohort": "ADNI",
        "window": "90d",
        "endpoint_family": "tau_binary",
        "endpoint_label": "tau positivity",
        "best_csv": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau90/best_model_per_experiment.csv",
        "subject_csv": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau90/subject_level_input_table.csv",
        "subject_col": "subject_id",
        "target_col": "y_target",
        "demo_features": ["age_h", "education_years_h", "apoe_e4_count_h", "apoe_e4_carrier_h", "sex_h"],
        "severity_features": ["cdr_global_h", "cdr_sumboxes_h", "faq_total_h", "moca_h"],
        "note": "",
        "supplement_only": False,
    },
    {
        "cohort_key": "adni_tau180",
        "cohort_label": "ADNI tau180",
        "cohort": "ADNI",
        "window": "180d",
        "endpoint_family": "tau_binary",
        "endpoint_label": "tau positivity",
        "best_csv": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau180/best_model_per_experiment.csv",
        "subject_csv": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau180/subject_level_input_table.csv",
        "subject_col": "subject_id",
        "target_col": "y_target",
        "demo_features": ["age_h", "education_years_h", "apoe_e4_count_h", "apoe_e4_carrier_h", "sex_h"],
        "severity_features": ["cdr_global_h", "cdr_sumboxes_h", "faq_total_h", "moca_h"],
        "note": "",
        "supplement_only": False,
    },
    {
        "cohort_key": "oasis3_tau90",
        "cohort_label": "OASIS3 tau90",
        "cohort": "OASIS3",
        "window": "90d",
        "endpoint_family": "tau_binary",
        "endpoint_label": "tau positivity (derived)",
        "best_csv": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau90/best_model_per_experiment.csv",
        "subject_csv": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau90/subject_level_input_table.csv",
        "subject_col": "subject_id",
        "target_col": "y_target",
        "demo_features": ["age_h", "education_years_h", "apoe_e4_count_h", "apoe_e4_carrier_h", "sex_h"],
        "severity_features": ["cdr_global_h", "cdr_sumboxes_h", "faq_total_h", "moca_h"],
        "note": "",
        "supplement_only": False,
    },
    {
        "cohort_key": "oasis3_tau180",
        "cohort_label": "OASIS3 tau180",
        "cohort": "OASIS3",
        "window": "180d",
        "endpoint_family": "tau_binary",
        "endpoint_label": "tau positivity (derived)",
        "best_csv": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau180/best_model_per_experiment.csv",
        "subject_csv": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau180/subject_level_input_table.csv",
        "subject_col": "subject_id",
        "target_col": "y_target",
        "demo_features": ["age_h", "education_years_h", "apoe_e4_count_h", "apoe_e4_carrier_h", "sex_h"],
        "severity_features": ["cdr_global_h", "cdr_sumboxes_h", "faq_total_h", "moca_h"],
        "note": "",
        "supplement_only": False,
    },
]

NACC_ABLATION_BEST = "/project/aereditato/abhat/NACC_17259/phase2_tau_amyloid_90d/model_ready_v3_from_master/ablation_v1/best_model_per_experiment.csv"
NACC_STRIP_BEST = "/project/aereditato/abhat/NACC_17259/phase2_tau_amyloid_90d/model_ready_v3_from_master/ablation_severity_strip/best_model_per_experiment.csv"

NACC_MAIN_MODEL = "/project/aereditato/abhat/NACC_17259/phase2_tau_amyloid_90d/model_ready_v3_from_master/nacc_AT_strict_model_table_nodx_withtracer_v3.csv"
NACC_TR6_MODEL = "/project/aereditato/abhat/NACC_17259/phase2_tau_amyloid_90d/model_ready_v3_from_master/nacc_AT_strict_tracer6_model_table_nodx_v3.csv"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

METRIC_MAP = {
    "auc": "roc_auc_mean",
    "ap": "ap_mean",
    "bal_acc": "bal_acc_mean",
    "f1": "f1_mean",
}

EXPERIMENT_ALIAS_MAP = {
    "ref": "reference_combined",
    "severity": "severity_only",
    "demo": "demo_only",
    "minus_all_cdr": "minus_all_cdr",
    "minus_cdr_faq": "minus_cdr_faq",
    "minus_cdr_moca": "minus_cdr_moca",
    "minus_faq_moca": "minus_faq_moca",
    "minus_moca": "minus_moca",
    "minus_faq": "minus_faq",
    "minus_cdr_global": "minus_cdr_global",
    "minus_cdr_sumboxes": "minus_cdr_sumboxes",
    "minus_fullstrip": "minus_cdr_faq_moca",
}


def read_csv_any(path):
    return pd.read_csv(path, low_memory=False, compression="infer")


def safe_get(df, experiment, metric_col):
    x = df[df["experiment"] == experiment]
    if x.empty or metric_col not in x.columns:
        return np.nan
    return float(x.iloc[0][metric_col])


def safe_div(a, b):
    if pd.notna(a) and pd.notna(b) and b != 0:
        return a / b
    return np.nan


def load_target_info(subject_csv, subject_col, target_col):
    df = read_csv_any(subject_csv).copy()
    if subject_col in df.columns:
        df = df.drop_duplicates(subset=[subject_col], keep="first").copy()
        n_subjects = int(df[subject_col].nunique())
    else:
        n_subjects = int(len(df))

    y = pd.to_numeric(df[target_col], errors="coerce")
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_missing = int(y.isna().sum())
    pos_rate = float(n_pos / (n_pos + n_neg)) if (n_pos + n_neg) > 0 else np.nan
    return df, n_subjects, n_pos, n_neg, n_missing, pos_rate


def feature_availability(df, demo_features, severity_features):
    demo_avail = [c for c in demo_features if c in df.columns and df[c].notna().sum() > 0]
    sev_avail = [c for c in severity_features if c in df.columns and df[c].notna().sum() > 0]
    return demo_avail, sev_avail


def add_metrics_from_best(row, best_df, alias_to_exp):
    for alias, exp in alias_to_exp.items():
        for short_name, metric_col in METRIC_MAP.items():
            row[f"{short_name}_{alias}"] = safe_get(best_df, exp, metric_col)
    return row


def add_derived_fields(row):
    auc_ref = row.get("auc_ref", np.nan)
    auc_sev = row.get("auc_severity", np.nan)
    auc_demo = row.get("auc_demo", np.nan)
    auc_all_cdr = row.get("auc_minus_all_cdr", np.nan)
    auc_fullstrip = row.get("auc_minus_fullstrip", np.nan)
    auc_moca = row.get("auc_minus_moca", np.nan)
    auc_faq = row.get("auc_minus_faq", np.nan)
    auc_cdrg = row.get("auc_minus_cdr_global", np.nan)
    auc_cdrsb = row.get("auc_minus_cdr_sumboxes", np.nan)

    row["severity_retention_auc_frac"] = safe_div(auc_sev, auc_ref)
    row["demo_retention_auc_frac"] = safe_div(auc_demo, auc_ref)
    row["full_strip_retention_auc_frac"] = safe_div(auc_fullstrip, auc_ref)

    row["severity_minus_demo_auc"] = (auc_sev - auc_demo) if pd.notna(auc_sev) and pd.notna(auc_demo) else np.nan

    row["cdr_block_drop_auc"] = (auc_ref - auc_all_cdr) if pd.notna(auc_ref) and pd.notna(auc_all_cdr) else np.nan
    row["full_strip_drop_auc"] = (auc_ref - auc_fullstrip) if pd.notna(auc_ref) and pd.notna(auc_fullstrip) else np.nan
    row["moca_unique_drop_auc"] = (auc_ref - auc_moca) if pd.notna(auc_ref) and pd.notna(auc_moca) else np.nan
    row["faq_unique_drop_auc"] = (auc_ref - auc_faq) if pd.notna(auc_ref) and pd.notna(auc_faq) else np.nan
    row["cdr_global_unique_drop_auc"] = (auc_ref - auc_cdrg) if pd.notna(auc_ref) and pd.notna(auc_cdrg) else np.nan
    row["cdr_sumboxes_unique_drop_auc"] = (auc_ref - auc_cdrsb) if pd.notna(auc_ref) and pd.notna(auc_cdrsb) else np.nan

    return row


def build_standard_row(cfg):
    best = read_csv_any(cfg["best_csv"])
    subj_df, n_subjects, n_pos, n_neg, n_missing, pos_rate = load_target_info(
        cfg["subject_csv"], cfg["subject_col"], cfg["target_col"]
    )
    demo_avail, sev_avail = feature_availability(subj_df, cfg["demo_features"], cfg["severity_features"])

    row = {
        "cohort_key": cfg["cohort_key"],
        "cohort_label": cfg["cohort_label"],
        "cohort": cfg["cohort"],
        "window": cfg["window"],
        "endpoint_family": cfg["endpoint_family"],
        "endpoint_label": cfg["endpoint_label"],
        "supplement_only": cfg["supplement_only"],
        "n_subjects": n_subjects,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_missing_target": n_missing,
        "pos_rate": pos_rate,
        "available_demo_features": "|".join(demo_avail),
        "available_severity_features": "|".join(sev_avail),
        "n_demo_features_available": len(demo_avail),
        "n_severity_features_available": len(sev_avail),
        "note": cfg["note"],
    }

    row = add_metrics_from_best(row, best, EXPERIMENT_ALIAS_MAP)
    row = add_derived_fields(row)
    return row, best


def build_nacc_main_row():
    ab1 = read_csv_any(NACC_ABLATION_BEST)
    strip_best = read_csv_any(NACC_STRIP_BEST)
    subj_df, n_subjects, n_pos, n_neg, n_missing, pos_rate = load_target_info(
        NACC_MAIN_MODEL, "NACCID", "y_AT_strict"
    )

    demo_features = ["age", "education_years", "apoe_e4_count", "apoe_e4_carrier", "sex"]
    severity_features = ["cdr_global", "cdr_sumboxes", "faq_total", "moca"]
    demo_avail, sev_avail = feature_availability(subj_df, demo_features, severity_features)

    row = {
        "cohort_key": "nacc_strict_at_main",
        "cohort_label": "NACC strict A/T main",
        "cohort": "NACC",
        "window": "90d",
        "endpoint_family": "strict_AT_binary",
        "endpoint_label": "strict A+/T+ vs A-/T-",
        "supplement_only": False,
        "n_subjects": n_subjects,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_missing_target": n_missing,
        "pos_rate": pos_rate,
        "available_demo_features": "|".join(demo_avail),
        "available_severity_features": "|".join(sev_avail),
        "n_demo_features_available": len(demo_avail),
        "n_severity_features_available": len(sev_avail),
        "note": "Different endpoint family from ADNI/OASIS. Reference and demo rows include tau_tracer in current NACC main outputs.",
    }

    # reference / demo / severity from ablation_v1
    main_ab1 = {
        "ref": "main_combined_withtracer",
        "demo": "main_demo_apoe_withtracer",
        "severity": "main_severity_only",
    }
    for alias, exp in main_ab1.items():
        for short_name, metric_col in METRIC_MAP.items():
            row[f"{short_name}_{alias}"] = safe_get(ab1, exp, metric_col)

    # strip experiments from severity-strip best file, dataset-filtered
    sb = strip_best[strip_best["dataset"] == "strict_main_withtracer"].copy()
    strip_aliases = {
        "minus_all_cdr": "minus_all_cdr",
        "minus_cdr_faq": "minus_cdr_faq",
        "minus_cdr_moca": "minus_cdr_moca",
        "minus_faq_moca": "minus_faq_moca",
        "minus_moca": "minus_moca",
        "minus_faq": "minus_faq",
        "minus_cdr_global": "minus_cdr_global",
        "minus_cdr_sumboxes": "minus_cdr_sumboxes",
        "minus_fullstrip": "minus_cdr_faq_moca",
    }
    row = add_metrics_from_best(row, sb, strip_aliases)
    row = add_derived_fields(row)
    return row, ab1, sb


def build_nacc_tracer6_row():
    ab1 = read_csv_any(NACC_ABLATION_BEST)
    strip_best = read_csv_any(NACC_STRIP_BEST)
    subj_df, n_subjects, n_pos, n_neg, n_missing, pos_rate = load_target_info(
        NACC_TR6_MODEL, "NACCID", "y_AT_strict"
    )

    demo_features = ["age", "education_years", "apoe_e4_count", "apoe_e4_carrier", "sex"]
    severity_features = ["cdr_global", "cdr_sumboxes", "faq_total", "moca"]
    demo_avail, sev_avail = feature_availability(subj_df, demo_features, severity_features)

    row = {
        "cohort_key": "nacc_strict_at_tracer6",
        "cohort_label": "NACC strict A/T tracer-6",
        "cohort": "NACC",
        "window": "90d",
        "endpoint_family": "strict_AT_binary",
        "endpoint_label": "strict A+/T+ vs A-/T- (tracer-6 subset)",
        "supplement_only": True,
        "n_subjects": n_subjects,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_missing_target": n_missing,
        "pos_rate": pos_rate,
        "available_demo_features": "|".join(demo_avail),
        "available_severity_features": "|".join(sev_avail),
        "n_demo_features_available": len(demo_avail),
        "n_severity_features_available": len(sev_avail),
        "note": "Sensitivity row. Current tracer-6 outputs do not include separate demo-only or severity-only experiments.",
    }

    # reference from ablation_v1
    for short_name, metric_col in METRIC_MAP.items():
        row[f"{short_name}_ref"] = safe_get(ab1, "tracer6_combined_notracer", metric_col)

    # unavailable in current outputs
    for short_name in METRIC_MAP.keys():
        row[f"{short_name}_demo"] = np.nan
        row[f"{short_name}_severity"] = np.nan

    # strip experiments from severity-strip best file, dataset-filtered
    sb = strip_best[strip_best["dataset"] == "strict_tracer6_notracer"].copy()
    strip_aliases = {
        "minus_all_cdr": "minus_all_cdr",
        "minus_cdr_faq": "minus_cdr_faq",
        "minus_cdr_moca": "minus_cdr_moca",
        "minus_faq_moca": "minus_faq_moca",
        "minus_moca": "minus_moca",
        "minus_faq": "minus_faq",
        "minus_cdr_global": "minus_cdr_global",
        "minus_cdr_sumboxes": "minus_cdr_sumboxes",
        "minus_fullstrip": "minus_cdr_faq_moca",
    }
    row = add_metrics_from_best(row, sb, strip_aliases)
    row = add_derived_fields(row)
    return row, ab1, sb


def build_long_table(wide_df):
    exp_cols = {
        "ref": "reference_combined",
        "severity": "severity_only",
        "demo": "demo_only",
        "minus_all_cdr": "minus_all_cdr",
        "minus_cdr_faq": "minus_cdr_faq",
        "minus_cdr_moca": "minus_cdr_moca",
        "minus_faq_moca": "minus_faq_moca",
        "minus_moca": "minus_moca",
        "minus_faq": "minus_faq",
        "minus_cdr_global": "minus_cdr_global",
        "minus_cdr_sumboxes": "minus_cdr_sumboxes",
        "minus_fullstrip": "minus_cdr_faq_moca",
    }

    rows = []
    for _, r in wide_df.iterrows():
        for alias, exp_name in exp_cols.items():
            rows.append(
                {
                    "cohort_key": r["cohort_key"],
                    "cohort_label": r["cohort_label"],
                    "endpoint_family": r["endpoint_family"],
                    "endpoint_label": r["endpoint_label"],
                    "supplement_only": r["supplement_only"],
                    "experiment_alias": alias,
                    "experiment_name": exp_name,
                    "auc": r.get(f"auc_{alias}", np.nan),
                    "ap": r.get(f"ap_{alias}", np.nan),
                    "bal_acc": r.get(f"bal_acc_{alias}", np.nan),
                    "f1": r.get(f"f1_{alias}", np.nan),
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Build rows
# -----------------------------------------------------------------------------

rows = []

for cfg in STANDARD_COHORTS:
    row, _ = build_standard_row(cfg)
    rows.append(row)

nacc_main_row, _, _ = build_nacc_main_row()
rows.append(nacc_main_row)

nacc_tr6_row, _, _ = build_nacc_tracer6_row()
rows.append(nacc_tr6_row)

wide_df = pd.DataFrame(rows)

# sort
cohort_order = {
    "adni_tau90": 1,
    "adni_tau180": 2,
    "oasis3_tau90": 3,
    "oasis3_tau180": 4,
    "nacc_strict_at_main": 5,
    "nacc_strict_at_tracer6": 6,
}
wide_df["sort_order"] = wide_df["cohort_key"].map(cohort_order).fillna(999)
wide_df = wide_df.sort_values("sort_order").drop(columns=["sort_order"]).reset_index(drop=True)

long_df = build_long_table(wide_df)

# -----------------------------------------------------------------------------
# Output tables
# -----------------------------------------------------------------------------

wide_csv = OUTDIR / "crosscohort_severity_summary_wide.csv"
long_csv = OUTDIR / "crosscohort_severity_summary_long.csv"
wide_df.to_csv(wide_csv, index=False)
long_df.to_csv(long_csv, index=False)

primary_core_cols = [
    "cohort_key",
    "cohort_label",
    "cohort",
    "window",
    "endpoint_family",
    "endpoint_label",
    "n_subjects",
    "n_pos",
    "n_neg",
    "pos_rate",
    "available_demo_features",
    "available_severity_features",
    "auc_ref",
    "auc_severity",
    "auc_demo",
    "auc_minus_all_cdr",
    "auc_minus_moca",
    "auc_minus_faq",
    "auc_minus_cdr_global",
    "auc_minus_cdr_sumboxes",
    "auc_minus_fullstrip",
    "severity_retention_auc_frac",
    "demo_retention_auc_frac",
    "full_strip_retention_auc_frac",
    "cdr_block_drop_auc",
    "full_strip_drop_auc",
    "moca_unique_drop_auc",
    "faq_unique_drop_auc",
    "cdr_global_unique_drop_auc",
    "cdr_sumboxes_unique_drop_auc",
    "severity_minus_demo_auc",
    "note",
]

primary_df = wide_df[wide_df["supplement_only"] == False].copy()
primary_core_df = primary_df[primary_core_cols].copy()
primary_core_csv = OUTDIR / "crosscohort_primary_core_table.csv"
primary_core_df.to_csv(primary_core_csv, index=False)

tau_only_df = primary_df[primary_df["endpoint_family"] == "tau_binary"].copy()
tau_only_core_df = tau_only_df[primary_core_cols].copy()
tau_only_csv = OUTDIR / "crosscohort_tau_binary_only_core_table.csv"
tau_only_core_df.to_csv(tau_only_csv, index=False)

nacc_supp_df = wide_df[wide_df["cohort_key"].str.contains("nacc", case=False, na=False)].copy()
nacc_supp_csv = OUTDIR / "crosscohort_nacc_anchor_and_sensitivity_table.csv"
nacc_supp_df.to_csv(nacc_supp_csv, index=False)

# compact manuscript-style display
display_cols = [
    "cohort_label",
    "endpoint_family",
    "n_subjects",
    "n_pos",
    "pos_rate",
    "auc_ref",
    "auc_severity",
    "auc_demo",
    "auc_minus_all_cdr",
    "auc_minus_fullstrip",
    "severity_retention_auc_frac",
    "demo_retention_auc_frac",
    "cdr_block_drop_auc",
    "full_strip_drop_auc",
]

print("\n" + "=" * 100)
print("PRIMARY CROSS-COHORT SUMMARY")
print("=" * 100)
print(primary_df[display_cols].to_string(index=False))

print("\n" + "=" * 100)
print("TAU-BINARY ONLY SUMMARY (ADNI + OASIS)")
print("=" * 100)
print(tau_only_df[display_cols].to_string(index=False))

print("\nSaved:")
print(" ", wide_csv)
print(" ", long_csv)
print(" ", primary_core_csv)
print(" ", tau_only_csv)
print(" ", nacc_supp_csv)