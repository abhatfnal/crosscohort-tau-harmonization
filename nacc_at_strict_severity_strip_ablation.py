# nacc_at_strict_severity_strip_ablation.py

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


# =============================================================================
# PATHS
# =============================================================================
BASE = Path("/project/aereditato/abhat/NACC_17259/phase2_tau_amyloid_90d/model_ready_v3_from_master")
MAIN_CSV = BASE / "nacc_AT_strict_model_table_nodx_withtracer_v3.csv"
TR6_CSV = BASE / "nacc_AT_strict_tracer6_model_table_nodx_v3.csv"
OUTDIR = BASE / "ablation_severity_strip"
OUTDIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SETTINGS
# =============================================================================
MAX_MISSING_PCT = 35.0
N_SPLITS = 5
RANDOM_STATE = 42

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================
DEMO_NUMERIC = [
    "age",
    "education_years",
    "apoe_e4_count",
    "apoe_e4_carrier",
]

DEMO_CATEGORICAL = [
    "sex",
    "race",
    "hispanic",
    "handedness",
]

SEVERITY_NUMERIC = [
    "cdr_global",
    "cdr_sumboxes",
    "faq_total",
    "moca",
    "animal_fluency",
    "vegetable_fluency",
]

TRACER_CATEGORICAL = [
    "tau_tracer",
]

# experiments are defined as features removed from the full combined set
EXPERIMENTS = [
    {"name": "reference_combined", "drop": []},
    {"name": "minus_cdr_global", "drop": ["cdr_global"]},
    {"name": "minus_cdr_sumboxes", "drop": ["cdr_sumboxes"]},
    {"name": "minus_all_cdr", "drop": ["cdr_global", "cdr_sumboxes"]},
    {"name": "minus_faq", "drop": ["faq_total"]},
    {"name": "minus_moca", "drop": ["moca"]},
    {"name": "minus_cdr_faq", "drop": ["cdr_global", "cdr_sumboxes", "faq_total"]},
    {"name": "minus_cdr_moca", "drop": ["cdr_global", "cdr_sumboxes", "moca"]},
    {"name": "minus_faq_moca", "drop": ["faq_total", "moca"]},
    {"name": "minus_cdr_faq_moca", "drop": ["cdr_global", "cdr_sumboxes", "faq_total", "moca"]},
]

SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
}


# =============================================================================
# HELPERS
# =============================================================================
def build_feature_lists(df, numeric_features, categorical_features, max_missing_pct=35.0):
    usable_numeric = []
    usable_categorical = []
    dropped = []

    for c in numeric_features:
        if c in df.columns:
            miss = 100.0 * df[c].isna().mean()
            if miss <= max_missing_pct:
                usable_numeric.append(c)
            else:
                dropped.append((c, round(float(miss), 2), "high_missing_numeric"))
        else:
            dropped.append((c, None, "missing_column_numeric"))

    for c in categorical_features:
        if c in df.columns:
            miss = 100.0 * df[c].isna().mean()
            if miss <= max_missing_pct:
                usable_categorical.append(c)
            else:
                dropped.append((c, round(float(miss), 2), "high_missing_categorical"))
        else:
            dropped.append((c, None, "missing_column_categorical"))

    return usable_numeric, usable_categorical, dropped


def build_preprocessor(num_cols, cat_cols):
    transformers = []

    if len(num_cols) > 0:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, num_cols))

    if len(cat_cols) > 0:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_pipe, cat_cols))

    if len(transformers) == 0:
        raise RuntimeError("No numeric or categorical features were supplied to the preprocessor.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_models():
    return {
        "logreg_balanced": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def requested_feature_set(include_tracer, drop_features):
    num = DEMO_NUMERIC + SEVERITY_NUMERIC
    cat = DEMO_CATEGORICAL + (TRACER_CATEGORICAL if include_tracer else [])

    drop_features = set(drop_features)
    num = [c for c in num if c not in drop_features]
    cat = [c for c in cat if c not in drop_features]

    return num, cat


def evaluate_experiment(df, dataset_name, experiment_name, include_tracer, drop_features):
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {dataset_name} | {experiment_name}")
    print("=" * 80)
    print("Rows:", len(df))
    print("Subjects:", df["NACCID"].nunique())
    print("Target counts:")
    print(df["y_AT_strict"].value_counts(dropna=False).to_string())

    req_num, req_cat = requested_feature_set(include_tracer=include_tracer, drop_features=drop_features)
    num_cols, cat_cols, dropped = build_feature_lists(
        df, req_num, req_cat, max_missing_pct=MAX_MISSING_PCT
    )

    feature_cols = num_cols + cat_cols
    if len(feature_cols) == 0:
        raise RuntimeError(f"No usable feature columns for {dataset_name} | {experiment_name}")

    print("\nRequested numeric:", req_num)
    print("Requested categorical:", req_cat)
    print("Using numeric:", num_cols)
    print("Using categorical:", cat_cols)
    if dropped:
        print("\nDropped features:")
        for x in dropped:
            print(" ", x)

    X = df[feature_cols].copy()
    y = df["y_AT_strict"].astype(int).values

    preproc = build_preprocessor(num_cols, cat_cols)
    models = get_models()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for model_name, model in models.items():
        pipe = Pipeline([
            ("preproc", preproc),
            ("model", model),
        ])

        cvres = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
            return_train_score=False,
        )

        rows.append({
            "dataset": dataset_name,
            "experiment": experiment_name,
            "removed_features": ";".join(drop_features) if len(drop_features) > 0 else "",
            "model": model_name,
            "n_rows": len(df),
            "n_subjects": df["NACCID"].nunique(),
            "n_features_requested": len(req_num) + len(req_cat),
            "n_features_used": len(feature_cols),
            "numeric_features_used": "|".join(num_cols),
            "categorical_features_used": "|".join(cat_cols),
            "roc_auc_mean": float(np.mean(cvres["test_roc_auc"])),
            "roc_auc_std": float(np.std(cvres["test_roc_auc"])),
            "ap_mean": float(np.mean(cvres["test_average_precision"])),
            "ap_std": float(np.std(cvres["test_average_precision"])),
            "bal_acc_mean": float(np.mean(cvres["test_balanced_accuracy"])),
            "bal_acc_std": float(np.std(cvres["test_balanced_accuracy"])),
            "f1_mean": float(np.mean(cvres["test_f1"])),
            "f1_std": float(np.std(cvres["test_f1"])),
        })

    metrics_df = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    print("\nCV metrics:")
    print(metrics_df.to_string(index=False))
    return metrics_df


def summarize_best_models(all_metrics):
    best = (
        all_metrics
        .sort_values(["dataset", "experiment", "roc_auc_mean"], ascending=[True, True, False])
        .groupby(["dataset", "experiment"], as_index=False)
        .first()
    )
    return best


def build_delta_table(best_df, reference_experiment="reference_combined"):
    out = []

    for dataset in sorted(best_df["dataset"].unique()):
        sub = best_df[best_df["dataset"] == dataset].copy()
        ref = sub[sub["experiment"] == reference_experiment]
        if len(ref) != 1:
            continue

        ref = ref.iloc[0]
        ref_auc = ref["roc_auc_mean"]
        ref_ap = ref["ap_mean"]
        ref_bal = ref["bal_acc_mean"]
        ref_f1 = ref["f1_mean"]

        for _, row in sub.iterrows():
            out.append({
                "dataset": dataset,
                "reference_experiment": reference_experiment,
                "experiment": row["experiment"],
                "removed_features": row["removed_features"],
                "best_model": row["model"],
                "roc_auc_mean": row["roc_auc_mean"],
                "ap_mean": row["ap_mean"],
                "bal_acc_mean": row["bal_acc_mean"],
                "f1_mean": row["f1_mean"],
                "drop_in_roc_auc_vs_reference": float(ref_auc - row["roc_auc_mean"]),
                "drop_in_ap_vs_reference": float(ref_ap - row["ap_mean"]),
                "drop_in_bal_acc_vs_reference": float(ref_bal - row["bal_acc_mean"]),
                "drop_in_f1_vs_reference": float(ref_f1 - row["f1_mean"]),
            })

    delta_df = pd.DataFrame(out)
    delta_df = delta_df.sort_values(
        ["dataset", "drop_in_roc_auc_vs_reference", "experiment"],
        ascending=[True, False, True]
    )
    return delta_df


# =============================================================================
# MAIN
# =============================================================================
print("=" * 80)
print("LOAD MODEL-READY TABLES")
print("=" * 80)

main_df = pd.read_csv(MAIN_CSV, low_memory=False)
tr6_df = pd.read_csv(TR6_CSV, low_memory=False)

print("\nMain strict cohort:")
print("Rows:", len(main_df), " Subjects:", main_df["NACCID"].nunique())
print(main_df["AT_group"].value_counts(dropna=False).to_string())

print("\nTracer-6 strict cohort:")
print("Rows:", len(tr6_df), " Subjects:", tr6_df["NACCID"].nunique())
print(tr6_df["AT_group"].value_counts(dropna=False).to_string())

all_metrics = []

# Main cohort: includes tracer
for exp in EXPERIMENTS:
    metrics = evaluate_experiment(
        df=main_df,
        dataset_name="strict_main_withtracer",
        experiment_name=exp["name"],
        include_tracer=True,
        drop_features=exp["drop"],
    )
    all_metrics.append(metrics)

# Tracer-6 cohort: no tracer feature needed
for exp in EXPERIMENTS:
    metrics = evaluate_experiment(
        df=tr6_df,
        dataset_name="strict_tracer6_notracer",
        experiment_name=exp["name"],
        include_tracer=False,
        drop_features=exp["drop"],
    )
    all_metrics.append(metrics)

all_metrics_df = pd.concat(all_metrics, ignore_index=True)
all_metrics_csv = OUTDIR / "all_severity_strip_metrics.csv"
all_metrics_df.to_csv(all_metrics_csv, index=False)

best_df = summarize_best_models(all_metrics_df)
best_csv = OUTDIR / "best_model_per_experiment.csv"
best_df.to_csv(best_csv, index=False)

delta_df = build_delta_table(best_df, reference_experiment="reference_combined")
delta_csv = OUTDIR / "severity_strip_deltas_vs_reference.csv"
delta_df.to_csv(delta_csv, index=False)

print("\n" + "=" * 80)
print("BEST MODEL PER EXPERIMENT")
print("=" * 80)
print(best_df.to_string(index=False))

print("\n" + "=" * 80)
print("DROP VS REFERENCE COMBINED")
print("=" * 80)
print(delta_df.to_string(index=False))

summary = {
    "main_csv": str(MAIN_CSV),
    "tracer6_csv": str(TR6_CSV),
    "outdir": str(OUTDIR),
    "max_missing_pct": MAX_MISSING_PCT,
    "n_splits": N_SPLITS,
    "experiments": EXPERIMENTS,
    "outputs": {
        "all_metrics_csv": str(all_metrics_csv),
        "best_model_csv": str(best_csv),
        "delta_csv": str(delta_csv),
    },
}

summary_json = OUTDIR / "severity_strip_summary.json"
with open(summary_json, "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print("Saved:")
print(" ", all_metrics_csv)
print(" ", best_csv)
print(" ", delta_csv)
print(" ", summary_json)