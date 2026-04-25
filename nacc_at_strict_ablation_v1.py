# nacc_at_strict_ablation_v1.py
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE = Path("/project/aereditato/abhat/NACC_17259/phase2_tau_amyloid_90d/model_ready_v3_from_master")
MAIN_CSV = BASE / "nacc_AT_strict_model_table_nodx_withtracer_v3.csv"
TR6_CSV = BASE / "nacc_AT_strict_tracer6_model_table_nodx_v3.csv"
OUTDIR = BASE / "ablation_v1"
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MAX_MISSING_PCT = 35.0
N_SPLITS = 5
RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# Feature sets
# -----------------------------------------------------------------------------
# These are harmonized feature names from your v3 model-ready tables.
FEATURE_SETS = {
    # Pure background / risk structure without tracer
    "demo_apoe_notracer": {
        "numeric": [
            "age",
            "education_years",
            "apoe_e4_count",
            "apoe_e4_carrier",
        ],
        "categorical": [
            "sex",
            "race",
            "hispanic",
            "handedness",
        ],
    },

    # Same as above, but explicitly includes tracer
    "demo_apoe_withtracer": {
        "numeric": [
            "age",
            "education_years",
            "apoe_e4_count",
            "apoe_e4_carrier",
        ],
        "categorical": [
            "sex",
            "race",
            "hispanic",
            "handedness",
            "tau_tracer",
        ],
    },

    # Clinical severity / phenotype only
    "severity_only": {
        "numeric": [
            "cdr_global",
            "cdr_sumboxes",
            "faq_total",
            "moca",
            "animal_fluency",
            "vegetable_fluency",
        ],
        "categorical": [],
    },

    # Main non-diagnostic phenotype model, no tracer
    "combined_notracer": {
        "numeric": [
            "age",
            "education_years",
            "apoe_e4_count",
            "apoe_e4_carrier",
            "cdr_global",
            "cdr_sumboxes",
            "faq_total",
            "moca",
            "animal_fluency",
            "vegetable_fluency",
        ],
        "categorical": [
            "sex",
            "race",
            "hispanic",
            "handedness",
        ],
    },

    # Same combined model, but add tracer
    "combined_withtracer": {
        "numeric": [
            "age",
            "education_years",
            "apoe_e4_count",
            "apoe_e4_carrier",
            "cdr_global",
            "cdr_sumboxes",
            "faq_total",
            "moca",
            "animal_fluency",
            "vegetable_fluency",
        ],
        "categorical": [
            "sex",
            "race",
            "hispanic",
            "handedness",
            "tau_tracer",
        ],
    },
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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


def make_preprocessor(num_cols, cat_cols):
    transformers = []

    if num_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, num_cols))

    if cat_cols:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_pipe, cat_cols))

    if not transformers:
        raise RuntimeError("No usable numeric or categorical columns were provided.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def evaluate_experiment(df, experiment_name, feature_set_name, numeric_features, categorical_features):
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 80)
    print("Rows:", len(df))
    print("Subjects:", df["NACCID"].nunique())
    print("Target counts:")
    print(df["y_AT_strict"].value_counts(dropna=False).to_string())

    num_cols, cat_cols, dropped = build_feature_lists(
        df=df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        max_missing_pct=MAX_MISSING_PCT,
    )

    feature_cols = num_cols + cat_cols
    if len(feature_cols) == 0:
        raise RuntimeError(f"No usable feature columns for {experiment_name}")

    print("\nUsing numeric features:", num_cols)
    print("Using categorical features:", cat_cols)

    if dropped:
        print("\nDropped features:")
        for item in dropped:
            print(" ", item)

    X = df[feature_cols].copy()
    y = df["y_AT_strict"].astype(int).values

    preproc = make_preprocessor(num_cols, cat_cols)

    models = {
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

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
    }

    metric_rows = []
    best_model_name = None
    best_auc = -np.inf
    best_pipe = None

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
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        row = {
            "experiment": experiment_name,
            "feature_set": feature_set_name,
            "model": model_name,
            "n_rows": int(len(df)),
            "n_subjects": int(df["NACCID"].nunique()),
            "n_features_requested": int(len(numeric_features) + len(categorical_features)),
            "n_features_used": int(len(feature_cols)),
            "roc_auc_mean": float(np.mean(cvres["test_roc_auc"])),
            "roc_auc_std": float(np.std(cvres["test_roc_auc"])),
            "ap_mean": float(np.mean(cvres["test_average_precision"])),
            "ap_std": float(np.std(cvres["test_average_precision"])),
            "bal_acc_mean": float(np.mean(cvres["test_balanced_accuracy"])),
            "bal_acc_std": float(np.std(cvres["test_balanced_accuracy"])),
            "f1_mean": float(np.mean(cvres["test_f1"])),
            "f1_std": float(np.std(cvres["test_f1"])),
        }
        metric_rows.append(row)

        if row["roc_auc_mean"] > best_auc:
            best_auc = row["roc_auc_mean"]
            best_model_name = model_name
            best_pipe = pipe

    metrics_df = pd.DataFrame(metric_rows).sort_values("roc_auc_mean", ascending=False)
    print("\nCV metrics:")
    print(metrics_df.to_string(index=False))

    metrics_csv = OUTDIR / f"{experiment_name}_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # OOF predictions for best model
    oof_proba = cross_val_predict(
        best_pipe,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    oof_df = df[["NACCID", "AT_group", "y_AT_strict"]].copy()
    oof_df["experiment"] = experiment_name
    oof_df["feature_set"] = feature_set_name
    oof_df["oof_pred_proba"] = oof_proba

    oof_csv = OUTDIR / f"{experiment_name}_best_oof_predictions.csv"
    oof_df.to_csv(oof_csv, index=False)

    info = {
        "experiment": experiment_name,
        "feature_set": feature_set_name,
        "numeric_features_requested": numeric_features,
        "categorical_features_requested": categorical_features,
        "numeric_features_used": num_cols,
        "categorical_features_used": cat_cols,
        "dropped_features": dropped,
        "best_model": best_model_name,
        "best_auc_mean": float(best_auc),
        "metrics_csv": str(metrics_csv),
        "oof_csv": str(oof_csv),
    }

    info_json = OUTDIR / f"{experiment_name}_feature_info.json"
    with open(info_json, "w") as f:
        json.dump(info, f, indent=2)

    return metrics_df, info


def build_tracer_effect_table(all_metrics_df):
    """
    Compare matched ablation pairs to quantify tracer contribution.
    We compare the best ROC AUC in each pair.
    """
    pairs = [
        ("main_demo_apoe_notracer", "main_demo_apoe_withtracer"),
        ("main_combined_notracer", "main_combined_withtracer"),
    ]

    rows = []
    for no_tracer_name, with_tracer_name in pairs:
        a = all_metrics_df[all_metrics_df["experiment"] == no_tracer_name].sort_values("roc_auc_mean", ascending=False)
        b = all_metrics_df[all_metrics_df["experiment"] == with_tracer_name].sort_values("roc_auc_mean", ascending=False)

        if len(a) == 0 or len(b) == 0:
            continue

        a_best = a.iloc[0]
        b_best = b.iloc[0]

        rows.append({
            "without_tracer_experiment": no_tracer_name,
            "without_tracer_best_model": a_best["model"],
            "without_tracer_best_auc": float(a_best["roc_auc_mean"]),
            "with_tracer_experiment": with_tracer_name,
            "with_tracer_best_model": b_best["model"],
            "with_tracer_best_auc": float(b_best["roc_auc_mean"]),
            "delta_auc_with_minus_without": float(b_best["roc_auc_mean"] - a_best["roc_auc_mean"]),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Run experiments
# -----------------------------------------------------------------------------
experiments = [
    {
        "experiment_name": "main_demo_apoe_notracer",
        "feature_set_name": "demo_apoe_notracer",
        "df": main_df,
    },
    {
        "experiment_name": "main_demo_apoe_withtracer",
        "feature_set_name": "demo_apoe_withtracer",
        "df": main_df,
    },
    {
        "experiment_name": "main_severity_only",
        "feature_set_name": "severity_only",
        "df": main_df,
    },
    {
        "experiment_name": "main_combined_notracer",
        "feature_set_name": "combined_notracer",
        "df": main_df,
    },
    {
        "experiment_name": "main_combined_withtracer",
        "feature_set_name": "combined_withtracer",
        "df": main_df,
    },
    {
        "experiment_name": "tracer6_combined_notracer",
        "feature_set_name": "combined_notracer",
        "df": tr6_df,
    },
]

all_metrics = []
all_infos = []

for exp in experiments:
    fs = FEATURE_SETS[exp["feature_set_name"]]
    metrics_df, info = evaluate_experiment(
        df=exp["df"],
        experiment_name=exp["experiment_name"],
        feature_set_name=exp["feature_set_name"],
        numeric_features=fs["numeric"],
        categorical_features=fs["categorical"],
    )
    all_metrics.append(metrics_df)
    all_infos.append(info)

all_metrics_df = pd.concat(all_metrics, ignore_index=True)
all_metrics_csv = OUTDIR / "all_ablation_metrics.csv"
all_metrics_df.to_csv(all_metrics_csv, index=False)

best_only_df = (
    all_metrics_df
    .sort_values(["experiment", "roc_auc_mean"], ascending=[True, False])
    .groupby("experiment", as_index=False)
    .first()
    .sort_values("roc_auc_mean", ascending=False)
)
best_only_csv = OUTDIR / "best_model_per_experiment.csv"
best_only_df.to_csv(best_only_csv, index=False)

tracer_effect_df = build_tracer_effect_table(all_metrics_df)
tracer_effect_csv = OUTDIR / "tracer_effect_comparisons.csv"
tracer_effect_df.to_csv(tracer_effect_csv, index=False)

summary = {
    "main_csv": str(MAIN_CSV),
    "tracer6_csv": str(TR6_CSV),
    "max_missing_pct": MAX_MISSING_PCT,
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE,
    "experiments_run": [x["experiment_name"] for x in experiments],
    "all_metrics_csv": str(all_metrics_csv),
    "best_only_csv": str(best_only_csv),
    "tracer_effect_csv": str(tracer_effect_csv),
    "feature_sets": FEATURE_SETS,
}

summary_json = OUTDIR / "ablation_summary.json"
with open(summary_json, "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print("BEST MODEL PER EXPERIMENT")
print("=" * 80)
print(best_only_df.to_string(index=False))

print("\n" + "=" * 80)
print("TRACER EFFECT COMPARISONS")
print("=" * 80)
if len(tracer_effect_df):
    print(tracer_effect_df.to_string(index=False))
else:
    print("No tracer comparison rows were produced.")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print("Saved:")
print(" ", all_metrics_csv)
print(" ", best_only_csv)
print(" ", tracer_effect_csv)
print(" ", summary_json)