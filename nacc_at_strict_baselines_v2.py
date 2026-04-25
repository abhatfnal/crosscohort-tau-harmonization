# nacc_at_strict_baselines_v2.py
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



BASE = Path("/project/aereditato/abhat/NACC_17259/phase2_tau_amyloid_90d/model_ready_v3_from_master")
MAIN_CSV = BASE / "nacc_AT_strict_model_table_nodx_withtracer_v3.csv"
TR6_CSV = BASE / "nacc_AT_strict_tracer6_model_table_nodx_v3.csv"
MANIFEST_JSON = BASE / "nacc_AT_strict_model_manifest_v3.json"
OUTDIR = BASE / "baselines_v3"
OUTDIR.mkdir(parents=True, exist_ok=True)

MAX_MISSING_PCT = 35.0
N_SPLITS = 5
RANDOM_STATE = 42


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
                dropped.append((c, round(miss, 2), "high_missing_numeric"))

    for c in categorical_features:
        if c in df.columns:
            miss = 100.0 * df[c].isna().mean()
            if miss <= max_missing_pct:
                usable_categorical.append(c)
            else:
                dropped.append((c, round(miss, 2), "high_missing_categorical"))

    return usable_numeric, usable_categorical, dropped


def evaluate_dataset(df, dataset_name, numeric_features, categorical_features):
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name}")
    print("=" * 80)
    print("Rows:", len(df))
    print("Subjects:", df["NACCID"].nunique())
    print("Target counts:")
    print(df["y_AT_strict"].value_counts(dropna=False).to_string())

    num_cols, cat_cols, dropped = build_feature_lists(
        df, numeric_features, categorical_features, max_missing_pct=MAX_MISSING_PCT
    )

    feature_cols = num_cols + cat_cols
    if len(feature_cols) == 0:
        raise RuntimeError(f"No usable feature columns found for {dataset_name}")

    print("\nUsing numeric features:", num_cols)
    print("Using categorical features:", cat_cols)
    if dropped:
        print("\nDropped features:")
        for x in dropped:
            print(" ", x)

    X = df[feature_cols].copy()
    y = df["y_AT_strict"].astype(int).values

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

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

    rows = []
    best_model_name = None
    best_auc = -np.inf
    best_pipe = None

    for name, model in models.items():
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
            "dataset": dataset_name,
            "model": name,
            "n_rows": len(df),
            "n_subjects": df["NACCID"].nunique(),
            "n_features": len(feature_cols),
            "roc_auc_mean": np.mean(cvres["test_roc_auc"]),
            "roc_auc_std": np.std(cvres["test_roc_auc"]),
            "ap_mean": np.mean(cvres["test_average_precision"]),
            "ap_std": np.std(cvres["test_average_precision"]),
            "bal_acc_mean": np.mean(cvres["test_balanced_accuracy"]),
            "bal_acc_std": np.std(cvres["test_balanced_accuracy"]),
            "f1_mean": np.mean(cvres["test_f1"]),
            "f1_std": np.std(cvres["test_f1"]),
        }
        rows.append(row)

        if row["roc_auc_mean"] > best_auc:
            best_auc = row["roc_auc_mean"]
            best_model_name = name
            best_pipe = pipe

    metrics_df = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    print("\nCV metrics:")
    print(metrics_df.to_string(index=False))

    metrics_csv = OUTDIR / f"{dataset_name}_baseline_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    oof_proba = cross_val_predict(
        best_pipe,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    oof_df = df[["NACCID", "AT_group", "y_AT_strict"]].copy()
    oof_df["oof_pred_proba"] = oof_proba
    oof_csv = OUTDIR / f"{dataset_name}_best_oof_predictions.csv"
    oof_df.to_csv(oof_csv, index=False)

    feature_info = {
        "dataset": dataset_name,
        "numeric_features_used": num_cols,
        "categorical_features_used": cat_cols,
        "dropped_features": dropped,
        "best_model": best_model_name,
        "best_auc_mean": float(best_auc),
        "metrics_csv": str(metrics_csv),
        "oof_csv": str(oof_csv),
    }
    with open(OUTDIR / f"{dataset_name}_feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)

    return metrics_df


print("=" * 80)
print("LOAD MODEL-READY TABLES")
print("=" * 80)

main_df = pd.read_csv(MAIN_CSV, low_memory=False)
tr6_df = pd.read_csv(TR6_CSV, low_memory=False)

with open(MANIFEST_JSON, "r") as f:
    manifest = json.load(f)

numeric_features = manifest["numeric_features"]
categorical_features = manifest["categorical_features"]

main_metrics = evaluate_dataset(
    main_df,
    dataset_name="strict_main_nodx_withtracer",
    numeric_features=numeric_features,
    categorical_features=categorical_features,
)

tr6_metrics = evaluate_dataset(
    tr6_df,
    dataset_name="strict_tracer6_nodx",
    numeric_features=numeric_features,
    categorical_features=[c for c in categorical_features if c != "tau_tracer"],
)

combined = pd.concat([main_metrics, tr6_metrics], ignore_index=True)
combined_csv = OUTDIR / "combined_baseline_metrics.csv"
combined.to_csv(combined_csv, index=False)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print("Saved:", combined_csv)