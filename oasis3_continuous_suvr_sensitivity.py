#!/usr/bin/env python3
"""
Within-OASIS3 continuous SUVR sensitivity analysis.

This analysis uses the continuous OASIS3 Tauopathy SUVR summary as the target
instead of binarizing it at the GMM midpoint. It keeps the same matched
demographic/genetic and severity feature blocks used in the primary
ADNI/OASIS3 analysis so the result can be interpreted as a threshold-free
OASIS3 sensitivity check.
"""
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5
PRIMARY_MODEL = "ridge"
TARGET_COL = "Tauopathy"

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "oasis3_continuous_suvr_sensitivity"
OUTDIR.mkdir(parents=True, exist_ok=True)

COHORTS = {
    "oasis3_tau180": {
        "cohort_label": "OASIS3 tau180",
        "window": "180d",
        "src": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau180/subject_level_input_table.csv",
    },
    "oasis3_tau90": {
        "cohort_label": "OASIS3 tau90",
        "window": "90d",
        "src": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau90/subject_level_input_table.csv",
    },
}

DEMO_NUM = ["age_h", "education_years_h", "apoe_e4_count_h", "apoe_e4_carrier_h"]
DEMO_CAT = ["sex_h"]
SEVERITY_NUM = ["cdr_sumboxes_h", "faq_total_h"]

EXPERIMENTS = {
    "reference_combined": {
        "numeric": DEMO_NUM + SEVERITY_NUM,
        "categorical": DEMO_CAT,
        "label": "Reference",
    },
    "demo_only": {
        "numeric": DEMO_NUM,
        "categorical": DEMO_CAT,
        "label": "Demo",
    },
    "severity_only": {
        "numeric": SEVERITY_NUM,
        "categorical": [],
        "label": "Severity",
    },
    "minus_all_cdr": {
        "numeric": DEMO_NUM + ["faq_total_h"],
        "categorical": DEMO_CAT,
        "label": "-CDR-SB",
    },
    "minus_faq": {
        "numeric": DEMO_NUM + ["cdr_sumboxes_h"],
        "categorical": DEMO_CAT,
        "label": "-FAQ",
    },
}


def make_preproc(num_cols, cat_cols):
    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_models():
    return {
        "ridge": Ridge(alpha=1.0, solver="lsqr"),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def pearson_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_corr(y_true, y_pred):
    true_rank = pd.Series(y_true).rank(method="average").to_numpy()
    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy()
    return pearson_corr(true_rank, pred_rank)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def target_stratified_folds(y):
    ranks = pd.Series(y).rank(method="first")
    bins = pd.qcut(ranks, q=N_SPLITS, labels=False)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(np.zeros(len(y)), bins))


def available(cols, df):
    return [c for c in cols if c in df.columns and df[c].notna().sum() > 0]


def evaluate(df, cohort_key, cohort_label, window, experiment, exp_cfg, model_name, model):
    num_cols = available(exp_cfg["numeric"], df)
    cat_cols = available(exp_cfg["categorical"], df)
    feature_cols = num_cols + cat_cols
    if not feature_cols:
        return None

    X = df[feature_cols].copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    folds = target_stratified_folds(y)

    oof_pred = np.full(len(y), np.nan, dtype=float)
    fold_rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        pipe = Pipeline(
            steps=[
                ("preproc", make_preproc(num_cols, cat_cols)),
                ("model", model),
            ]
        )
        pipe.fit(X.iloc[train_idx], y[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        oof_pred[test_idx] = pred

        fold_rows.append(
            {
                "fold": fold_idx,
                "r2": r2_score(y[test_idx], pred),
                "pearson_r": pearson_corr(y[test_idx], pred),
                "spearman_r": spearman_corr(y[test_idx], pred),
                "mae": mean_absolute_error(y[test_idx], pred),
                "rmse": rmse(y[test_idx], pred),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    return {
        "cohort_key": cohort_key,
        "cohort_label": cohort_label,
        "window": window,
        "target": TARGET_COL,
        "experiment": experiment,
        "experiment_label": exp_cfg["label"],
        "model": model_name,
        "n_subjects": int(len(df)),
        "n_features_used": int(len(feature_cols)),
        "numeric_features_used": "|".join(num_cols),
        "categorical_features_used": "|".join(cat_cols),
        "target_mean": float(np.mean(y)),
        "target_sd": float(np.std(y, ddof=1)),
        "target_min": float(np.min(y)),
        "target_max": float(np.max(y)),
        "oof_r2": float(r2_score(y, oof_pred)),
        "oof_pearson_r": pearson_corr(y, oof_pred),
        "oof_spearman_r": spearman_corr(y, oof_pred),
        "oof_mae": float(mean_absolute_error(y, oof_pred)),
        "oof_rmse": rmse(y, oof_pred),
        "fold_r2_mean": float(fold_df["r2"].mean()),
        "fold_r2_std": float(fold_df["r2"].std(ddof=0)),
        "fold_pearson_mean": float(fold_df["pearson_r"].mean()),
        "fold_pearson_std": float(fold_df["pearson_r"].std(ddof=0)),
        "fold_spearman_mean": float(fold_df["spearman_r"].mean()),
        "fold_spearman_std": float(fold_df["spearman_r"].std(ddof=0)),
        "fold_mae_mean": float(fold_df["mae"].mean()),
        "fold_mae_std": float(fold_df["mae"].std(ddof=0)),
        "fold_rmse_mean": float(fold_df["rmse"].mean()),
        "fold_rmse_std": float(fold_df["rmse"].std(ddof=0)),
    }


def build_primary_summary(metrics_df):
    primary = metrics_df[metrics_df["model"] == PRIMARY_MODEL].copy()
    rows = []
    for cohort_key, sub in primary.groupby("cohort_key", sort=False):
        sub = sub.set_index("experiment")
        row = {
            "cohort_key": cohort_key,
            "cohort_label": sub["cohort_label"].iloc[0],
            "window": sub["window"].iloc[0],
            "target": TARGET_COL,
            "model": PRIMARY_MODEL,
            "n_subjects": int(sub["n_subjects"].iloc[0]),
            "target_mean": float(sub["target_mean"].iloc[0]),
            "target_sd": float(sub["target_sd"].iloc[0]),
        }
        for metric in ["oof_r2", "oof_pearson_r", "oof_spearman_r", "oof_mae", "oof_rmse"]:
            for exp in EXPERIMENTS:
                row[f"{exp}_{metric}"] = float(sub.loc[exp, metric])
        row["ref_minus_demo_r2"] = row["reference_combined_oof_r2"] - row["demo_only_oof_r2"]
        row["ref_minus_severity_r2"] = row["reference_combined_oof_r2"] - row["severity_only_oof_r2"]
        row["ref_minus_minus_all_cdr_r2"] = (
            row["reference_combined_oof_r2"] - row["minus_all_cdr_oof_r2"]
        )
        row["ref_minus_minus_faq_r2"] = row["reference_combined_oof_r2"] - row["minus_faq_oof_r2"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_primary(metrics_df):
    primary = metrics_df[metrics_df["model"] == PRIMARY_MODEL].copy()
    exp_order = list(EXPERIMENTS)
    labels = [EXPERIMENTS[e]["label"] for e in exp_order]
    colors = ["#2f6f9f", "#d08c2f", "#4f8a58", "#8b6bb1", "#9b4f58"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, (cohort_key, cohort_cfg) in zip(axes, COHORTS.items()):
        sub = primary[primary["cohort_key"] == cohort_key].set_index("experiment")
        vals = [sub.loc[e, "oof_r2"] for e in exp_order]
        ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(cohort_cfg["cohort_label"], fontsize=12)
        ax.set_ylabel("Out-of-fold R2" if ax is axes[0] else "")
        ax.set_ylim(min(-0.25, min(vals) - 0.05), max(0.35, max(vals) + 0.05))
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        for idx, val in enumerate(vals):
            va = "bottom" if val >= 0 else "top"
            offset = 0.012 if val >= 0 else -0.012
            ax.text(idx, val + offset, f"{val:.3f}", ha="center", va=va, fontsize=8)

    fig.suptitle("Within-OASIS3 continuous Tauopathy SUVR sensitivity", fontsize=13, fontweight="bold")
    fig.tight_layout()
    png = OUTDIR / "continuous_suvr_r2_by_feature_block.png"
    pdf = OUTDIR / "continuous_suvr_r2_by_feature_block.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    rows = []
    models = make_models()

    for cohort_key, cfg in COHORTS.items():
        df = pd.read_csv(cfg["src"], low_memory=False).copy()
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        df = df[df[TARGET_COL].notna()].reset_index(drop=True)
        print(f"\n{cfg['cohort_label']}: N={len(df)}")
        print(df[TARGET_COL].describe().to_string())

        for experiment, exp_cfg in EXPERIMENTS.items():
            for model_name, model in models.items():
                row = evaluate(
                    df=df,
                    cohort_key=cohort_key,
                    cohort_label=cfg["cohort_label"],
                    window=cfg["window"],
                    experiment=experiment,
                    exp_cfg=exp_cfg,
                    model_name=model_name,
                    model=model,
                )
                if row is not None:
                    rows.append(row)
                    print(
                        f"  {experiment:18s} {model_name:13s} "
                        f"R2={row['oof_r2']:.3f} "
                        f"r={row['oof_pearson_r']:.3f} "
                        f"rho={row['oof_spearman_r']:.3f}"
                    )

    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTDIR / "continuous_suvr_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    summary_df = build_primary_summary(metrics_df)
    summary_path = OUTDIR / "continuous_suvr_primary_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    png, pdf = plot_primary(metrics_df)

    print("\nPrimary ridge summary:")
    display_cols = [
        "cohort_label",
        "n_subjects",
        "reference_combined_oof_r2",
        "demo_only_oof_r2",
        "severity_only_oof_r2",
        "ref_minus_demo_r2",
        "ref_minus_severity_r2",
    ]
    print(summary_df[display_cols].to_string(index=False))
    print("\nSaved:")
    print(" ", metrics_path)
    print(" ", summary_path)
    print(" ", png)
    print(" ", pdf)


if __name__ == "__main__":
    main()
