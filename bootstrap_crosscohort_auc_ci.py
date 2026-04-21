# bootstrap_crosscohort_auc_ci.py
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
N_SPLITS = 5

# Primary paper cohorts
DATASETS = {
    "adni_tau90": {
        "label": "ADNI tau90",
        "subject_table": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau90/subject_level_input_table.csv",
    },
    "adni_tau180": {
        "label": "ADNI tau180",
        "subject_table": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau180/subject_level_input_table.csv",
    },
    "oasis3_tau180": {
        "label": "OASIS3 tau180",
        "subject_table": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau180/subject_level_input_table.csv",
    },
    # optional supplementary
    "oasis3_tau90": {
        "label": "OASIS3 tau90",
        "subject_table": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau90/subject_level_input_table.csv",
    },
}

MATCHED_RERUN_DIR = Path(
    "/project/aereditato/abhat/adni-mri-classification/crosscohort_matched_rerun"
)
BEST_MODELS_CSV = MATCHED_RERUN_DIR / "best_model_per_experiment.csv"

# Keep this EXACTLY aligned with the matched-feature publication tables
FEATURES = {
    "reference_combined": {
        "num": [
            "apoe_e4_carrier_h",
            "apoe_e4_count_h",
            "education_years_h",
            "cdr_sumboxes_h",
            "faq_total_h",
        ],
        "cat": ["sex_h"],
    },
    "severity_only": {
        "num": ["cdr_sumboxes_h", "faq_total_h"],
        "cat": [],
    },
    "demo_only": {
        "num": [
            "apoe_e4_carrier_h",
            "apoe_e4_count_h",
            "education_years_h",
        ],
        "cat": ["sex_h"],
    },
    "minus_all_cdr": {
        "num": [
            "apoe_e4_carrier_h",
            "apoe_e4_count_h",
            "education_years_h",
            "faq_total_h",
        ],
        "cat": ["sex_h"],
    },
}

EXPERIMENTS = [
    "reference_combined",
    "severity_only",
    "demo_only",
    "minus_all_cdr",
]


def make_preprocessor(num_cols, cat_cols):
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


def make_model(model_name: str):
    if model_name == "logreg_balanced":
        return LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
    if model_name == "random_forest_balanced":
        return RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,  # avoid nested oversubscription
        )
    if model_name == "extra_trees_balanced":
        return ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,  # avoid nested oversubscription
        )
    raise ValueError(f"Unknown model name: {model_name}")


def read_subject_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "y_target" not in df.columns:
        raise KeyError(f"Missing y_target in {path}")
    if "subject_id" not in df.columns:
        raise KeyError(f"Missing subject_id in {path}")
    df = df[df["y_target"].notna()].copy()
    df["y_target"] = df["y_target"].astype(int)
    return df.reset_index(drop=True)


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
    boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)

    boot_idx = np.concatenate([boot_neg, boot_pos])
    rng.shuffle(boot_idx)
    return boot_idx


def mean_cv_auc(df: pd.DataFrame, exp_name: str, model_name: str) -> float:
    spec = FEATURES[exp_name]
    num_cols = [c for c in spec["num"] if c in df.columns]
    cat_cols = [c for c in spec["cat"] if c in df.columns]
    feat_cols = num_cols + cat_cols

    if not feat_cols:
        raise RuntimeError(f"No features present for {exp_name}")

    X = df[feat_cols].copy()
    y = df["y_target"].to_numpy()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    preproc = make_preprocessor(num_cols, cat_cols)
    base_model = make_model(model_name)

    fold_aucs = []

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = clone(base_model)
        pipe = Pipeline(
            steps=[
                ("preproc", preproc),
                ("model", model),
            ]
        )
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        fold_aucs.append(roc_auc_score(y_test, prob))

    return float(np.mean(fold_aucs))


def one_bootstrap_replicate(
    df: pd.DataFrame,
    best_model_map: dict,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    y = df["y_target"].to_numpy()
    boot_idx = stratified_bootstrap_indices(y, rng)
    boot_df = df.iloc[boot_idx].reset_index(drop=True)

    out = {}
    for exp_name in EXPERIMENTS:
        model_name = best_model_map[exp_name]
        out[exp_name] = mean_cv_auc(boot_df, exp_name, model_name)

    # paired differences from same bootstrap sample
    out["diff_ref_minus_severity"] = (
        out["reference_combined"] - out["severity_only"]
    )
    out["diff_ref_minus_demo"] = (
        out["reference_combined"] - out["demo_only"]
    )
    out["diff_ref_minus_minusallcdr"] = (
        out["reference_combined"] - out["minus_all_cdr"]
    )
    return out


def percentile_ci(x, alpha=0.05):
    lo = np.quantile(x, alpha / 2.0)
    hi = np.quantile(x, 1.0 - alpha / 2.0)
    return float(lo), float(hi)


def load_best_models() -> pd.DataFrame:
    df = pd.read_csv(BEST_MODELS_CSV)

    # Prefer cohort_key as the canonical cohort identifier
    if "dataset" not in df.columns:
        if "cohort_key" in df.columns:
            df = df.rename(columns={"cohort_key": "dataset"})
        elif "cohort" in df.columns:
            df = df.rename(columns={"cohort": "dataset"})

    # If duplicate column names somehow still exist, keep the first copy only
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    print("\nLoaded best-model table columns:")
    print(df.columns.tolist())

    required = ["dataset", "experiment", "model", "roc_auc_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns in {BEST_MODELS_CSV}: {missing}. "
            f"Columns are: {df.columns.tolist()}"
        )

    keep = df[df["experiment"].isin(EXPERIMENTS)].copy()
    return keep


def summarize_cohort(
    cohort_key: str,
    df: pd.DataFrame,
    best_models_df: pd.DataFrame,
    n_bootstrap: int,
    n_jobs: int,
    outdir: Path,
    seed: int,
):
    cohort_label = DATASETS[cohort_key]["label"]

    if "dataset" not in best_models_df.columns:
        raise KeyError(
            f"best_models_df is missing 'dataset'. Columns: {best_models_df.columns.tolist()}"
        )

    print("Available dataset values:", sorted(best_models_df["dataset"].dropna().unique().tolist()))
    cohort_best = best_models_df.loc[best_models_df["dataset"] == cohort_key].copy()

    if cohort_best.empty:
        raise RuntimeError(
            f"No rows found in best_model_per_experiment.csv for dataset='{cohort_key}'. "
            f"Available dataset values: {sorted(best_models_df['dataset'].dropna().unique().tolist())}"
        )

    cohort_best = cohort_best.set_index("experiment")

    missing = [e for e in EXPERIMENTS if e not in cohort_best.index]
    if missing:
        raise RuntimeError(f"{cohort_key}: missing best-model rows for {missing}")

    best_model_map = {
        exp: cohort_best.loc[exp, "model"]
        for exp in EXPERIMENTS
    }
    point_auc_map = {
        exp: float(cohort_best.loc[exp, "roc_auc_mean"])
        for exp in EXPERIMENTS
    }

    seeds = [seed + i for i in range(n_bootstrap)]
    reps = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(one_bootstrap_replicate)(df, best_model_map, s)
        for s in seeds
    )
    reps_df = pd.DataFrame(reps)

    reps_path = outdir / f"{cohort_key}_bootstrap_replicates.csv.gz"
    reps_df.to_csv(reps_path, index=False, compression="gzip")

    summary_rows = []
    for exp in EXPERIMENTS:
        lo, hi = percentile_ci(reps_df[exp].values)
        summary_rows.append(
            {
                "cohort_key": cohort_key,
                "cohort_label": cohort_label,
                "experiment": exp,
                "model": best_model_map[exp],
                "n_subjects": int(df["subject_id"].nunique()),
                "n_pos": int(df["y_target"].sum()),
                "n_neg": int((1 - df["y_target"]).sum()),
                "auc_point": point_auc_map[exp],
                "auc_boot_mean": float(reps_df[exp].mean()),
                "auc_ci_low": lo,
                "auc_ci_high": hi,
                "n_bootstrap": int(n_bootstrap),
            }
        )

    diff_specs = {
        "diff_ref_minus_severity": (
            "reference_combined",
            "severity_only",
        ),
        "diff_ref_minus_demo": (
            "reference_combined",
            "demo_only",
        ),
        "diff_ref_minus_minusallcdr": (
            "reference_combined",
            "minus_all_cdr",
        ),
    }

    diff_rows = []
    for diff_name, (a, b) in diff_specs.items():
        point = point_auc_map[a] - point_auc_map[b]
        lo, hi = percentile_ci(reps_df[diff_name].values)
        diff_rows.append(
            {
                "cohort_key": cohort_key,
                "cohort_label": cohort_label,
                "difference": diff_name,
                "auc_diff_point": float(point),
                "auc_diff_boot_mean": float(reps_df[diff_name].mean()),
                "auc_diff_ci_low": lo,
                "auc_diff_ci_high": hi,
                "n_bootstrap": int(n_bootstrap),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(diff_rows), best_model_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=["adni_tau90", "adni_tau180", "oasis3_tau180"],
        choices=sorted(DATASETS.keys()),
        help="Primary cohorts by default. Add oasis3_tau90 if you want the supplement.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--outdir",
        default="/project/aereditato/abhat/adni-mri-classification/crosscohort_bootstrap_ci",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    best_models_df = load_best_models()

    all_summary = []
    all_diffs = []
    manifest = {
        "cohorts": args.cohorts,
        "n_bootstrap": args.n_bootstrap,
        "n_splits": N_SPLITS,
        "seed": args.seed,
        "features": FEATURES,
        "best_models_csv": str(BEST_MODELS_CSV),
    }

    for cohort_key in args.cohorts:
        print("\n" + "=" * 100)
        print(f"BOOTSTRAP CI: {cohort_key} ({DATASETS[cohort_key]['label']})")
        print("=" * 100)

        df = read_subject_table(DATASETS[cohort_key]["subject_table"])
        print(f"rows={len(df)} subjects={df['subject_id'].nunique()} pos={df['y_target'].sum()}")

        summary_df, diff_df, best_map = summarize_cohort(
            cohort_key=cohort_key,
            df=df,
            best_models_df=best_models_df,
            n_bootstrap=args.n_bootstrap,
            n_jobs=args.n_jobs,
            outdir=outdir,
            seed=args.seed,
        )
        manifest[f"{cohort_key}_best_models"] = best_map

        print("\nAUC summary")
        print(summary_df.to_string(index=False))

        print("\nAUC differences")
        print(diff_df.to_string(index=False))

        all_summary.append(summary_df)
        all_diffs.append(diff_df)

    all_summary_df = pd.concat(all_summary, ignore_index=True)
    all_diffs_df = pd.concat(all_diffs, ignore_index=True)

    all_summary_df.to_csv(outdir / "bootstrap_auc_summary.csv", index=False)
    all_diffs_df.to_csv(outdir / "bootstrap_auc_differences.csv", index=False)

    with open(outdir / "bootstrap_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nSaved:")
    print(" ", outdir / "bootstrap_auc_summary.csv")
    print(" ", outdir / "bootstrap_auc_differences.csv")
    print(" ", outdir / "bootstrap_manifest.json")


if __name__ == "__main__":
    main()