# crosscohort_tau_matched_rerun.py
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


RANDOM_STATE = 42
N_SPLITS = 5

COHORTS = {
    "adni_tau90": {
        "cohort_label": "ADNI tau90",
        "cohort": "ADNI",
        "window": "90d",
        "endpoint_family": "tau_binary",
        "endpoint_label": "tau positivity",
        "src": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau90/subject_level_input_table.csv",
    },
    "adni_tau180": {
        "cohort_label": "ADNI tau180",
        "cohort": "ADNI",
        "window": "180d",
        "endpoint_family": "tau_binary",
        "endpoint_label": "tau positivity",
        "src": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau180/subject_level_input_table.csv",
    },
    "oasis3_tau90": {
        "cohort_label": "OASIS3 tau90",
        "cohort": "OASIS3",
        "window": "90d",
        "endpoint_family": "tau_binary",
        "endpoint_label": "tau positivity (derived)",
        "src": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau90/subject_level_input_table.csv",
    },
    "oasis3_tau180": {
        "cohort_label": "OASIS3 tau180",
        "cohort": "OASIS3",
        "window": "180d",
        "endpoint_family": "tau_binary",
        "endpoint_label": "tau positivity (derived)",
        "src": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau180/subject_level_input_table.csv",
    },
}

OUTDIR = Path("/project/aereditato/abhat/adni-mri-classification/crosscohort_matched_rerun")
OUTDIR.mkdir(parents=True, exist_ok=True)

DEMO_NUM = ["age_h", "education_years_h", "apoe_e4_count_h", "apoe_e4_carrier_h"]
DEMO_CAT = ["sex_h"]
SEV_NUM = ["cdr_global_h", "cdr_sumboxes_h", "faq_total_h", "moca_h"]


def read_csv_any(path):
    return pd.read_csv(path, low_memory=False, compression="infer")


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


def available_features(df, feature_list):
    return {c for c in feature_list if c in df.columns and df[c].notna().sum() > 0}


def evaluate_experiment(df, cohort_key, cohort_meta, experiment, num_cols, cat_cols, exp_outdir):
    use_num = [c for c in num_cols if c in df.columns]
    use_cat = [c for c in cat_cols if c in df.columns]
    feature_cols = use_num + use_cat

    if len(feature_cols) == 0:
        return None

    X = df[feature_cols].copy()
    y = df["y_target"].astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        return None

    preproc = make_preproc(use_num, use_cat)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

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

    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
    }

    rows = []
    for model_name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preproc", preproc),
                ("model", model),
            ]
        )

        cvres = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        rows.append(
            {
                "cohort_key": cohort_key,
                "cohort_label": cohort_meta["cohort_label"],
                "cohort": cohort_meta["cohort"],
                "window": cohort_meta["window"],
                "endpoint_family": cohort_meta["endpoint_family"],
                "endpoint_label": cohort_meta["endpoint_label"],
                "experiment": experiment,
                "model": model_name,
                "n_rows": int(len(df)),
                "n_subjects": int(df["subject_id"].nunique()),
                "n_pos": int((df["y_target"] == 1).sum()),
                "n_neg": int((df["y_target"] == 0).sum()),
                "n_features_used": int(len(feature_cols)),
                "numeric_features_used": "|".join(use_num),
                "categorical_features_used": "|".join(use_cat),
                "roc_auc_mean": float(np.mean(cvres["test_roc_auc"])),
                "roc_auc_std": float(np.std(cvres["test_roc_auc"])),
                "ap_mean": float(np.mean(cvres["test_average_precision"])),
                "ap_std": float(np.std(cvres["test_average_precision"])),
                "bal_acc_mean": float(np.mean(cvres["test_balanced_accuracy"])),
                "bal_acc_std": float(np.std(cvres["test_balanced_accuracy"])),
                "f1_mean": float(np.mean(cvres["test_f1"])),
                "f1_std": float(np.std(cvres["test_f1"])),
            }
        )

    out = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    out.to_csv(exp_outdir / f"{cohort_key}_{experiment}_metrics.csv", index=False)
    return out


def get_best_metric(best_df, experiment, metric="roc_auc_mean"):
    sub = best_df[best_df["experiment"] == experiment]
    if len(sub) == 0:
        return np.nan
    return float(sub.iloc[0][metric])


def build_core_row(best_df, cohort_key, cohort_meta, df, shared_demo_num, shared_demo_cat, shared_sev_num):
    auc_ref = get_best_metric(best_df, "reference_combined")
    auc_sev = get_best_metric(best_df, "severity_only")
    auc_demo = get_best_metric(best_df, "demo_only")
    auc_minus_all_cdr = get_best_metric(best_df, "minus_all_cdr")
    auc_minus_fullstrip = auc_demo

    auc_minus_moca = get_best_metric(best_df, "minus_moca")
    auc_minus_faq = get_best_metric(best_df, "minus_faq")
    auc_minus_cdr_global = get_best_metric(best_df, "minus_cdr_global")
    auc_minus_cdr_sumboxes = get_best_metric(best_df, "minus_cdr_sumboxes")

    row = {
        "cohort_key": cohort_key,
        "cohort_label": cohort_meta["cohort_label"],
        "cohort": cohort_meta["cohort"],
        "window": cohort_meta["window"],
        "endpoint_family": cohort_meta["endpoint_family"],
        "endpoint_label": cohort_meta["endpoint_label"],
        "n_subjects": int(df["subject_id"].nunique()),
        "n_pos": int((df["y_target"] == 1).sum()),
        "n_neg": int((df["y_target"] == 0).sum()),
        "pos_rate": float((df["y_target"] == 1).mean()),
        "available_demo_features": "|".join(shared_demo_num + shared_demo_cat),
        "available_severity_features": "|".join(shared_sev_num),
        "auc_ref": auc_ref,
        "auc_severity": auc_sev,
        "auc_demo": auc_demo,
        "auc_minus_all_cdr": auc_minus_all_cdr,
        "auc_minus_moca": auc_minus_moca,
        "auc_minus_faq": auc_minus_faq,
        "auc_minus_cdr_global": auc_minus_cdr_global,
        "auc_minus_cdr_sumboxes": auc_minus_cdr_sumboxes,
        "auc_minus_fullstrip": auc_minus_fullstrip,
        "severity_retention_auc_frac": float(auc_sev / auc_ref) if pd.notna(auc_sev) and pd.notna(auc_ref) and auc_ref != 0 else np.nan,
        "demo_retention_auc_frac": float(auc_demo / auc_ref) if pd.notna(auc_demo) and pd.notna(auc_ref) and auc_ref != 0 else np.nan,
        "full_strip_retention_auc_frac": float(auc_minus_fullstrip / auc_ref) if pd.notna(auc_minus_fullstrip) and pd.notna(auc_ref) and auc_ref != 0 else np.nan,
        "cdr_block_drop_auc": float(auc_ref - auc_minus_all_cdr) if pd.notna(auc_ref) and pd.notna(auc_minus_all_cdr) else np.nan,
        "full_strip_drop_auc": float(auc_ref - auc_minus_fullstrip) if pd.notna(auc_ref) and pd.notna(auc_minus_fullstrip) else np.nan,
        "moca_unique_drop_auc": float(auc_ref - auc_minus_moca) if pd.notna(auc_ref) and pd.notna(auc_minus_moca) else np.nan,
        "faq_unique_drop_auc": float(auc_ref - auc_minus_faq) if pd.notna(auc_ref) and pd.notna(auc_minus_faq) else np.nan,
        "cdr_global_unique_drop_auc": float(auc_ref - auc_minus_cdr_global) if pd.notna(auc_ref) and pd.notna(auc_minus_cdr_global) else np.nan,
        "cdr_sumboxes_unique_drop_auc": float(auc_ref - auc_minus_cdr_sumboxes) if pd.notna(auc_ref) and pd.notna(auc_minus_cdr_sumboxes) else np.nan,
        "severity_minus_demo_auc": float(auc_sev - auc_demo) if pd.notna(auc_sev) and pd.notna(auc_demo) else np.nan,
        "note": "Matched-feature rerun using only shared ADNI/OASIS harmonized columns.",
    }
    return row


def main():
    print("=" * 100)
    print("LOAD SUBJECT-LEVEL TABLES")
    print("=" * 100)

    dfs = {}
    availability_rows = []

    for cohort_key, meta in COHORTS.items():
        df = read_csv_any(meta["src"]).copy()
        df = df[df["y_target"].notna()].copy()

        if "subject_id" not in df.columns:
            raise KeyError(f"{cohort_key}: missing subject_id")
        if "y_target" not in df.columns:
            raise KeyError(f"{cohort_key}: missing y_target")

        dfs[cohort_key] = df

        demo_num_avail = available_features(df, DEMO_NUM)
        demo_cat_avail = available_features(df, DEMO_CAT)
        sev_avail = available_features(df, SEV_NUM)

        for fam, feats in [
            ("demo_num", DEMO_NUM),
            ("demo_cat", DEMO_CAT),
            ("severity", SEV_NUM),
        ]:
            for f in feats:
                availability_rows.append(
                    {
                        "cohort_key": cohort_key,
                        "cohort_label": meta["cohort_label"],
                        "feature_family": fam,
                        "feature": f,
                        "present": int(f in df.columns),
                        "nonmissing_n": int(df[f].notna().sum()) if f in df.columns else 0,
                    }
                )

        print(f"\n[{cohort_key}] {meta['cohort_label']}")
        print(f"rows={len(df)} subjects={df['subject_id'].nunique()} pos={int((df['y_target']==1).sum())} neg={int((df['y_target']==0).sum())}")
        print("demo numeric available:", sorted(demo_num_avail))
        print("demo categorical available:", sorted(demo_cat_avail))
        print("severity available:", sorted(sev_avail))

    availability_df = pd.DataFrame(availability_rows)
    availability_df.to_csv(OUTDIR / "feature_availability_by_cohort.csv", index=False)

    shared_demo_num = sorted(set.intersection(*[available_features(dfs[k], DEMO_NUM) for k in COHORTS]))
    shared_demo_cat = sorted(set.intersection(*[available_features(dfs[k], DEMO_CAT) for k in COHORTS]))
    shared_sev_num = sorted(set.intersection(*[available_features(dfs[k], SEV_NUM) for k in COHORTS]))

    manifest = {
        "cohorts": {k: COHORTS[k]["src"] for k in COHORTS},
        "shared_demo_num": shared_demo_num,
        "shared_demo_cat": shared_demo_cat,
        "shared_severity_num": shared_sev_num,
        "note": "Shared features are defined as harmonized columns with at least one non-missing value in every ADNI/OASIS tau cohort.",
    }

    with open(OUTDIR / "shared_feature_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 100)
    print("SHARED FEATURES USED FOR MATCHED RERUN")
    print("=" * 100)
    print("shared_demo_num :", shared_demo_num)
    print("shared_demo_cat :", shared_demo_cat)
    print("shared_severity :", shared_sev_num)

    experiments = []
    experiments.append(("reference_combined", shared_demo_num + shared_sev_num, shared_demo_cat))
    experiments.append(("demo_only", shared_demo_num, shared_demo_cat))
    experiments.append(("severity_only", shared_sev_num, []))

    if any(c.startswith("cdr_") for c in shared_sev_num):
        experiments.append(
            (
                "minus_all_cdr",
                shared_demo_num + [c for c in shared_sev_num if not c.startswith("cdr_")],
                shared_demo_cat,
            )
        )

    if "cdr_global_h" in shared_sev_num:
        experiments.append(
            (
                "minus_cdr_global",
                shared_demo_num + [c for c in shared_sev_num if c != "cdr_global_h"],
                shared_demo_cat,
            )
        )

    if "cdr_sumboxes_h" in shared_sev_num:
        experiments.append(
            (
                "minus_cdr_sumboxes",
                shared_demo_num + [c for c in shared_sev_num if c != "cdr_sumboxes_h"],
                shared_demo_cat,
            )
        )

    if "faq_total_h" in shared_sev_num:
        experiments.append(
            (
                "minus_faq",
                shared_demo_num + [c for c in shared_sev_num if c != "faq_total_h"],
                shared_demo_cat,
            )
        )

    if "moca_h" in shared_sev_num:
        experiments.append(
            (
                "minus_moca",
                shared_demo_num + [c for c in shared_sev_num if c != "moca_h"],
                shared_demo_cat,
            )
        )

    # de-duplicate experiments by exact feature-set signature
    dedup = []
    seen = set()
    for name, num_cols, cat_cols in experiments:
        sig = (tuple(num_cols), tuple(cat_cols))
        if sig in seen:
            continue
        seen.add(sig)
        dedup.append((name, num_cols, cat_cols))
    experiments = dedup

    all_metrics = []
    core_rows = []

    for cohort_key, meta in COHORTS.items():
        df = dfs[cohort_key]
        cohort_outdir = OUTDIR / cohort_key
        cohort_outdir.mkdir(parents=True, exist_ok=True)

        cohort_metrics = []

        for exp_name, num_cols, cat_cols in experiments:
            print("\n" + "=" * 100)
            print(f"RUN {cohort_key} | {exp_name}")
            print("=" * 100)
            print("num:", num_cols)
            print("cat:", cat_cols)

            met = evaluate_experiment(
                df=df,
                cohort_key=cohort_key,
                cohort_meta=meta,
                experiment=exp_name,
                num_cols=num_cols,
                cat_cols=cat_cols,
                exp_outdir=cohort_outdir,
            )
            if met is not None:
                cohort_metrics.append(met)
                print(
                    met[
                        [
                            "experiment",
                            "model",
                            "n_features_used",
                            "roc_auc_mean",
                            "ap_mean",
                            "bal_acc_mean",
                            "f1_mean",
                        ]
                    ].to_string(index=False)
                )

        if not cohort_metrics:
            raise RuntimeError(f"No experiments completed for {cohort_key}")

        cohort_all = pd.concat(cohort_metrics, ignore_index=True)
        cohort_all.to_csv(cohort_outdir / "all_metrics.csv", index=False)

        best = (
            cohort_all.sort_values(["experiment", "roc_auc_mean"], ascending=[True, False])
            .drop_duplicates(subset=["experiment"], keep="first")
            .reset_index(drop=True)
        )
        best.to_csv(cohort_outdir / "best_model_per_experiment.csv", index=False)

        all_metrics.append(cohort_all)
        core_rows.append(
            build_core_row(
                best_df=best,
                cohort_key=cohort_key,
                cohort_meta=meta,
                df=df,
                shared_demo_num=shared_demo_num,
                shared_demo_cat=shared_demo_cat,
                shared_sev_num=shared_sev_num,
            )
        )

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_metrics_df.to_csv(OUTDIR / "all_metrics.csv", index=False)

    best_global = (
        all_metrics_df.sort_values(["cohort_key", "experiment", "roc_auc_mean"], ascending=[True, True, False])
        .drop_duplicates(subset=["cohort_key", "experiment"], keep="first")
        .reset_index(drop=True)
    )
    best_global.to_csv(OUTDIR / "best_model_per_experiment.csv", index=False)

    core_df = pd.DataFrame(core_rows).sort_values(["cohort", "window"]).reset_index(drop=True)
    core_df.to_csv(OUTDIR / "matched_core_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("MATCHED-CORE SUMMARY")
    print("=" * 100)
    print(
        core_df[
            [
                "cohort_label",
                "n_subjects",
                "n_pos",
                "pos_rate",
                "available_demo_features",
                "available_severity_features",
                "auc_ref",
                "auc_severity",
                "auc_demo",
                "auc_minus_all_cdr",
                "auc_minus_fullstrip",
                "severity_retention_auc_frac",
                "demo_retention_auc_frac",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(" ", OUTDIR / "feature_availability_by_cohort.csv")
    print(" ", OUTDIR / "shared_feature_manifest.json")
    print(" ", OUTDIR / "all_metrics.csv")
    print(" ", OUTDIR / "best_model_per_experiment.csv")
    print(" ", OUTDIR / "matched_core_summary.csv")


if __name__ == "__main__":
    main()