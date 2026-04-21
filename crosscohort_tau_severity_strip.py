# crosscohort_tau_severity_strip.py
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import re

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

try:
    from sklearn.mixture import GaussianMixture
    HAVE_GMM = True
except Exception:
    HAVE_GMM = False


RANDOM_STATE = 42
N_SPLITS = 5
MAX_MISSING_PCT = 35.0
ADNI_MASTER = "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/adni_master_visits_06Mar2026.csv.gz"
DATASETS = {
    "adni_tau90": {
        "kind": "adni",
        "src": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/cohort_taupos_plasma_amy_mri_90d.csv",
        "outdir": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau90",
    },
    "adni_tau180": {
        "kind": "adni",
        "src": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/cohort_taupos_plasma_amy_mri_180d.csv",
        "outdir": "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/severity_strip_tau180",
    },
    "oasis3_tau90": {
        "kind": "oasis3",
        "src": "/project/aereditato/abhat/oasis/phase0/oasis3_clinical_master_v7_tau90_amy90_fs90_fixed.csv",
        "outdir": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau90",
    },
    "oasis3_tau180": {
        "kind": "oasis3",
        "src": "/project/aereditato/abhat/oasis/phase0/oasis3_clinical_master_v7_tau180_amy180_fs180_fixed.csv",
        "outdir": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau180",
    },
}


def read_csv_any(path):
    path = Path(path)
    return pd.read_csv(path, low_memory=False, compression="infer")


def pick_col(df, candidates, required=False, label="column"):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Could not find required {label}. Tried: {candidates}")
    return None


def to_num(s):
    return pd.to_numeric(s, errors="coerce")




def coerce_binary_target(s):
    if s is None:
        return pd.Series(dtype="float")
    s_num = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype="float")

    out.loc[s_num == 0] = 0.0
    out.loc[s_num == 1] = 1.0

    s_str = s.astype(str).str.strip().str.upper()
    pos = {"T+", "POS", "POSITIVE", "TAU+", "1", "TRUE", "YES"}
    neg = {"T-", "NEG", "NEGATIVE", "TAU-", "0", "FALSE", "NO"}

    out.loc[s_str.isin(pos)] = 1.0
    out.loc[s_str.isin(neg)] = 0.0
    return out


def gmm_midpoint_threshold(x):
    x = pd.Series(x).dropna().astype(float)
    if len(x) < 25 or not HAVE_GMM:
        thr = float(x.median())
        return {
            "method": "median_fallback",
            "threshold": thr,
            "n": int(len(x)),
            "means": [],
            "weights": [],
        }

    arr = x.to_numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE, n_init=10)
    gmm.fit(arr)

    means = gmm.means_.flatten()
    weights = gmm.weights_.flatten()
    order = np.argsort(means)
    means = means[order]
    weights = weights[order]
    thr = float((means[0] + means[1]) / 2.0)

    return {
        "method": "gmm_midpoint",
        "threshold": thr,
        "n": int(len(x)),
        "means": [float(m) for m in means],
        "weights": [float(w) for w in weights],
    }


def best_one_row_per_subject(df, sid_col, sort_cols):
    tmp = df.copy()
    by = [sid_col] + sort_cols
    tmp = tmp.sort_values(by=by, kind="mergesort")
    tmp = tmp.drop_duplicates(subset=[sid_col], keep="first").reset_index(drop=True)
    return tmp

def _norm_col(x):
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())

def _find_col(df, aliases):
    cols = list(df.columns)
    norm_pairs = [(_norm_col(c), c) for c in cols]
    norm_map = {nc: c for nc, c in norm_pairs}

    # 1) exact raw match
    for a in aliases:
        if a in df.columns:
            return a

    # 2) exact normalized match
    for a in aliases:
        na = _norm_col(a)
        if na in norm_map:
            return norm_map[na]

    # 3) unique suffix match
    for a in aliases:
        na = _norm_col(a)
        hits = [c for nc, c in norm_pairs if nc.endswith(na)]
        if len(hits) == 1:
            return hits[0]

    # 4) unique substring match
    for a in aliases:
        na = _norm_col(a)
        hits = [c for nc, c in norm_pairs if na in nc]
        if len(hits) == 1:
            return hits[0]

    return None

def _num(s):
    x = pd.to_numeric(s, errors="coerce")
    # common coded missings across cohorts
    x = x.replace([
        -4, -3, -2, -1,
        88, 888, 8888,
        99, 999, 9999,
        999.0, 9999.0,
    ], np.nan)
    return x

def _parse_apoe_e4_count(s):
    """
    Convert APOE representations into E4 allele count {0,1,2}.
    Handles:
      - numeric counts already encoded as 0/1/2
      - genotype strings like '3/3', '3/4', '4/4', '2/4', 'E3/E4'
    """
    if s is None:
        return pd.Series(dtype="float")

    out = pd.Series(np.nan, index=s.index, dtype="float")

    # Case 1: already numeric count
    num = pd.to_numeric(s, errors="coerce")
    out.loc[num.isin([0, 1, 2])] = num.loc[num.isin([0, 1, 2])].astype(float)

    # Case 2: genotype strings
    ss = s.astype(str).str.upper().str.strip()
    ss = ss.replace({
        "": np.nan,
        "NAN": np.nan,
        "NONE": np.nan,
        "-4": np.nan,
        "-3": np.nan,
        "-2": np.nan,
        "-1": np.nan,
        "UNK": np.nan,
        "UNKNOWN": np.nan,
    })

    def count_e4(x):
        if pd.isna(x):
            return np.nan
        vals = re.findall(r"[234]", str(x))
        if len(vals) == 0:
            return np.nan
        return float(sum(v == "4" for v in vals))

    parsed = ss.map(count_e4)
    out = out.fillna(parsed)

    # keep only 0/1/2
    out = out.where(out.isin([0.0, 1.0, 2.0]), np.nan)
    return out

def _harmonize_sex(s):
    if s is None:
        return pd.Series(np.nan, index=[])

    out = s.astype(str).str.strip().str.upper()
    out = out.replace({
        "1": "M", "1.0": "M",
        "2": "F", "2.0": "F",
        "MALE": "M", "FEMALE": "F",
        "M": "M", "F": "F",
        "0": np.nan, "0.0": np.nan,
        "88": np.nan, "88.0": np.nan,
        "-4": np.nan, "-4.0": np.nan,
        "NAN": np.nan, "NONE": np.nan,
    })
    out = out.where(out.isin(["M", "F"]), np.nan)
    return out

def add_shared_features(df):
    df = df.copy()
    meta = {"source_columns": {}}

    age_col = _find_col(df, [
        "age_h", "AGE", "age", "NACCAGE", "age_baseline", "age_at_visit",
        "PTAGE", "AGE_YRS", "AGEYEARS", "DEM_AGE"
    ])

    sex_col = _find_col(df, [
        "sex_h", "SEX", "sex", "PTGENDER", "DEM_PTGENDER", "gender", "GENDER"
    ])

    educ_col = _find_col(df, [
        "education_years_h", "EDUC", "education_years", "education",
        "PTEDUCAT", "DEM_PTEDUCAT"
    ])

    apoe_col = _find_col(df, [
        "apoe_e4_count_h",
        "apoe_e4_count",
        "APOE_E4_COUNT",
        "APOE4",
        "APOE",
        "APOE_GENOTYPE",
        "NACCNE4S",
    ])

    apoe_car_col = _find_col(df, [
        "apoe_e4_carrier_h", "apoe_e4_carrier", "APOE_E4_CARRIER",
        "APOE4_BIN", "E4_CARRIER"
    ])

    cdrg_col = _find_col(df, [
        "cdr_global_h", "cdr_global", "CDRGLOB", "CDGLOBAL", "CDRGLOBAL",
        "CDR_CDGLOBAL"
    ])

    cdrsb_col = _find_col(df, [
        "cdr_sumboxes_h", "cdr_sumboxes", "CDRSUM", "CDRSB",
        "CDR_SUM", "CDR_SUMBOXES", "CDR_CDRSB"
    ])

    faq_col = _find_col(df, [
        "faq_total_h", "faq_total", "FAQTOTAL", "FAQ_TOTAL",
        "FAQ_FAQTOTAL"
    ])

    moca_col = _find_col(df, [
        "moca_h", "moca", "MOCA", "MOCA_TOTAL", "MOCA_MOCA",
        "NACCMOCA", "MOCATOTS"
    ])

    meta["source_columns"] = {
        "age_h": age_col,
        "sex_h": sex_col,
        "education_years_h": educ_col,
        "apoe_e4_count_h": apoe_col,
        "apoe_e4_carrier_h": apoe_car_col,
        "cdr_global_h": cdrg_col,
        "cdr_sumboxes_h": cdrsb_col,
        "faq_total_h": faq_col,
        "moca_h": moca_col,
    }

    if age_col is not None:
        df["age_h"] = _num(df[age_col])

    if sex_col is not None:
        df["sex_h"] = _harmonize_sex(df[sex_col])

    if educ_col is not None:
        df["education_years_h"] = _num(df[educ_col])

    if apoe_col is not None:
        # robustly parse both numeric-count and genotype-string representations
        df["apoe_e4_count_h"] = _parse_apoe_e4_count(df[apoe_col])

    if "apoe_e4_count_h" in df.columns:
        df["apoe_e4_carrier_h"] = np.where(
            df["apoe_e4_count_h"].notna(),
            (df["apoe_e4_count_h"] > 0).astype(float),
            np.nan
        )
    elif apoe_car_col is not None:
        x = _num(df[apoe_car_col])
        df["apoe_e4_carrier_h"] = np.where(x.notna(), (x > 0).astype(float), np.nan)

    if cdrg_col is not None:
        df["cdr_global_h"] = _num(df[cdrg_col])

    if cdrsb_col is not None:
        df["cdr_sumboxes_h"] = _num(df[cdrsb_col])

    if faq_col is not None:
        df["faq_total_h"] = _num(df[faq_col])

    if moca_col is not None:
        df["moca_h"] = _num(df[moca_col])

    return df, meta


def build_adni_subject_table(src):
    labels = read_csv_any(src)
    master = read_csv_any(ADNI_MASTER)
    print("\nADNI master likely feature columns:")
    cand = [c for c in master.columns if any(k in c.upper() for k in [
        "AGE", "SEX", "GENDER", "EDUC", "APOE", "CDR", "FAQ", "MOCA"
    ])]
    print(cand[:120])

    master_tmp, feat_meta_tmp = add_shared_features(master)
    print("\nRecovered ADNI feature source columns:")
    print(feat_meta_tmp["source_columns"])
    print("\nNon-missing in master after harmonization:")
    for c in ["age_h","sex_h","education_years_h","apoe_e4_count_h","apoe_e4_carrier_h",
            "cdr_global_h","cdr_sumboxes_h","faq_total_h","moca_h"]:
        print(c, int(master_tmp[c].notna().sum()) if c in master_tmp.columns else "missing_col")

    sid_lab = pick_col(labels, ["subject_id", "RID", "PTID", "Subject", "RIDNUM"], required=True, label="ADNI label subject id")
    sid_mas = pick_col(master, ["subject_id", "RID", "PTID", "Subject", "RIDNUM"], required=True, label="ADNI master subject id")

    target_col = pick_col(
        labels,
        ["tau_pos", "T_pos", "TAU_POS", "tau_positive"],
        required=True,
        label="ADNI tau target",
    )

    # Try to recover the visit key from the final cohort file
    date_lab = pick_col(
        labels,
        ["exam_date", "EXAMDATE", "visit_date", "VISDATE", "mri_date", "amy_date", "tau_date"],
        required=False,
        label="ADNI label date",
    )
    date_mas = pick_col(
        master,
        ["exam_date", "EXAMDATE", "visit_date", "VISDATE"],
        required=False,
        label="ADNI master date",
    )

    labels = labels.copy()
    labels["y_target"] = coerce_binary_target(labels[target_col])
    labels = labels[labels["y_target"].notna()].copy()

    # Harmonize BOTH label-side cohort file and master
    labels, feat_meta_labels = add_shared_features(labels)
    master, feat_meta_master = add_shared_features(master)

    shared_cols = [
        "age_h", "sex_h", "education_years_h", "apoe_e4_count_h",
        "apoe_e4_carrier_h", "cdr_global_h", "cdr_sumboxes_h",
        "faq_total_h", "moca_h"
    ]

    # Keep label-side harmonized features if they already exist there
    keep_cols = [sid_lab, target_col, "y_target"]
    if date_lab:
        keep_cols.append(date_lab)
    keep_cols += [c for c in shared_cols if c in labels.columns]
    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in labels.columns]))
    labels = labels[keep_cols].copy()

    master_core_cols = [
    c for c in [
        "age_h", "sex_h", "education_years_h", "apoe_e4_count_h",
        "apoe_e4_carrier_h", "cdr_global_h", "cdr_sumboxes_h",
        "faq_total_h", "moca_h"
    ]
    if c in master.columns
]

    if master_core_cols:
        master["_missing_core"] = master[master_core_cols].isna().sum(axis=1)
    else:
        master["_missing_core"] = 999

    feat_meta = {
        "labels_source_columns": feat_meta_labels["source_columns"],
        "master_source_columns": feat_meta_master["source_columns"],
    }

    # Exact subject + date merge when possible
    if date_lab is not None and date_mas is not None:
        labels["_join_date"] = pd.to_datetime(labels[date_lab], errors="coerce")
        master["_join_date"] = pd.to_datetime(master[date_mas], errors="coerce")

        merged = labels.merge(
            master,
            left_on=[sid_lab, "_join_date"],
            right_on=[sid_mas, "_join_date"],
            how="left",
            suffixes=("", "_master"),
        )
    else:
        merged = labels.merge(
            master,
            left_on=sid_lab,
            right_on=sid_mas,
            how="left",
            suffixes=("", "_master"),
        )
    for c in shared_cols:
        master_c = f"{c}_master"
        if c in merged.columns and master_c in merged.columns:
            merged[c] = merged[c].combine_first(merged[master_c])
        elif master_c in merged.columns and c not in merged.columns:
            merged[c] = merged[master_c]


    # If exact merge failed for some rows, fall back subject-only and choose best visit
    feature_cols = [
        "age_h", "sex_h", "education_years_h", "apoe_e4_count_h",
        "apoe_e4_carrier_h", "cdr_global_h", "cdr_sumboxes_h",
        "faq_total_h", "moca_h"
    ]
    present_feature_cols = [c for c in feature_cols if c in merged.columns]

    if present_feature_cols:
        matched_mask = merged[present_feature_cols].notna().any(axis=1)
    else:
        matched_mask = pd.Series(False, index=merged.index)
    unmatched = merged.loc[~matched_mask, [sid_lab, "y_target"] + ([date_lab] if date_lab else [])].copy()

    if len(unmatched) > 0:
        fallback_parts = []

        # Prepare master copy for fallback ranking
        m2 = master.copy()
        if date_mas is not None:
            m2["_master_date"] = pd.to_datetime(m2[date_mas], errors="coerce")

        for _, row in unmatched.iterrows():
            sid_val = row[sid_lab]
            cand = m2[m2[sid_mas] == sid_val].copy()
            if cand.empty:
                continue

            if date_lab is not None and date_mas is not None:
                lab_date = pd.to_datetime(row[date_lab], errors="coerce")
                cand["_abs_date_diff"] = (cand["_master_date"] - lab_date).abs().dt.days
                cand["_abs_date_diff"] = pd.to_numeric(cand["_abs_date_diff"], errors="coerce").fillna(10**9)
            else:
                cand["_abs_date_diff"] = 10**9

            cand = cand.sort_values(["_abs_date_diff", "_missing_core"], kind="mergesort")
            best = cand.head(1).copy()

            for c in [sid_lab, "y_target"] + ([date_lab] if date_lab else []):
                best[c] = row[c]

            fallback_parts.append(best)

        if fallback_parts:
            fallback = pd.concat(fallback_parts, ignore_index=True)

            # Keep already matched exact rows
            matched = merged.loc[matched_mask].copy()

            # Remove unmatched rows from exact merged and replace with fallback
            merged = pd.concat([matched, fallback], ignore_index=True, sort=False)

    # Final subject-level collapse
    if date_lab and date_lab in merged.columns:
        merged["_date_sort"] = pd.to_datetime(merged[date_lab], errors="coerce")
    else:
        merged["_date_sort"] = pd.NaT

    final_core_cols = [
        c for c in [
            "age_h", "sex_h", "education_years_h", "apoe_e4_count_h",
            "apoe_e4_carrier_h", "cdr_global_h", "cdr_sumboxes_h",
            "faq_total_h", "moca_h"
        ]
        if c in merged.columns
    ]

    if final_core_cols:
        merged["_missing_core_final"] = merged[final_core_cols].isna().sum(axis=1)
    else:
        merged["_missing_core_final"] = 999

    subj = merged.sort_values(
        ["_missing_core_final", "_date_sort"],
        kind="mergesort"
    ).drop_duplicates(subset=[sid_lab], keep="first").reset_index(drop=True)

    subj = subj.rename(columns={sid_lab: "subject_id"})

    target_meta = {
        "target_mode": "existing_tau_pos_from_final_cohort",
        "target_source_col": target_col,
        "feature_source": ADNI_MASTER,
        "label_source": src,
        "label_date_col": date_lab,
        "master_date_col": date_mas,
    }

    return subj, feat_meta, target_meta


def build_oasis_subject_table(src):
    df = read_csv_any(src)
    sid = pick_col(df, ["OASISID", "subject_id"], required=True, label="subject id")
    date_col = pick_col(df, ["days_to_visit", "visit_date"], required=False, label="visit date")

    df, feat_meta = add_shared_features(df)

    tau_col = pick_col(
        df,
        [
            "Tauopathy",
            "Braak5_6",
            "PET_fSUVR_TOT_CORTMEAN",
            "PET_fSUVR_TOT_CTX_ENTORHINAL",
            "PET_fSUVR_TOT_CTX_PARAHPCMPL",
            "PET_fSUVR_TOT_CTX_INFERTMP",
        ],
        required=True,
        label="oasis tau measure",
    )

    tau_day_diff_col = pick_col(df, ["tau_day_diff"], required=False, label="tau day diff")
    if tau_day_diff_col:
        df["_abs_tau_diff"] = to_num(df[tau_day_diff_col]).abs()
    else:
        df["_abs_tau_diff"] = np.inf

    if date_col and date_col in df.columns:
        df["_date_sort"] = to_num(df[date_col])
    else:
        df["_date_sort"] = np.inf

    core_cols = [
        c for c in [
            "age_h", "sex_h", "education_years_h", "apoe_e4_count_h",
            "cdr_global_h", "cdr_sumboxes_h", "faq_total_h", "moca_h"
        ]
        if c in df.columns
    ]

    if core_cols:
        df["_missing_core"] = df[core_cols].isna().sum(axis=1)
    else:
        df["_missing_core"] = 999

    df = df[df[tau_col].notna()].copy()
    subj = best_one_row_per_subject(
        df,
        sid_col=sid,
        sort_cols=["_abs_tau_diff", "_missing_core", "_date_sort"],
    )
    subj = subj.rename(columns={sid: "subject_id"})

    thr_info = gmm_midpoint_threshold(subj[tau_col])
    thr = thr_info["threshold"]
    subj["y_target"] = np.where(to_num(subj[tau_col]) >= thr, 1.0, 0.0)

    target_meta = {
        "target_mode": "derived_tau_pos_gmm",
        "target_source_col": tau_col,
        "threshold_info": thr_info,
    }

    return subj, feat_meta, target_meta


def build_feature_lists(df, numeric_features, categorical_features, max_missing_pct):
    usable_num, usable_cat, dropped = [], [], []

    for c in numeric_features:
        if c in df.columns:
            miss = 100.0 * df[c].isna().mean()
            if miss <= max_missing_pct:
                usable_num.append(c)
            else:
                dropped.append((c, round(miss, 2), "high_missing_numeric"))

    for c in categorical_features:
        if c in df.columns:
            miss = 100.0 * df[c].isna().mean()
            if miss <= max_missing_pct:
                usable_cat.append(c)
            else:
                dropped.append((c, round(miss, 2), "high_missing_categorical"))

    return usable_num, usable_cat, dropped


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


def evaluate_experiment(df, dataset_name, experiment_name, num_req, cat_req, outdir):
    num_cols, cat_cols, dropped = build_feature_lists(df, num_req, cat_req, MAX_MISSING_PCT)
    feature_cols = num_cols + cat_cols
    if not feature_cols:
        return None

    X = df[feature_cols].copy()
    y = df["y_target"].astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        return None

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    preproc = make_preproc(num_cols, cat_cols)

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
                "dataset": dataset_name,
                "experiment": experiment_name,
                "model": model_name,
                "n_rows": int(len(df)),
                "n_subjects": int(df["subject_id"].nunique()),
                "n_features_requested": int(len(num_req) + len(cat_req)),
                "n_features_used": int(len(feature_cols)),
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
                "dropped_features": ";".join([f"{a}:{b}:{c}" for a, b, c in dropped]),
            }
        )

    met = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    met.to_csv(outdir / f"{experiment_name}_metrics.csv", index=False)
    return met


def run_all_experiments(df, dataset_name, outdir):
    num_demo = ["age_h", "education_years_h", "apoe_e4_count_h", "apoe_e4_carrier_h"]
    cat_demo = ["sex_h"]
    sev = ["cdr_global_h", "cdr_sumboxes_h", "faq_total_h", "moca_h"]

    available = [c for c in num_demo + cat_demo + sev if c in df.columns]
    print("\nAvailable harmonized columns:")
    print(available)
    available_set = set(df.columns)

    experiments = [
    ("reference_combined", [c for c in num_demo + sev if c in available_set], [c for c in cat_demo if c in available_set]),
    ("demo_only", [c for c in num_demo if c in available_set], [c for c in cat_demo if c in available_set]),
    ("severity_only", [c for c in sev if c in available_set], []),
    ("minus_cdr_global", [c for c in num_demo + [x for x in sev if x != "cdr_global_h"] if c in available_set], [c for c in cat_demo if c in available_set]),
    ("minus_cdr_sumboxes", [c for c in num_demo + [x for x in sev if x != "cdr_sumboxes_h"] if c in available_set], [c for c in cat_demo if c in available_set]),
    ("minus_all_cdr", [c for c in num_demo + [x for x in sev if x not in {"cdr_global_h", "cdr_sumboxes_h"}] if c in available_set], [c for c in cat_demo if c in available_set]),
    ("minus_faq", [c for c in num_demo + [x for x in sev if x != "faq_total_h"] if c in available_set], [c for c in cat_demo if c in available_set]),
    ("minus_moca", [c for c in num_demo + [x for x in sev if x != "moca_h"] if c in available_set], [c for c in cat_demo if c in available_set]),
    ("minus_cdr_faq", [c for c in num_demo + [x for x in sev if x not in {"cdr_global_h", "cdr_sumboxes_h", "faq_total_h"}] if c in available_set], [c for c in cat_demo if c in available_set]),
    ("minus_cdr_moca", [c for c in num_demo + [x for x in sev if x not in {"cdr_global_h", "cdr_sumboxes_h", "moca_h"}] if c in available_set], [c for c in cat_demo if c in available_set]),
    ("minus_faq_moca", [c for c in num_demo + [x for x in sev if x not in {"faq_total_h", "moca_h"}] if c in available_set], [c for c in cat_demo if c in available_set]),
    ("minus_cdr_faq_moca", [c for c in num_demo + [x for x in sev if x not in {"cdr_global_h", "cdr_sumboxes_h", "faq_total_h", "moca_h"}] if c in available_set], [c for c in cat_demo if c in available_set]),
    ]

    all_metrics = []
    for exp_name, num_req, cat_req in experiments:
        req = [c for c in num_req + cat_req if c in df.columns]
        ref_req = [c for c in (num_demo + sev + cat_demo) if c in df.columns]

        if exp_name != "reference_combined" and set(req) == set(ref_req):
            print(f"\nSkipping {exp_name}: no relevant columns available to remove.")
            continue

        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {dataset_name} | {exp_name}")
        print("=" * 80)
        print("Rows:", len(df))
        print("Subjects:", df["subject_id"].nunique())
        print("Target counts:")
        print(df["y_target"].value_counts(dropna=False).to_string())

        met = evaluate_experiment(df, dataset_name, exp_name, num_req, cat_req, outdir)
        if met is not None:
            print("\nCV metrics:")
            print(
                met[
                    [
                        "dataset",
                        "experiment",
                        "model",
                        "n_rows",
                        "n_subjects",
                        "n_features_used",
                        "roc_auc_mean",
                        "ap_mean",
                        "bal_acc_mean",
                        "f1_mean",
                    ]
                ].to_string(index=False)
            )
            all_metrics.append(met)

    if not all_metrics:
        raise RuntimeError("No experiments ran successfully.")

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_metrics_df.to_csv(outdir / "all_severity_strip_metrics.csv", index=False)

    best = (
        all_metrics_df.sort_values(["experiment", "roc_auc_mean"], ascending=[True, False])
        .drop_duplicates(subset=["experiment"], keep="first")
        .reset_index(drop=True)
    )
    best.to_csv(outdir / "best_model_per_experiment.csv", index=False)

    ref = best[best["experiment"] == "reference_combined"].iloc[0]
    delta = best.copy()
    delta["drop_in_roc_auc_vs_reference"] = ref["roc_auc_mean"] - delta["roc_auc_mean"]
    delta["drop_in_ap_vs_reference"] = ref["ap_mean"] - delta["ap_mean"]
    delta["drop_in_bal_acc_vs_reference"] = ref["bal_acc_mean"] - delta["bal_acc_mean"]
    delta["drop_in_f1_vs_reference"] = ref["f1_mean"] - delta["f1_mean"]
    delta.to_csv(outdir / "severity_strip_deltas_vs_reference.csv", index=False)

    return all_metrics_df, best, delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"LOAD DATASET: {args.dataset}")
    print("=" * 80)
    print("Source:", cfg["src"])

    if cfg["kind"] == "adni":
        subj, feat_meta, target_meta = build_adni_subject_table(cfg["src"])
    elif cfg["kind"] == "oasis3":
        subj, feat_meta, target_meta = build_oasis_subject_table(cfg["src"])
    else:
        raise ValueError(f"Unknown dataset kind: {cfg['kind']}")

    subj.to_csv(outdir / "subject_level_input_table.csv", index=False)

    print("\nSubject-level rows:", len(subj))
    print("Unique subjects:", subj["subject_id"].nunique())
    print("Target counts:")
    print(subj["y_target"].value_counts(dropna=False).to_string())

    meta = {
        "dataset": args.dataset,
        "src": cfg["src"],
        "outdir": str(outdir),
        "n_subject_rows": int(len(subj)),
        "n_subjects": int(subj["subject_id"].nunique()),
        "feature_column_map": feat_meta,
        "target_meta": target_meta,
    }
    with open(outdir / "build_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


    print("\nNon-missing counts in harmonized ADNI/OASIS features:")
    for c in ["age_h", "sex_h", "education_years_h", "apoe_e4_count_h",
            "apoe_e4_carrier_h", "cdr_global_h", "cdr_sumboxes_h",
            "faq_total_h", "moca_h"]:
        if c in subj.columns:
            print(f"{c}: {int(subj[c].notna().sum())} / {len(subj)}")

    all_metrics_df, best, delta = run_all_experiments(subj, args.dataset, outdir)

    print("\n" + "=" * 80)
    print("BEST MODEL PER EXPERIMENT")
    print("=" * 80)
    print(
        best[
            [
                "dataset",
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

    print("\n" + "=" * 80)
    print("DROP VS REFERENCE")
    print("=" * 80)
    print(
        delta[
            [
                "dataset",
                "experiment",
                "model",
                "roc_auc_mean",
                "drop_in_roc_auc_vs_reference",
                "drop_in_ap_vs_reference",
                "drop_in_bal_acc_vs_reference",
                "drop_in_f1_vs_reference",
            ]
        ]
        .sort_values("drop_in_roc_auc_vs_reference", ascending=False)
        .to_string(index=False)
    )

    summary = {
        "dataset": args.dataset,
        "best_reference_model": best.loc[best["experiment"] == "reference_combined", "model"].iloc[0],
        "best_reference_auc": float(best.loc[best["experiment"] == "reference_combined", "roc_auc_mean"].iloc[0]),
        "outdir": str(outdir),
    }
    with open(outdir / "severity_strip_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(" ", outdir / "subject_level_input_table.csv")
    print(" ", outdir / "build_metadata.json")
    print(" ", outdir / "all_severity_strip_metrics.csv")
    print(" ", outdir / "best_model_per_experiment.csv")
    print(" ", outdir / "severity_strip_deltas_vs_reference.csv")
    print(" ", outdir / "severity_strip_summary.json")


if __name__ == "__main__":
    main()