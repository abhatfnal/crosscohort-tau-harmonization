#!/usr/bin/env python3
"""
run_mismatch_decision_experiments_v2.py

Three decision experiments for the ADNI tau-clinical mismatch paper path:

1) Signed vs absolute mismatch
2) Same-subset nested comparisons
3) Adjusted hippocampus model

This version is robust to odd sex coding in DEM_PTGENDER and will
automatically drop sex from the baseline model if sex cannot be recovered.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
MASTER_PATH = Path("/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/adni_master_visits_06Mar2026.csv.gz")
ANCHOR_PATH = Path("/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/resilience_tau_anchor_followup24m.csv")
OUT_DIR = Path("/project/aereditato/abhat/ADNI/ARC_06Mar2026/output")

OUT_SCORED = OUT_DIR / "mismatch_decision_experiments_v2_scored.csv"
OUT_SUMMARY = OUT_DIR / "mismatch_decision_experiments_v2_summary.json"
OUT_MODELS = OUT_DIR / "mismatch_decision_experiments_v2_model_table.csv"


# =============================================================================
# HELPERS
# =============================================================================
def header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def compute_age_years(visit_date: pd.Series, dob: pd.Series) -> pd.Series:
    v = pd.to_datetime(visit_date, errors="coerce")
    b = pd.to_datetime(dob, errors="coerce")
    return (v - b).dt.days / 365.25


def safe_pearson(x: pd.Series, y: pd.Series):
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p), n


def safe_spearman(x: pd.Series, y: pd.Series):
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, n
    r, p = stats.spearmanr(x[mask], y[mask])
    return float(r), float(p), n


def fit_linear_with_standardized_coefs(X: pd.DataFrame, y: pd.Series):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(Xs, y)
    return model, scaler


def cv_scores(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    r2 = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
    mae = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error")

    pipe.fit(X, y)
    pred = pipe.predict(X)

    return {
        "cv_r2_mean": float(np.mean(r2)),
        "cv_r2_std": float(np.std(r2)),
        "cv_mae_mean": float(np.mean(mae)),
        "cv_mae_std": float(np.std(mae)),
        "train_r2": float(r2_score(y, pred)),
        "train_mae": float(mean_absolute_error(y, pred)),
    }


def ols_fit_report(df: pd.DataFrame, y_col: str, x_cols: list[str], label: str):
    use = [y_col] + x_cols
    d = df[use].dropna().copy()
    if len(d) < 20:
        return {
            "label": label,
            "n": int(len(d)),
            "status": "too_few_samples",
        }

    X = d[x_cols].astype(float)
    y = d[y_col].astype(float)

    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)

    return {
        "label": label,
        "n": int(len(d)),
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "features": x_cols,
        "coefficients": {c: float(b) for c, b in zip(x_cols, model.coef_)},
        "intercept": float(model.intercept_),
    }


def make_tertiles(series: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=object)
    valid = series.dropna()
    if len(valid) < 10 or valid.nunique() < 3:
        return out
    q = pd.qcut(valid, 3, labels=["Low", "Mid", "High"], duplicates="drop")
    out.loc[q.index] = q.astype(str)
    return out


def robust_encode_sex(series: pd.Series) -> pd.Series:
    """
    Returns:
        1.0 for male
        0.0 for female
        NaN otherwise
    Handles common codings:
        Male/Female, M/F, 1/2, 0/1, 1.0/2.0
    """
    s = series.copy()
    out = pd.Series(np.nan, index=s.index, dtype=float)

    # raw string form
    s_str = s.astype(str).str.strip().str.upper()

    male_tokens = {"M", "MALE", "1", "1.0"}
    female_tokens = {"F", "FEMALE", "2", "2.0", "0", "0.0"}

    out[s_str.isin(male_tokens)] = 1.0
    out[s_str.isin(female_tokens)] = 0.0

    # numeric fallback
    s_num = pd.to_numeric(s, errors="coerce")
    out[(out.isna()) & (s_num == 1)] = 1.0
    out[(out.isna()) & (s_num == 2)] = 0.0
    out[(out.isna()) & (s_num == 0)] = 0.0

    return out


# =============================================================================
# STEP 1: LOAD TAU-ANCHORED COHORT
# =============================================================================
header("STEP 1: LOAD TAU-ANCHORED COHORT")
df = pd.read_csv(ANCHOR_PATH, low_memory=False)
print(f"Loaded cohort: {len(df)} subjects")

numeric_cols = [
    "CDR_CDRSB",
    "delta_CDR_CDRSB",
    "MMSE_MMSCORE",
    "delta_MMSE_MMSCORE",
    "TAU6MM_META_TEMPORAL_SUVR",
    "AMY6MM_CENTILOIDS",
    "FS7_HIPPO_BILAT_ICVnorm",
    "WMH_TOTAL_WMH",
    "PLASMA_pT217_F",
    "PLASMA_NfL_Q",
    "PLASMA_GFAP_Q",
]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "APOE_GENOTYPE" in df.columns:
    df["apoe4_carrier"] = df["APOE_GENOTYPE"].astype(str).str.contains("4", na=False).astype(float)
    df.loc[df["APOE_GENOTYPE"].isna(), "apoe4_carrier"] = np.nan
else:
    df["apoe4_carrier"] = np.nan

df["VISCODE2_STD"] = df["VISCODE2"].astype(str).str.strip().str.lower()


# =============================================================================
# STEP 2: MERGE DEMOGRAPHICS
# =============================================================================
header("STEP 2: MERGE DEMOGRAPHICS FROM MASTER")
master_cols = pd.read_csv(MASTER_PATH, nrows=0).columns.tolist()
usecols = [c for c in ["PTID", "DEM_VISCODE", "DEM_VISDATE", "DEM_PTGENDER", "DEM_PTEDUCAT", "DEM_PTDOB"] if c in master_cols]
master = pd.read_csv(MASTER_PATH, usecols=usecols, low_memory=False)
print("Master demographic columns:", usecols)

master["DEM_VISCODE_STD"] = master["DEM_VISCODE"].astype(str).str.strip().str.lower()
master["AGE_DEM"] = compute_age_years(master["DEM_VISDATE"], master["DEM_PTDOB"])

# exact match by PTID + visitcode
demo_exact = (
    master[["PTID", "DEM_VISCODE_STD", "DEM_PTGENDER", "DEM_PTEDUCAT", "AGE_DEM"]]
    .drop_duplicates(subset=["PTID", "DEM_VISCODE_STD"], keep="first")
    .rename(columns={"DEM_VISCODE_STD": "VISCODE2_STD"})
)

df = df.merge(demo_exact, on=["PTID", "VISCODE2_STD"], how="left")

# fallback by PTID
demo_fb = (
    master.sort_values(["PTID", "DEM_VISDATE"])
    .groupby("PTID", as_index=False)
    .first()[["PTID", "DEM_PTGENDER", "DEM_PTEDUCAT", "AGE_DEM"]]
    .rename(columns={
        "DEM_PTGENDER": "DEM_PTGENDER_FB",
        "DEM_PTEDUCAT": "DEM_PTEDUCAT_FB",
        "AGE_DEM": "AGE_DEM_FB",
    })
)

df = df.merge(demo_fb, on="PTID", how="left")

df["SEX_RAW"] = df["DEM_PTGENDER"].fillna(df["DEM_PTGENDER_FB"])
df["EDUCATION"] = pd.to_numeric(df["DEM_PTEDUCAT"], errors="coerce")
df["EDUCATION"] = df["EDUCATION"].fillna(pd.to_numeric(df["DEM_PTEDUCAT_FB"], errors="coerce"))

df["AGE_FINAL"] = pd.to_numeric(df["AGE_DEM"], errors="coerce")
df["AGE_FINAL"] = df["AGE_FINAL"].fillna(pd.to_numeric(df["AGE_DEM_FB"], errors="coerce"))

# diagnose sex coding
print("\nRaw DEM_PTGENDER / fallback values (top 20):")
print(df["SEX_RAW"].astype(str).value_counts(dropna=False).head(20).to_string())

df["SEX_MALE"] = robust_encode_sex(df["SEX_RAW"])

print(f"\nWith sex:       {int(df['SEX_MALE'].notna().sum())}")
print(f"With education: {int(df['EDUCATION'].notna().sum())}")
print(f"With age:       {int(df['AGE_FINAL'].notna().sum())}")
print(f"With all demos: {int(df[['SEX_MALE', 'EDUCATION', 'AGE_FINAL']].notna().all(axis=1).sum())}")


# =============================================================================
# STEP 3: BASELINE MISMATCH MODEL
# =============================================================================
header("STEP 3: COMPUTE BASELINE MISMATCH")

pathology_features = ["TAU6MM_META_TEMPORAL_SUVR", "AMY6MM_CENTILOIDS"]
demo_candidates = ["AGE_FINAL", "EDUCATION", "SEX_MALE"]

# only keep demographic features that actually have enough data
demo_features = [c for c in demo_candidates if c in df.columns and df[c].notna().sum() >= 0.7 * len(df)]
mismatch_features = pathology_features + demo_features

print("Mismatch model features:", mismatch_features)

base_mask = df["CDR_CDRSB"].notna()
for c in mismatch_features:
    base_mask &= df[c].notna()

df_base = df.loc[base_mask].copy()
print(f"Complete cases for mismatch model: {len(df_base)}")

if len(df_base) < 20:
    raise RuntimeError(
        "Too few complete cases for the mismatch model after demographic filtering. "
        "Check the printed raw sex values and demographic merge."
    )

X_base = df_base[mismatch_features].astype(float)
y_base = df_base["CDR_CDRSB"].astype(float)

std_model, std_scaler = fit_linear_with_standardized_coefs(X_base, y_base)
print(f"Baseline mismatch model R² = {std_model.score(std_scaler.transform(X_base), y_base):.3f}")
print("Standardized coefficients:")
for feat, coef in zip(mismatch_features, std_model.coef_):
    print(f"  {feat:<25} {coef:+.3f}")

pred_pipe = make_pipeline(StandardScaler(), LinearRegression())
pred_pipe.fit(X_base, y_base)
df_base["CDR_expected"] = pred_pipe.predict(X_base)
df_base["mismatch_score"] = df_base["CDR_CDRSB"] - df_base["CDR_expected"]

df = df.merge(df_base[["PTID", "CDR_expected", "mismatch_score"]], on="PTID", how="left")

df["abs_mismatch"] = df["mismatch_score"].abs()
df["mismatch_pos"] = df["mismatch_score"].clip(lower=0)
df["mismatch_neg"] = (-df["mismatch_score"]).clip(lower=0)

ms = df["mismatch_score"].dropna()
print(f"Mismatch score: N={len(ms)}, mean={ms.mean():.3f}, std={ms.std():.3f}")


# =============================================================================
# DECISION EXPERIMENT 1
# =============================================================================
header("DECISION EXPERIMENT 1: SIGNED VS ABSOLUTE MISMATCH")
prog = df[df["delta_CDR_CDRSB"].notna() & df["mismatch_score"].notna()].copy()
print(f"Subjects with mismatch + future delta_CDR_CDRSB: {len(prog)}")

signed_r_p, signed_p_p, n1 = safe_pearson(prog["mismatch_score"], prog["delta_CDR_CDRSB"])
signed_r_s, signed_p_s, _ = safe_spearman(prog["mismatch_score"], prog["delta_CDR_CDRSB"])

abs_r_p, abs_p_p, n2 = safe_pearson(prog["abs_mismatch"], prog["delta_CDR_CDRSB"])
abs_r_s, abs_p_s, _ = safe_spearman(prog["abs_mismatch"], prog["delta_CDR_CDRSB"])

print("Primary outcome: delta_CDR_CDRSB")
print(f"  Signed mismatch   Pearson  r={signed_r_p:+.3f}, p={signed_p_p:.4g}, N={n1}")
print(f"  Signed mismatch   Spearman r={signed_r_s:+.3f}, p={signed_p_s:.4g}")
print(f"  Absolute mismatch Pearson  r={abs_r_p:+.3f}, p={abs_p_p:.4g}, N={n2}")
print(f"  Absolute mismatch Spearman r={abs_r_s:+.3f}, p={abs_p_s:.4g}")

exp1_models = []
for label, extra in [
    ("Base_pathology+demo", []),
    ("Base+signed_mismatch", ["mismatch_score"]),
    ("Base+abs_mismatch", ["abs_mismatch"]),
    ("Base+pos_neg_mismatch", ["mismatch_pos", "mismatch_neg"]),
]:
    feats = mismatch_features + extra
    rep = ols_fit_report(prog, "delta_CDR_CDRSB", feats, label)
    exp1_models.append(rep)

print("\nOLS descriptive fits for future delta_CDR_CDRSB:")
for rep in exp1_models:
    if rep.get("status") == "too_few_samples":
        print(f"  {rep['label']}: too few samples")
    else:
        print(f"  {rep['label']:<24} N={rep['n']:>3}  R²={rep['r2']:.3f}  MAE={rep['mae']:.3f}")


# =============================================================================
# DECISION EXPERIMENT 2
# =============================================================================
header("DECISION EXPERIMENT 2: SAME-SUBSET NESTED COMPARISONS")
model_rows = []

# MRI same-subset
mri_feats = ["FS7_HIPPO_BILAT_ICVnorm", "WMH_TOTAL_WMH"]
mri_cols = mismatch_features + ["mismatch_score", "abs_mismatch"] + mri_feats + ["delta_CDR_CDRSB"]
mri_subset = prog[mri_cols].dropna().copy()
print(f"MRI/WMH same-subset N = {len(mri_subset)}")

mri_tiers = {
    "MRI_subset_T1_path+demo": mismatch_features,
    "MRI_subset_T2_+signed": mismatch_features + ["mismatch_score"],
    "MRI_subset_T2b_+abs": mismatch_features + ["abs_mismatch"],
    "MRI_subset_T3_+MRI_WMH": mismatch_features + ["abs_mismatch"] + mri_feats,
}

for name, feats in mri_tiers.items():
    if len(mri_subset) < 30:
        break
    X = mri_subset[feats].astype(float)
    y = mri_subset["delta_CDR_CDRSB"].astype(float)
    scores = cv_scores(X, y, n_splits=5)
    model_rows.append({
        "experiment": "same_subset_mri",
        "model": name,
        "n": int(len(mri_subset)),
        "features": "|".join(feats),
        **scores,
    })
    print(f"{name:<28} CV R²={scores['cv_r2_mean']:+.3f}±{scores['cv_r2_std']:.2f}   CV MAE={scores['cv_mae_mean']:.3f}±{scores['cv_mae_std']:.2f}")

# Plasma same-subset
plasma_feats = ["PLASMA_pT217_F", "PLASMA_NfL_Q", "PLASMA_GFAP_Q", "apoe4_carrier"]
plasma_cols = mismatch_features + ["mismatch_score", "abs_mismatch"] + plasma_feats + ["delta_CDR_CDRSB"]
plasma_subset = prog[plasma_cols].dropna().copy()
print(f"\nPlasma/APOE same-subset N = {len(plasma_subset)}")

plasma_tiers = {
    "Plasma_subset_T1_path+demo": mismatch_features,
    "Plasma_subset_T2_+signed": mismatch_features + ["mismatch_score"],
    "Plasma_subset_T2b_+abs": mismatch_features + ["abs_mismatch"],
    "Plasma_subset_T4_+plasma": mismatch_features + ["abs_mismatch"] + plasma_feats,
}

for name, feats in plasma_tiers.items():
    if len(plasma_subset) < 30:
        break
    X = plasma_subset[feats].astype(float)
    y = plasma_subset["delta_CDR_CDRSB"].astype(float)
    scores = cv_scores(X, y, n_splits=5)
    model_rows.append({
        "experiment": "same_subset_plasma",
        "model": name,
        "n": int(len(plasma_subset)),
        "features": "|".join(feats),
        **scores,
    })
    print(f"{name:<28} CV R²={scores['cv_r2_mean']:+.3f}±{scores['cv_r2_std']:.2f}   CV MAE={scores['cv_mae_mean']:.3f}±{scores['cv_mae_std']:.2f}")


# =============================================================================
# DECISION EXPERIMENT 3
# =============================================================================
header("DECISION EXPERIMENT 3: ADJUSTED HIPPOCAMPUS MODEL")

# 3A. hippocampus explains mismatch
hippo_explain_feats = mismatch_features + ["FS7_HIPPO_BILAT_ICVnorm"]
hippo_explain = df[hippo_explain_feats + ["mismatch_score"]].dropna().copy()
print(f"Hippocampus->mismatch adjusted model N = {len(hippo_explain)}")

if len(hippo_explain) >= 20:
    X1 = hippo_explain[mismatch_features].astype(float)
    y1 = hippo_explain["mismatch_score"].astype(float)
    s1 = cv_scores(X1, y1, n_splits=5)

    X2 = hippo_explain[mismatch_features + ["FS7_HIPPO_BILAT_ICVnorm"]].astype(float)
    y2 = hippo_explain["mismatch_score"].astype(float)
    s2 = cv_scores(X2, y2, n_splits=5)

    print("Predict mismatch_score")
    print(f"  Base pathology+demo        CV R²={s1['cv_r2_mean']:+.3f}±{s1['cv_r2_std']:.2f}   MAE={s1['cv_mae_mean']:.3f}")
    print(f"  + Hippocampus/ICV          CV R²={s2['cv_r2_mean']:+.3f}±{s2['cv_r2_std']:.2f}   MAE={s2['cv_mae_mean']:.3f}")

    hippo_rep = ols_fit_report(
        hippo_explain,
        y_col="mismatch_score",
        x_cols=mismatch_features + ["FS7_HIPPO_BILAT_ICVnorm"],
        label="hippocampus_explains_mismatch"
    )

    model_rows.append({
        "experiment": "hippo_explains_mismatch",
        "model": "base",
        "n": int(len(hippo_explain)),
        "features": "|".join(mismatch_features),
        **s1,
    })
    model_rows.append({
        "experiment": "hippo_explains_mismatch",
        "model": "plus_hippo",
        "n": int(len(hippo_explain)),
        "features": "|".join(mismatch_features + ["FS7_HIPPO_BILAT_ICVnorm"]),
        **s2,
    })
else:
    hippo_rep = {"status": "too_few_samples"}

# 3B. hippocampus predicts future decline
hippo_prog_feats = mismatch_features + ["abs_mismatch", "FS7_HIPPO_BILAT_ICVnorm"]
hippo_prog = prog[hippo_prog_feats + ["delta_CDR_CDRSB"]].dropna().copy()
print(f"\nHippocampus->future decline adjusted model N = {len(hippo_prog)}")

if len(hippo_prog) >= 20:
    Xp1 = hippo_prog[mismatch_features + ["abs_mismatch"]].astype(float)
    yp1 = hippo_prog["delta_CDR_CDRSB"].astype(float)
    sp1 = cv_scores(Xp1, yp1, n_splits=5)

    Xp2 = hippo_prog[mismatch_features + ["abs_mismatch", "FS7_HIPPO_BILAT_ICVnorm"]].astype(float)
    yp2 = hippo_prog["delta_CDR_CDRSB"].astype(float)
    sp2 = cv_scores(Xp2, yp2, n_splits=5)

    print("Predict delta_CDR_CDRSB")
    print(f"  Base + abs_mismatch        CV R²={sp1['cv_r2_mean']:+.3f}±{sp1['cv_r2_std']:.2f}   MAE={sp1['cv_mae_mean']:.3f}")
    print(f"  + Hippocampus/ICV          CV R²={sp2['cv_r2_mean']:+.3f}±{sp2['cv_r2_std']:.2f}   MAE={sp2['cv_mae_mean']:.3f}")

    hippo_prog_rep = ols_fit_report(
        hippo_prog,
        y_col="delta_CDR_CDRSB",
        x_cols=mismatch_features + ["abs_mismatch", "FS7_HIPPO_BILAT_ICVnorm"],
        label="hippocampus_predicts_decline"
    )

    model_rows.append({
        "experiment": "hippo_predicts_decline",
        "model": "base_plus_abs",
        "n": int(len(hippo_prog)),
        "features": "|".join(mismatch_features + ["abs_mismatch"]),
        **sp1,
    })
    model_rows.append({
        "experiment": "hippo_predicts_decline",
        "model": "plus_hippo",
        "n": int(len(hippo_prog)),
        "features": "|".join(mismatch_features + ["abs_mismatch", "FS7_HIPPO_BILAT_ICVnorm"]),
        **sp2,
    })
else:
    hippo_prog_rep = {"status": "too_few_samples"}


# =============================================================================
# SAVE
# =============================================================================
header("STEP 6: SAVE OUTPUTS")
model_table = pd.DataFrame(model_rows)
model_table.to_csv(OUT_MODELS, index=False)
df.to_csv(OUT_SCORED, index=False)

summary = {
    "n_total": int(len(df)),
    "n_with_mismatch": int(df["mismatch_score"].notna().sum()),
    "n_with_prog": int(len(prog)),
    "mismatch_features": mismatch_features,
    "raw_sex_value_counts_top20": df["SEX_RAW"].astype(str).value_counts(dropna=False).head(20).to_dict(),
    "decision_experiment_1": {
        "signed_pearson_r": None if pd.isna(signed_r_p) else float(signed_r_p),
        "signed_pearson_p": None if pd.isna(signed_p_p) else float(signed_p_p),
        "signed_spearman_r": None if pd.isna(signed_r_s) else float(signed_r_s),
        "signed_spearman_p": None if pd.isna(signed_p_s) else float(signed_p_s),
        "abs_pearson_r": None if pd.isna(abs_r_p) else float(abs_r_p),
        "abs_pearson_p": None if pd.isna(abs_p_p) else float(abs_p_p),
        "abs_spearman_r": None if pd.isna(abs_r_s) else float(abs_r_s),
        "abs_spearman_p": None if pd.isna(abs_p_s) else float(abs_p_s),
        "ols_models": exp1_models,
    },
    "decision_experiment_2": model_table.to_dict(orient="records"),
    "decision_experiment_3": {
        "hippo_explains_mismatch": hippo_rep,
        "hippo_predicts_decline": hippo_prog_rep,
    },
}

with open(OUT_SUMMARY, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Saved scored cohort: {OUT_SCORED}")
print(f"Saved model table:   {OUT_MODELS}")
print(f"Saved summary JSON:  {OUT_SUMMARY}")
print("\nDone.")