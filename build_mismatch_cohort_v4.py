#!/usr/bin/env python3
"""
Tau-Clinical Mismatch Analysis v4
=================================

Research question
-----------------
Given similar tau/amyloid burden, why do some people have worse clinical
severity and faster decline, and can MRI / WMH / plasma explain that mismatch?

Primary endpoint:
    delta_CDR_CDRSB

Secondary endpoint:
    delta_MMSE_MMSCORE

What this script does
---------------------
1. Loads the tau-anchored follow-up cohort
2. Pulls demographics from the ARC master file
3. Computes a baseline mismatch score:
       baseline CDR-SB - expected CDR-SB from pathology + demographics
4. Tests whether mismatch predicts future decline
5. Compares nested prognostic models:
       pathology+demo
       + mismatch
       + MRI/WMH
       + plasma/APOE
6. Saves scored cohort + summary outputs
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
# CONFIGURATION
# =============================================================================
MASTER_PATH = Path("/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/adni_master_visits_06Mar2026.csv.gz")
ANCHOR_PATH = Path("/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/resilience_tau_anchor_followup24m.csv")
OUTPUT_DIR = Path("/project/aereditato/abhat/ADNI/ARC_06Mar2026/output")

OUT_SCORED = OUTPUT_DIR / "mismatch_analysis_v4_scored.csv"
OUT_RESULTS_JSON = OUTPUT_DIR / "mismatch_analysis_v4_summary.json"
OUT_TIER_CSV = OUTPUT_DIR / "mismatch_analysis_v4_tiers.csv"


# =============================================================================
# HELPERS
# =============================================================================
def log_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def normalize_viscode(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def compute_age_years(visit_date: pd.Series, dob: pd.Series) -> pd.Series:
    visit_dt = pd.to_datetime(visit_date, errors="coerce")
    dob_dt = pd.to_datetime(dob, errors="coerce")
    return (visit_dt - dob_dt).dt.days / 365.25


def safe_pearson(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p), n


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, n
    r, p = stats.spearmanr(x[mask], y[mask])
    return float(r), float(p), n


def make_tertiles(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    out = pd.Series(np.nan, index=series.index, dtype=object)
    if valid.nunique() < 3:
        return out
    try:
        labels = ["Low", "Mid", "High"]
        cut = pd.qcut(valid, 3, labels=labels, duplicates="drop")
        out.loc[cut.index] = cut.astype(str)
        return out
    except Exception:
        return out


def cv_regression_scores(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    r2_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
    mae_scores = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error")

    # fit once on full data for descriptive output
    pipe.fit(X, y)
    y_hat = pipe.predict(X)

    return {
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "train_r2": float(r2_score(y, y_hat)),
        "train_mae": float(mean_absolute_error(y, y_hat)),
    }


# =============================================================================
# STEP 1: LOAD BASE COHORT
# =============================================================================
log_header("STEP 1: Load tau-anchored cohort")
df = pd.read_csv(ANCHOR_PATH, low_memory=False)
print(f"Loaded tau-anchored cohort: {len(df)} subjects")

# Ensure key columns are numeric if present
numeric_candidates = [
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
for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# APOE4 carrier
if "APOE_GENOTYPE" in df.columns:
    df["apoe4_carrier"] = df["APOE_GENOTYPE"].astype(str).str.contains("4", na=False).astype(float)
    df.loc[df["APOE_GENOTYPE"].isna(), "apoe4_carrier"] = np.nan
else:
    df["apoe4_carrier"] = np.nan


# =============================================================================
# STEP 2: LOAD DEMOGRAPHICS FROM MASTER
# =============================================================================
log_header("STEP 2: Add demographics from master file")

master_cols = pd.read_csv(MASTER_PATH, nrows=0).columns.tolist()
needed_demo_cols = ["PTID", "DEM_VISCODE", "DEM_VISDATE", "DEM_PTGENDER", "DEM_PTEDUCAT", "DEM_PTDOB"]
usecols = [c for c in needed_demo_cols if c in master_cols]

master_demo = pd.read_csv(MASTER_PATH, usecols=usecols, low_memory=False)
print("Loaded master demographic columns:", usecols)

# Standardize visit labels
if "VISCODE2" in df.columns:
    df["VISCODE2_STD"] = normalize_viscode(df["VISCODE2"])
else:
    df["VISCODE2_STD"] = np.nan

master_demo["VISCODE2_STD"] = normalize_viscode(master_demo["DEM_VISCODE"]) if "DEM_VISCODE" in master_demo.columns else np.nan

# Compute age from DEM_VISDATE - DEM_PTDOB
if {"DEM_VISDATE", "DEM_PTDOB"}.issubset(master_demo.columns):
    master_demo["AGE_DEM"] = compute_age_years(master_demo["DEM_VISDATE"], master_demo["DEM_PTDOB"])
else:
    master_demo["AGE_DEM"] = np.nan

# Exact PTID + VISCODE match first
demo_exact_cols = ["PTID", "VISCODE2_STD"]
for c in ["DEM_PTGENDER", "DEM_PTEDUCAT", "AGE_DEM"]:
    if c in master_demo.columns:
        demo_exact_cols.append(c)

demo_exact = (
    master_demo[demo_exact_cols]
    .sort_values(["PTID", "VISCODE2_STD"])
    .drop_duplicates(subset=["PTID", "VISCODE2_STD"], keep="first")
    .copy()
)

df = df.merge(demo_exact, on=["PTID", "VISCODE2_STD"], how="left")

# Fallback by PTID only for missing demographics
fallback_cols = ["PTID"]
for c in ["DEM_PTGENDER", "DEM_PTEDUCAT", "AGE_DEM"]:
    if c in master_demo.columns:
        fallback_cols.append(c)

demo_fallback = (
    master_demo.sort_values(["PTID", "DEM_VISDATE"] if "DEM_VISDATE" in master_demo.columns else ["PTID"])
    .groupby("PTID", as_index=False)
    .first()[fallback_cols]
    .rename(
        columns={
            "DEM_PTGENDER": "DEM_PTGENDER_FB",
            "DEM_PTEDUCAT": "DEM_PTEDUCAT_FB",
            "AGE_DEM": "AGE_DEM_FB",
        }
    )
)

df = df.merge(demo_fallback, on="PTID", how="left")

# Final demographic fields
df["SEX"] = df["DEM_PTGENDER"]
df["SEX"] = df["SEX"].fillna(df.get("DEM_PTGENDER_FB"))

df["EDUCATION"] = pd.to_numeric(df["DEM_PTEDUCAT"], errors="coerce")
if "DEM_PTEDUCAT_FB" in df.columns:
    df["EDUCATION"] = df["EDUCATION"].fillna(pd.to_numeric(df["DEM_PTEDUCAT_FB"], errors="coerce"))

df["AGE_FINAL"] = pd.to_numeric(df["AGE_DEM"], errors="coerce")
if "AGE_DEM_FB" in df.columns:
    df["AGE_FINAL"] = df["AGE_FINAL"].fillna(pd.to_numeric(df["AGE_DEM_FB"], errors="coerce"))

df["SEX_MALE"] = np.nan
if "SEX" in df.columns:
    df.loc[df["SEX"].astype(str).str.upper() == "MALE", "SEX_MALE"] = 1.0
    df.loc[df["SEX"].astype(str).str.upper() == "FEMALE", "SEX_MALE"] = 0.0

print(f"With sex:        {int(df['SEX'].notna().sum())}")
print(f"With education:  {int(df['EDUCATION'].notna().sum())}")
print(f"With age:        {int(df['AGE_FINAL'].notna().sum())}")
print(f"With all demos:  {int(df[['SEX_MALE', 'EDUCATION', 'AGE_FINAL']].notna().all(axis=1).sum())}")


# =============================================================================
# STEP 3: DEFINE BASELINE MISMATCH SCORE
# =============================================================================
log_header("STEP 3: Compute baseline mismatch score")

pathology_cols = [c for c in ["TAU6MM_META_TEMPORAL_SUVR", "AMY6MM_CENTILOIDS"] if c in df.columns]
demo_cols = []

# Use age if reasonably available, otherwise skip it
if "AGE_FINAL" in df.columns and df["AGE_FINAL"].notna().sum() >= 0.7 * len(df):
    demo_cols.append("AGE_FINAL")
if "EDUCATION" in df.columns and df["EDUCATION"].notna().sum() >= 0.7 * len(df):
    demo_cols.append("EDUCATION")
if "SEX_MALE" in df.columns and df["SEX_MALE"].notna().sum() >= 0.7 * len(df):
    demo_cols.append("SEX_MALE")

mismatch_features = pathology_cols + demo_cols
print("Mismatch model features:", mismatch_features)

valid = df["CDR_CDRSB"].notna()
for col in mismatch_features:
    valid &= df[col].notna()

df_model = df.loc[valid].copy()
print(f"Complete cases for mismatch model: {len(df_model)}")

X = df_model[mismatch_features].astype(float).copy()
y = df_model["CDR_CDRSB"].astype(float).copy()

mismatch_pipe = make_pipeline(StandardScaler(), LinearRegression())
mismatch_pipe.fit(X, y)

df_model["CDR_expected"] = mismatch_pipe.predict(X)
df_model["mismatch_score"] = df_model["CDR_CDRSB"] - df_model["CDR_expected"]

# Merge back to full cohort
df = df.merge(
    df_model[["PTID", "CDR_expected", "mismatch_score"]],
    on="PTID",
    how="left",
)

# Refit linear regression directly for readable coefficients
coef_model = LinearRegression()
coef_scaler = StandardScaler()
X_scaled = coef_scaler.fit_transform(X)
coef_model.fit(X_scaled, y)

print(f"\nBaseline mismatch model R² = {coef_model.score(X_scaled, y):.3f}")
print("Standardized coefficients:")
for feat, coef in zip(mismatch_features, coef_model.coef_):
    print(f"  {feat:<25} {coef:+.3f}")

ms = df["mismatch_score"].dropna()
print(f"\nMismatch score distribution: N={len(ms)}, mean={ms.mean():.3f}, std={ms.std():.3f}")


# =============================================================================
# STEP 4: NESTED PROGNOSTIC MODELS
# =============================================================================
log_header("STEP 4: Nested prognostic models for future decline")

prog_df = df[df["mismatch_score"].notna() & df["delta_CDR_CDRSB"].notna()].copy()
print(f"Subjects with mismatch + follow-up delta CDR-SB: {len(prog_df)}")

# Feature blocks
base_prog_features = pathology_cols + demo_cols
tier_definitions = {
    "Tier1_Pathology+Demo": base_prog_features,
    "Tier2_+Mismatch": base_prog_features + ["mismatch_score"],
    "Tier3_+MRI_WMH": base_prog_features + ["mismatch_score", "FS7_HIPPO_BILAT_ICVnorm", "WMH_TOTAL_WMH"],
    "Tier4_+Plasma_APOE": base_prog_features + ["mismatch_score", "FS7_HIPPO_BILAT_ICVnorm", "WMH_TOTAL_WMH",
                                                "PLASMA_pT217_F", "PLASMA_NfL_Q", "PLASMA_GFAP_Q", "apoe4_carrier"],
}

tier_rows = []
print(f"\n{'Tier':<22} {'N':>6} {'CV R²':>16} {'CV MAE':>16}")
print("-" * 66)

for tier_name, raw_feats in tier_definitions.items():
    feats = [f for f in raw_feats if f in prog_df.columns]
    tier_dat = prog_df[feats + ["delta_CDR_CDRSB"]].dropna().copy()

    if len(tier_dat) < 30:
        print(f"{tier_name:<22} {'N/A':>6} {'too few':>16} {'samples':>16}")
        continue

    X_tier = tier_dat[feats].astype(float)
    y_tier = tier_dat["delta_CDR_CDRSB"].astype(float)

    scores = cv_regression_scores(X_tier, y_tier, n_splits=5)

    print(
        f"{tier_name:<22} "
        f"{len(tier_dat):>6} "
        f"{scores['r2_mean']:+.3f}±{scores['r2_std']:.2f}".rjust(16) + " "
        f"{scores['mae_mean']:.3f}±{scores['mae_std']:.2f}".rjust(16)
    )

    tier_rows.append(
        {
            "tier": tier_name,
            "n": int(len(tier_dat)),
            "features": "|".join(feats),
            **scores,
        }
    )

tier_df = pd.DataFrame(tier_rows)
tier_df.to_csv(OUT_TIER_CSV, index=False)
print(f"\nSaved tier comparison: {OUT_TIER_CSV}")


# =============================================================================
# STEP 5: DOES MISMATCH PREDICT FUTURE DECLINE?
# =============================================================================
log_header("STEP 5: Validate mismatch vs future decline")

r_p, p_p, n_p = safe_pearson(prog_df["mismatch_score"], prog_df["delta_CDR_CDRSB"])
r_s, p_s, n_s = safe_spearman(prog_df["mismatch_score"], prog_df["delta_CDR_CDRSB"])

print("Continuous validation: mismatch_score vs delta_CDR_CDRSB")
print(f"  Pearson : r = {r_p:+.3f}, p = {p_p:.4g}, N = {n_p}")
print(f"  Spearman: r = {r_s:+.3f}, p = {p_s:.4g}, N = {n_s}")

if np.isfinite(r_p):
    if r_p > 0 and p_p < 0.05:
        print("  → Higher mismatch (worse clinical status than expected) predicts faster decline.")
    elif r_p < 0 and p_p < 0.05:
        print("  → Higher mismatch predicts slower decline.")
    else:
        print("  → Weak / non-significant continuous relationship.")

prog_df["mismatch_tertile"] = make_tertiles(prog_df["mismatch_score"])

print("\nDelta CDR-SB by mismatch tertile:")
print(f"{'Tertile':<10} {'N':>6} {'Mean Δ':>10} {'Median':>10} {'SD':>10}")
print("-" * 52)

anova_groups = []
for tertile in ["Low", "Mid", "High"]:
    vals = prog_df.loc[prog_df["mismatch_tertile"] == tertile, "delta_CDR_CDRSB"].dropna()
    if len(vals) > 0:
        print(f"{tertile:<10} {len(vals):>6} {vals.mean():>+10.3f} {vals.median():>10.3f} {vals.std():>10.3f}")
    if len(vals) >= 3:
        anova_groups.append(vals.values)

if len(anova_groups) == 3:
    f_stat, p_anova = stats.f_oneway(*anova_groups)
    h_stat, p_kw = stats.kruskal(*anova_groups)
    print(f"\nANOVA:          F = {f_stat:.2f}, p = {p_anova:.4g}")
    print(f"Kruskal-Wallis: H = {h_stat:.2f}, p = {p_kw:.4g}")

# Secondary MMSE summary
if "delta_MMSE_MMSCORE" in prog_df.columns:
    mmse_df = prog_df[prog_df["delta_MMSE_MMSCORE"].notna()].copy()
    if len(mmse_df) > 0:
        print("\nSecondary outcome: delta_MMSE_MMSCORE by mismatch tertile")
        print(f"{'Tertile':<10} {'N':>6} {'Mean Δ':>10} {'Median':>10} {'SD':>10}")
        print("-" * 52)
        for tertile in ["Low", "Mid", "High"]:
            vals = mmse_df.loc[mmse_df["mismatch_tertile"] == tertile, "delta_MMSE_MMSCORE"].dropna()
            if len(vals) > 0:
                print(f"{tertile:<10} {len(vals):>6} {vals.mean():>+10.3f} {vals.median():>10.3f} {vals.std():>10.3f}")


# =============================================================================
# STEP 6: WHAT EXPLAINS MISMATCH?
# =============================================================================
log_header("STEP 6: Correlates of mismatch")

base_mismatch_df = df[df["mismatch_score"].notna()].copy()

explain_features = [
    ("FS7_HIPPO_BILAT_ICVnorm", "Hippocampal volume (ICV-norm)"),
    ("WMH_TOTAL_WMH", "White matter hyperintensities"),
    ("PLASMA_pT217_F", "Plasma p-tau217"),
    ("PLASMA_NfL_Q", "Plasma NfL"),
    ("PLASMA_GFAP_Q", "Plasma GFAP"),
    ("AMY6MM_CENTILOIDS", "Amyloid centiloids"),
    ("TAU6MM_META_TEMPORAL_SUVR", "Tau meta-temporal SUVR"),
]

print(f"{'Feature':<35} {'r':>8} {'p':>10} {'N':>6}")
print("-" * 65)

mismatch_corrs = []
for col, label in explain_features:
    if col in base_mismatch_df.columns:
        r, p, n = safe_pearson(base_mismatch_df[col], base_mismatch_df["mismatch_score"])
        if n >= 10:
            sig = "*" if p < 0.05 else ""
            print(f"{label:<35} {r:+8.3f} {p:>10.4f} {n:>6} {sig}")
            mismatch_corrs.append(
                {
                    "feature": col,
                    "label": label,
                    "r": None if pd.isna(r) else float(r),
                    "p": None if pd.isna(p) else float(p),
                    "n": int(n),
                }
            )


# =============================================================================
# STEP 7: TAU+ SUBGROUP
# =============================================================================
log_header("STEP 7: Tau-positive subgroup")

taup_summary = {}
if "tau_pos" in prog_df.columns:
    taup = prog_df[prog_df["tau_pos"] == 1].copy()
    print(f"Tau+ subjects with mismatch + follow-up: {len(taup)}")

    if len(taup) >= 10:
        r_t, p_t, n_t = safe_pearson(taup["mismatch_score"], taup["delta_CDR_CDRSB"])
        print(f"Pearson(mismatch, delta_CDR_CDRSB): r = {r_t:+.3f}, p = {p_t:.4g}, N = {n_t}")

        taup["mismatch_tertile"] = make_tertiles(taup["mismatch_score"])
        print("\nDelta CDR-SB by mismatch tertile (tau+ only):")
        for tertile in ["Low", "Mid", "High"]:
            vals = taup.loc[taup["mismatch_tertile"] == tertile, "delta_CDR_CDRSB"].dropna()
            if len(vals) > 0:
                print(f"  {tertile:<4} N={len(vals):>3}, mean Δ={vals.mean():+.3f}, median={vals.median():.3f}")

        taup_summary = {
            "n": int(len(taup)),
            "pearson_r": None if pd.isna(r_t) else float(r_t),
            "pearson_p": None if pd.isna(p_t) else float(p_t),
        }


# =============================================================================
# STEP 8: SAVE OUTPUTS
# =============================================================================
log_header("STEP 8: Save outputs")

df.to_csv(OUT_SCORED, index=False)
print(f"Saved scored cohort: {OUT_SCORED}")

summary = {
    "n_total": int(len(df)),
    "n_with_mismatch": int(df["mismatch_score"].notna().sum()),
    "n_with_followup_cdrsb": int(df["delta_CDR_CDRSB"].notna().sum()) if "delta_CDR_CDRSB" in df.columns else 0,
    "n_prog": int(len(prog_df)),
    "mismatch_features": mismatch_features,
    "baseline_model_r2": float(coef_model.score(X_scaled, y)),
    "mismatch_mean": float(ms.mean()) if len(ms) else None,
    "mismatch_std": float(ms.std()) if len(ms) else None,
    "pearson_mismatch_vs_delta_cdrsb_r": None if pd.isna(r_p) else float(r_p),
    "pearson_mismatch_vs_delta_cdrsb_p": None if pd.isna(p_p) else float(p_p),
    "spearman_mismatch_vs_delta_cdrsb_r": None if pd.isna(r_s) else float(r_s),
    "spearman_mismatch_vs_delta_cdrsb_p": None if pd.isna(p_s) else float(p_s),
    "tier_results": tier_df.to_dict(orient="records"),
    "tau_positive_summary": taup_summary,
    "mismatch_correlations": mismatch_corrs,
}

with open(OUT_RESULTS_JSON, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Saved summary JSON: {OUT_RESULTS_JSON}")

print("\nDone.")