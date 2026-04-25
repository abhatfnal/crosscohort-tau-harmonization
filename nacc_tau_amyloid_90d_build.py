from pathlib import Path
import json
import numpy as np
import pandas as pd

try:
    from sklearn.mixture import GaussianMixture
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
INPUT = Path("/project/aereditato/abhat/NACC_17259/phase1_v2/cohorts_90d/tau_plus_amyloid_subject_level.csv")
TAU_SUMMARY = Path("/project/aereditato/abhat/NACC_17259/phase0/nacc_tau_summary.csv")

OUTDIR = Path("/project/aereditato/abhat/NACC_17259/phase2_tau_amyloid_90d")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_TABLE = OUTDIR / "nacc_tau_amyloid_90d_analysis_table.csv"
OUT_THRESH = OUTDIR / "nacc_tau_tracer_thresholds.csv"
OUT_AT_COUNTS = OUTDIR / "nacc_tau_amyloid_AT_counts.csv"
OUT_DX_BY_AT = OUTDIR / "nacc_tau_amyloid_dx_by_AT.csv"
OUT_SUMMARY = OUTDIR / "nacc_tau_amyloid_90d_summary.json"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def pick_col(df, candidates, required=False, label="column"):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Could not find required {label}. Tried: {candidates}")
    return None


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def safe_date(s):
    return pd.to_datetime(s, errors="coerce")


def mode_or_nan(x):
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    m = x.mode(dropna=True)
    return m.iloc[0] if len(m) else np.nan


def build_gmm_threshold(values: pd.Series):
    x = values.dropna().to_numpy().reshape(-1, 1)
    if len(x) < 40 or len(np.unique(x)) < 10:
        return None

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=42,
        n_init=10
    )
    gmm.fit(x)

    means = gmm.means_.ravel()
    weights = gmm.weights_.ravel()
    order = np.argsort(means)

    lo_mean = float(means[order[0]])
    hi_mean = float(means[order[1]])
    lo_weight = float(weights[order[0]])
    hi_weight = float(weights[order[1]])
    thr = float((lo_mean + hi_mean) / 2.0)

    return {
        "method": "gmm_midpoint",
        "threshold": thr,
        "lo_mean": lo_mean,
        "hi_mean": hi_mean,
        "lo_weight": lo_weight,
        "hi_weight": hi_weight,
    }


def build_q75_threshold(values: pd.Series):
    vals = values.dropna()
    if len(vals) == 0:
        return None
    return {
        "method": "q75_fallback",
        "threshold": float(vals.quantile(0.75)),
        "lo_mean": float(vals.mean()),
        "hi_mean": np.nan,
        "lo_weight": 1.0,
        "hi_weight": np.nan,
    }


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------
print("=" * 80)
print("LOAD SUBJECT-LEVEL TAU+AMYLOID 90D COHORT")
print("=" * 80)
print("Input:", INPUT)

df = pd.read_csv(INPUT, low_memory=False)
print("Rows:", len(df))
print("Unique subjects:", df["NACCID"].nunique())

if df["NACCID"].duplicated().any():
    print("\nFound duplicated NACCID rows. Keeping row with most non-missing biomarker fields.")
    priority_cols = [c for c in [
        "TAU_META_TEMPORAL_SUVR",
        "TAU_CTX_ENTORHINAL_SUVR",
        "AMY_CENTILOIDS",
        "AMY_GAAIN_SUMMARY_SUVR",
        "AMY_NPDKA_SUMMARY_SUVR",
        "NACCMMSE",
        "CDRSUM",
        "FAQ_TOTAL",
        "apoe_e4_count",
        "apoe_e4_carrier",
    ] if c in df.columns]

    score = df[priority_cols].notna().sum(axis=1) if priority_cols else pd.Series(0, index=df.index)
    df = (
        df.assign(_keep_score=score)
          .sort_values(["NACCID", "_keep_score"], ascending=[True, False])
          .drop_duplicates(subset=["NACCID"], keep="first")
          .drop(columns=["_keep_score"])
          .reset_index(drop=True)
    )
    print("Rows after de-dup:", len(df))


# ---------------------------------------------------------------------
# Identify columns from cohort file
# ---------------------------------------------------------------------
col_subject = "NACCID"
col_dx = pick_col(df, ["dx_harmonized"], required=True, label="diagnosis")
col_visit_date = pick_col(df, ["visit_date"])
col_age = pick_col(df, ["NACCAGE", "age", "AGE"])
col_sex = pick_col(df, ["sex_harmonized", "SEX"])
col_educ = pick_col(df, ["EDUC", "education_years"])
col_mmse = pick_col(df, ["NACCMMSE", "MMSE"])
col_cdrglob = pick_col(df, ["CDRGLOB", "CDRTOT"])
col_cdrsum = pick_col(df, ["CDRSUM"])
col_faq = pick_col(df, ["FAQ_TOTAL", "faq_total"])
col_apoe_count = pick_col(df, ["apoe_e4_count", "NACCNE4S"])
col_apoe_carrier = pick_col(df, ["apoe_e4_carrier"])

# Amyloid
col_amy_date = pick_col(df, ["amy_date"])
col_amy_diff = pick_col(df, ["amy_day_diff"])
col_amy_tracer = pick_col(df, ["AMY_TRACER", "amy_tracer"])
col_amy_status = pick_col(df, ["AMY_AMYLOID_STATUS", "amyloid_status_raw"])
col_amy_cent = pick_col(df, ["AMY_CENTILOIDS"])
col_amy_gaain = pick_col(df, ["AMY_GAAIN_SUMMARY_SUVR"])
col_amy_npdka = pick_col(df, ["AMY_NPDKA_SUMMARY_SUVR"])
col_amy_postcg = pick_col(df, ["AMY_CTX_POSTERIORCINGULATE_SUVR"])
col_amy_prec = pick_col(df, ["AMY_CTX_PRECUNEUS_SUVR"])

# Tau
col_tau_date = pick_col(df, ["tau_date"], required=True, label="tau_date")
col_tau_diff = pick_col(df, ["tau_day_diff"])
col_tau_tracer = pick_col(df, ["TAU_TRACER", "tau_tracer"])
col_tau_meta = pick_col(df, ["TAU_META_TEMPORAL_SUVR"], required=True, label="tau meta temporal")
col_tau_ent = pick_col(df, ["TAU_CTX_ENTORHINAL_SUVR"], required=True, label="tau entorhinal")


# ---------------------------------------------------------------------
# Recover tau tracer if absent from cohort file
# ---------------------------------------------------------------------
tau_tracer_source = "cohort_file"

if col_tau_tracer is None:
    print("\nTAU_TRACER not present in cohort file. Attempting recovery from phase0 tau summary:")
    print(TAU_SUMMARY)

    if TAU_SUMMARY.exists():
        tau_sum = pd.read_csv(TAU_SUMMARY, low_memory=False)

        sum_id = pick_col(tau_sum, ["NACCID"], required=True, label="tau summary NACCID")
        sum_date = pick_col(tau_sum, ["tau_date", "SCANDATE", "scan_date"])
        sum_tracer = pick_col(tau_sum, ["TAU_TRACER", "TRACER", "tau_tracer", "tracer"])

        if sum_date is not None and sum_tracer is not None:
            left = df[[col_subject, col_tau_date]].copy()
            left["_tau_date_key"] = safe_date(left[col_tau_date]).dt.normalize()

            right = tau_sum[[sum_id, sum_date, sum_tracer]].copy()
            right["_tau_date_key"] = safe_date(right[sum_date]).dt.normalize()
            right = (
                right.rename(columns={sum_id: "NACCID", sum_tracer: "tau_tracer_recovered"})
                     [["NACCID", "_tau_date_key", "tau_tracer_recovered"]]
                     .drop_duplicates()
            )

            df = df.merge(
                right,
                left_on=[col_subject, safe_date(df[col_tau_date]).dt.normalize()],
                right_on=["NACCID", "_tau_date_key"],
                how="left"
            )

            # cleanup merge artifacts
            drop_cols = [c for c in ["NACCID_y", "_tau_date_key"] if c in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)
            if "NACCID_x" in df.columns:
                df = df.rename(columns={"NACCID_x": "NACCID"})

            col_tau_tracer = "tau_tracer_recovered"
            tau_tracer_source = "phase0_tau_summary_exact_date"

            matched = df[col_tau_tracer].notna().sum()
            print("Recovered tracer for rows:", int(matched), "out of", len(df))

            # subject-level fallback if exact date merge missed some rows
            if matched < len(df):
                subj_mode = (
                    tau_sum.rename(columns={sum_id: "NACCID", sum_tracer: "tau_tracer_tmp"})
                           .groupby("NACCID")["tau_tracer_tmp"]
                           .agg(mode_or_nan)
                           .reset_index()
                           .rename(columns={"tau_tracer_tmp": "tau_tracer_subject_mode"})
                )
                df = df.merge(subj_mode, on="NACCID", how="left")
                miss = df[col_tau_tracer].isna() & df["tau_tracer_subject_mode"].notna()
                df.loc[miss, col_tau_tracer] = df.loc[miss, "tau_tracer_subject_mode"]
                df = df.drop(columns=["tau_tracer_subject_mode"])
                matched2 = df[col_tau_tracer].notna().sum()
                print("Recovered tracer after subject-mode fallback:", int(matched2), "out of", len(df))
        else:
            print("Could not identify tracer/date columns in phase0 tau summary.")
    else:
        print("Tau summary file not found.")

if col_tau_tracer is None or col_tau_tracer not in df.columns:
    print("\nTracer recovery failed. Falling back to pooled tau thresholding.")
    df["tau_tracer_pooled"] = "ALL_TRACERS_POOLED"
    col_tau_tracer = "tau_tracer_pooled"
    tau_tracer_source = "pooled_fallback"
else:
    missing_after = df[col_tau_tracer].isna().sum()
    if missing_after > 0:
        print(f"\n{missing_after} rows still missing tracer after recovery. Assigning pooled label for those rows.")
        df[col_tau_tracer] = df[col_tau_tracer].fillna("ALL_TRACERS_POOLED")


# ---------------------------------------------------------------------
# Build analysis table
# ---------------------------------------------------------------------
out = pd.DataFrame({
    "NACCID": df[col_subject].astype(str),
    "dx_harmonized": df[col_dx].astype(str),
})

if col_visit_date:
    out["visit_date"] = safe_date(df[col_visit_date])
if col_age:
    out["age"] = to_num(df[col_age])
if col_sex:
    out["sex"] = df[col_sex]
if col_educ:
    out["education_years"] = to_num(df[col_educ])
if col_mmse:
    out["MMSE"] = to_num(df[col_mmse])
if col_cdrglob:
    out["CDRGLOB"] = to_num(df[col_cdrglob])
if col_cdrsum:
    out["CDRSUM"] = to_num(df[col_cdrsum])
if col_faq:
    out["FAQ_TOTAL"] = to_num(df[col_faq])
if col_apoe_count:
    out["apoe_e4_count"] = to_num(df[col_apoe_count])
if col_apoe_carrier:
    out["apoe_e4_carrier"] = to_num(df[col_apoe_carrier])

# Amyloid
if col_amy_date:
    out["amy_date"] = safe_date(df[col_amy_date])
if col_amy_diff:
    out["amy_day_diff"] = to_num(df[col_amy_diff])
if col_amy_tracer:
    out["amy_tracer"] = df[col_amy_tracer]
if col_amy_status:
    out["amyloid_status_raw"] = to_num(df[col_amy_status])
if col_amy_cent:
    out["AMY_CENTILOIDS"] = to_num(df[col_amy_cent])
if col_amy_gaain:
    out["AMY_GAAIN_SUMMARY_SUVR"] = to_num(df[col_amy_gaain])
if col_amy_npdka:
    out["AMY_NPDKA_SUMMARY_SUVR"] = to_num(df[col_amy_npdka])
if col_amy_postcg:
    out["AMY_CTX_POSTERIORCINGULATE_SUVR"] = to_num(df[col_amy_postcg])
if col_amy_prec:
    out["AMY_CTX_PRECUNEUS_SUVR"] = to_num(df[col_amy_prec])

# Tau
out["tau_date"] = safe_date(df[col_tau_date])
if col_tau_diff:
    out["tau_day_diff"] = to_num(df[col_tau_diff])
out["tau_tracer"] = df[col_tau_tracer].astype(str)
out["TAU_META_TEMPORAL_SUVR"] = to_num(df[col_tau_meta])
out["TAU_CTX_ENTORHINAL_SUVR"] = to_num(df[col_tau_ent])


# ---------------------------------------------------------------------
# A status
# ---------------------------------------------------------------------
out["A_status"] = pd.Series(index=out.index, dtype="object")
out["A_status_source"] = pd.Series(index=out.index, dtype="object")

if "amyloid_status_raw" in out.columns:
    mask = out["amyloid_status_raw"].isin([0, 1])
    out.loc[mask & (out["amyloid_status_raw"] == 1), "A_status"] = "A+"
    out.loc[mask & (out["amyloid_status_raw"] == 0), "A_status"] = "A-"
    out.loc[mask, "A_status_source"] = "AMY_AMYLOID_STATUS"

if "AMY_CENTILOIDS" in out.columns:
    fallback = out["A_status"].isna() & out["AMY_CENTILOIDS"].notna()
    out.loc[fallback & (out["AMY_CENTILOIDS"] >= 20), "A_status"] = "A+"
    out.loc[fallback & (out["AMY_CENTILOIDS"] < 20), "A_status"] = "A-"
    out.loc[fallback, "A_status_source"] = "CENTILOIDS_GE20_FALLBACK"


# ---------------------------------------------------------------------
# T status
# ---------------------------------------------------------------------
print("\n" + "=" * 80)
print("BUILD PROVISIONAL T STATUS")
print("=" * 80)
print("sklearn available:", HAVE_SKLEARN)
print("tau tracer source:", tau_tracer_source)

out["T_status"] = pd.Series(index=out.index, dtype="object")
out["T_status_source"] = pd.Series(index=out.index, dtype="object")
out["T_threshold_used"] = np.nan
out["TAU_META_TEMPORAL_z_within_tracer"] = np.nan

threshold_rows = []

tau_meta = out["TAU_META_TEMPORAL_SUVR"]
tau_tracer = out["tau_tracer"].fillna("ALL_TRACERS_POOLED").astype(str)

for tracer in sorted(tau_tracer.unique()):
    idx = out.index[tau_tracer == tracer]
    vals = tau_meta.loc[idx].dropna()
    n = len(vals)

    if n == 0:
        continue

    mu = float(vals.mean())
    sd = float(vals.std(ddof=0))
    if sd > 0:
        out.loc[idx, "TAU_META_TEMPORAL_z_within_tracer"] = (tau_meta.loc[idx] - mu) / sd

    thr_info = None
    if HAVE_SKLEARN:
        thr_info = build_gmm_threshold(vals)
    if thr_info is None:
        thr_info = build_q75_threshold(vals)

    thr = thr_info["threshold"]
    method = thr_info["method"]

    use_mask = (tau_tracer == tracer) & tau_meta.notna()
    out.loc[use_mask & (tau_meta >= thr), "T_status"] = "T+"
    out.loc[use_mask & (tau_meta < thr), "T_status"] = "T-"
    out.loc[use_mask, "T_status_source"] = method
    out.loc[use_mask, "T_threshold_used"] = thr

    threshold_rows.append({
        "tau_tracer": tracer,
        "n_subjects": int(n),
        "method": method,
        "threshold": float(thr),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=0)),
        "min": float(vals.min()),
        "median": float(vals.median()),
        "max": float(vals.max()),
        "lo_mean": thr_info["lo_mean"],
        "hi_mean": thr_info["hi_mean"],
        "lo_weight": thr_info["lo_weight"],
        "hi_weight": thr_info["hi_weight"],
    })

thr_df = pd.DataFrame(threshold_rows).sort_values(["tau_tracer"]).reset_index(drop=True)
thr_df.to_csv(OUT_THRESH, index=False)

print("Wrote thresholds:", OUT_THRESH)
if len(thr_df):
    print(thr_df.to_string(index=False))
else:
    print("No thresholds written.")


# ---------------------------------------------------------------------
# AT groups
# ---------------------------------------------------------------------
out["AT_group"] = pd.Series(index=out.index, dtype="object")
both = out["A_status"].notna() & out["T_status"].notna()
out.loc[both, "AT_group"] = out.loc[both, "A_status"] + "/" + out.loc[both, "T_status"]

out["dx_harmonized"] = out["dx_harmonized"].astype(str)


# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------
out.to_csv(OUT_TABLE, index=False)

a_counts = (
    out["A_status"]
    .value_counts(dropna=False)
    .rename_axis("A_status")
    .reset_index(name="n")
)
t_counts = (
    out["T_status"]
    .value_counts(dropna=False)
    .rename_axis("T_status")
    .reset_index(name="n")
)
at_counts = (
    out["AT_group"]
    .value_counts(dropna=False)
    .rename_axis("AT_group")
    .reset_index(name="n")
)
at_counts.to_csv(OUT_AT_COUNTS, index=False)

dx_by_at = pd.crosstab(out["AT_group"], out["dx_harmonized"], dropna=False).reset_index()
dx_by_at.to_csv(OUT_DX_BY_AT, index=False)

summary = {
    "input": str(INPUT),
    "tau_summary": str(TAU_SUMMARY),
    "output_table": str(OUT_TABLE),
    "output_thresholds": str(OUT_THRESH),
    "rows": int(len(out)),
    "unique_subjects": int(out["NACCID"].nunique()),
    "have_sklearn": bool(HAVE_SKLEARN),
    "tau_tracer_source": tau_tracer_source,
    "tau_tracer_counts": (
        out["tau_tracer"].value_counts(dropna=False).rename_axis("tau_tracer").reset_index(name="n").to_dict(orient="records")
    ),
    "A_status_counts": a_counts.to_dict(orient="records"),
    "T_status_counts": t_counts.to_dict(orient="records"),
    "AT_group_counts": at_counts.to_dict(orient="records"),
    "dx_counts": (
        out["dx_harmonized"].value_counts(dropna=False).rename_axis("dx_harmonized").reset_index(name="n").to_dict(orient="records")
    ),
    "tau_thresholds": thr_df.to_dict(orient="records"),
}
with open(OUT_SUMMARY, "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("\n" + "=" * 80)
print("ANALYSIS TABLE")
print("=" * 80)
print("Wrote:", OUT_TABLE)
print("Rows:", len(out))
print("Unique subjects:", out["NACCID"].nunique())

print("\n" + "=" * 80)
print("TAU TRACER COUNTS")
print("=" * 80)
print(out["tau_tracer"].value_counts(dropna=False).to_string())

print("\n" + "=" * 80)
print("A STATUS")
print("=" * 80)
print(a_counts.to_string(index=False))

print("\n" + "=" * 80)
print("T STATUS")
print("=" * 80)
print(t_counts.to_string(index=False))

print("\n" + "=" * 80)
print("AT GROUP")
print("=" * 80)
print(at_counts.to_string(index=False))

print("\n" + "=" * 80)
print("DX BY AT GROUP")
print("=" * 80)
print(dx_by_at.to_string(index=False))

print("\nSaved:")
print(OUT_TABLE)
print(OUT_THRESH)
print(OUT_AT_COUNTS)
print(OUT_DX_BY_AT)
print(OUT_SUMMARY)

print("\nDone.")
