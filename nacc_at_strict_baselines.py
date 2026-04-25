# nacc_at_strict_model_ready_build.py
from pathlib import Path
import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
STRICT_IN = Path(
    "/project/aereditato/abhat/NACC_17259/phase2_tau_amyloid_90d/derived_cohorts/"
    "nacc_tau_amyloid_90d_AT_strict_AposTpos_vs_AnegTneg.csv"
)

OUTDIR = Path(
    "/project/aereditato/abhat/NACC_17259/phase2_tau_amyloid_90d/model_ready"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def harmonize_sex(s):
    if s is None:
        return s
    out = s.copy()
    if pd.api.types.is_numeric_dtype(out):
        out = out.map({1: "M", 2: "F"})
    else:
        out = out.astype(str).str.strip().replace(
            {
                "1": "M",
                "2": "F",
                "male": "M",
                "Male": "M",
                "m": "M",
                "female": "F",
                "Female": "F",
                "f": "F",
            }
        )
    return out

def as_cat_string(s):
    if s is None:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("Int64").astype("string")
    return s.astype("string")

# ---------------------------------------------------------------------
# Load strict cohort
# ---------------------------------------------------------------------
print("=" * 80)
print("LOAD STRICT A+/T+ vs A-/T- COHORT")
print("=" * 80)
print("Input:", STRICT_IN)

df = pd.read_csv(STRICT_IN, low_memory=False)

if "AT_group" not in df.columns:
    raise KeyError("AT_group not found in strict cohort file")

df = df[df["AT_group"].isin(["A-/T-", "A+/T+"])].copy()
df["y_AT_strict"] = (df["AT_group"] == "A+/T+").astype(int)

print("Rows:", len(df))
print("Unique subjects:", df["NACCID"].nunique())
print(df["AT_group"].value_counts().to_string())

# ---------------------------------------------------------------------
# Metadata columns to preserve (not all used as model features)
# ---------------------------------------------------------------------
meta_candidates = {
    "NACCID": ["NACCID"],
    "AT_group": ["AT_group"],
    "y_AT_strict": ["y_AT_strict"],
    "dx_harmonized": ["dx_harmonized"],
    "tau_tracer": ["tau_tracer"],
    "visit_date": ["visit_date"],
    "amy_date": ["amy_date"],
    "tau_date": ["tau_date"],
    "amy_day_diff": ["amy_day_diff"],
    "tau_day_diff": ["tau_day_diff"],
    "AMY_CENTILOIDS": ["AMY_CENTILOIDS"],
    "TAU_META_TEMPORAL_SUVR": ["TAU_META_TEMPORAL_SUVR"],
    "TAU_CTX_ENTORHINAL_SUVR": ["TAU_CTX_ENTORHINAL_SUVR"],
}

# ---------------------------------------------------------------------
# Feature map
# These are intended to be leakage-safe features for biomarker prediction.
# We intentionally exclude amyloid/tau measurements and AT labels.
# ---------------------------------------------------------------------
feature_map = {
    # demographics / genetics
    "age": ["NACCAGE"],
    "sex": ["sex_harmonized", "SEX"],
    "education_years": ["EDUC"],
    "race": ["RACE"],
    "hispanic": ["HISPANIC"],
    "handedness": ["HANDED"],
    "apoe_e4_count": ["apoe_e4_count", "NACCNE4S"],
    "apoe_e4_carrier": ["apoe_e4_carrier"],

    # global cognition / function / mood
    "mmse": ["NACCMMSE"],
    "moca": ["NACCMOCA"],
    "cdr_global": ["CDRGLOB"],
    "cdr_sumboxes": ["CDRSUM"],
    "faq_total": ["FAQ_TOTAL"],
    "gds_total": ["NACCGDS"],

    # common neuropsych features across UDS eras
    "logical_memory_old": ["LOGIMEM"],
    "craft_story_delayed": ["CRAFTDVR"],
    "digits_forward": ["DIGIF"],
    "digits_forward_span": ["DIGIFLEN"],
    "digits_backward": ["DIGIB"],
    "digits_backward_span": ["DIGIBLEN"],
    "animal_fluency": ["ANIMALS"],
    "vegetable_fluency": ["VEG"],
    "trail_a": ["TRAILA"],
    "trail_b": ["TRAILB"],
    "boston_naming": ["BOSTON"],
    "mint_total": ["MINTTOTS"],
}

categorical_std = {"sex", "race", "hispanic", "handedness", "tau_tracer"}
numeric_std = set(feature_map.keys()) - categorical_std

# ---------------------------------------------------------------------
# Build output table
# ---------------------------------------------------------------------
out = pd.DataFrame(index=df.index)

# metadata
meta_manifest = {}
for out_col, candidates in meta_candidates.items():
    src = pick_col(df, candidates)
    meta_manifest[out_col] = src
    if src is not None:
        out[out_col] = df[src]
    elif out_col in ["AT_group", "y_AT_strict"]:
        out[out_col] = df[out_col]
    else:
        out[out_col] = np.nan

# standardized features
feature_manifest = {}
for std_col, candidates in feature_map.items():
    src = pick_col(df, candidates)
    feature_manifest[std_col] = src
    if src is None:
        out[std_col] = np.nan
    else:
        out[std_col] = df[src]

# harmonize categoricals
if "sex" in out.columns:
    out["sex"] = harmonize_sex(out["sex"])

for c in ["race", "hispanic", "handedness", "tau_tracer"]:
    if c in out.columns:
        out[c] = as_cat_string(out[c])

# numeric coercion
for c in numeric_std.union({"y_AT_strict"}):
    if c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

# ---------------------------------------------------------------------
# Missingness report
# ---------------------------------------------------------------------
missing_rows = []
for c in feature_map.keys():
    if c in out.columns:
        missing_rows.append({
            "feature": c,
            "source_col": feature_manifest.get(c),
            "missing_n": int(out[c].isna().sum()),
            "missing_pct": round(100.0 * out[c].isna().mean(), 2),
        })

missing_df = pd.DataFrame(missing_rows).sort_values(
    ["missing_pct", "feature"], ascending=[False, True]
)

# ---------------------------------------------------------------------
# Save master tables
# ---------------------------------------------------------------------
full_csv = OUTDIR / "nacc_AT_strict_model_table_full.csv"
out.to_csv(full_csv, index=False)

# no-dx, with tau tracer covariate
nodx_withtracer_cols = (
    ["NACCID", "AT_group", "y_AT_strict", "tau_tracer"]
    + list(feature_map.keys())
)
nodx_withtracer_cols = [c for c in nodx_withtracer_cols if c in out.columns]
nodx_withtracer = out[nodx_withtracer_cols].copy()
nodx_withtracer_csv = OUTDIR / "nacc_AT_strict_model_table_nodx_withtracer.csv"
nodx_withtracer.to_csv(nodx_withtracer_csv, index=False)

# no-dx, no tracer covariate
nodx_notracer_cols = (
    ["NACCID", "AT_group", "y_AT_strict"]
    + [c for c in feature_map.keys() if c != "tau_tracer"]
)
nodx_notracer_cols = [c for c in nodx_notracer_cols if c in out.columns]
nodx_notracer = out[nodx_notracer_cols].copy()
nodx_notracer_csv = OUTDIR / "nacc_AT_strict_model_table_nodx_notracer.csv"
nodx_notracer.to_csv(nodx_notracer_csv, index=False)

# exploratory with dx
withdx_cols = (
    ["NACCID", "AT_group", "y_AT_strict", "dx_harmonized", "tau_tracer"]
    + list(feature_map.keys())
)
withdx_cols = [c for c in withdx_cols if c in out.columns]
withdx = out[withdx_cols].copy()
withdx_csv = OUTDIR / "nacc_AT_strict_model_table_withdx.csv"
withdx.to_csv(withdx_csv, index=False)

# tracer-6 sensitivity subset
tr6 = out[out["tau_tracer"].astype("string") == "6"].copy()
tr6_nodx_cols = [c for c in nodx_notracer_cols if c != "tau_tracer"]
tr6_nodx = tr6[tr6_nodx_cols].copy()
tr6_csv = OUTDIR / "nacc_AT_strict_tracer6_model_table_nodx.csv"
tr6_nodx.to_csv(tr6_csv, index=False)

# ---------------------------------------------------------------------
# Save manifest
# ---------------------------------------------------------------------
manifest = {
    "input_csv": str(STRICT_IN),
    "rows": int(len(out)),
    "subjects": int(out["NACCID"].nunique()),
    "class_counts": out["AT_group"].value_counts(dropna=False).to_dict(),
    "meta_sources": meta_manifest,
    "feature_sources": feature_manifest,
    "categorical_features": [c for c in categorical_std if c in out.columns],
    "numeric_features": [c for c in numeric_std if c in out.columns],
    "saved_files": {
        "full": str(full_csv),
        "nodx_withtracer": str(nodx_withtracer_csv),
        "nodx_notracer": str(nodx_notracer_csv),
        "withdx": str(withdx_csv),
        "tracer6_nodx": str(tr6_csv),
        "missingness_csv": str(OUTDIR / "nacc_AT_strict_feature_missingness.csv"),
    },
}

missing_csv = OUTDIR / "nacc_AT_strict_feature_missingness.csv"
missing_df.to_csv(missing_csv, index=False)

manifest_json = OUTDIR / "nacc_AT_strict_model_manifest.json"
with open(manifest_json, "w") as f:
    json.dump(manifest, f, indent=2)

# ---------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------
print("\n" + "=" * 80)
print("MODEL-READY TABLES SAVED")
print("=" * 80)
print("Full:", full_csv)
print("No-dx with tracer:", nodx_withtracer_csv)
print("No-dx no tracer:", nodx_notracer_csv)
print("With dx:", withdx_csv)
print("Tracer-6 no-dx:", tr6_csv)
print("Missingness:", missing_csv)
print("Manifest:", manifest_json)

print("\nTop feature missingness:")
print(missing_df.head(20).to_string(index=False))

print("\nTracer counts in strict cohort:")
if "tau_tracer" in out.columns:
    print(out["tau_tracer"].value_counts(dropna=False).to_string())

print("\nTracer-6 strict subset:")
print("rows:", len(tr6), "subjects:", tr6["NACCID"].nunique())
if "AT_group" in tr6.columns:
    print(tr6["AT_group"].value_counts(dropna=False).to_string())