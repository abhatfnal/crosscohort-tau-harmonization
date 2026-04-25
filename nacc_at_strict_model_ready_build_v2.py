# nacc_at_strict_model_ready_build_v2.py
from pathlib import Path
import json
import numpy as np
import pandas as pd

STRICT_IN = Path(
    "/project/aereditato/abhat/NACC_17259/phase2_tau_amyloid_90d/derived_cohorts/"
    "nacc_tau_amyloid_90d_AT_strict_AposTpos_vs_AnegTneg.csv"
)

FEATURE_IN = Path(
    "/project/aereditato/abhat/NACC_17259/phase1_v2/cohorts_90d/"
    "tau_plus_amyloid_subject_level.csv"
)

OUTDIR = Path(
    "/project/aereditato/abhat/NACC_17259/phase2_tau_amyloid_90d/model_ready_v2"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def harmonize_sex(s):
    if s is None:
        return s
    s = s.copy()
    if pd.api.types.is_numeric_dtype(s):
        return s.map({1: "M", 2: "F"})
    return (
        s.astype(str).str.strip().replace(
            {
                "1": "M", "2": "F",
                "male": "M", "Male": "M", "m": "M",
                "female": "F", "Female": "F", "f": "F",
            }
        )
    )


def as_cat_string(s):
    if s is None:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("Int64").astype("string")
    return s.astype("string")


def coalesce(df, preferred, backup, outcol=None):
    outcol = outcol or preferred
    if preferred in df.columns and backup in df.columns:
        df[outcol] = df[preferred].combine_first(df[backup])
    elif preferred in df.columns:
        df[outcol] = df[preferred]
    elif backup in df.columns:
        df[outcol] = df[backup]
    else:
        df[outcol] = np.nan


print("=" * 80)
print("LOAD STRICT LABELS + FEATURE SOURCE")
print("=" * 80)
print("Strict labels:", STRICT_IN)
print("Feature source:", FEATURE_IN)

strict = pd.read_csv(STRICT_IN, low_memory=False)
feat = pd.read_csv(FEATURE_IN, low_memory=False)

if "NACCID" not in strict.columns:
    raise KeyError("NACCID not found in strict file")
if "NACCID" not in feat.columns:
    raise KeyError("NACCID not found in feature source file")

strict = strict.drop_duplicates(subset=["NACCID"]).copy()
feat = feat.drop_duplicates(subset=["NACCID"]).copy()

print("Strict rows:", len(strict), "subjects:", strict["NACCID"].nunique())
print("Feature rows:", len(feat), "subjects:", feat["NACCID"].nunique())

merged = strict.merge(feat, on="NACCID", how="left", suffixes=("", "__feat"))

print("Merged rows:", len(merged))
print("Subjects matched to feature source:", int(merged.filter(regex="__feat$").notna().any(axis=1).sum()))

# important label/meta fields
for c in [
    "AT_group", "dx_harmonized", "tau_tracer", "A_status", "T_status",
    "visit_date", "amy_date", "tau_date", "amy_day_diff", "tau_day_diff",
    "AMY_CENTILOIDS", "TAU_META_TEMPORAL_SUVR", "TAU_CTX_ENTORHINAL_SUVR"
]:
    coalesce(merged, c, f"{c}__feat", c)

if "y_AT_strict" not in merged.columns:
    merged["y_AT_strict"] = (merged["AT_group"] == "A+/T+").astype(int)

# derive APOE carrier if needed
if "apoe_e4_count" not in merged.columns:
    if "apoe_e4_count__feat" in merged.columns:
        merged["apoe_e4_count"] = pd.to_numeric(merged["apoe_e4_count__feat"], errors="coerce")
    elif "NACCNE4S" in merged.columns:
        merged["apoe_e4_count"] = pd.to_numeric(merged["NACCNE4S"], errors="coerce")
    elif "NACCNE4S__feat" in merged.columns:
        merged["apoe_e4_count"] = pd.to_numeric(merged["NACCNE4S__feat"], errors="coerce")
    else:
        merged["apoe_e4_count"] = np.nan

if "apoe_e4_carrier" not in merged.columns:
    merged["apoe_e4_carrier"] = (merged["apoe_e4_count"] > 0).astype(float)

# feature map
feature_map = {
    "age": ["NACCAGE"],
    "sex": ["sex_harmonized", "SEX"],
    "education_years": ["EDUC"],
    "race": ["RACE"],
    "hispanic": ["HISPANIC"],
    "handedness": ["HANDED"],
    "apoe_e4_count": ["apoe_e4_count", "NACCNE4S"],
    "apoe_e4_carrier": ["apoe_e4_carrier"],

    "mmse": ["NACCMMSE"],
    "moca": ["NACCMOCA"],
    "cdr_global": ["CDRGLOB"],
    "cdr_sumboxes": ["CDRSUM"],
    "faq_total": ["FAQ_TOTAL"],
    "gds_total": ["NACCGDS"],

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

out = pd.DataFrame(index=merged.index)

meta_cols = [
    "NACCID", "AT_group", "y_AT_strict", "dx_harmonized", "tau_tracer",
    "A_status", "T_status", "visit_date", "amy_date", "tau_date",
    "amy_day_diff", "tau_day_diff", "AMY_CENTILOIDS",
    "TAU_META_TEMPORAL_SUVR", "TAU_CTX_ENTORHINAL_SUVR"
]
for c in meta_cols:
    out[c] = merged[c] if c in merged.columns else np.nan

feature_manifest = {}
for std_col, candidates in feature_map.items():
    src = None
    for cand in candidates:
        if cand in merged.columns:
            src = cand
            break
        if f"{cand}__feat" in merged.columns:
            src = f"{cand}__feat"
            break
    feature_manifest[std_col] = src
    out[std_col] = merged[src] if src is not None else np.nan

# harmonize / coerce
if "sex" in out.columns:
    out["sex"] = harmonize_sex(out["sex"])

for c in ["race", "hispanic", "handedness", "tau_tracer"]:
    if c in out.columns:
        out[c] = as_cat_string(out[c])

for c in list(numeric_std) + ["y_AT_strict", "AMY_CENTILOIDS", "TAU_META_TEMPORAL_SUVR", "TAU_CTX_ENTORHINAL_SUVR"]:
    if c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

missing_rows = []
for c in feature_map.keys():
    missing_rows.append({
        "feature": c,
        "source_col": feature_manifest.get(c),
        "missing_n": int(out[c].isna().sum()),
        "missing_pct": round(100.0 * out[c].isna().mean(), 2),
    })
missing_df = pd.DataFrame(missing_rows).sort_values(["missing_pct", "feature"], ascending=[False, True])

full_csv = OUTDIR / "nacc_AT_strict_model_table_full_v2.csv"
nodx_withtracer_csv = OUTDIR / "nacc_AT_strict_model_table_nodx_withtracer_v2.csv"
nodx_notracer_csv = OUTDIR / "nacc_AT_strict_model_table_nodx_notracer_v2.csv"
withdx_csv = OUTDIR / "nacc_AT_strict_model_table_withdx_v2.csv"
tr6_csv = OUTDIR / "nacc_AT_strict_tracer6_model_table_nodx_v2.csv"
missing_csv = OUTDIR / "nacc_AT_strict_feature_missingness_v2.csv"
manifest_json = OUTDIR / "nacc_AT_strict_model_manifest_v2.json"

out.to_csv(full_csv, index=False)

nodx_withtracer_cols = ["NACCID", "AT_group", "y_AT_strict", "tau_tracer"] + list(feature_map.keys())
nodx_withtracer_cols = [c for c in nodx_withtracer_cols if c in out.columns]
out[nodx_withtracer_cols].to_csv(nodx_withtracer_csv, index=False)

nodx_notracer_cols = ["NACCID", "AT_group", "y_AT_strict"] + [c for c in feature_map.keys() if c != "tau_tracer"]
nodx_notracer_cols = [c for c in nodx_notracer_cols if c in out.columns]
out[nodx_notracer_cols].to_csv(nodx_notracer_csv, index=False)

withdx_cols = ["NACCID", "AT_group", "y_AT_strict", "dx_harmonized", "tau_tracer"] + list(feature_map.keys())
withdx_cols = [c for c in withdx_cols if c in out.columns]
out[withdx_cols].to_csv(withdx_csv, index=False)

tr6 = out[out["tau_tracer"].astype("string") == "6"].copy()
tr6_cols = [c for c in nodx_notracer_cols if c in tr6.columns]
tr6[tr6_cols].to_csv(tr6_csv, index=False)

missing_df.to_csv(missing_csv, index=False)

manifest = {
    "strict_input": str(STRICT_IN),
    "feature_input": str(FEATURE_IN),
    "rows": int(len(out)),
    "subjects": int(out["NACCID"].nunique()),
    "class_counts": out["AT_group"].value_counts(dropna=False).to_dict(),
    "feature_sources": feature_manifest,
    "categorical_features": [c for c in categorical_std if c in out.columns],
    "numeric_features": [c for c in numeric_std if c in out.columns],
    "saved_files": {
        "full": str(full_csv),
        "nodx_withtracer": str(nodx_withtracer_csv),
        "nodx_notracer": str(nodx_notracer_csv),
        "withdx": str(withdx_csv),
        "tracer6_nodx": str(tr6_csv),
        "missingness": str(missing_csv),
    }
}
with open(manifest_json, "w") as f:
    json.dump(manifest, f, indent=2)

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

print("\nTracer counts:")
print(out["tau_tracer"].value_counts(dropna=False).to_string())

print("\nStrict class counts:")
print(out["AT_group"].value_counts(dropna=False).to_string())