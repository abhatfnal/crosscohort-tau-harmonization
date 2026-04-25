# nacc_at_strict_model_ready_build_v3_from_master.py
from pathlib import Path
import json
import numpy as np
import pandas as pd

STRICT_IN = Path(
    "/project/aereditato/abhat/NACC_17259/phase2_tau_amyloid_90d/derived_cohorts/"
    "nacc_tau_amyloid_90d_AT_strict_AposTpos_vs_AnegTneg.csv"
)

MASTER_IN = Path(
    "/project/aereditato/abhat/NACC_17259/phase1_v2/"
    "nacc_master_90d_v2_clean.csv"
)

OUTDIR = Path(
    "/project/aereditato/abhat/NACC_17259/phase2_tau_amyloid_90d/model_ready_v3_from_master"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


def pick_first(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def harmonize_sex(s):
    if s is None:
        return s
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


def to_cat_string(s):
    if s is None:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("Int64").astype("string")
    return s.astype("string")


print("=" * 80)
print("LOAD STRICT LABELS + FULL 90D CLEAN MASTER")
print("=" * 80)

strict = pd.read_csv(STRICT_IN, low_memory=False)
master = pd.read_csv(MASTER_IN, low_memory=False)

for c in ["visit_date", "amy_date", "tau_date"]:
    if c in strict.columns:
        strict[c] = pd.to_datetime(strict[c], errors="coerce")
    if c in master.columns:
        master[c] = pd.to_datetime(master[c], errors="coerce")

strict = strict.drop_duplicates(subset=["NACCID", "visit_date"]).copy()
master = master.drop_duplicates(subset=["NACCID", "visit_date"]).copy()

print("Strict rows:", len(strict), "subjects:", strict["NACCID"].nunique())
print("Master rows:", len(master), "subjects:", master["NACCID"].nunique())

# exact merge on subject + visit date
merged = strict.merge(
    master,
    on=["NACCID", "visit_date"],
    how="left",
    suffixes=("", "__master"),
)

print("Merged rows:", len(merged))
print("Exact visit matches:", int(merged.filter(regex="__master$").notna().any(axis=1).sum()))

# preserve / coalesce key labels from strict first
for c in [
    "AT_group", "A_status", "T_status", "dx_harmonized", "tau_tracer",
    "amy_date", "tau_date", "amy_day_diff", "tau_day_diff",
    "AMY_CENTILOIDS", "TAU_META_TEMPORAL_SUVR", "TAU_CTX_ENTORHINAL_SUVR"
]:
    if c not in merged.columns and f"{c}__master" in merged.columns:
        merged[c] = merged[f"{c}__master"]
    elif c in merged.columns and f"{c}__master" in merged.columns:
        merged[c] = merged[c].combine_first(merged[f"{c}__master"])

if "y_AT_strict" not in merged.columns:
    merged["y_AT_strict"] = (merged["AT_group"] == "A+/T+").astype(int)

# APOE
if "apoe_e4_count" not in merged.columns:
    src = pick_first(merged, ["NACCNE4S", "apoe_e4_count", "NACCNE4S__master"])
    merged["apoe_e4_count"] = pd.to_numeric(merged[src], errors="coerce") if src else np.nan

if "apoe_e4_carrier" not in merged.columns:
    merged["apoe_e4_carrier"] = (merged["apoe_e4_count"] > 0).astype(float)

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
    "NACCID", "visit_date", "AT_group", "y_AT_strict", "dx_harmonized", "tau_tracer",
    "A_status", "T_status", "amy_date", "tau_date",
    "amy_day_diff", "tau_day_diff", "AMY_CENTILOIDS",
    "TAU_META_TEMPORAL_SUVR", "TAU_CTX_ENTORHINAL_SUVR"
]
for c in meta_cols:
    out[c] = merged[c] if c in merged.columns else np.nan

feature_manifest = {}
for std_col, candidates in feature_map.items():
    src = pick_first(merged, candidates + [f"{x}__master" for x in candidates])
    feature_manifest[std_col] = src
    out[std_col] = merged[src] if src else np.nan

if "sex" in out.columns:
    out["sex"] = harmonize_sex(out["sex"])

for c in ["race", "hispanic", "handedness", "tau_tracer"]:
    if c in out.columns:
        out[c] = to_cat_string(out[c])

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
missing_df = pd.DataFrame(missing_rows).sort_values(
    ["missing_pct", "feature"], ascending=[False, True]
)

full_csv = OUTDIR / "nacc_AT_strict_model_table_full_v3.csv"
nodx_withtracer_csv = OUTDIR / "nacc_AT_strict_model_table_nodx_withtracer_v3.csv"
nodx_notracer_csv = OUTDIR / "nacc_AT_strict_model_table_nodx_notracer_v3.csv"
withdx_csv = OUTDIR / "nacc_AT_strict_model_table_withdx_v3.csv"
tr6_csv = OUTDIR / "nacc_AT_strict_tracer6_model_table_nodx_v3.csv"
missing_csv = OUTDIR / "nacc_AT_strict_feature_missingness_v3.csv"
manifest_json = OUTDIR / "nacc_AT_strict_model_manifest_v3.json"

out.to_csv(full_csv, index=False)

nodx_withtracer_cols = ["NACCID", "visit_date", "AT_group", "y_AT_strict", "tau_tracer"] + list(feature_map.keys())
nodx_withtracer_cols = [c for c in nodx_withtracer_cols if c in out.columns]
out[nodx_withtracer_cols].to_csv(nodx_withtracer_csv, index=False)

nodx_notracer_cols = ["NACCID", "visit_date", "AT_group", "y_AT_strict"] + [c for c in feature_map.keys() if c != "tau_tracer"]
nodx_notracer_cols = [c for c in nodx_notracer_cols if c in out.columns]
out[nodx_notracer_cols].to_csv(nodx_notracer_csv, index=False)

withdx_cols = ["NACCID", "visit_date", "AT_group", "y_AT_strict", "dx_harmonized", "tau_tracer"] + list(feature_map.keys())
withdx_cols = [c for c in withdx_cols if c in out.columns]
out[withdx_cols].to_csv(withdx_csv, index=False)

tr6 = out[out["tau_tracer"].astype("string") == "6"].copy()
tr6_cols = [c for c in nodx_notracer_cols if c in tr6.columns]
tr6[tr6_cols].to_csv(tr6_csv, index=False)

missing_df.to_csv(missing_csv, index=False)

manifest = {
    "strict_input": str(STRICT_IN),
    "master_input": str(MASTER_IN),
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
print(missing_df.head(25).to_string(index=False))

print("\nStrict class counts:")
print(out["AT_group"].value_counts(dropna=False).to_string())