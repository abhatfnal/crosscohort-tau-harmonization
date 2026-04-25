#!/usr/bin/env python3

import re
import json
import numpy as np
import pandas as pd

# =============================================================================
# PATHS
# =============================================================================
# ---- OASIS-3
o3_fs_path = "/project/aereditato/abhat/oasis/docs/oasis3_data_files/OASIS3_data_files/scans/FS-Freesurfer_output/resources/csv/files/OASIS3_Freesurfer_output.csv"
o3_main_in = "/project/aereditato/abhat/oasis/phase0/oasis3_clinical_master_v5_tau180_amy180_fs180.csv"
o3_strict_in = "/project/aereditato/abhat/oasis/phase0/oasis3_clinical_master_v5_tau90_amy90_fs90.csv"

o3_fs_fixed_out = "/project/aereditato/abhat/oasis/phase0/oasis3_fs_summary_table_v3_fixed.csv"
o3_main_fixed_out = "/project/aereditato/abhat/oasis/phase0/oasis3_clinical_master_v7_tau180_amy180_fs180_fixed.csv"
o3_strict_fixed_out = "/project/aereditato/abhat/oasis/phase0/oasis3_clinical_master_v7_tau90_amy90_fs90_fixed.csv"

# ---- OASIS-4
o4_demo_path = "/project/aereditato/abhat/oasis/docs/oasis4_data_files/OASIS4_data_files/scans/demo-demographics/resources/csv/files/OASIS4_data_demographics.csv"
o4_clin_path = "/project/aereditato/abhat/oasis/docs/oasis4_data_files/OASIS4_data_files/scans/clinical-OASIS4_data_clinical/resources/csv/files/OASIS4_data_clinical.csv"
o4_cdr_path = "/project/aereditato/abhat/oasis/docs/oasis4_data_files/OASIS4_data_files/scans/CDR-OASIS4_data_CDR/resources/csv/files/OASIS4_data_CDR.csv"
o4_neuro_path = "/project/aereditato/abhat/oasis/docs/oasis4_data_files/OASIS4_data_files/scans/neuropsych-OASIS4_data_Cognitive_Assessments/resources/csv/files/OASIS4_data_Neuropsychometric.csv"
o4_img_path = "/project/aereditato/abhat/oasis/docs/oasis4_data_files/OASIS4_data_files/scans/imaging-OASIS4_data_imaging/resources/csv/files/OASIS4_data_imaging.csv"
o4_csf_path = "/project/aereditato/abhat/oasis/docs/oasis4_data_files/OASIS4_data_files/scans/CSF-OASIS4_data_CSF/resources/csv/files/OASIS4_data_CSF.csv"

o4_master_out = "/project/aereditato/abhat/oasis/phase0/oasis4_clinical_master_v2.csv"
o4_summary_out = "/project/aereditato/abhat/oasis/phase0/oasis4_clinical_master_v2_summary.json"

# =============================================================================
# HELPERS
# =============================================================================
def print_block(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def extract_days(val):
    if pd.isna(val):
        return np.nan
    m = re.search(r"_d(\d+)", str(val))
    return float(m.group(1)) if m else np.nan

def safe_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def detect_subject_col(df):
    preferred = [
        "OASISID", "OASIS_ID", "oasis_id", "OASISID_x", "Subject", "subject_id",
        "subject", "OASISID_y"
    ]
    for c in preferred:
        if c in df.columns:
            return c
    for c in df.columns:
        nc = norm(c)
        if nc in {"oasisid", "oasis id", "subject", "subject id"}:
            return c
    raise KeyError("Could not detect subject column")

def detect_day_col(df):
    preferred = [
        "days_to_visit", "visit_days", "day", "days", "demographics_firstvisit",
        "age at visit", "image_age"
    ]
    for c in preferred:
        if c in df.columns:
            return c
    # Fall back to parsing an ID column with _d####
    for c in df.columns:
        s = df[c].astype(str)
        if s.str.contains(r"_d\d+", regex=True, na=False).sum() > 0:
            return c
    return None

def make_subject_key(df, colname="OASISID"):
    subj_col = detect_subject_col(df)
    df = df.copy()
    df[colname] = df[subj_col].astype(str)
    return df

def find_column_by_tokens(columns, must_have, must_not_have=None):
    must_not_have = must_not_have or []
    hits = []
    for c in columns:
        nc = norm(c)
        if all(tok in nc for tok in must_have) and not any(tok in nc for tok in must_not_have):
            hits.append(c)
    if not hits:
        return None
    hits = sorted(hits, key=lambda x: (len(norm(x)), x))
    return hits[0]

def asof_merge_by_subject(left, right, left_day, right_day, tolerance):
    parts = []

    left = left.copy()
    right = right.copy()

    left[left_day] = pd.to_numeric(left[left_day], errors="coerce").astype(float)
    right[right_day] = pd.to_numeric(right[right_day], errors="coerce").astype(float)

    left = left.dropna(subset=["OASISID", left_day]).copy()
    right = right.dropna(subset=["OASISID", right_day]).copy()

    for sid, lsub in left.groupby("OASISID", sort=False):
        rsub = right[right["OASISID"] == sid].copy()
        lsub = lsub.sort_values(left_day).reset_index(drop=True)

        if rsub.empty:
            out = lsub.copy()
            for c in right.columns:
                if c not in out.columns:
                    out[c] = np.nan
        else:
            rsub = rsub.sort_values(right_day).reset_index(drop=True)
            rsub = rsub.drop(columns=["OASISID"], errors="ignore")
            out = pd.merge_asof(
                lsub,
                rsub,
                left_on=left_day,
                right_on=right_day,
                direction="nearest",
                tolerance=float(tolerance),
            )

        out["OASISID"] = sid
        parts.append(out)

    return pd.concat(parts, ignore_index=True)

# =============================================================================
# PART 1: OASIS-3 FS FIX
# =============================================================================
print_block("PART 1: FIX OASIS-3 FREESURFER TABLE")

fs = pd.read_csv(o3_fs_path, low_memory=False)
fs = make_subject_key(fs)

# Standard keys
if "MR_session" in fs.columns:
    fs["fs_days_to_visit"] = fs["MR_session"].map(extract_days)
else:
    raise KeyError("MR_session not found in OASIS3 FS table")

fs = fs.dropna(subset=["OASISID", "fs_days_to_visit"]).copy()
fs_cols = list(fs.columns)

entorh_candidates = [c for c in fs_cols if "entorh" in norm(c)]
print("Entorhinal candidate columns:", entorh_candidates)

left_hippo_col = find_column_by_tokens(fs_cols, ["left", "hippo"])
right_hippo_col = find_column_by_tokens(fs_cols, ["right", "hippo"])
if left_hippo_col is None:
    left_hippo_col = find_column_by_tokens(fs_cols, ["lh", "hippo"])
if right_hippo_col is None:
    right_hippo_col = find_column_by_tokens(fs_cols, ["rh", "hippo"])

icv_col = (
    find_column_by_tokens(fs_cols, ["intracranial"]) or
    find_column_by_tokens(fs_cols, ["intra", "cranial"]) or
    find_column_by_tokens(fs_cols, ["icv"]) or
    find_column_by_tokens(fs_cols, ["etiv"])
)

# Stronger entorhinal detection
entorh_l_col = (
    find_column_by_tokens(fs_cols, ["lh", "entorh"], ["rh", "right"]) or
    find_column_by_tokens(fs_cols, ["left", "entorh"], ["rh", "right"]) or
    find_column_by_tokens(fs_cols, ["l", "entorh"], ["rh", "right"])
)
entorh_r_col = (
    find_column_by_tokens(fs_cols, ["rh", "entorh"], ["lh", "left"]) or
    find_column_by_tokens(fs_cols, ["right", "entorh"], ["lh", "left"]) or
    find_column_by_tokens(fs_cols, ["r", "entorh"], ["lh", "left"])
)

# Final fallback using direct substrings
if entorh_l_col is None:
    for c in fs_cols:
        lc = str(c).lower()
        if "lh_entorh" in lc or "left-entorh" in lc or "left_entorh" in lc:
            entorh_l_col = c
            break

if entorh_r_col is None:
    for c in fs_cols:
        lc = str(c).lower()
        if "rh_entorh" in lc or "right-entorh" in lc or "right_entorh" in lc:
            entorh_r_col = c
            break

detected = {
    "left_hippocampus": left_hippo_col,
    "right_hippocampus": right_hippo_col,
    "icv": icv_col,
    "left_entorhinal": entorh_l_col,
    "right_entorhinal": entorh_r_col,
}

print("Detected columns:")
for k, v in detected.items():
    print(f"  {k:<20} -> {v}")

keep_cols = ["OASISID", "fs_days_to_visit"]
for c in [left_hippo_col, right_hippo_col, icv_col, entorh_l_col, entorh_r_col]:
    if c is not None and c not in keep_cols:
        keep_cols.append(c)

fs_small = fs[keep_cols].copy()
fs_small = safe_num(fs_small, [c for c in keep_cols if c not in ["OASISID", "fs_days_to_visit"]])

if left_hippo_col is not None and right_hippo_col is not None:
    fs_small["FS_HIPPO_BILAT"] = fs_small[left_hippo_col] + fs_small[right_hippo_col]

if icv_col is not None and "FS_HIPPO_BILAT" in fs_small.columns:
    fs_small["FS_HIPPO_BILAT_ICVnorm"] = fs_small["FS_HIPPO_BILAT"] / fs_small[icv_col]

if entorh_l_col is not None and entorh_r_col is not None:
    fs_small["FS_ENTORH_BILAT"] = fs_small[entorh_l_col] + fs_small[entorh_r_col]

fs_small = fs_small.sort_values(["OASISID", "fs_days_to_visit"]).drop_duplicates(
    subset=["OASISID", "fs_days_to_visit"], keep="first"
)
fs_small.to_csv(o3_fs_fixed_out, index=False)

print("Wrote:", o3_fs_fixed_out)
print("Rows:", len(fs_small), " Unique subjects:", fs_small["OASISID"].nunique())
for c in ["FS_HIPPO_BILAT", "FS_HIPPO_BILAT_ICVnorm", "FS_ENTORH_BILAT"]:
    if c in fs_small.columns:
        print(f"With {c}: {int(fs_small[c].notna().sum())}")

# rebuild OASIS-3 masters
for in_path, out_path, label, tol in [
    (o3_main_in, o3_main_fixed_out, "MAIN", 180),
    (o3_strict_in, o3_strict_fixed_out, "STRICT", 90),
]:
    master = pd.read_csv(in_path, low_memory=False)
    master["days_to_visit"] = pd.to_numeric(master["days_to_visit"], errors="coerce").astype(float)

    stale_cols = [c for c in master.columns if c.startswith("FS_") or c in ["fs_days_to_visit", "fs_day_diff"]]
    master = master.drop(columns=stale_cols, errors="ignore")

    merged = asof_merge_by_subject(
        master, fs_small,
        left_day="days_to_visit",
        right_day="fs_days_to_visit",
        tolerance=tol
    )
    merged["fs_day_diff"] = merged["fs_days_to_visit"] - merged["days_to_visit"]
    merged.to_csv(out_path, index=False)

    print_block(f"OASIS-3 {label} MASTER REBUILT")
    print("Wrote:", out_path)
    print("Rows:", len(merged))
    print("Matched FS:", int(merged["fs_days_to_visit"].notna().sum()))
    if "FS_ENTORH_BILAT" in merged.columns:
        print("With FS_ENTORH_BILAT:", int(merged["FS_ENTORH_BILAT"].notna().sum()))
    if merged["fs_day_diff"].notna().any():
        print("Median abs FS diff:", float(merged["fs_day_diff"].abs().median()))
        print("90th pct abs FS diff:", float(merged["fs_day_diff"].abs().quantile(0.90)))

# =============================================================================
# PART 2: BUILD OASIS-4 MASTER
# =============================================================================
print_block("PART 2: BUILD OASIS-4 CLINICAL MASTER")

demo = pd.read_csv(o4_demo_path, low_memory=False)
clin = pd.read_csv(o4_clin_path, low_memory=False)
cdr = pd.read_csv(o4_cdr_path, low_memory=False)
neuro = pd.read_csv(o4_neuro_path, low_memory=False)
img = pd.read_csv(o4_img_path, low_memory=False)
csf = pd.read_csv(o4_csf_path, low_memory=False)

print("Loaded rows:")
print("  demo :", len(demo))
print("  clin :", len(clin))
print("  cdr  :", len(cdr))
print("  neuro:", len(neuro))
print("  img  :", len(img))
print("  csf  :", len(csf))

# ---- standardize keys
demo = make_subject_key(demo)
clin = make_subject_key(clin)
cdr = make_subject_key(cdr)
neuro = make_subject_key(neuro)
img = make_subject_key(img)
csf = make_subject_key(csf)

# ---- subject-level demo table (no visit-day assumption)
demo_subject = demo.copy()
demo_drop = [c for c in demo_subject.columns if c.lower() in {"demographics_firstvisit", "visit_days", "days_to_visit"}]
demo_subject = demo_subject.drop(columns=demo_drop, errors="ignore")
demo_subject = demo_subject.groupby("OASISID", as_index=False).first()

# prefix demo extras to avoid collisions
rename_demo = {}
for c in demo_subject.columns:
    if c != "OASISID":
        rename_demo[c] = f"demo_{c}"
demo_subject = demo_subject.rename(columns=rename_demo)

# ---- clinical subject-level table
clin["days_to_visit"] = pd.to_numeric(clin.get("demographics_firstvisit", np.nan), errors="coerce")
clin["age_baseline"] = pd.to_numeric(clin.get("age", np.nan), errors="coerce")
clin["education_years"] = pd.to_numeric(clin.get("edu", np.nan), errors="coerce")
clin["sex_harmonized"] = clin.get("sex", np.nan).map({1: "M", 2: "F"})

dx_txt = clin.get("final_dx", pd.Series("", index=clin.index)).astype(str).str.lower()
clin["dx_harmonized"] = "Other/Unknown"
clin.loc[dx_txt.str.contains("normal|cognitively normal", na=False), "dx_harmonized"] = "CN"
clin.loc[dx_txt.str.contains("mci|mild cognitive", na=False), "dx_harmonized"] = "MCI"
clin.loc[dx_txt.str.contains("alzheimer|dementia|ad possible|ad probable", na=False), "dx_harmonized"] = "Dementia"

clin_subject_cols = [c for c in [
    "OASISID", "days_to_visit", "sex_harmonized", "age_baseline",
    "education_years", "race", "hispanic", "bmi",
    "final_dx", "final_dx_categorized", "dx_harmonized"
] if c in clin.columns]
clin_subject = clin[clin_subject_cols].sort_values(["OASISID", "days_to_visit"]).groupby("OASISID", as_index=False).first()

# ---- CDR visit table
cdr["days_to_visit"] = pd.to_numeric(cdr.get("visit_days", np.nan), errors="coerce")
cdr_small = cdr[[c for c in [
    "OASISID", "days_to_visit", "cdr", "sumbox",
    "memory", "orient", "judgement", "community", "homehobb", "perscare"
] if c in cdr.columns]].copy()
cdr_small = cdr_small.rename(columns={"cdr": "CDRTOT", "sumbox": "CDRSUM"})
cdr_small = safe_num(cdr_small, [c for c in cdr_small.columns if c not in ["OASISID", "days_to_visit"]])

# ---- neuropsych visit table
neuro["days_to_visit"] = pd.to_numeric(neuro.get("visit_days", np.nan), errors="coerce")
neuro_small = neuro[[c for c in [
    "OASISID", "days_to_visit", "mmse", "mem_word", "recall_word", "short_blessed",
    "logimem", "traila_sec", "trailb_sec", "digitsym_second", "gds_total",
    "boston", "verb_fleunc"
] if c in neuro.columns]].copy()
neuro_small = neuro_small.rename(columns={
    "mmse": "MMSE",
    "logimem": "LOGIMEM",
    "verb_fleunc": "ANIMALS",
})
neuro_small = safe_num(neuro_small, [c for c in neuro_small.columns if c not in ["OASISID", "days_to_visit"]])

# ---- imaging visit table
img["days_to_visit"] = pd.to_numeric(img.get("visit_days", np.nan), errors="coerce")
img_small = img[[c for c in [
    "OASISID", "days_to_visit", "imaging_id", "Central_accession",
    "scanner", "contrast", "contrastx", "image_age"
] if c in img.columns]].copy()
img_small = safe_num(img_small, [c for c in ["image_age", "scanner", "contrast"] if c in img_small.columns])

# ---- CSF visit table
csf_day_col = None
for cand in ["visit_days", "days_to_visit"]:
    if cand in csf.columns:
        csf_day_col = cand
        break
if csf_day_col is None:
    csf_day_col = detect_day_col(csf)

if csf_day_col is not None:
    if csf_day_col in csf.columns and csf_day_col not in {"visit_days", "days_to_visit"}:
        csf["days_to_visit"] = csf[csf_day_col].map(extract_days)
    else:
        csf["days_to_visit"] = pd.to_numeric(csf[csf_day_col], errors="coerce")
else:
    csf["days_to_visit"] = np.nan

csf_keep = ["OASISID", "days_to_visit"]
for c in csf.columns:
    nc = norm(c)
    if any(k in nc for k in ["abeta", "amyloid", "tau", "ptau", "ttau", "ratio"]):
        if c not in csf_keep:
            csf_keep.append(c)
csf_small = csf[[c for c in csf_keep if c in csf.columns]].copy()
csf_small = safe_num(csf_small, [c for c in csf_small.columns if c not in ["OASISID", "days_to_visit"]])

# ---- Merge around CDR anchor
anchor = cdr_small.rename(columns={"days_to_visit": "cdr_days_to_visit"})

neuro_r = neuro_small.rename(columns={"days_to_visit": "neuro_days_to_visit"})
m1 = asof_merge_by_subject(anchor, neuro_r, "cdr_days_to_visit", "neuro_days_to_visit", 365)
m1["neuro_day_diff"] = m1["neuro_days_to_visit"] - m1["cdr_days_to_visit"]

img_r = img_small.rename(columns={"days_to_visit": "img_days_to_visit"})
m2 = asof_merge_by_subject(m1, img_r, "cdr_days_to_visit", "img_days_to_visit", 365)
m2["img_day_diff"] = m2["img_days_to_visit"] - m2["cdr_days_to_visit"]

if csf_small["days_to_visit"].notna().any():
    csf_r = csf_small.rename(columns={"days_to_visit": "csf_days_to_visit"})
    m3 = asof_merge_by_subject(m2, csf_r, "cdr_days_to_visit", "csf_days_to_visit", 365)
    m3["csf_day_diff"] = m3["csf_days_to_visit"] - m3["cdr_days_to_visit"]
else:
    m3 = m2.copy()

master = m3.merge(clin_subject.drop(columns=["days_to_visit"], errors="ignore"), on="OASISID", how="left")
master = master.merge(demo_subject, on="OASISID", how="left")
master = master.rename(columns={"cdr_days_to_visit": "days_to_visit"})

master.to_csv(o4_master_out, index=False)

print_block("OASIS-4 MASTER COMPLETE")
print("Wrote:", o4_master_out)
print("Rows:", len(master))
print("Unique subjects:", master["OASISID"].nunique())
print("With CDR:", int(master["CDRTOT"].notna().sum()) if "CDRTOT" in master.columns else 0)
print("With MMSE:", int(master["MMSE"].notna().sum()) if "MMSE" in master.columns else 0)
print("With LOGIMEM:", int(master["LOGIMEM"].notna().sum()) if "LOGIMEM" in master.columns else 0)
print("With imaging match:", int(master["img_days_to_visit"].notna().sum()) if "img_days_to_visit" in master.columns else 0)
if "csf_days_to_visit" in master.columns:
    print("With CSF match:", int(master["csf_days_to_visit"].notna().sum()))

if "neuro_day_diff" in master.columns and master["neuro_day_diff"].notna().any():
    print("Median abs neuro diff:", float(master["neuro_day_diff"].abs().median()))
    print("90th pct abs neuro diff:", float(master["neuro_day_diff"].abs().quantile(0.90)))

if "img_day_diff" in master.columns and master["img_day_diff"].notna().any():
    print("Median abs img diff:", float(master["img_day_diff"].abs().median()))
    print("90th pct abs img diff:", float(master["img_day_diff"].abs().quantile(0.90)))

print("\nDiagnosis distribution:")
if "dx_harmonized" in master.columns:
    print(master["dx_harmonized"].value_counts(dropna=False).to_string())

print("\nExample rows:")
show_cols = [c for c in [
    "OASISID", "days_to_visit", "CDRTOT", "CDRSUM", "MMSE", "LOGIMEM",
    "sex_harmonized", "age_baseline", "education_years", "dx_harmonized",
    "img_days_to_visit", "img_day_diff", "imaging_id", "Central_accession"
] if c in master.columns]
print(master[show_cols].head(12).to_string(index=False))

summary = {
    "oasis3_detected_columns": detected,
    "oasis3_fs_rows": int(len(fs_small)),
    "oasis3_fs_subjects": int(fs_small["OASISID"].nunique()),
    "oasis4_rows": int(len(master)),
    "oasis4_subjects": int(master["OASISID"].nunique()),
    "oasis4_with_cdr": int(master["CDRTOT"].notna().sum()) if "CDRTOT" in master.columns else 0,
    "oasis4_with_mmse": int(master["MMSE"].notna().sum()) if "MMSE" in master.columns else 0,
    "oasis4_with_imaging": int(master["img_days_to_visit"].notna().sum()) if "img_days_to_visit" in master.columns else 0,
}

if "dx_harmonized" in master.columns:
    summary["oasis4_dx_counts"] = master["dx_harmonized"].value_counts(dropna=False).to_dict()

with open(o4_summary_out, "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved summary JSON:", o4_summary_out)
print("\nDone.")