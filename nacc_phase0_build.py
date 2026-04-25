#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================
BASE = Path("/project/aereditato/abhat/NACC_17259")
TABLES = BASE / "tables"
SCAN_MRI = BASE / "scan_mri"
SCAN_PET = BASE / "scan_pet"
OUTDIR = BASE / "phase0"
OUTDIR.mkdir(parents=True, exist_ok=True)

UDS_PATH = TABLES / "investigator_nacc72.csv"
MIXED_MRI_PATH = TABLES / "investigator_mri_nacc72.csv"
CSF_PATH = TABLES / "investigator_fcsf_nacc72.csv"

SCAN_MRI_QC_PATH = SCAN_MRI / "investigator_scan_mriqc_nacc72.csv"
SCAN_MRI_SBM_PATH = SCAN_MRI / "investigator_scan_mrisbm_nacc72.csv"

SCAN_AMY_GAAIN_PATH = SCAN_PET / "investigator_scan_amyloidpetgaain_nacc72.csv"
SCAN_AMY_NPDKA_PATH = SCAN_PET / "investigator_scan_amyloidpetnpdka_nacc72.csv"
SCAN_FDG_NPDKA_PATH = SCAN_PET / "investigator_scan_fdgpetnpdka_nacc72.csv"
SCAN_TAU_NPDKA_PATH = SCAN_PET / "investigator_scan_taupetnpdka_nacc72.csv"

MP_AMY_GAAIN_PATH = SCAN_PET / "investigator_scan_mp_amyloidpetgaain_nacc72.csv"
MP_AMY_NPDKA_PATH = SCAN_PET / "investigator_scan_mp_amyloidpetnpdka_nacc72.csv"
MP_TAU_NPDKA_PATH = SCAN_PET / "investigator_scan_mp_taupetnpdka_nacc72.csv"

# If later you get a mixed-protocol FDG mp file, add it here if it exists
MP_FDG_NPDKA_PATH = SCAN_PET / "investigator_scan_mp_fdgpetnpdka_nacc72.csv"

TOLERANCES = [180, 90]


# =============================================================================
# Helpers
# =============================================================================
def read_csv(path):
    print(f"Loading: {path}")
    return pd.read_csv(path, low_memory=False)


def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def replace_nacc_missing_codes(df):
    """
    Broad phase-0 cleanup for common NACC sentinel codes.
    This is intentionally aggressive for numeric columns only.
    """
    sentinels = {
        -4, -4.0,
        8, 8.0,
        88, 88.0,
        888, 888.0,
        8888, 8888.0,
        999, 999.0,
        9999, 9999.0,
        99999, 99999.0,
        9999.999, 99999.999,
    }

    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = df[c].where(~df[c].isin(sentinels), np.nan)
    return df


def build_date_from_ymd(df, year_col, mo_col, day_col, out_col):
    y = pd.to_numeric(df[year_col], errors="coerce") if year_col in df.columns else np.nan
    m = pd.to_numeric(df[mo_col], errors="coerce") if mo_col in df.columns else np.nan
    d = pd.to_numeric(df[day_col], errors="coerce") if day_col in df.columns else np.nan

    tmp = pd.DataFrame({"year": y, "month": m, "day": d})
    tmp = tmp.where(pd.notnull(tmp), np.nan)
    df[out_col] = pd.to_datetime(tmp, errors="coerce")
    return df


def build_date_from_string(df, src_col, out_col):
    if src_col in df.columns:
        df[out_col] = pd.to_datetime(df[src_col], errors="coerce")
    else:
        df[out_col] = pd.NaT
    return df


def harmonize_sex(series):
    mapping = {1: "M", 2: "F"}
    s = pd.to_numeric(series, errors="coerce")
    return s.map(mapping)


def compute_dx_harmonized(df):
    df = df.copy()

    cn_col = first_existing(df, ["NORMCOG"])
    dem_col = first_existing(df, ["DEMENTED"])
    mci_candidates = [c for c in ["NACCTMCI", "NACCMCIL", "NACCMCIA", "NACCMCIE", "NACCMCIV", "NACCMCII"] if c in df.columns]

    if cn_col:
        df[cn_col] = pd.to_numeric(df[cn_col], errors="coerce")
    if dem_col:
        df[dem_col] = pd.to_numeric(df[dem_col], errors="coerce")
    for c in mci_candidates:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    dx = pd.Series("Other/Unknown", index=df.index, dtype=object)

    if cn_col:
        dx.loc[df[cn_col] == 1] = "CN"
    if mci_candidates:
        mci_mask = df[mci_candidates].fillna(0).eq(1).any(axis=1)
        dx.loc[mci_mask] = "MCI"
    if dem_col:
        dx.loc[df[dem_col] == 1] = "Dementia"

    return dx


def dedupe_best_row(df, subset_cols):
    """
    Keep the row with the highest non-null count per key.
    """
    df = df.copy()
    df["_nn"] = df.notna().sum(axis=1)
    df = df.sort_values(subset_cols + ["_nn"], ascending=[True] * len(subset_cols) + [False])
    df = df.drop_duplicates(subset=subset_cols, keep="first")
    return df.drop(columns="_nn")


def asof_merge_by_subject_date(left, right, left_date_col, right_date_col, subject_col="NACCID", tolerance_days=180):
    left = left.copy()
    right = right.copy()

    left[left_date_col] = pd.to_datetime(left[left_date_col], errors="coerce")
    right[right_date_col] = pd.to_datetime(right[right_date_col], errors="coerce")

    left = left.dropna(subset=[subject_col, left_date_col]).copy()
    right = right.dropna(subset=[subject_col, right_date_col]).copy()

    # Drop overlapping non-key columns from the right table so repeated merges
    # do not create NACCADC_x / NACCADC_y / duplicate suffix collisions.
    protected = {subject_col, right_date_col}
    overlap = [c for c in right.columns if c in left.columns and c not in protected]
    if overlap:
        right = right.drop(columns=overlap)

    out_parts = []
    tol = pd.Timedelta(days=tolerance_days)

    for sid, lsub in left.groupby(subject_col, sort=False):
        rsub = right[right[subject_col] == sid].copy()
        lsub = lsub.sort_values(left_date_col).reset_index(drop=True)

        if rsub.empty:
            out = lsub.copy()
            for c in right.columns:
                if c not in out.columns:
                    out[c] = np.nan
        else:
            rsub = rsub.sort_values(right_date_col).reset_index(drop=True)
            out = pd.merge_asof(
                lsub,
                rsub,
                left_on=left_date_col,
                right_on=right_date_col,
                direction="nearest",
                tolerance=tol,
                allow_exact_matches=True,
                suffixes=("", "_r"),
            )

            # Clean up accidental duplicate subject columns if pandas makes them
            if f"{subject_col}_r" in out.columns:
                out = out.drop(columns=[f"{subject_col}_r"])

        out_parts.append(out)

    merged = pd.concat(out_parts, ignore_index=True)
    return merged

def print_block(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# 1) Build UDS visit-level clinical backbone
# =============================================================================
print_block("BUILD UDS VISIT MASTER")

uds = read_csv(UDS_PATH)
uds = replace_nacc_missing_codes(uds)

uds = build_date_from_ymd(uds, "VISITYR", "VISITMO", "VISITDAY", "visit_date")

faq_items = [c for c in ["BILLS", "TAXES", "SHOPPING", "GAMES", "STOVE", "MEALPREP", "EVENTS", "PAYATTN", "REMDATES", "TRAVEL"] if c in uds.columns]
if faq_items:
    uds = safe_num(uds, faq_items)
    uds["FAQ_TOTAL"] = uds[faq_items].sum(axis=1, min_count=1)
else:
    uds["FAQ_TOTAL"] = np.nan

uds["sex_harmonized"] = harmonize_sex(uds["SEX"]) if "SEX" in uds.columns else np.nan
uds["dx_harmonized"] = compute_dx_harmonized(uds)

if "NACCNE4S" in uds.columns:
    uds["apoe_e4_count"] = pd.to_numeric(uds["NACCNE4S"], errors="coerce")
    uds["apoe_e4_carrier"] = (uds["apoe_e4_count"] > 0).astype("float")
else:
    uds["apoe_e4_count"] = np.nan
    uds["apoe_e4_carrier"] = np.nan

keep_uds = [c for c in [
    "NACCID", "NACCADC",
    "visit_date", "NACCDAYS", "NACCVNUM", "PACKET", "FORMVER",
    "NACCAGE", "SEX", "sex_harmonized", "EDUC", "RACE", "HISPANIC", "HANDED",
    "NACCNE4S", "apoe_e4_count", "apoe_e4_carrier",
    "NORMCOG", "DEMENTED", "NACCTMCI", "NACCMCIL", "NACCMCIA", "NACCMCIE", "NACCMCIV", "NACCMCII", "dx_harmonized",
    "CDRGLOB", "CDRSUM", "MEMORY", "ORIENT", "JUDGMENT", "COMMUN", "HOMEHOBB", "PERSCARE",
    "NACCMMSE", "NACCMOCA",
    "LOGIMEM", "ANIMALS", "VEG", "DIGIF", "DIGIB", "BOSTON",
    "FAQ_TOTAL",
    "CSFABETA", "CSFPTAU", "CSFTTAU",  # these may not exist here; harmless if absent
] if c in uds.columns]

uds_master = uds[keep_uds].copy()
uds_master = dedupe_best_row(uds_master, ["NACCID", "visit_date"])
uds_master_out = OUTDIR / "nacc_uds_visit_master.csv"
uds_master.to_csv(uds_master_out, index=False)

print("Wrote:", uds_master_out)
print("Rows:", len(uds_master))
print("Unique subjects:", uds_master["NACCID"].nunique())
print("Diagnosis distribution:")
print(uds_master["dx_harmonized"].value_counts(dropna=False).to_string())


# =============================================================================
# 2) Mixed-protocol MRI summary table
# =============================================================================
print_block("BUILD MIXED-PROTOCOL MRI SUMMARY")

mixed_mri = read_csv(MIXED_MRI_PATH)
mixed_mri = replace_nacc_missing_codes(mixed_mri)
mixed_mri = build_date_from_ymd(mixed_mri, "MRIYR", "MRIMO", "MRIDY", "mri_date")

mixed_cols = [c for c in [
    "NACCID", "NACCADC", "mri_date", "NACCMRFI", "NACCMNUM", "NACCMRDY", "NACCMRSA",
    "MRIT1", "MRIT2", "MRIFLAIR", "MRIDTI", "MRIDWI",
    "NACCMVOL", "NACCICV", "NACCBRNV", "NACCWMVL", "WMHVOL",
    "HIPPOVOL", "LHIPPO", "RHIPPO",
    "LENT", "RENT",
    "LFUS", "RFUS",
    "LPARHIP", "RPARHIP",
    "LTEMPCOR", "RTEMPCOR", "TEMPCOR",
] if c in mixed_mri.columns]

mixed_mri = mixed_mri[mixed_cols].copy()

rename_mixed = {}
for c in mixed_mri.columns:
    if c not in ["NACCID", "NACCADC", "mri_date"]:
        rename_mixed[c] = f"MPMRI_{c}"
mixed_mri = mixed_mri.rename(columns=rename_mixed)
mixed_mri = dedupe_best_row(mixed_mri, ["NACCID", "mri_date"])

mixed_mri_out = OUTDIR / "nacc_mixed_mri_summary.csv"
mixed_mri.to_csv(mixed_mri_out, index=False)

print("Wrote:", mixed_mri_out)
print("Rows:", len(mixed_mri))
print("Unique subjects:", mixed_mri["NACCID"].nunique())


# =============================================================================
# 3) SCAN MRI morphometry table
# =============================================================================
print_block("BUILD SCAN MRI SUMMARY")

scan_mri = read_csv(SCAN_MRI_SBM_PATH)
scan_mri = replace_nacc_missing_codes(scan_mri)
scan_mri = build_date_from_string(scan_mri, "SCANDT", "scan_mri_date")

scan_mri_cols = [c for c in [
    "NACCID", "NACCADC", "scan_mri_date", "LONI_IMAGE_T1", "DESCRIPTION_T1", "FREESURFER_VERSION",
    "ESTIMATEDTOTALINTRACRANIALVOL", "BRAINSEG", "CORTEX", "CEREBRALWHITEMATTER", "WM_HYPOINTENSITIES",
    "LEFT_HIPPOCAMPUS", "RIGHT_HIPPOCAMPUS", "HIPPOCAMPUS",
    "LH_ENTORHINAL_GVOL", "RH_ENTORHINAL_GVOL",
    "LH_ENTORHINAL_AVGTH", "RH_ENTORHINAL_AVGTH",
    "LH_PARAHIPPOCAMPAL_GVOL", "RH_PARAHIPPOCAMPAL_GVOL",
    "LH_FUSIFORM_GVOL", "RH_FUSIFORM_GVOL",
    "LEFT_AMYGDALA", "RIGHT_AMYGDALA",
] if c in scan_mri.columns]

scan_mri = scan_mri[scan_mri_cols].copy()

rename_scan_mri = {}
for c in scan_mri.columns:
    if c not in ["NACCID", "NACCADC", "scan_mri_date"]:
        rename_scan_mri[c] = f"SCANMRI_{c}"
scan_mri = scan_mri.rename(columns=rename_scan_mri)

scan_mri = dedupe_best_row(scan_mri, ["NACCID", "scan_mri_date"])
scan_mri_out = OUTDIR / "nacc_scan_mri_summary.csv"
scan_mri.to_csv(scan_mri_out, index=False)

print("Wrote:", scan_mri_out)
print("Rows:", len(scan_mri))
print("Unique subjects:", scan_mri["NACCID"].nunique())


# =============================================================================
# 4) Amyloid summary table (SCAN + mixed protocol)
# =============================================================================
print_block("BUILD AMYLOID SUMMARY")

amy_gaain = read_csv(SCAN_AMY_GAAIN_PATH)
amy_np = read_csv(SCAN_AMY_NPDKA_PATH)
amy_gaain = replace_nacc_missing_codes(amy_gaain)
amy_np = replace_nacc_missing_codes(amy_np)
amy_gaain = build_date_from_string(amy_gaain, "SCANDATE", "amy_date")
amy_np = build_date_from_string(amy_np, "SCANDATE", "amy_date")

amy_gaain_keep = [c for c in [
    "NACCID", "NACCADC", "amy_date", "LONIUID", "TRACER", "AMYLOID_STATUS", "CENTILOIDS",
    "GAAIN_SUMMARY_SUVR", "GAAIN_WHOLECEREBELLUM_SUVR", "GAAIN_COMPOSITE_REF_SUVR"
] if c in amy_gaain.columns]
amy_np_keep = [c for c in [
    "NACCID", "NACCADC", "amy_date", "LONIUID", "TRACER",
    "NPDKA_SUMMARY_SUVR",
    "CTX_POSTERIORCINGULATE_SUVR", "CTX_PRECUNEUS_SUVR", "CTX_ENTORHINAL_SUVR",
    "CTX_PARAHIPPOCAMPAL_SUVR", "CTX_INFERIORTEMPORAL_SUVR", "CTX_FUSIFORM_SUVR",
    "HIPPOCAMPUS_SUVR", "AMYGDALA_SUVR"
] if c in amy_np.columns]

amy_scan = pd.merge(
    amy_gaain[amy_gaain_keep],
    amy_np[amy_np_keep],
    on=[c for c in ["NACCID", "NACCADC", "amy_date", "LONIUID", "TRACER"] if c in amy_gaain_keep and c in amy_np_keep],
    how="outer"
)
amy_scan["AMY_SOURCE"] = "SCAN"

mp_amy_gaain = read_csv(MP_AMY_GAAIN_PATH)
mp_amy_np = read_csv(MP_AMY_NPDKA_PATH)
mp_amy_gaain = replace_nacc_missing_codes(mp_amy_gaain)
mp_amy_np = replace_nacc_missing_codes(mp_amy_np)
mp_amy_gaain = build_date_from_string(mp_amy_gaain, "SCANDATE", "amy_date")
mp_amy_np = build_date_from_string(mp_amy_np, "SCANDATE", "amy_date")

mp_amy_gaain_keep = [c for c in [
    "NACCID", "NACCADC", "amy_date", "LONIUID", "TRACER", "SCAN_PROJECT", "AMYLOID_STATUS", "CENTILOIDS",
    "GAAIN_SUMMARY_SUVR", "GAAIN_WHOLECEREBELLUM_SUVR", "GAAIN_COMPOSITE_REF_SUVR"
] if c in mp_amy_gaain.columns]
mp_amy_np_keep = [c for c in [
    "NACCID", "NACCADC", "amy_date", "LONIUID", "TRACER", "SCAN_PROJECT",
    "NPDKA_SUMMARY_SUVR",
    "CTX_POSTERIORCINGULATE_SUVR", "CTX_PRECUNEUS_SUVR", "CTX_ENTORHINAL_SUVR",
    "CTX_PARAHIPPOCAMPAL_SUVR", "CTX_INFERIORTEMPORAL_SUVR", "CTX_FUSIFORM_SUVR",
    "HIPPOCAMPUS_SUVR", "AMYGDALA_SUVR"
] if c in mp_amy_np.columns]

amy_mp = pd.merge(
    mp_amy_gaain[mp_amy_gaain_keep],
    mp_amy_np[mp_amy_np_keep],
    on=[c for c in ["NACCID", "NACCADC", "amy_date", "LONIUID", "TRACER", "SCAN_PROJECT"] if c in mp_amy_gaain_keep and c in mp_amy_np_keep],
    how="outer"
)
amy_mp["AMY_SOURCE"] = "MIXED_PROTOCOL"

amy = pd.concat([amy_scan, amy_mp], ignore_index=True, sort=False)

rename_amy = {}
for c in amy.columns:
    if c not in ["NACCID", "NACCADC", "amy_date"]:
        rename_amy[c] = f"AMY_{c}"
amy = amy.rename(columns=rename_amy)

amy = dedupe_best_row(amy, ["NACCID", "amy_date"])
amy_out = OUTDIR / "nacc_amyloid_summary.csv"
amy.to_csv(amy_out, index=False)

print("Wrote:", amy_out)
print("Rows:", len(amy))
print("Unique subjects:", amy["NACCID"].nunique())
for c in ["AMY_CENTILOIDS", "AMY_GAAIN_SUMMARY_SUVR", "AMY_NPDKA_SUMMARY_SUVR"]:
    if c in amy.columns:
        print(f"With {c}: {amy[c].notna().sum()}")


# =============================================================================
# 5) Tau summary table (SCAN + mixed protocol)
# =============================================================================
print_block("BUILD TAU SUMMARY")

tau_scan = read_csv(SCAN_TAU_NPDKA_PATH)
tau_scan = replace_nacc_missing_codes(tau_scan)
tau_scan = build_date_from_string(tau_scan, "SCANDATE", "tau_date")

tau_scan_keep = [c for c in [
    "NACCID", "NACCADC", "tau_date", "LONIUID", "TRACER",
    "META_TEMPORAL_SUVR", "CTX_ENTORHINAL_SUVR", "CTX_PARAHIPPOCAMPAL_SUVR",
    "CTX_INFERIORTEMPORAL_SUVR", "CTX_FUSIFORM_SUVR",
    "CTX_POSTERIORCINGULATE_SUVR", "CTX_PRECUNEUS_SUVR",
    "HIPPOCAMPUS_SUVR", "AMYGDALA_SUVR"
] if c in tau_scan.columns]
tau_scan = tau_scan[tau_scan_keep].copy()
tau_scan["TAU_SOURCE"] = "SCAN"

tau_mp = read_csv(MP_TAU_NPDKA_PATH)
tau_mp = replace_nacc_missing_codes(tau_mp)
tau_mp = build_date_from_string(tau_mp, "SCANDATE", "tau_date")

tau_mp_keep = [c for c in [
    "NACCID", "NACCADC", "tau_date", "LONIUID", "TRACER", "SCAN_PROJECT",
    "META_TEMPORAL_SUVR", "CTX_ENTORHINAL_SUVR", "CTX_PARAHIPPOCAMPAL_SUVR",
    "CTX_INFERIORTEMPORAL_SUVR", "CTX_FUSIFORM_SUVR",
    "CTX_POSTERIORCINGULATE_SUVR", "CTX_PRECUNEUS_SUVR",
    "HIPPOCAMPUS_SUVR", "AMYGDALA_SUVR"
] if c in tau_mp.columns]
tau_mp = tau_mp[tau_mp_keep].copy()
tau_mp["TAU_SOURCE"] = "MIXED_PROTOCOL"

tau = pd.concat([tau_scan, tau_mp], ignore_index=True, sort=False)

rename_tau = {}
for c in tau.columns:
    if c not in ["NACCID", "NACCADC", "tau_date"]:
        rename_tau[c] = f"TAU_{c}"
tau = tau.rename(columns=rename_tau)

tau = dedupe_best_row(tau, ["NACCID", "tau_date"])
tau_out = OUTDIR / "nacc_tau_summary.csv"
tau.to_csv(tau_out, index=False)

print("Wrote:", tau_out)
print("Rows:", len(tau))
print("Unique subjects:", tau["NACCID"].nunique())
for c in ["TAU_META_TEMPORAL_SUVR", "TAU_CTX_ENTORHINAL_SUVR"]:
    if c in tau.columns:
        print(f"With {c}: {tau[c].notna().sum()}")


# =============================================================================
# 6) FDG summary table (SCAN + mixed protocol if present)
# =============================================================================
print_block("BUILD FDG SUMMARY")

fdg_scan = read_csv(SCAN_FDG_NPDKA_PATH)
fdg_scan = replace_nacc_missing_codes(fdg_scan)
fdg_scan = build_date_from_string(fdg_scan, "SCANDATE", "fdg_date")

fdg_scan_keep = [c for c in [
    "NACCID", "NACCADC", "fdg_date", "LONIUID", "TRACER",
    "FDG_METAROI_SUVR",
    "CTX_POSTERIORCINGULATE_SUVR", "CTX_PRECUNEUS_SUVR",
    "CTX_INFERIORTEMPORAL_SUVR", "CTX_PARAHIPPOCAMPAL_SUVR",
    "CTX_ENTORHINAL_SUVR", "HIPPOCAMPUS_SUVR", "AMYGDALA_SUVR"
] if c in fdg_scan.columns]
fdg_scan = fdg_scan[fdg_scan_keep].copy()
fdg_scan["FDG_SOURCE"] = "SCAN"

fdg_frames = [fdg_scan]

if MP_FDG_NPDKA_PATH.exists():
    fdg_mp = read_csv(MP_FDG_NPDKA_PATH)
    fdg_mp = replace_nacc_missing_codes(fdg_mp)
    fdg_mp = build_date_from_string(fdg_mp, "SCANDATE", "fdg_date")
    fdg_mp_keep = [c for c in [
        "NACCID", "NACCADC", "fdg_date", "LONIUID", "TRACER", "SCAN_PROJECT",
        "FDG_METAROI_SUVR",
        "CTX_POSTERIORCINGULATE_SUVR", "CTX_PRECUNEUS_SUVR",
        "CTX_INFERIORTEMPORAL_SUVR", "CTX_PARAHIPPOCAMPAL_SUVR",
        "CTX_ENTORHINAL_SUVR", "HIPPOCAMPUS_SUVR", "AMYGDALA_SUVR"
    ] if c in fdg_mp.columns]
    fdg_mp = fdg_mp[fdg_mp_keep].copy()
    fdg_mp["FDG_SOURCE"] = "MIXED_PROTOCOL"
    fdg_frames.append(fdg_mp)

fdg = pd.concat(fdg_frames, ignore_index=True, sort=False)

rename_fdg = {}
for c in fdg.columns:
    if c not in ["NACCID", "NACCADC", "fdg_date"]:
        rename_fdg[c] = f"FDG_{c}"
fdg = fdg.rename(columns=rename_fdg)

fdg = dedupe_best_row(fdg, ["NACCID", "fdg_date"])
fdg_out = OUTDIR / "nacc_fdg_summary.csv"
fdg.to_csv(fdg_out, index=False)

print("Wrote:", fdg_out)
print("Rows:", len(fdg))
print("Unique subjects:", fdg["NACCID"].nunique())
for c in ["FDG_FDG_METAROI_SUVR", "FDG_CTX_POSTERIORCINGULATE_SUVR"]:
    if c in fdg.columns:
        print(f"With {c}: {fdg[c].notna().sum()}")


# =============================================================================
# 7) CSF summary table
# =============================================================================
print_block("BUILD CSF SUMMARY")

csf = read_csv(CSF_PATH)
csf = replace_nacc_missing_codes(csf)

# LP date first; if absent, fall back to biomarker assay dates
csf = build_date_from_ymd(csf, "CSFLPYR", "CSFLPMO", "CSFLPDY", "csf_date_lp")
csf = build_date_from_ymd(csf, "CSFABYR", "CSFABMO", "CSFABDY", "csf_date_ab")
csf = build_date_from_ymd(csf, "CSFPTYR", "CSFPTMO", "CSFPTDY", "csf_date_pt")
csf = build_date_from_ymd(csf, "CSFTTYR", "CSFTTMO", "CSFTTDY", "csf_date_tt")

csf["csf_date"] = csf["csf_date_lp"]
for alt in ["csf_date_ab", "csf_date_pt", "csf_date_tt"]:
    csf["csf_date"] = csf["csf_date"].fillna(csf[alt])

csf = safe_num(csf, ["CSFABETA", "CSFPTAU", "CSFTTAU"])
csf["CSF_PTAU_ABETA_RATIO"] = csf["CSFPTAU"] / csf["CSFABETA"]
csf["CSF_TTAU_ABETA_RATIO"] = csf["CSFTTAU"] / csf["CSFABETA"]

csf_keep = [c for c in [
    "NACCID", "NACCADC", "csf_date",
    "CSFABETA", "CSFPTAU", "CSFTTAU",
    "CSF_PTAU_ABETA_RATIO", "CSF_TTAU_ABETA_RATIO"
] if c in csf.columns]
csf = csf[csf_keep].copy()

rename_csf = {}
for c in csf.columns:
    if c not in ["NACCID", "NACCADC", "csf_date"]:
        rename_csf[c] = f"CSF_{c}"
csf = csf.rename(columns=rename_csf)

csf = dedupe_best_row(csf, ["NACCID", "csf_date"])
csf_out = OUTDIR / "nacc_csf_summary.csv"
csf.to_csv(csf_out, index=False)

print("Wrote:", csf_out)
print("Rows:", len(csf))
print("Unique subjects:", csf["NACCID"].nunique())


# =============================================================================
# 8) Build 180d and 90d integrated masters
# =============================================================================
summary = {}

for tol in TOLERANCES:
    print_block(f"BUILD MASTER WITH {tol} DAY MATCHING")

    master = uds_master.copy()

    master = asof_merge_by_subject_date(master, mixed_mri, "visit_date", "mri_date", tolerance_days=tol)
    master["mixed_mri_day_diff"] = (master["mri_date"] - master["visit_date"]).dt.days

    master = asof_merge_by_subject_date(master, scan_mri, "visit_date", "scan_mri_date", tolerance_days=tol)
    master["scan_mri_day_diff"] = (master["scan_mri_date"] - master["visit_date"]).dt.days

    master = asof_merge_by_subject_date(master, amy, "visit_date", "amy_date", tolerance_days=tol)
    master["amy_day_diff"] = (master["amy_date"] - master["visit_date"]).dt.days

    master = asof_merge_by_subject_date(master, tau, "visit_date", "tau_date", tolerance_days=tol)
    master["tau_day_diff"] = (master["tau_date"] - master["visit_date"]).dt.days

    master = asof_merge_by_subject_date(master, fdg, "visit_date", "fdg_date", tolerance_days=tol)
    master["fdg_day_diff"] = (master["fdg_date"] - master["visit_date"]).dt.days

    master = asof_merge_by_subject_date(master, csf, "visit_date", "csf_date", tolerance_days=tol)
    master["csf_day_diff"] = (master["csf_date"] - master["visit_date"]).dt.days

    out_csv = OUTDIR / f"nacc_master_{tol}d.csv"
    master.to_csv(out_csv, index=False)

    stats = {
        "rows": int(len(master)),
        "unique_subjects": int(master["NACCID"].nunique()),
        "matched_mixed_mri": int(master["mri_date"].notna().sum()) if "mri_date" in master.columns else 0,
        "matched_scan_mri": int(master["scan_mri_date"].notna().sum()) if "scan_mri_date" in master.columns else 0,
        "matched_amyloid": int(master["amy_date"].notna().sum()) if "amy_date" in master.columns else 0,
        "matched_tau": int(master["tau_date"].notna().sum()) if "tau_date" in master.columns else 0,
        "matched_fdg": int(master["fdg_date"].notna().sum()) if "fdg_date" in master.columns else 0,
        "matched_csf": int(master["csf_date"].notna().sum()) if "csf_date" in master.columns else 0,
        "median_abs_amy_diff": float(master["amy_day_diff"].abs().median()) if "amy_day_diff" in master.columns and master["amy_day_diff"].notna().any() else None,
        "median_abs_tau_diff": float(master["tau_day_diff"].abs().median()) if "tau_day_diff" in master.columns and master["tau_day_diff"].notna().any() else None,
        "median_abs_fdg_diff": float(master["fdg_day_diff"].abs().median()) if "fdg_day_diff" in master.columns and master["fdg_day_diff"].notna().any() else None,
        "median_abs_scan_mri_diff": float(master["scan_mri_day_diff"].abs().median()) if "scan_mri_day_diff" in master.columns and master["scan_mri_day_diff"].notna().any() else None,
        "median_abs_mixed_mri_diff": float(master["mixed_mri_day_diff"].abs().median()) if "mixed_mri_day_diff" in master.columns and master["mixed_mri_day_diff"].notna().any() else None,
        "median_abs_csf_diff": float(master["csf_day_diff"].abs().median()) if "csf_day_diff" in master.columns and master["csf_day_diff"].notna().any() else None,
    }
    summary[f"master_{tol}d"] = stats

    print("Wrote:", out_csv)
    print(json.dumps(stats, indent=2))

    # Small preview
    preview_cols = [c for c in [
        "NACCID", "visit_date", "dx_harmonized", "NACCAGE", "NACCMMSE", "CDRGLOB", "CDRSUM",
        "mri_date", "mixed_mri_day_diff", "MPMRI_HIPPOVOL", "MPMRI_LHIPPO", "MPMRI_RHIPPO",
        "scan_mri_date", "scan_mri_day_diff", "SCANMRI_HIPPOCAMPUS", "SCANMRI_LH_ENTORHINAL_GVOL", "SCANMRI_RH_ENTORHINAL_GVOL",
        "amy_date", "amy_day_diff", "AMY_CENTILOIDS", "AMY_GAAIN_SUMMARY_SUVR", "AMY_CTX_POSTERIORCINGULATE_SUVR", "AMY_CTX_PRECUNEUS_SUVR",
        "tau_date", "tau_day_diff", "TAU_META_TEMPORAL_SUVR", "TAU_CTX_ENTORHINAL_SUVR",
        "fdg_date", "fdg_day_diff", "FDG_FDG_METAROI_SUVR", "FDG_CTX_POSTERIORCINGULATE_SUVR",
        "csf_date", "csf_day_diff", "CSF_CSFABETA", "CSF_CSFPTAU", "CSF_CSFTTAU",
    ] if c in master.columns]

    if preview_cols:
        print("\nExample rows:")
        print(master[preview_cols].head(10).to_string(index=False))


# =============================================================================
# 9) Save overall summary
# =============================================================================
overall = {
    "uds_rows": int(len(uds_master)),
    "uds_subjects": int(uds_master["NACCID"].nunique()),
    "mixed_mri_rows": int(len(mixed_mri)),
    "mixed_mri_subjects": int(mixed_mri["NACCID"].nunique()),
    "scan_mri_rows": int(len(scan_mri)),
    "scan_mri_subjects": int(scan_mri["NACCID"].nunique()),
    "amy_rows": int(len(amy)),
    "amy_subjects": int(amy["NACCID"].nunique()),
    "tau_rows": int(len(tau)),
    "tau_subjects": int(tau["NACCID"].nunique()),
    "fdg_rows": int(len(fdg)),
    "fdg_subjects": int(fdg["NACCID"].nunique()),
    "csf_rows": int(len(csf)),
    "csf_subjects": int(csf["NACCID"].nunique()),
}
overall.update(summary)

summary_path = OUTDIR / "nacc_phase0_summary.json"
with open(summary_path, "w") as f:
    json.dump(overall, f, indent=2)

print_block("DONE")
print("Saved summary:", summary_path)
print(json.dumps(overall, indent=2))