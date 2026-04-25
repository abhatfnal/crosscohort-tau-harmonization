#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/project/aereditato/abhat/NACC_17259")
PHASE0 = ROOT / "phase0"
OUT = ROOT / "phase1_v2"
OUT.mkdir(parents=True, exist_ok=True)

MASTER_180_SRC = PHASE0 / "nacc_master_180d.csv"
MASTER_90_SRC = PHASE0 / "nacc_master_90d.csv"
UDS_SRC = PHASE0 / "nacc_uds_visit_master.csv"

MASTER_180_CLEAN = OUT / "nacc_master_180d_v2_clean.csv"
MASTER_90_CLEAN = OUT / "nacc_master_90d_v2_clean.csv"
SUMMARY_JSON = OUT / "nacc_phase1_v2_summary.json"

CHUNKSIZE = 25000

# MRI below means SCAN MRI (the cleaner standardized MRI subset)
COHORT_SPECS = {
    "amyloid": ["has_amyloid"],
    "tau": ["has_tau"],
    "scan_mri": ["has_scan_mri"],
    "tau_plus_amyloid": ["has_tau", "has_amyloid"],
    "tau_plus_mri": ["has_tau", "has_scan_mri"],
    "amyloid_plus_mri": ["has_amyloid", "has_scan_mri"],
    "tau_plus_amyloid_plus_mri": ["has_tau", "has_amyloid", "has_scan_mri"],
}

SELECTED_COLS = [
    "NACCID",
    "visit_date",
    "dx_harmonized",
    "sex_harmonized",
    "NACCAGE",
    "NACCMMSE",
    "CDRGLOB",
    "CDRSUM",
    "FAQ_TOTAL",
    "apoe_e4_count",
    "apoe_e4_carrier",
    "scan_mri_date",
    "scan_mri_day_diff",
    "SCANMRI_HIPPOCAMPUS",
    "SCANMRI_LH_ENTORHINAL_GVOL",
    "SCANMRI_RH_ENTORHINAL_GVOL",
    "amy_date",
    "amy_day_diff",
    "AMY_CENTILOIDS",
    "AMY_GAAIN_SUMMARY_SUVR",
    "AMY_NPDKA_SUMMARY_SUVR",
    "AMY_CTX_POSTERIORCINGULATE_SUVR",
    "AMY_CTX_PRECUNEUS_SUVR",
    "tau_date",
    "tau_day_diff",
    "TAU_META_TEMPORAL_SUVR",
    "TAU_CTX_ENTORHINAL_SUVR",
]

DATE_COLS = ["visit_date", "scan_mri_date", "amy_date", "tau_date"]


def print_header(msg: str) -> None:
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


def normalize_numeric_like_strings(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.str.replace(r"\.0+$", "", regex=True)
    return s


def clean_placeholder_codes(df: pd.DataFrame, preserve_cols=None):
    """
    Safe first-pass NACC cleanup:
      - repeated 8/9 sentinels: 88, 888, 9999, 88.8, 9999.999, ...
      - negative missing sentinels: -4, -4.4, -44, -8, -9, ...
    This intentionally does NOT replace single positive 8/9 codes globally,
    because those can be legitimate category values in some variables.
    """
    preserve_cols = set(preserve_cols or [])
    out = df.copy()
    total_replaced = 0
    per_col = {}

    for col in out.columns:
        if col in preserve_cols:
            continue

        s = out[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue

        ss = normalize_numeric_like_strings(s)
        compact = ss.str.replace(".", "", regex=False)

        mask = compact.str.fullmatch(r"(8{2,}|9{2,}|-(4+|8+|9+))", na=False)
        n = int(mask.sum())

        if n > 0:
            out.loc[mask, col] = np.nan
            total_replaced += n
            per_col[col] = n

    return out, total_replaced, per_col


def clean_csv_in_chunks(src: Path, dst: Path, chunksize: int = CHUNKSIZE):
    if dst.exists():
        dst.unlink()

    total_rows = 0
    total_replaced = 0
    all_hits = {}
    first = True

    for chunk in pd.read_csv(src, low_memory=False, chunksize=chunksize):
        clean_chunk, n_replaced, hits = clean_placeholder_codes(
            chunk,
            preserve_cols={"NACCID"},
        )
        clean_chunk.to_csv(dst, mode="w" if first else "a", header=first, index=False)
        first = False

        total_rows += len(clean_chunk)
        total_replaced += n_replaced
        for k, v in hits.items():
            all_hits[k] = all_hits.get(k, 0) + v

    top_hits = sorted(all_hits.items(), key=lambda x: x[1], reverse=True)[:25]
    return {
        "src": str(src),
        "dst": str(dst),
        "rows": total_rows,
        "total_replaced": total_replaced,
        "top_replaced_columns": top_hits,
    }


def quantify_missing_visit_dates(uds_path: Path):
    uds = pd.read_csv(
        uds_path,
        usecols=["NACCID", "visit_date", "dx_harmonized"],
        low_memory=False,
        parse_dates=["visit_date"],
    )

    total_rows = len(uds)
    total_subjects = uds["NACCID"].nunique()

    miss_mask = uds["visit_date"].isna()
    missing_rows = int(miss_mask.sum())
    nonmissing_rows = int((~miss_mask).sum())

    missing_subjects_any = int(uds.loc[miss_mask, "NACCID"].nunique())

    subj_all_missing = (
        uds.groupby("NACCID")["visit_date"]
        .apply(lambda x: x.isna().all())
        .rename("all_missing")
        .reset_index()
    )
    missing_subjects_all = int(subj_all_missing["all_missing"].sum())

    dx_dist_missing = (
        uds.loc[miss_mask, "dx_harmonized"]
        .fillna("Missing")
        .value_counts(dropna=False)
        .to_dict()
    )

    return {
        "uds_rows_total": total_rows,
        "uds_subjects_total": total_subjects,
        "uds_rows_missing_visit_date": missing_rows,
        "uds_rows_nonmissing_visit_date": nonmissing_rows,
        "uds_subjects_with_any_missing_visit_date": missing_subjects_any,
        "uds_subjects_with_all_visits_missing_visit_date": missing_subjects_all,
        "dx_distribution_among_missing_visit_date_rows": dx_dist_missing,
    }


def load_selected_master(master_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(
        master_csv,
        usecols=lambda c: c in set(SELECTED_COLS),
        low_memory=False,
        parse_dates=DATE_COLS,
    )

    for c in [
        "NACCAGE",
        "NACCMMSE",
        "CDRGLOB",
        "CDRSUM",
        "FAQ_TOTAL",
        "apoe_e4_count",
        "apoe_e4_carrier",
        "scan_mri_day_diff",
        "SCANMRI_HIPPOCAMPUS",
        "SCANMRI_LH_ENTORHINAL_GVOL",
        "SCANMRI_RH_ENTORHINAL_GVOL",
        "amy_day_diff",
        "AMY_CENTILOIDS",
        "AMY_GAAIN_SUMMARY_SUVR",
        "AMY_NPDKA_SUMMARY_SUVR",
        "AMY_CTX_POSTERIORCINGULATE_SUVR",
        "AMY_CTX_PRECUNEUS_SUVR",
        "tau_day_diff",
        "TAU_META_TEMPORAL_SUVR",
        "TAU_CTX_ENTORHINAL_SUVR",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def add_modality_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    amy_cols = [
        c for c in [
            "AMY_CENTILOIDS",
            "AMY_GAAIN_SUMMARY_SUVR",
            "AMY_NPDKA_SUMMARY_SUVR",
            "AMY_CTX_POSTERIORCINGULATE_SUVR",
            "AMY_CTX_PRECUNEUS_SUVR",
        ] if c in df.columns
    ]
    tau_cols = [
        c for c in ["TAU_META_TEMPORAL_SUVR", "TAU_CTX_ENTORHINAL_SUVR"]
        if c in df.columns
    ]
    scan_cols = [
        c for c in [
            "SCANMRI_HIPPOCAMPUS",
            "SCANMRI_LH_ENTORHINAL_GVOL",
            "SCANMRI_RH_ENTORHINAL_GVOL",
        ] if c in df.columns
    ]

    df["has_amyloid"] = df["amy_date"].notna() & df[amy_cols].notna().any(axis=1)
    df["has_tau"] = df["tau_date"].notna() & df[tau_cols].notna().any(axis=1)
    df["has_scan_mri"] = df["scan_mri_date"].notna() & df[scan_cols].notna().any(axis=1)

    return df


def choose_one_row_per_subject(df: pd.DataFrame, required_flags, required_diff_cols):
    mask = pd.Series(True, index=df.index)
    for f in required_flags:
        mask &= df[f].fillna(False)

    sub = df.loc[mask].copy()
    if sub.empty:
        return sub

    diff_cols = [c for c in required_diff_cols if c in sub.columns]
    for c in diff_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    if diff_cols:
        absdiff = sub[diff_cols].abs()
        sub["_score_n_missing"] = absdiff.isna().sum(axis=1)
        sub["_score_sumdiff"] = absdiff.sum(axis=1, min_count=1).fillna(np.inf)
        sub["_score_maxdiff"] = absdiff.max(axis=1, skipna=True).fillna(np.inf)
    else:
        sub["_score_n_missing"] = 0
        sub["_score_sumdiff"] = 0.0
        sub["_score_maxdiff"] = 0.0

    sub = sub.sort_values(
        ["NACCID", "_score_n_missing", "_score_sumdiff", "_score_maxdiff", "visit_date"],
        ascending=[True, True, True, True, False],
    )

    best = sub.groupby("NACCID", as_index=False).first()
    best = best.drop(columns=["_score_n_missing", "_score_sumdiff", "_score_maxdiff"])
    return best


def dx_distribution(df: pd.DataFrame):
    if df.empty:
        return []

    vc = df["dx_harmonized"].fillna("Missing").value_counts(dropna=False)
    n = len(df)
    return [
        {"dx_harmonized": str(k), "n": int(v), "pct": round(100.0 * float(v) / n, 2)}
        for k, v in vc.items()
    ]


def build_candidate_cohorts(master_csv: Path, label: str):
    print_header(f"BUILD ONE-ROW-PER-SUBJECT COHORTS ({label})")
    df = load_selected_master(master_csv)
    df = add_modality_flags(df)

    out_dir = OUT / f"cohorts_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    summary_json = {}

    diff_map = {
        "has_amyloid": "amy_day_diff",
        "has_tau": "tau_day_diff",
        "has_scan_mri": "scan_mri_day_diff",
    }

    for cohort_name, flags in COHORT_SPECS.items():
        diff_cols = [diff_map[f] for f in flags if f in diff_map]
        cohort = choose_one_row_per_subject(df, flags, diff_cols)

        out_csv = out_dir / f"{cohort_name}_subject_level.csv"
        cohort.to_csv(out_csv, index=False)

        dist = dx_distribution(cohort)
        summary_json[cohort_name] = {
            "rows": int(len(cohort)),
            "unique_subjects": int(cohort["NACCID"].nunique()) if not cohort.empty else 0,
            "dx_distribution": dist,
            "csv": str(out_csv),
        }

        summary_rows.append({
            "cohort": cohort_name,
            "rows": int(len(cohort)),
            "unique_subjects": int(cohort["NACCID"].nunique()) if not cohort.empty else 0,
            "n_CN": int(sum(d["n"] for d in dist if d["dx_harmonized"] == "CN")),
            "n_MCI": int(sum(d["n"] for d in dist if d["dx_harmonized"] == "MCI")),
            "n_Dementia": int(sum(d["n"] for d in dist if d["dx_harmonized"] == "Dementia")),
            "n_OtherUnknown": int(sum(d["n"] for d in dist if d["dx_harmonized"] == "Other/Unknown")),
            "n_MissingDX": int(sum(d["n"] for d in dist if d["dx_harmonized"] == "Missing")),
        })

        print(f"\n[{cohort_name}] rows={len(cohort)} subjects={cohort['NACCID'].nunique() if not cohort.empty else 0}")
        if dist:
            print(pd.DataFrame(dist).to_string(index=False))
        else:
            print("Empty cohort")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / f"candidate_cohort_summary_{label}.csv"
    summary_df.to_csv(summary_csv, index=False)
    summary_json["_summary_csv"] = str(summary_csv)

    summary_json_path = out_dir / f"candidate_cohort_summary_{label}.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_json, f, indent=2)

    print("\nSaved:")
    print(summary_csv)
    print(summary_json_path)

    return {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json_path),
        "cohorts": summary_json,
    }


def main():
    print_header("STEP 1: QUANTIFY UDS VISITS DROPPED DUE TO MISSING visit_date")
    uds_stats = quantify_missing_visit_dates(UDS_SRC)
    print(json.dumps(uds_stats, indent=2))

    print_header("STEP 2: CLEAN PHASE-0 MASTER 180D -> MASTER V2")
    clean180_stats = clean_csv_in_chunks(MASTER_180_SRC, MASTER_180_CLEAN, chunksize=CHUNKSIZE)
    print(json.dumps(clean180_stats, indent=2))

    print_header("STEP 3: CLEAN PHASE-0 MASTER 90D -> MASTER V2")
    clean90_stats = clean_csv_in_chunks(MASTER_90_SRC, MASTER_90_CLEAN, chunksize=CHUNKSIZE)
    print(json.dumps(clean90_stats, indent=2))

    print_header("STEP 4: BUILD CANDIDATE COHORTS FROM 180D CLEAN MASTER")
    cohorts180 = build_candidate_cohorts(MASTER_180_CLEAN, "180d")

    print_header("STEP 5: BUILD CANDIDATE COHORTS FROM 90D CLEAN MASTER")
    cohorts90 = build_candidate_cohorts(MASTER_90_CLEAN, "90d")

    summary = {
        "uds_missing_visit_date": uds_stats,
        "clean_master_180d": clean180_stats,
        "clean_master_90d": clean90_stats,
        "candidate_cohorts_180d": cohorts180,
        "candidate_cohorts_90d": cohorts90,
    }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print_header("DONE")
    print(f"Saved summary: {SUMMARY_JSON}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()