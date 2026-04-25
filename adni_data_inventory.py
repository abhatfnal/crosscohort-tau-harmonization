#!/usr/bin/env python3
"""
ADNI data inventory script.

Quick mode (default): scans CSV headers and filenames to detect key ADNI tables + modalities.
Deep mode (--deep): for detected key tables, loads a small set of columns to estimate
rows, unique subjects, and missingness.

Usage examples:
  python adni_data_inventory.py
  python adni_data_inventory.py --data-roots /project/aereditato/abhat/ADNI
  python adni_data_inventory.py --data-roots /project/aereditato/abhat/ADNI --deep
  python adni_data_inventory.py --data-roots /project/aereditato/abhat/ADNI --deep --count-imaging
"""

from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


# -----------------------------
# Heuristics / patterns
# -----------------------------
KEY_TABLE_HINTS = {
    "ADNIMERGE": ["ADNIMERGE"],
    "DXSUM": ["DXSUM", "DIAGNOSIS SUMMARY", "DX_SUM"],
    "DXCHANGE": ["DXCHANGE", "DX_CHANGE"],
    "APOERES": ["APOERES", "APOE"],
    "UPENNBIOMK": ["UPENNBIOMK", "CSF", "BIOMK"],
    "BLENNOWPLASMA": ["BLENNOWPLASMA", "PLASMA"],
    "UCBERKELEY_AV45": ["UCBERKELEYAV45", "AV45", "AMYLOID PET", "FBB", "FLORBETAPIR"],
    "UCBERKELEY_FDG": ["UCBERKELEYFDG", "FDG"],
    "UCBERKELEY_TAU": ["UCBERKELEYTAU", "AV1451", "FLORTAUCIPIR", "FTP", "TAU PET", "TAU"],
}

# Column keyword patterns (header-based)
RX_SUBJECT = re.compile(r"\b(RID|PTID|SUBJECT|SUBJECT_ID|PARTICIPANT)\b", re.IGNORECASE)
RX_VISIT = re.compile(r"\b(VISCODE|VISIT|VISITCODE|EXAMDATE|SCANDATE|DATE)\b", re.IGNORECASE)

RX_DX = re.compile(r"\b(DX|DIAG|DXCHANGE|DXCURREN|DXSUM)\b", re.IGNORECASE)

RX_COG = re.compile(r"\b(MMSE|CDR|CDRSB|ADAS|FAQ|RAVLT|LOGICAL)\b", re.IGNORECASE)

RX_APOE = re.compile(r"\b(APOE|APOE4|E4)\b", re.IGNORECASE)

RX_CSF = re.compile(r"\b(ABETA|A\W?BETA|AB42|AB40|TAU|PTAU|P\W?TAU|TTAU)\b", re.IGNORECASE)

RX_PLASMA = re.compile(r"\b(PTAU217|P\W?TAU217|GFAP|NFL|NfL|AB42|AB40|ABETA)\b", re.IGNORECASE)

RX_PET_AMY = re.compile(r"\b(AV45|FBB|FLORBETAPIR|PIB|CENTILOID|AMYLOID|SUVR)\b", re.IGNORECASE)
RX_PET_FDG = re.compile(r"\b(FDG|SUVR)\b", re.IGNORECASE)
RX_PET_TAU = re.compile(r"\b(AV1451|FLORTAUCIPIR|FTP|TAU|SUVR)\b", re.IGNORECASE)


def safe_read_header(csv_path: Path, nrows: int = 5) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(csv_path, nrows=nrows, low_memory=False)
    except Exception:
        return None


def guess_table_type(path: Path) -> List[str]:
    """Guess which key tables this file might correspond to based on filename."""
    name = path.name.upper()
    hits = []
    for key, hints in KEY_TABLE_HINTS.items():
        for h in hints:
            if h.upper() in name:
                hits.append(key)
                break
    return hits


def summarize_columns(cols: List[str]) -> Dict[str, bool]:
    colstr = " | ".join(cols)
    return {
        "has_subject_id": bool(RX_SUBJECT.search(colstr)),
        "has_visit_time": bool(RX_VISIT.search(colstr)),
        "has_dx": bool(RX_DX.search(colstr)),
        "has_cognition": bool(RX_COG.search(colstr)),
        "has_apoe": bool(RX_APOE.search(colstr)),
        "has_csf_markers": bool(RX_CSF.search(colstr)),
        "has_plasma_markers": bool(RX_PLASMA.search(colstr)),
        "has_amyloid_pet_terms": bool(RX_PET_AMY.search(colstr)),
        "has_fdg_pet_terms": bool(RX_PET_FDG.search(colstr)),
        "has_tau_pet_terms": bool(RX_PET_TAU.search(colstr)),
    }


def choose_best_subject_col(cols: List[str]) -> Optional[str]:
    candidates = ["RID", "PTID", "subject_id", "Subject", "SUBJECT", "participant_id"]
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    # heuristic fallback
    for c in cols:
        if RX_SUBJECT.search(c):
            return c
    return None


def choose_best_visit_col(cols: List[str]) -> Optional[str]:
    candidates = ["VISCODE", "VISCODE2", "VISITCODE", "VISIT", "EXAMDATE", "SCANDATE", "DATE"]
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in cols:
        if RX_VISIT.search(c):
            return c
    return None


def deep_stats(csv_path: Path, cols: List[str]) -> Dict:
    """
    Load minimal columns to estimate:
    - rows
    - unique subjects
    - unique visits (if available)
    - missingness for key biomarker/cog/dx fields
    """
    subject_col = choose_best_subject_col(cols)
    visit_col = choose_best_visit_col(cols)

    # keep a small set of "interesting" columns if present
    keep = []
    for c in cols:
        if (subject_col and c == subject_col) or (visit_col and c == visit_col):
            keep.append(c)
            continue
        if RX_DX.search(c) or RX_COG.search(c) or RX_APOE.search(c) or RX_CSF.search(c) or RX_PLASMA.search(c):
            keep.append(c)

    # cap to avoid loading too wide tables
    keep = keep[:80]

    out = {
        "rows": None,
        "unique_subjects": None,
        "unique_visits": None,
        "subject_col": subject_col,
        "visit_col": visit_col,
        "missingness": {},
    }

    try:
        df = pd.read_csv(csv_path, usecols=keep, low_memory=False)
    except Exception as e:
        out["error"] = f"Failed to load for deep stats: {e}"
        return out

    out["rows"] = int(len(df))
    if subject_col and subject_col in df.columns:
        out["unique_subjects"] = int(df[subject_col].nunique(dropna=True))
    if visit_col and visit_col in df.columns:
        out["unique_visits"] = int(df[visit_col].nunique(dropna=True))

    # missingness for key markers (if present)
    key_cols = []
    for c in df.columns:
        if RX_DX.search(c) or RX_COG.search(c) or RX_PLASMA.search(c) or RX_CSF.search(c) or RX_APOE.search(c):
            key_cols.append(c)

    for c in key_cols[:60]:
        miss = float(df[c].isna().mean())
        out["missingness"][c] = miss

    return out


def count_imaging_files(data_roots: List[Path]) -> Dict[str, int]:
    """
    Optional: count NIfTI files by basic heuristics.
    This can be expensive on huge trees; keep it lightweight.
    """
    exts = (".nii", ".nii.gz")
    counts = {"nii_total": 0, "mri_like": 0, "pet_like": 0}
    for root in data_roots:
        if not root.exists():
            continue
        # limit traversal depth a bit by not using recursive glob on enormous trees in a tight loop
        for p in root.rglob("*"):
            if p.is_file() and p.name.endswith(exts):
                counts["nii_total"] += 1
                name = p.name.lower()
                if any(k in name for k in ["mri", "t1", "mprage", "spgr"]):
                    counts["mri_like"] += 1
                if any(k in name for k in ["pet", "fdg", "av45", "av1451", "tau"]):
                    counts["pet_like"] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".", help="Path to adni-mri-classification repo (default: .)")
    ap.add_argument("--data-roots", nargs="*", default=[], help="Additional roots to scan for CSVs (ADNI metadata dirs)")
    ap.add_argument("--deep", action="store_true", help="Compute deep stats for key tables (loads some columns)")
    ap.add_argument("--count-imaging", action="store_true", help="Also count NIfTI files (can be slow on huge trees)")
    ap.add_argument("--out", default="adni_inventory_report.json", help="Output JSON path")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_roots = [Path(p).resolve() for p in args.data_roots] if args.data_roots else []
    scan_roots = [repo_root] + data_roots

    csv_files: List[Path] = []
    for root in scan_roots:
        if root.exists():
            csv_files.extend(root.rglob("*.csv"))
    csv_files = sorted(set(csv_files))

    report = {
        "repo_root": str(repo_root),
        "data_roots": [str(p) for p in data_roots],
        "csv_files_scanned": len(csv_files),
        "key_tables_found": {},
        "signals": {
            "longitudinal_dx": False,
            "tau_pet": False,
            "amyloid_pet": False,
            "fdg_pet": False,
            "csf": False,
            "plasma": False,
            "apoe": False,
            "cognition": False,
        },
        "files": [],
        "imaging_counts": None,
    }

    # Quick scan headers
    for csv_path in csv_files:
        header_df = safe_read_header(csv_path, nrows=5)
        if header_df is None:
            continue
        cols = list(header_df.columns)
        colsum = summarize_columns(cols)
        guessed = guess_table_type(csv_path)

        # update key tables
        for g in guessed:
            report["key_tables_found"].setdefault(g, []).append(str(csv_path))

        # update signals (any evidence anywhere)
        report["signals"]["longitudinal_dx"] |= colsum["has_dx"] and colsum["has_visit_time"] and colsum["has_subject_id"]
        report["signals"]["tau_pet"] |= colsum["has_tau_pet_terms"] and ("TAU" in csv_path.name.upper() or "AV1451" in csv_path.name.upper() or "UCBERKELEY" in csv_path.name.upper())
        report["signals"]["amyloid_pet"] |= colsum["has_amyloid_pet_terms"] and ("AV45" in csv_path.name.upper() or "AMY" in csv_path.name.upper() or "UCBERKELEY" in csv_path.name.upper())
        report["signals"]["fdg_pet"] |= colsum["has_fdg_pet_terms"] and ("FDG" in csv_path.name.upper() or "UCBERKELEY" in csv_path.name.upper())
        report["signals"]["csf"] |= colsum["has_csf_markers"] and ("CSF" in csv_path.name.upper() or "UPENN" in csv_path.name.upper() or "BIOMK" in csv_path.name.upper())
        report["signals"]["plasma"] |= colsum["has_plasma_markers"] and ("PLASMA" in csv_path.name.upper() or "BLOOD" in csv_path.name.upper() or "BLENNOW" in csv_path.name.upper())
        report["signals"]["apoe"] |= colsum["has_apoe"]
        report["signals"]["cognition"] |= colsum["has_cognition"]

        report["files"].append({
            "path": str(csv_path),
            "ncols": len(cols),
            "table_guess": guessed,
            "column_signals": colsum,
        })

    # Deep stats on key tables if requested
    if args.deep:
        deep_block = {}
        # choose files that look like key tables (from guesses)
        for key, paths in report["key_tables_found"].items():
            deep_block[key] = []
            for p in paths[:6]:  # cap
                pth = Path(p)
                hdr = safe_read_header(pth, nrows=5)
                if hdr is None:
                    continue
                deep_block[key].append({
                    "path": p,
                    "stats": deep_stats(pth, list(hdr.columns)),
                })
        report["deep_stats"] = deep_block

    if args.count_imaging and data_roots:
        report["imaging_counts"] = count_imaging_files(data_roots)

    # Save JSON
    out_path = Path(args.out).resolve()
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report: {out_path}\n")

    # Print a short human summary
    print("=== ADNI Inventory Summary ===")
    print(f"CSV scanned: {report['csv_files_scanned']}")
    print("Key tables found:")
    for k, v in sorted(report["key_tables_found"].items()):
        print(f"  - {k}: {len(v)} file(s)")
    print("Signals:")
    for k, v in report["signals"].items():
        print(f"  - {k}: {v}")

    if args.deep and "deep_stats" in report:
        print("\nDeep stats (first few):")
        for k, lst in report["deep_stats"].items():
            if not lst:
                continue
            print(f"  [{k}]")
            for item in lst[:2]:
                st = item["stats"]
                print(f"    - {Path(item['path']).name}: rows={st.get('rows')} uniq_subj={st.get('unique_subjects')} uniq_visit={st.get('unique_visits')}")

    print("\nNext step: paste the 'Signals' + 'Key tables found' section here (or attach the JSON) and I’ll recommend the best paper direction.\n")


if __name__ == "__main__":
    main()