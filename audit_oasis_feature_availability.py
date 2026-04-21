# audit_oasis_feature_availability.py
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd


RAW = {
    "oasis3_tau90_raw": "/project/aereditato/abhat/oasis/phase0/oasis3_clinical_master_v7_tau90_amy90_fs90_fixed.csv",
    "oasis3_tau180_raw": "/project/aereditato/abhat/oasis/phase0/oasis3_clinical_master_v7_tau180_amy180_fs180_fixed.csv",
}

SUBJECT = {
    "oasis3_tau90_subject": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau90/subject_level_input_table.csv",
    "oasis3_tau180_subject": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau180/subject_level_input_table.csv",
}

OUTDIR = Path("/project/aereditato/abhat/adni-mri-classification/crosscohort_matched_rerun/oasis_feature_audit")
OUTDIR.mkdir(parents=True, exist_ok=True)


FEATURE_GROUPS = {
    "age": [
        "age_h", "AGE", "age", "Age", "age_at_visit", "age_at_entry", "baseline_age",
        "entry_age", "visit_age", "days_to_visit"
    ],
    "apoe": [
        "apoe_e4_count_h", "apoe_e4_carrier_h", "APOE", "APOE4", "E4", "APOE_GENOTYPE",
        "apoe_genotype", "apoe_status", "apoe_e4_count", "apoe_e4_carrier"
    ],
    "moca": [
        "moca_h", "MOCA", "moca", "MOCA_TOTAL", "MOCATOTS", "MOCA_MOCA"
    ],
    "cdr_global": [
        "cdr_global_h", "CDRGLOBAL", "CDR_GLOBAL", "CDR_CDGLOBAL", "cdr_global", "CDGLOBAL"
    ],
    "cdr_sumboxes": [
        "cdr_sumboxes_h", "CDRSUM", "CDRSB", "cdrsb", "cdr_sumboxes", "CDR_CDRSB"
    ],
    "faq": [
        "faq_total_h", "FAQTOTAL", "FAQ_TOTAL", "FAQ_FAQTOTAL", "faq_total", "FAQ"
    ],
}


def read_csv_any(path):
    return pd.read_csv(path, low_memory=False, compression="infer")


def norm(x):
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())


def candidate_columns(df, family):
    pats = {
        "age": ["AGE"],
        "apoe": ["APOE", "E4", "GENOTYPE"],
        "moca": ["MOCA"],
        "cdr_global": ["CDR", "GLOBAL"],
        "cdr_sumboxes": ["CDR", "SUM", "SB"],
        "faq": ["FAQ"],
    }
    wanted = pats[family]
    out = []
    for c in df.columns:
        uc = str(c).upper()
        if any(w in uc for w in wanted):
            out.append(c)
    return out


def find_alias(df, aliases):
    norm_map = {norm(c): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
    for a in aliases:
        na = norm(a)
        if na in norm_map:
            return norm_map[na]
    return None


def nonmissing_summary(series):
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        vals = vals.replace(
            [-4, -3, -2, -1, 88, 888, 8888, 99, 999, 9999, 999.0, 9999.0],
            np.nan,
        )
        return int(vals.notna().sum()), int(vals.dropna().nunique())

    s = s.replace(["", " ", "NA", "NaN", "nan"], np.nan)
    return int(s.notna().sum()), int(s.dropna().nunique())


def audit_one(name, path):
    df = read_csv_any(path)
    rows = []

    for family, aliases in FEATURE_GROUPS.items():
        matched = find_alias(df, aliases)
        cands = candidate_columns(df, family)

        if matched is not None:
            nn, nu = nonmissing_summary(df[matched])
        else:
            nn, nu = 0, 0

        rows.append(
            {
                "dataset": name,
                "path": path,
                "feature_family": family,
                "matched_col": matched,
                "nonmissing_n": nn,
                "unique_nonnull": nu,
                "candidate_columns": "|".join(cands[:50]),
            }
        )

    return pd.DataFrame(rows), df.columns.tolist()


def main():
    all_rows = []
    columns_dump = {}

    print("=" * 100)
    print("AUDIT RAW OASIS FILES")
    print("=" * 100)
    for name, path in RAW.items():
        out, cols = audit_one(name, path)
        columns_dump[name] = cols
        all_rows.append(out)
        print(f"\n{name}")
        print(out[["feature_family", "matched_col", "nonmissing_n", "unique_nonnull"]].to_string(index=False))

    print("\n" + "=" * 100)
    print("AUDIT SUBJECT-LEVEL INPUT TABLES")
    print("=" * 100)
    for name, path in SUBJECT.items():
        out, cols = audit_one(name, path)
        columns_dump[name] = cols
        all_rows.append(out)
        print(f"\n{name}")
        print(out[["feature_family", "matched_col", "nonmissing_n", "unique_nonnull"]].to_string(index=False))

    audit_df = pd.concat(all_rows, ignore_index=True)
    audit_df.to_csv(OUTDIR / "oasis_feature_audit.csv", index=False)

    with open(OUTDIR / "oasis_feature_audit_columns.json", "w") as f:
        json.dump(columns_dump, f, indent=2)

    print("\nSaved:")
    print(" ", OUTDIR / "oasis_feature_audit.csv")
    print(" ", OUTDIR / "oasis_feature_audit_columns.json")


if __name__ == "__main__":
    main()