# rebuild_crosscohort_publication_tables.py
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent
MATCHED_SUMMARY = ROOT / "crosscohort_matched_rerun" / "matched_core_summary.csv"
EXISTING_PRIMARY = ROOT / "crosscohort_severity_summary" / "crosscohort_primary_core_table.csv"

OUTDIR = ROOT / "crosscohort_publication_tables"
OUTDIR.mkdir(parents=True, exist_ok=True)


PRIMARY_KEYS = ["adni_tau90", "adni_tau180", "oasis3_tau180"]
SUPP_KEYS = ["oasis3_tau90", "nacc_strict_at_main"]


DISPLAY_COLS = [
    "cohort_key",
    "cohort_label",
    "cohort",
    "window",
    "endpoint_family",
    "endpoint_label",
    "n_subjects",
    "n_pos",
    "n_neg",
    "pos_rate",
    "available_demo_features",
    "available_severity_features",
    "auc_ref",
    "auc_severity",
    "auc_demo",
    "auc_minus_all_cdr",
    "auc_minus_fullstrip",
    "severity_retention_auc_frac",
    "demo_retention_auc_frac",
    "full_strip_retention_auc_frac",
    "cdr_block_drop_auc",
    "full_strip_drop_auc",
    "moca_unique_drop_auc",
    "faq_unique_drop_auc",
    "cdr_global_unique_drop_auc",
    "cdr_sumboxes_unique_drop_auc",
    "severity_minus_demo_auc",
    "note",
]


def main():
    matched = pd.read_csv(MATCHED_SUMMARY)
    existing = pd.read_csv(EXISTING_PRIMARY)

    # Keep matched rows for ADNI/OASIS
    matched_needed = matched[matched["cohort_key"].isin(["adni_tau90", "adni_tau180", "oasis3_tau90", "oasis3_tau180"])].copy()
    matched_needed["table_source"] = "matched_feature_rerun"

    # Keep NACC from existing table
    nacc = existing[existing["cohort_key"] == "nacc_strict_at_main"].copy()
    nacc["table_source"] = "existing_nacc_summary"

    combined = pd.concat([matched_needed, nacc], ignore_index=True, sort=False)

    primary = combined[combined["cohort_key"].isin(PRIMARY_KEYS)].copy()
    supp = combined[combined["cohort_key"].isin(SUPP_KEYS)].copy()

    primary = primary[DISPLAY_COLS + ["table_source"]]
    supp = supp[DISPLAY_COLS + ["table_source"]]
    combined_out = combined[DISPLAY_COLS + ["table_source"]].copy()

    combined_out.to_csv(OUTDIR / "crosscohort_publication_combined.csv", index=False)
    primary.to_csv(OUTDIR / "crosscohort_primary_table.csv", index=False)
    supp.to_csv(OUTDIR / "crosscohort_supplementary_table.csv", index=False)

    print("=" * 100)
    print("PRIMARY TABLE")
    print("=" * 100)
    print(primary.to_string(index=False))

    print("\n" + "=" * 100)
    print("SUPPLEMENTARY TABLE")
    print("=" * 100)
    print(supp.to_string(index=False))

    print("\nSaved:")
    print(" ", OUTDIR / "crosscohort_publication_combined.csv")
    print(" ", OUTDIR / "crosscohort_primary_table.csv")
    print(" ", OUTDIR / "crosscohort_supplementary_table.csv")


if __name__ == "__main__":
    main()
