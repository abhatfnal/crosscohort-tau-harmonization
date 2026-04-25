#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/project/aereditato/abhat/NACC_17259/phase1_v2")
SUMMARY_JSON = ROOT / "nacc_phase1_v2_summary.json"

def print_header(msg):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)

with open(SUMMARY_JSON) as f:
    summary = json.load(f)

print_header("UDS MISSING visit_date")
print(json.dumps(summary["uds_missing_visit_date"], indent=2))

for label in ["180d", "90d"]:
    csv_path = ROOT / f"cohorts_{label}" / f"candidate_cohort_summary_{label}.csv"
    print_header(f"CANDIDATE COHORTS: {label}")
    df = pd.read_csv(csv_path)
    print(df.to_string(index=False))