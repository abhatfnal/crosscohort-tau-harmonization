#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DEFAULT_PRE_HARMONIZATION_TABLE = (
    "/project/aereditato/abhat/adni-mri-classification/"
    "crosscohort_severity_summary/crosscohort_primary_core_table.csv"
)

DEFAULT_POST_HARMONIZATION_TABLE = (
    "/project/aereditato/abhat/adni-mri-classification/"
    "crosscohort_publication_tables/crosscohort_primary_table.csv"
)

DEFAULT_OUTDIR = (
    "/project/aereditato/abhat/adni-mri-classification/"
    "paper_tables_and_figures"
)

COHORT_CONFIG = [
    {
        "cohort_key": "adni_tau90",
        "cohort_label": "ADNI tau90",
        "group": "primary",
        "endpoint_label": "tau positivity",
        "csv": (
            "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/"
            "severity_strip_tau90/subject_level_input_table.csv"
        ),
    },
    {
        "cohort_key": "adni_tau180",
        "cohort_label": "ADNI tau180",
        "group": "primary",
        "endpoint_label": "tau positivity",
        "csv": (
            "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/"
            "severity_strip_tau180/subject_level_input_table.csv"
        ),
    },
    {
        "cohort_key": "oasis3_tau180",
        "cohort_label": "OASIS3 tau180",
        "group": "primary",
        "endpoint_label": "tau positivity (derived)",
        "csv": (
            "/project/aereditato/abhat/oasis/phase2_tau/"
            "severity_strip_tau180/subject_level_input_table.csv"
        ),
    },
    {
        "cohort_key": "oasis3_tau90",
        "cohort_label": "OASIS3 tau90",
        "group": "supplementary",
        "endpoint_label": "tau positivity (derived)",
        "csv": (
            "/project/aereditato/abhat/oasis/phase2_tau/"
            "severity_strip_tau90/subject_level_input_table.csv"
        ),
    },
    # NACC can be added at runtime with --nacc-subject-csv
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def pick_col(df: pd.DataFrame, aliases):
    for c in aliases:
        if c in df.columns:
            return c
    return None


def to_num(s):
    x = pd.to_numeric(s, errors="coerce")
    x = x.replace(
        [
            -4, -3, -2, -1,
            88, 888, 8888,
            99, 999, 9999,
            999.0, 9999.0,
        ],
        np.nan,
    )
    return x


def female_mask(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=bool)

    ss = s.astype(str).str.strip().str.upper()
    num = pd.to_numeric(s, errors="coerce")

    mask = (
        ss.isin(["F", "FEMALE"]) |
        num.eq(2) |
        num.eq(0) & pd.Series(False, index=s.index)  # never true; keeps shape simple
    )
    return mask.fillna(False)


def apoe_carrier_series(df: pd.DataFrame) -> pd.Series:
    carrier_col = pick_col(df, ["apoe_e4_carrier_h", "apoe_e4_carrier", "APOE_E4_CARRIER"])
    count_col = pick_col(df, ["apoe_e4_count_h", "apoe_e4_count", "APOE_E4_COUNT"])

    if carrier_col is not None:
        x = to_num(df[carrier_col])
        return np.where(x.notna(), (x > 0).astype(float), np.nan)

    if count_col is not None:
        x = to_num(df[count_col])
        return np.where(x.notna(), (x > 0).astype(float), np.nan)

    return pd.Series(np.nan, index=df.index, dtype="float")


def fmt_mean_sd(s: pd.Series) -> str:
    x = to_num(s).dropna()
    if len(x) == 0:
        return "NA"
    return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"


def fmt_n_pct(n: int, denom: int) -> str:
    if denom == 0:
        return "NA"
    return f"{n} ({100.0 * n / denom:.1f})"


def summarize_subject_table(
    cohort_key: str,
    cohort_label: str,
    endpoint_label: str,
    csv_path: str,
):
    df = pd.read_csv(csv_path, low_memory=False)

    sid_col = pick_col(df, ["subject_id", "OASISID", "RID", "PTID"])
    y_col = pick_col(df, ["y_target", "tau_pos", "target"])
    sex_col = pick_col(df, ["sex_h", "sex"])
    age_col = pick_col(df, ["age_h", "age"])
    educ_col = pick_col(df, ["education_years_h", "education_years", "education"])
    cdrsb_col = pick_col(df, ["cdr_sumboxes_h", "cdr_sumboxes", "cdrsb", "CDRSUM"])
    faq_col = pick_col(df, ["faq_total_h", "faq_total", "FAQTOTAL", "FAQ"])

    # Age cleanup:
    # ADNI subject-level tables may still contain a previously mis-recovered age_h.
    # For Table 1, force ADNI age to missing until a valid chronological age source is recovered.
    if age_col is not None:
        age_series = to_num(df[age_col])
    else:
        age_series = pd.Series(np.nan, index=df.index, dtype="float")

    if cohort_key.startswith("adni_"):
        age_series = pd.Series(np.nan, index=df.index, dtype="float")

    n = int(df[sid_col].nunique()) if sid_col is not None else int(len(df))

    if y_col is not None:
        y = to_num(df[y_col])
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
    else:
        n_pos = np.nan
        n_neg = np.nan

    # female %
    if sex_col is not None:
        fmask = female_mask(df[sex_col])
        female_n = int(fmask.sum())
        female_pct = 100.0 * female_n / len(df) if len(df) else np.nan
    else:
        female_n = np.nan
        female_pct = np.nan

    # APOE e4 carrier %
    apoe_car = pd.Series(apoe_carrier_series(df), index=df.index).astype("float")
    apoe_nonmissing = apoe_car.notna().sum()
    apoe_car_n = int((apoe_car == 1).sum()) if apoe_nonmissing > 0 else np.nan
    apoe_car_pct = 100.0 * apoe_car_n / apoe_nonmissing if apoe_nonmissing > 0 else np.nan

    # CDR-SB = 0 prevalence
    if cdrsb_col is not None:
        cdrsb = to_num(df[cdrsb_col])
        cdrsb_nonmissing = cdrsb.notna().sum()
        cdrsb0_n = int((cdrsb == 0).sum()) if cdrsb_nonmissing > 0 else np.nan
        cdrsb0_pct = 100.0 * cdrsb0_n / cdrsb_nonmissing if cdrsb_nonmissing > 0 else np.nan
    else:
        cdrsb0_n = np.nan
        cdrsb0_pct = np.nan

    raw = {
        "cohort_key": cohort_key,
        "cohort_label": cohort_label,
        "endpoint_label": endpoint_label,
        "N": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate_pct": 100.0 * n_pos / n if n and pd.notna(n_pos) else np.nan,
        "female_n": female_n,
        "female_pct": female_pct,
        "age_mean": age_series.mean(),
        "age_sd": age_series.std(ddof=1),
        "education_mean": to_num(df[educ_col]).mean() if educ_col is not None else np.nan,
        "education_sd": to_num(df[educ_col]).std(ddof=1) if educ_col is not None else np.nan,
        "apoe_e4_carrier_n": apoe_car_n,
        "apoe_e4_carrier_pct": apoe_car_pct,
        "cdrsb_mean": to_num(df[cdrsb_col]).mean() if cdrsb_col is not None else np.nan,
        "cdrsb_sd": to_num(df[cdrsb_col]).std(ddof=1) if cdrsb_col is not None else np.nan,
        "cdrsb0_n": cdrsb0_n,
        "cdrsb0_pct": cdrsb0_pct,
        "faq_mean": to_num(df[faq_col]).mean() if faq_col is not None else np.nan,
        "faq_sd": to_num(df[faq_col]).std(ddof=1) if faq_col is not None else np.nan,
    }

    formatted = {
        "Cohort": cohort_label,
        "Endpoint": endpoint_label,
        "N": n,
        "Positive, n (%)": fmt_n_pct(int(n_pos), int(n)) if pd.notna(n_pos) else "NA",
        "Female, n (%)": fmt_n_pct(int(female_n), int(len(df))) if pd.notna(female_n) else "NA",
        "Age, mean ± SD": fmt_mean_sd(age_series),
        "Education (yrs), mean ± SD": fmt_mean_sd(df[educ_col]) if educ_col is not None else "NA",
        "APOE e4+, n (%)": fmt_n_pct(int(apoe_car_n), int(apoe_nonmissing)) if pd.notna(apoe_car_n) else "NA",
        "CDR-SB, mean ± SD": fmt_mean_sd(df[cdrsb_col]) if cdrsb_col is not None else "NA",
        "CDR-SB = 0, n (%)": fmt_n_pct(int(cdrsb0_n), int(cdrsb_nonmissing)) if pd.notna(cdrsb0_n) else "NA",
        "FAQ total, mean ± SD": fmt_mean_sd(df[faq_col]) if faq_col is not None else "NA",
    }

    return raw, formatted


def build_table1(outdir: Path, nacc_subject_csv: str | None):
    cohort_defs = list(COHORT_CONFIG)

    if nacc_subject_csv:
        cohort_defs.append(
            {
                "cohort_key": "nacc_strict_at_main",
                "cohort_label": "NACC strict A/T main",
                "group": "supplementary",
                "endpoint_label": "strict A+/T+ vs A-/T-",
                "csv": nacc_subject_csv,
            }
        )

    raw_rows = []
    fmt_rows = []
    primary_fmt_rows = []
    supp_fmt_rows = []

    for cfg in cohort_defs:
        path = Path(cfg["csv"])
        if not path.exists():
            print(f"[warn] missing file, skipping: {path}")
            continue

        raw, fmt = summarize_subject_table(
            cohort_key=cfg["cohort_key"],
            cohort_label=cfg["cohort_label"],
            endpoint_label=cfg["endpoint_label"],
            csv_path=str(path),
        )
        raw["group"] = cfg["group"]
        raw["source_csv"] = str(path)
        raw_rows.append(raw)
        fmt_rows.append({**fmt, "Group": cfg["group"]})

        if cfg["group"] == "primary":
            primary_fmt_rows.append(fmt)
        else:
            supp_fmt_rows.append(fmt)

    raw_df = pd.DataFrame(raw_rows)
    fmt_df = pd.DataFrame(fmt_rows)
    primary_df = pd.DataFrame(primary_fmt_rows)
    supp_df = pd.DataFrame(supp_fmt_rows)

    raw_df.to_csv(outdir / "table1_demographics_raw.csv", index=False)
    fmt_df.to_csv(outdir / "table1_demographics_formatted_all.csv", index=False)
    primary_df.to_csv(outdir / "table1_demographics_primary.csv", index=False)
    supp_df.to_csv(outdir / "table1_demographics_supplementary.csv", index=False)

    # LaTeX-friendly versions
    primary_df.to_latex(
        outdir / "table1_demographics_primary.tex",
        index=False,
        escape=False,
        longtable=False,
    )
    supp_df.to_latex(
        outdir / "table1_demographics_supplementary.tex",
        index=False,
        escape=False,
        longtable=False,
    )

    print("\nTable 1 summary")
    if not primary_df.empty:
        print(primary_df.to_string(index=False))
    if not supp_df.empty:
        print("\nSupplementary Table 1 summary")
        print(supp_df.to_string(index=False))

    return raw_df, primary_df, supp_df


def build_before_after_values(pre_csv: str, post_csv: str):
    pre = pd.read_csv(pre_csv)
    post = pd.read_csv(post_csv)

    needed = ["adni_tau180", "oasis3_tau180"]

    pre = pre[pre["cohort_key"].isin(needed)].copy()
    post = post[post["cohort_key"].isin(needed)].copy()

    pre = pre.set_index("cohort_key")
    post = post.set_index("cohort_key")

    rows = []
    for key in needed:
        rows.append(
            {
                "cohort_key": key,
                "cohort_label": pre.loc[key, "cohort_label"] if key in pre.index else post.loc[key, "cohort_label"],
                "auc_demo_before": float(pre.loc[key, "auc_demo"]),
                "auc_demo_after": float(post.loc[key, "auc_demo"]),
            }
        )

    df = pd.DataFrame(rows)
    df["gap_before"] = abs(df.loc[df["cohort_key"] == "adni_tau180", "auc_demo_before"].iloc[0]
                           - df.loc[df["cohort_key"] == "oasis3_tau180", "auc_demo_before"].iloc[0])
    df["gap_after"] = abs(df.loc[df["cohort_key"] == "adni_tau180", "auc_demo_after"].iloc[0]
                          - df.loc[df["cohort_key"] == "oasis3_tau180", "auc_demo_after"].iloc[0])
    return df


def plot_figure3(before_after_df: pd.DataFrame, outdir: Path):
    labels = ["ADNI tau180", "OASIS3 tau180"]
    before_vals = [
        float(before_after_df.loc[before_after_df["cohort_key"] == "adni_tau180", "auc_demo_before"].iloc[0]),
        float(before_after_df.loc[before_after_df["cohort_key"] == "oasis3_tau180", "auc_demo_before"].iloc[0]),
    ]
    after_vals = [
        float(before_after_df.loc[before_after_df["cohort_key"] == "adni_tau180", "auc_demo_after"].iloc[0]),
        float(before_after_df.loc[before_after_df["cohort_key"] == "oasis3_tau180", "auc_demo_after"].iloc[0]),
    ]

    gap_before = abs(before_vals[0] - before_vals[1])
    gap_after = abs(after_vals[0] - after_vals[1])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)

    x = np.arange(len(labels))

    for ax, vals, title, gap in zip(
        axes,
        [before_vals, after_vals],
        ["Before harmonization", "After harmonization"],
        [gap_before, gap_after],
    ):
        bars = ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.45, 0.90)
        ax.set_ylabel("Demo-only AUC")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

        for i, b in enumerate(bars):
            y = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                y + 0.01,
                f"{vals[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # gap annotation
        y_gap = max(vals) + 0.06
        ax.plot([x[0], x[0], x[1], x[1]], [y_gap - 0.005, y_gap, y_gap, y_gap - 0.005], lw=1.2)
        ax.text(
            np.mean(x),
            y_gap + 0.005,
            f"Gap = {gap:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.suptitle("Effect of feature harmonization on cross-cohort demo-only prediction", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    png = outdir / "figure3_before_after_harmonization.png"
    pdf = outdir / "figure3_before_after_harmonization.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return png, pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pre-harmonization-table",
        default=DEFAULT_PRE_HARMONIZATION_TABLE,
        help="Existing pre-matched summary table (expected to contain auc_demo).",
    )
    parser.add_argument(
        "--post-harmonization-table",
        default=DEFAULT_POST_HARMONIZATION_TABLE,
        help="Current matched-feature primary table (expected to contain auc_demo).",
    )
    parser.add_argument(
        "--nacc-subject-csv",
        default=None,
        help="Optional subject-level CSV for NACC strict A/T table-1 row.",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Table 1
    raw_df, primary_df, supp_df = build_table1(
        outdir=outdir,
        nacc_subject_csv=args.nacc_subject_csv,
    )

    # Figure 3 values + plot
    before_after_df = build_before_after_values(
        pre_csv=args.pre_harmonization_table,
        post_csv=args.post_harmonization_table,
    )
    before_after_df.to_csv(outdir / "figure3_before_after_values.csv", index=False)

    png, pdf = plot_figure3(before_after_df, outdir)

    print("\nFigure 3 values")
    print(before_after_df.to_string(index=False))

    print("\nSaved:")
    print(" ", outdir / "table1_demographics_raw.csv")
    print(" ", outdir / "table1_demographics_formatted_all.csv")
    print(" ", outdir / "table1_demographics_primary.csv")
    print(" ", outdir / "table1_demographics_supplementary.csv")
    print(" ", outdir / "table1_demographics_primary.tex")
    print(" ", outdir / "table1_demographics_supplementary.tex")
    print(" ", outdir / "figure3_before_after_values.csv")
    print(" ", png)
    print(" ", pdf)


if __name__ == "__main__":
    main()