# plot_crosscohort_severity_figure.py
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_SUMMARY = ROOT / "crosscohort_bootstrap_ci" / "bootstrap_auc_summary.csv"
DEFAULT_OUTDIR = ROOT / "crosscohort_bootstrap_ci" / "figures"
PRIMARY_MODEL = "logreg_balanced"


EXPERIMENT_ORDER = [
    "reference_combined",
    "severity_only",
    "demo_only",
]

EXPERIMENT_LABELS = {
    "reference_combined": "Reference",
    "severity_only": "Severity-only",
    "demo_only": "Demo-only",
}

COHORT_ORDER_PRIMARY = [
    "adni_tau90",
    "adni_tau180",
    "oasis3_tau180",
]

COHORT_LABELS = {
    "adni_tau90": "ADNI tau90",
    "adni_tau180": "ADNI tau180",
    "oasis3_tau180": "OASIS3 tau180",
    "oasis3_tau90": "OASIS3 tau90",
}

# Colour-blind-safe categorical trio (validated: worst all-pairs CVD dE 9.2);
# marker shape repeats the experiment identity for greyscale print.
EXPERIMENT_COLORS = {
    "reference_combined": "#2a78d6",
    "severity_only": "#eb6834",
    "demo_only": "#1baf7a",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=COHORT_ORDER_PRIMARY,
        choices=list(COHORT_LABELS.keys()),
    )
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--title",
        default="",
        help="Optional in-figure title; the manuscript caption carries the title.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv)
    df = df[df["experiment"].isin(EXPERIMENT_ORDER)].copy()
    df = df[df["cohort_key"].isin(args.cohorts)].copy()

    # Points and intervals must come from one bootstrap run of the primary model,
    # so the figure cannot drift from Table 3 or the CIs quoted in the text.
    if not df["model"].eq(PRIMARY_MODEL).all():
        raise ValueError(f"{args.summary_csv} contains non-{PRIMARY_MODEL} rows")
    if df["n_bootstrap"].nunique() != 1:
        raise ValueError(f"{args.summary_csv} mixes bootstrap runs: {df['n_bootstrap'].unique()}")

    # enforce order
    df["cohort_key"] = pd.Categorical(df["cohort_key"], categories=args.cohorts, ordered=True)
    df["experiment"] = pd.Categorical(df["experiment"], categories=EXPERIMENT_ORDER, ordered=True)
    df = df.sort_values(["cohort_key", "experiment"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    base_y = np.arange(len(args.cohorts))[::-1]
    offsets = {
        "reference_combined": 0.22,
        "severity_only": 0.00,
        "demo_only": -0.22,
    }
    markers = {
        "reference_combined": "o",
        "severity_only": "s",
        "demo_only": "^",
    }

    for exp in EXPERIMENT_ORDER:
        sub = df[df["experiment"] == exp].copy()
        y = np.array([base_y[args.cohorts.index(k)] + offsets[exp] for k in sub["cohort_key"]])

        # Point = five-fold CV AUC on the full cohort (the Table 3 value);
        # bar = 95% percentile interval from the bootstrap replicates.
        x = sub["auc_point"].to_numpy()
        xlo = sub["auc_ci_low"].to_numpy()
        xhi = sub["auc_ci_high"].to_numpy()

        left_err = np.maximum(0, x - xlo)
        right_err = np.maximum(0, xhi - x)
        xerr = np.vstack([left_err, right_err])

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt=markers[exp],
            color=EXPERIMENT_COLORS[exp],
            capsize=3,
            linewidth=1.3,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=EXPERIMENT_LABELS[exp],
        )

    ax.set_yticks(base_y)
    ax.set_yticklabels([COHORT_LABELS[k] for k in args.cohorts])

    ax.set_xlabel("Five-fold cross-validated AUC (bars: 95% bootstrap CI)")
    if args.title:
        ax.set_title(args.title)

    ax.set_xlim(0.6, 1.00)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best")

    plt.tight_layout()

    png_path = outdir / "figure_primary_auc_with_ci.png"
    pdf_path = outdir / "figure_primary_auc_with_ci.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(" ", png_path)
    print(" ", pdf_path)


if __name__ == "__main__":
    main()