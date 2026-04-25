# plot_crosscohort_severity_figure.py
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SUMMARY = (
    "/project/aereditato/abhat/adni-mri-classification/"
    "crosscohort_bootstrap_ci/bootstrap_auc_summary.csv"
)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=COHORT_ORDER_PRIMARY,
        choices=list(COHORT_LABELS.keys()),
    )
    parser.add_argument(
        "--outdir",
        default="/project/aereditato/abhat/adni-mri-classification/crosscohort_bootstrap_ci/figures",
    )
    parser.add_argument(
        "--title",
        default="Cross-cohort decomposition of tau-positivity prediction by feature domain",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary_csv)
    df = df[df["experiment"].isin(EXPERIMENT_ORDER)].copy()
    df = df[df["cohort_key"].isin(args.cohorts)].copy()

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

        x = sub["auc_boot_mean"].to_numpy()
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
            capsize=3,
            linewidth=1.3,
            markersize=6.5,
            label=EXPERIMENT_LABELS[exp],
        )

    ax.set_yticks(base_y)
    ax.set_yticklabels([COHORT_LABELS[k] for k in args.cohorts])

    ax.set_xlabel("Mean AUC with 95% bootstrap CI")
    ax.set_title(args.title)

    ax.set_xlim(0.6, 1.00)
    ax.grid(axis="x", alpha=0.3)
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