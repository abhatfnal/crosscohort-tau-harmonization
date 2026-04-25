#!/usr/bin/env python3
"""
OASIS3 GMM-Threshold Sensitivity Analysis
==========================================
The y_target in each OASIS3 subject table was derived from a GMM midpoint
threshold applied to PET_fSUVR_TOT_CORTMEAN_x.  This script sweeps ±30 % around
that threshold to verify that classification performance (logreg CV AUC) is not
sensitive to the exact cut-point choice.

For each cohort × threshold combination the script reports:
  - N_pos / N_neg
  - 5-fold stratified CV AUC (logreg_balanced)
  - Whether the result meets the minimum-positive-class requirement (N_pos ≥ 10)

Outputs:
  oasis3_threshold_sensitivity/
    threshold_sweep_results.csv
    threshold_sweep_plot.png / .pdf
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

OUTDIR = Path("/project/aereditato/abhat/adni-mri-classification/oasis3_threshold_sensitivity")
OUTDIR.mkdir(parents=True, exist_ok=True)

COHORTS = {
    "oasis3_tau180": {
        "label": "OASIS3 tau180",
        "path": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau180/subject_level_input_table.csv",
        "suvr_col": "PET_fSUVR_TOT_CORTMEAN_x",
    },
    "oasis3_tau90": {
        "label": "OASIS3 tau90",
        "path": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau90/subject_level_input_table.csv",
        "suvr_col": "PET_fSUVR_TOT_CORTMEAN_x",
    },
}

NUM_FEATURES = ["age_h", "apoe_e4_carrier_h", "apoe_e4_count_h", "education_years_h"]
CAT_FEATURES = ["sex_h"]
MIN_POS_FOR_AUC = 10
N_THRESHOLDS = 21   # odd number so midpoint is always included
SWEEP_PCT = 0.30    # ±30 % around GMM midpoint
N_SPLITS_CV = 5
RANDOM_STATE = 42


def gmm_midpoint(suvr: pd.Series) -> float:
    arr = suvr.dropna().values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE, n_init=10)
    gmm.fit(arr)
    means = np.sort(gmm.means_.flatten())
    return float(np.mean(means))


def make_pipe(num_cols, cat_cols):
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]),
            cat_cols,
        ))
    preproc = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([
        ("preproc", preproc),
        ("model", LogisticRegression(
            max_iter=5000, class_weight="balanced",
            solver="liblinear", random_state=RANDOM_STATE,
        )),
    ])


def cv_auc(df: pd.DataFrame, num_cols, cat_cols, n_splits=N_SPLITS_CV):
    use_num = [c for c in num_cols if c in df.columns]
    use_cat = [c for c in cat_cols if c in df.columns]
    if not (use_num + use_cat):
        return np.nan
    X = df[use_num + use_cat].copy()
    y = df["y_target"].astype(int).to_numpy()
    n_pos = int(y.sum())
    if n_pos < 2 or (len(y) - n_pos) < 2:
        return np.nan
    actual_splits = min(n_splits, n_pos, len(y) - n_pos)
    if actual_splits < 2:
        return np.nan
    cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=RANDOM_STATE)
    pipe = make_pipe(use_num, use_cat)
    fold_aucs = []
    for tr, te in cv.split(X, y):
        pipe.fit(X.iloc[tr], y[tr])
        if len(np.unique(y[te])) < 2:
            continue
        prob = pipe.predict_proba(X.iloc[te])[:, 1]
        fold_aucs.append(roc_auc_score(y[te], prob))
    return float(np.mean(fold_aucs)) if fold_aucs else np.nan


rows = []

for cohort_key, cfg in COHORTS.items():
    print(f"\n{'='*60}")
    print(f"{cfg['label']}")
    print(f"{'='*60}")

    df = pd.read_csv(cfg["path"], low_memory=False)
    suvr_col = cfg["suvr_col"]

    suvr = pd.to_numeric(df.get(suvr_col, pd.Series(dtype=float)), errors="coerce")
    df["_suvr"] = suvr

    # Drop rows without SUVR
    df = df[df["_suvr"].notna()].copy()
    n_total = len(df)

    # Refit GMM on this subset (same procedure as pipeline)
    gmm_thr = gmm_midpoint(df["_suvr"])
    print(f"GMM midpoint threshold: {gmm_thr:.4f}")
    print(f"N with SUVR: {n_total}")

    # Sweep thresholds ±SWEEP_PCT around GMM midpoint
    thresholds = np.linspace(
        gmm_thr * (1 - SWEEP_PCT),
        gmm_thr * (1 + SWEEP_PCT),
        N_THRESHOLDS,
    )

    for thr in thresholds:
        df["y_target"] = (df["_suvr"] >= thr).astype(int)
        n_pos = int(df["y_target"].sum())
        n_neg = n_total - n_pos
        reliable = n_pos >= MIN_POS_FOR_AUC and n_neg >= MIN_POS_FOR_AUC

        auc = cv_auc(df, NUM_FEATURES, CAT_FEATURES) if reliable else np.nan
        is_gmm = abs(thr - gmm_thr) < 1e-9

        print(
            f"  thr={thr:.4f}{'*' if is_gmm else ' '}: "
            f"N+={n_pos:3d}, N-={n_neg:3d}, "
            f"AUC={'NA' if np.isnan(auc) else f'{auc:.3f}'}{'  [UNRELIABLE]' if not reliable else ''}"
        )

        rows.append({
            "cohort_key": cohort_key,
            "cohort_label": cfg["label"],
            "threshold": round(float(thr), 5),
            "gmm_midpoint": round(float(gmm_thr), 5),
            "is_gmm_threshold": is_gmm,
            "pct_from_gmm": round(100 * (thr - gmm_thr) / gmm_thr, 1),
            "n_total": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auc": round(float(auc), 4) if not np.isnan(auc) else np.nan,
            "reliable": reliable,
        })

results_df = pd.DataFrame(rows)
results_df.to_csv(OUTDIR / "threshold_sweep_results.csv", index=False)
print(f"\nSaved: {OUTDIR / 'threshold_sweep_results.csv'}")


# ── Plot ──────────────────────────────────────────────────────────────────────
COHORT_COLORS = {
    "oasis3_tau180": "#2171b5",
    "oasis3_tau90":  "#6baed6",
}
COHORT_STYLES = {
    "oasis3_tau180": "-o",
    "oasis3_tau90":  "--s",
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.subplots_adjust(wspace=0.35)

ax_auc, ax_npos = axes

for cohort_key, cfg in COHORTS.items():
    sub = results_df[results_df["cohort_key"] == cohort_key].copy()
    reliable = sub[sub["reliable"]]
    unreliable = sub[~sub["reliable"]]

    color = COHORT_COLORS[cohort_key]
    style = COHORT_STYLES[cohort_key]

    # AUC panel
    if not reliable.empty:
        ax_auc.plot(
            reliable["threshold"], reliable["auc"],
            style, color=color, label=cfg["label"],
            markersize=5, linewidth=1.8,
        )
    if not unreliable.empty:
        ax_auc.scatter(
            unreliable["threshold"], [0.5] * len(unreliable),
            marker="x", color=color, s=30, zorder=3,
        )

    # Mark GMM midpoint
    gmm_row = sub[sub["is_gmm_threshold"]]
    if not gmm_row.empty and gmm_row["reliable"].iloc[0]:
        ax_auc.axvline(gmm_row["threshold"].iloc[0], color=color, alpha=0.4,
                       linestyle=":", linewidth=1.2)

    # N_pos panel
    ax_npos.plot(
        sub["threshold"], sub["n_pos"],
        style, color=color, label=cfg["label"],
        markersize=5, linewidth=1.8,
    )
    if not gmm_row.empty:
        ax_npos.axvline(gmm_row["threshold"].iloc[0], color=color, alpha=0.4,
                        linestyle=":", linewidth=1.2)

ax_auc.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax_auc.set_xlabel("SUVR threshold", fontsize=11)
ax_auc.set_ylabel("5-fold CV AUC (logreg balanced)", fontsize=11)
ax_auc.set_title("AUC vs tau positivity threshold", fontsize=12)
ax_auc.set_ylim(0.3, 1.05)
ax_auc.legend(fontsize=10, framealpha=0.8)
ax_auc.spines[["top", "right"]].set_visible(False)
ax_auc.grid(axis="y", alpha=0.3, linestyle="--")

ax_npos.axhline(MIN_POS_FOR_AUC, color="crimson", linestyle="--",
                linewidth=1.2, alpha=0.7, label=f"Min reliable (N={MIN_POS_FOR_AUC})")
ax_npos.set_xlabel("SUVR threshold", fontsize=11)
ax_npos.set_ylabel("N tau-positive", fontsize=11)
ax_npos.set_title("Tau-positive count vs threshold", fontsize=12)
ax_npos.legend(fontsize=10, framealpha=0.8)
ax_npos.spines[["top", "right"]].set_visible(False)
ax_npos.grid(axis="y", alpha=0.3, linestyle="--")

fig.suptitle(
    "OASIS3 GMM threshold sensitivity (±30 % around midpoint)",
    fontsize=13, fontweight="bold", y=1.01,
)

plt.tight_layout()
png = OUTDIR / "threshold_sweep_plot.png"
pdf = OUTDIR / "threshold_sweep_plot.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {png}")
print(f"Saved: {pdf}")

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY — reliable AUC range per cohort")
print("="*60)
for cohort_key in COHORTS:
    sub = results_df[(results_df["cohort_key"] == cohort_key) & results_df["reliable"]]
    if sub.empty:
        print(f"{cohort_key}: no reliable AUC points")
        continue
    gmm_auc = sub[sub["is_gmm_threshold"]]["auc"]
    gmm_auc_str = f"{gmm_auc.iloc[0]:.3f}" if not gmm_auc.empty else "NA"
    print(
        f"{cohort_key}: AUC {sub['auc'].min():.3f}–{sub['auc'].max():.3f} "
        f"(at GMM midpoint: {gmm_auc_str}), "
        f"N_pos range {sub['n_pos'].min()}–{sub['n_pos'].max()}"
    )
