#!/usr/bin/env python3
"""
OASIS3 GMM-Threshold Sensitivity Analysis
==========================================
The y_target in each OASIS3 subject table was derived from a GMM midpoint
threshold applied to the Tauopathy SUVR column (confirmed via build_metadata.json).
This script sweeps ±30 % around that threshold to verify that classification
performance (logreg CV AUC) is not sensitive to the exact cut-point choice.

For each cohort × threshold combination the script reports:
  - N_pos / N_neg
  - 5-fold stratified CV AUC (logreg_balanced)
  - Whether the result meets the minimum-positive-class requirement (N_pos ≥ 10)

The published OASIS-3 tauopathy cutoff (1.22; OASIS-3 Imaging Methods & Data
Dictionary v2.0, Table 18, citing Mishra et al. 2017) is evaluated alongside the
sweep. `Tauopathy` is the OASIS-distributed summary: the mean of the partial-
volume-corrected (RSF) SUVRs in amygdala, entorhinal, inferior temporal and
lateral occipital cortex, i.e. the same measure the cutoff was defined on.
At the GMM midpoint and at the published cutoff the script also reports how
many tau-positive participants are amyloid-negative or have CDR-SB = 0.

Outputs:
  oasis3_threshold_sensitivity/
    threshold_sweep_results.csv
    threshold_sweep_plot.png / .pdf
    published_cutoff_comparison.csv
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

OUTDIR = Path(__file__).resolve().parent / "oasis3_threshold_sensitivity"
OUTDIR.mkdir(parents=True, exist_ok=True)

COHORTS = {
    "oasis3_tau180": {
        "label": "OASIS3 tau180",
        "path": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau180/subject_level_input_table.csv",
        "suvr_col": "Tauopathy",
    },
    "oasis3_tau90": {
        "label": "OASIS3 tau90",
        "path": "/project/aereditato/abhat/oasis/phase2_tau/severity_strip_tau90/subject_level_input_table.csv",
        "suvr_col": "Tauopathy",
    },
}

NUM_FEATURES = ["age_h", "apoe_e4_carrier_h", "apoe_e4_count_h", "education_years_h"]
CAT_FEATURES = ["sex_h"]
SEV_FEATURES = ["cdr_sumboxes_h", "faq_total_h"]
MIN_POS_FOR_AUC = 10
N_THRESHOLDS = 21   # odd number so midpoint is always included
SWEEP_PCT = 0.30    # ±30 % around GMM midpoint
N_SPLITS_CV = 5
RANDOM_STATE = 42

# OASIS-3 Imaging Data Dictionary v2.0, Table 18 (tauopathy, cerebellar cortex reference)
PUBLISHED_CUTOFF = 1.22
# Same dictionary, Table 11: native cerebellar-cortex RSF cutoffs for mean cortical SUVR
AMYLOID_TRACER_COL = "tracer_y"
AMYLOID_SUVR_COL = "PET_fSUVR_rsf_TOT_CORTMEAN"
AMYLOID_RSF_CUTOFFS = {"PIB": 1.42, "AV45": 1.19}


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


def amyloid_status(df: pd.DataFrame) -> pd.Series:
    """1/0 amyloid positivity from the matched amyloid PET; NaN if unavailable."""
    tracer = df[AMYLOID_TRACER_COL].astype(str).str.upper().str.replace("-", "", regex=False)
    suvr = pd.to_numeric(df[AMYLOID_SUVR_COL], errors="coerce")
    cutoff = tracer.map(AMYLOID_RSF_CUTOFFS)
    return (suvr > cutoff).astype(float).where(suvr.notna() & cutoff.notna())


rows = []
cohort_frames = {}

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

    # Sweep thresholds ±SWEEP_PCT around GMM midpoint, plus the published cutoff
    thresholds = np.linspace(
        gmm_thr * (1 - SWEEP_PCT),
        gmm_thr * (1 + SWEEP_PCT),
        N_THRESHOLDS,
    )
    evaluated = [(thr, False) for thr in thresholds] + [(PUBLISHED_CUTOFF, True)]
    cohort_frames[cohort_key] = (df.copy(), gmm_thr)

    for thr, is_published in evaluated:
        df["y_target"] = (df["_suvr"] >= thr).astype(int)
        n_pos = int(df["y_target"].sum())
        n_neg = n_total - n_pos
        reliable = n_pos >= MIN_POS_FOR_AUC and n_neg >= MIN_POS_FOR_AUC

        auc = cv_auc(df, NUM_FEATURES, CAT_FEATURES) if reliable else np.nan
        is_gmm = abs(thr - gmm_thr) < 1e-9
        flag = "*" if is_gmm else ("P" if is_published else " ")

        print(
            f"  thr={thr:.4f}{flag}: "
            f"N+={n_pos:3d}, N-={n_neg:3d}, "
            f"AUC={'NA' if np.isnan(auc) else f'{auc:.3f}'}{'  [UNRELIABLE]' if not reliable else ''}"
        )

        rows.append({
            "cohort_key": cohort_key,
            "cohort_label": cfg["label"],
            "threshold": round(float(thr), 5),
            "gmm_midpoint": round(float(gmm_thr), 5),
            "is_gmm_threshold": is_gmm,
            "is_published_cutoff": is_published,
            "pct_from_gmm": round(100 * (thr - gmm_thr) / gmm_thr, 1),
            "n_total": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auc": round(float(auc), 4) if not np.isnan(auc) else np.nan,
            "auc_full": float(auc),   # unrounded, for reporting at 3 dp without double rounding
            "reliable": reliable,
        })

results_df = pd.DataFrame(rows).sort_values(["cohort_key", "threshold"]).reset_index(drop=True)
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

NE_Y = 0.36   # "not estimated" band, kept clear of the AUC scale

for cohort_key, cfg in COHORTS.items():
    cohort_rows = results_df[results_df["cohort_key"] == cohort_key]
    sub = cohort_rows[~cohort_rows["is_published_cutoff"]].copy()
    published = cohort_rows[cohort_rows["is_published_cutoff"]]
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
            unreliable["threshold"], [NE_Y] * len(unreliable),
            marker="x", color=color, s=30, zorder=3,
        )
    if not published.empty and published["reliable"].iloc[0]:
        ax_auc.scatter(
            published["threshold"], published["auc"],
            marker="*", s=150, color=color, edgecolor="#0b0b0b", linewidth=0.6, zorder=4,
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

published_label = f"OASIS-3 published cutoff ({PUBLISHED_CUTOFF:.2f})"
for ax in (ax_auc, ax_npos):
    ax.axvline(PUBLISHED_CUTOFF, color="#52514e", linestyle="-.", linewidth=1.2,
               label=published_label)

ax_auc.axhline(0.5, color="#c3c2b7", linewidth=0.9, zorder=0)   # chance
ax_auc.axhline(0.42, color="#c3c2b7", linewidth=0.8, zorder=0)  # separates the n.e. band
ax_auc.set_xlabel("SUVR threshold", fontsize=11)
ax_auc.set_ylabel("Five-fold CV AUC (balanced logistic regression)", fontsize=11)
ax_auc.set_title("AUC vs tau positivity threshold", fontsize=12)
ax_auc.set_ylim(0.3, 1.05)
ax_auc.set_yticks([NE_Y, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax_auc.set_yticklabels(["not est.", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
ax_auc.legend(fontsize=9, framealpha=0.9, loc="upper left")
ax_auc.spines[["top", "right"]].set_visible(False)
ax_auc.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax_auc.set_axisbelow(True)

ax_npos.axhline(MIN_POS_FOR_AUC, color="crimson", linestyle="--",
                linewidth=1.2, alpha=0.7, label=f"Min reliable (N={MIN_POS_FOR_AUC})")
ax_npos.set_xlabel("SUVR threshold", fontsize=11)
ax_npos.set_ylabel("N tau-positive", fontsize=11)
ax_npos.set_title("Tau-positive count vs threshold", fontsize=12)
ax_npos.legend(fontsize=9, framealpha=0.9)
ax_npos.spines[["top", "right"]].set_visible(False)
ax_npos.grid(axis="y", color="#e1e0d9", linewidth=0.8)
ax_npos.set_axisbelow(True)

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
    sub = results_df[(results_df["cohort_key"] == cohort_key) & results_df["reliable"]
                     & ~results_df["is_published_cutoff"]]
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


# ── Published cutoff vs GMM midpoint ──────────────────────────────────────────
# Who does each definition call tau-positive? An AD-type tau endpoint should be
# largely amyloid-positive; feature-block AUCs show how the decomposition moves.
comparison_rows = []
for cohort_key, cfg in COHORTS.items():
    df, gmm_thr = cohort_frames[cohort_key]
    amy = amyloid_status(df)
    cdr0 = pd.to_numeric(df["cdr_sumboxes_h"], errors="coerce").eq(0)

    for definition, thr in [("gmm_midpoint", gmm_thr), ("published_cutoff", PUBLISHED_CUTOFF)]:
        df["y_target"] = (df["_suvr"] >= thr).astype(int)
        tau_pos = df["y_target"].eq(1)
        tau_pos_amy = tau_pos & amy.notna()
        comparison_rows.append({
            "cohort_key": cohort_key,
            "cohort_label": cfg["label"],
            "definition": definition,
            "threshold": round(float(thr), 5),
            "n_total": len(df),
            "n_pos": int(tau_pos.sum()),
            "pct_pos": round(100 * tau_pos.mean(), 1),
            "n_with_amyloid": int(amy.notna().sum()),
            "pct_amyloid_pos": round(100 * amy.mean(), 1),
            "n_pos_with_amyloid": int(tau_pos_amy.sum()),
            "n_pos_amyloid_neg": int((tau_pos_amy & amy.eq(0)).sum()),
            "pct_pos_amyloid_neg": round(100 * amy[tau_pos_amy].eq(0).mean(), 1),
            "n_pos_cdrsb0": int((tau_pos & cdr0).sum()),
            "pct_pos_cdrsb0": round(100 * (tau_pos & cdr0).sum() / tau_pos.sum(), 1),
            "auc_reference": cv_auc(df, NUM_FEATURES + SEV_FEATURES, CAT_FEATURES),
            "auc_severity": cv_auc(df, SEV_FEATURES, []),
            "auc_demo": cv_auc(df, NUM_FEATURES, CAT_FEATURES),
        })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(OUTDIR / "published_cutoff_comparison.csv", index=False)
print("\n" + "=" * 60)
print("PUBLISHED CUTOFF vs GMM MIDPOINT")
print("=" * 60)
print(comparison_df.drop(columns=["cohort_key"]).to_string(index=False))
print(f"Saved: {OUTDIR / 'published_cutoff_comparison.csv'}")
