#!/usr/bin/env python3
"""
Tau-Clinical Mismatch Analysis v3
=================================
Research Question: Given similar tau/amyloid burden, why do some people 
have worse clinical severity and faster decline, and can MRI/WMH/plasma 
explain that mismatch?

Endpoint: delta_CDR_CDRSB (primary), delta_MMSE (secondary)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
MASTER_PATH = "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/adni_master_visits_06Mar2026.csv.gz"
ANCHOR_PATH = "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output/resilience_tau_anchor_followup24m.csv"
OUTPUT_DIR = "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output"

# =============================================================================
# STEP 1: LOAD AND ADD DEMOGRAPHICS
# =============================================================================
print("="*70)
print("STEP 1: Loading data and adding demographics")
print("="*70)

# Load anchor cohort
df = pd.read_csv(ANCHOR_PATH, low_memory=False)
print(f"Loaded {len(df)} tau-anchored subjects")

# Load master to get demographics
master = pd.read_csv(MASTER_PATH, usecols=[
    'PTID', 'VISCODE2', 'DEM_PTGENDER', 'DEM_PTEDUCAT', 'AGE'
], low_memory=False)

# Get demographics at baseline (first row per subject)
demo = master.groupby('PTID').first()[['DEM_PTGENDER', 'DEM_PTEDUCAT', 'AGE']].reset_index()
demo.columns = ['PTID', 'SEX', 'EDUCATION', 'AGE_MASTER']

# Merge
df = df.merge(demo, on='PTID', how='left')

# Use AGE from anchor if available, else from master
if 'AGE' in df.columns:
    df['AGE_FINAL'] = df['AGE'].fillna(df['AGE_MASTER'])
else:
    df['AGE_FINAL'] = df['AGE_MASTER']

# Encode sex
df['SEX_MALE'] = (df['SEX'] == 'Male').astype(int)

print(f"With demographics: {df[['AGE_FINAL', 'SEX', 'EDUCATION']].notna().all(axis=1).sum()}")

# =============================================================================
# STEP 2: DEFINE MISMATCH SCORE
# =============================================================================
print("\n" + "="*70)
print("STEP 2: Computing baseline mismatch score")
print("="*70)

# Model: CDR-SB ~ tau + amyloid + age + education + sex
pathology_cols = ['TAU6MM_META_TEMPORAL_SUVR', 'AMY6MM_CENTILOIDS']
demo_cols = ['AGE_FINAL', 'EDUCATION', 'SEX_MALE']
all_features = pathology_cols + demo_cols

# Complete cases
valid = df['CDR_CDRSB'].notna()
for col in all_features:
    if col in df.columns:
        valid &= df[col].notna()

df_model = df[valid].copy()
print(f"Complete cases for mismatch model: {len(df_model)}")

# Fit model
X = df_model[all_features].values.astype(float)
y = df_model['CDR_CDRSB'].values.astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

print(f"\nBaseline model R² = {model.score(X_scaled, y):.3f}")
print("Coefficients:")
for feat, coef in zip(all_features, model.coef_):
    print(f"  {feat}: {coef:.3f}")

# Predict and compute residuals
df_model['CDR_expected'] = model.predict(X_scaled)
df_model['mismatch_score'] = df_model['CDR_CDRSB'] - df_model['CDR_expected']

print(f"\nMismatch score: mean={df_model['mismatch_score'].mean():.3f}, std={df_model['mismatch_score'].std():.3f}")

# =============================================================================
# STEP 3: NESTED PROGNOSTIC MODELS
# =============================================================================
print("\n" + "="*70)
print("STEP 3: Nested prognostic models for future decline")
print("="*70)

# Filter to subjects with follow-up CDR-SB
has_fu = df_model['delta_CDR_CDRSB'].notna()
df_prog = df_model[has_fu].copy()
print(f"Subjects with CDR-SB follow-up: {len(df_prog)}")

# Outcome
y_prog = df_prog['delta_CDR_CDRSB'].values

# Define feature tiers
tier_definitions = {
    'Tier1_PathologyOnly': pathology_cols,
    'Tier2_Path+Mismatch': pathology_cols + ['mismatch_score'],
    'Tier3_Path+Mismatch+MRI': pathology_cols + ['mismatch_score', 'FS7_HIPPO_BILAT_ICVnorm', 'WMH_TOTAL_WMH'],
    'Tier4_Full': pathology_cols + ['mismatch_score', 'FS7_HIPPO_BILAT_ICVnorm', 'WMH_TOTAL_WMH', 'PLASMA_pT217_F', 'PLASMA_NfL_Q', 'apoe4_carrier'],
}

# Add APOE if not present
if 'apoe4_carrier' not in df_prog.columns and 'APOE_GENOTYPE' in df_prog.columns:
    df_prog['apoe4_carrier'] = df_prog['APOE_GENOTYPE'].astype(str).str.contains('4', na=False).astype(int)

print("\n--- NESTED MODEL COMPARISON (5-fold CV) ---")
print(f"{'Tier':<30} {'N':>6} {'R² (CV)':>12} {'MAE (CV)':>12}")
print("-" * 65)

results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for tier_name, feature_list in tier_definitions.items():
    # Check which features are available
    avail_features = [f for f in feature_list if f in df_prog.columns]
    
    # Complete cases for this tier
    tier_valid = df_prog[avail_features + ['delta_CDR_CDRSB']].dropna()
    
    if len(tier_valid) < 30:
        print(f"{tier_name:<30} {'N/A':>6} {'too few':>12} {'samples':>12}")
        continue
    
    X_tier = tier_valid[avail_features].values
    y_tier = tier_valid['delta_CDR_CDRSB'].values
    
    # Standardize
    X_tier_scaled = StandardScaler().fit_transform(X_tier)
    
    # Cross-validation
    r2_scores = cross_val_score(Ridge(alpha=1.0), X_tier_scaled, y_tier, cv=kf, scoring='r2')
    mae_scores = -cross_val_score(Ridge(alpha=1.0), X_tier_scaled, y_tier, cv=kf, scoring='neg_mean_absolute_error')
    
    print(f"{tier_name:<30} {len(tier_valid):>6} {r2_scores.mean():>10.3f}±{r2_scores.std():.2f} {mae_scores.mean():>10.3f}±{mae_scores.std():.2f}")
    
    results[tier_name] = {
        'n': len(tier_valid),
        'r2_mean': r2_scores.mean(),
        'r2_std': r2_scores.std(),
        'mae_mean': mae_scores.mean(),
        'mae_std': mae_scores.std(),
        'features': avail_features
    }

# =============================================================================
# STEP 4: MISMATCH vs DECLINE VALIDATION
# =============================================================================
print("\n" + "="*70)
print("STEP 4: Does mismatch predict future decline?")
print("="*70)

# Correlation
r, p = stats.pearsonr(df_prog['mismatch_score'], df_prog['delta_CDR_CDRSB'])
print(f"\nPearson correlation (mismatch vs delta_CDR_CDRSB):")
print(f"  r = {r:.3f}, p = {p:.4f}")

if r > 0 and p < 0.05:
    print("  → Higher mismatch (worse than expected) predicts FASTER decline ✓")
elif r < 0 and p < 0.05:
    print("  → Higher mismatch predicts SLOWER decline")
else:
    print("  → No significant relationship")

# Tertile analysis
df_prog['mismatch_tertile'] = pd.qcut(df_prog['mismatch_score'], 3, labels=['Low', 'Mid', 'High'])

print("\nDelta CDR-SB by mismatch tertile:")
print(f"{'Tertile':<10} {'N':>6} {'Mean Δ':>10} {'SD':>8}")
print("-" * 40)

for tertile in ['Low', 'Mid', 'High']:
    mask = df_prog['mismatch_tertile'] == tertile
    n = mask.sum()
    if n > 0:
        change = df_prog.loc[mask, 'delta_CDR_CDRSB']
        print(f"{tertile:<10} {n:>6} {change.mean():>+10.3f} {change.std():>8.3f}")

# ANOVA
from scipy.stats import f_oneway
groups = [df_prog.loc[df_prog['mismatch_tertile'] == t, 'delta_CDR_CDRSB'].values for t in ['Low', 'Mid', 'High']]
f_stat, p_anova = f_oneway(*groups)
print(f"\nANOVA: F = {f_stat:.2f}, p = {p_anova:.4f}")

# =============================================================================
# STEP 5: REGIONAL ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("STEP 5: What explains mismatch?")
print("="*70)

# Correlate mismatch with potential explanatory features
explain_features = [
    ('FS7_HIPPO_BILAT_ICVnorm', 'Hippocampal volume (ICV-norm)'),
    ('WMH_TOTAL_WMH', 'White matter hyperintensities'),
    ('PLASMA_pT217_F', 'Plasma p-tau217'),
    ('PLASMA_NfL_Q', 'Plasma NfL'),
    ('PLASMA_GFAP_Q', 'Plasma GFAP'),
]

print("\nCorrelations with mismatch score:")
print(f"{'Feature':<35} {'r':>8} {'p':>10} {'N':>6}")
print("-" * 65)

for col, name in explain_features:
    if col in df_prog.columns:
        valid_mask = df_prog[col].notna() & df_prog['mismatch_score'].notna()
        if valid_mask.sum() > 10:
            r, p = stats.pearsonr(
                df_prog.loc[valid_mask, col],
                df_prog.loc[valid_mask, 'mismatch_score']
            )
            sig = '*' if p < 0.05 else ''
            print(f"{name:<35} {r:>+8.3f} {p:>10.4f} {valid_mask.sum():>6} {sig}")

# =============================================================================
# STEP 6: TAU+ SUBGROUP ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("STEP 6: Tau-positive subgroup")
print("="*70)

if 'tau_pos' in df_prog.columns:
    df_taup = df_prog[df_prog['tau_pos'] == 1].copy()
    print(f"Tau+ subjects with follow-up: {len(df_taup)}")
    
    if len(df_taup) >= 20:
        r, p = stats.pearsonr(df_taup['mismatch_score'], df_taup['delta_CDR_CDRSB'])
        print(f"\nIn tau+ subgroup:")
        print(f"  Correlation (mismatch vs decline): r = {r:.3f}, p = {p:.4f}")
        
        # Tertile analysis
        df_taup['mismatch_tertile'] = pd.qcut(df_taup['mismatch_score'], 3, labels=['Low', 'Mid', 'High'], duplicates='drop')
        
        print("\nDelta CDR-SB by mismatch tertile (tau+ only):")
        for tertile in ['Low', 'Mid', 'High']:
            mask = df_taup['mismatch_tertile'] == tertile
            n = mask.sum()
            if n > 0:
                change = df_taup.loc[mask, 'delta_CDR_CDRSB']
                print(f"  {tertile}: N={n}, mean Δ={change.mean():+.3f}")

# =============================================================================
# SAVE
# =============================================================================
output_file = f"{OUTPUT_DIR}/mismatch_analysis_v3.csv"
df_model.to_csv(output_file, index=False)
print(f"\nSaved to: {output_file}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
