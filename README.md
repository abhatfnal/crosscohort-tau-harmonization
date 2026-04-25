# Cross-Cohort Tau Harmonization Analysis

Code for the cross-cohort tau-positivity harmonization study comparing ADNI, OASIS3, and NACC.

## Project summary

This repository contains the analysis code used for:
- within-cohort severity-strip ablation
- OASIS3 feature availability audit
- matched-feature rerun across ADNI and OASIS3
- bootstrap confidence intervals for primary AUC contrasts
- publication tables and figures
- sensitivity analyses for CDR-SB=0 restriction and OASIS3 threshold choices

## Main scripts

- `crosscohort_tau_severity_strip.py`  
  Runs within-cohort severity-strip ablation analyses.

- `audit_oasis_feature_availability.py`  
  Audits raw and subject-level OASIS3 tables for recoverable variables.

- `crosscohort_tau_matched_rerun.py`  
  Runs the matched-feature rerun across cohorts using only shared harmonized variables.

- `rebuild_crosscohort_publication_tables.py`  
  Rebuilds publication-ready primary and supplementary tables.

- `bootstrap_crosscohort_auc_ci.py`  
  Computes bootstrap confidence intervals for AUCs and AUC contrasts.

- `plot_crosscohort_severity_figure.py`  
  Generates the primary severity-decomposition figure.

- `build_table1_and_figure3.py`  
  Builds Table 1 demographics and the harmonization comparison figure.

- `summarize_crosscohort_severity_profiles.py`  
  Summarizes cohort-level severity profiles.

- `sensitivity_cdrsb0.py`  
  Sensitivity analysis for cognitively unimpaired / CDR-SB=0 subsets.

- `oasis3_threshold_sensitivity.py`  
  Sensitivity analysis for OASIS3 tau-threshold definitions.

## Repository structure

- `figures/` — manuscript figures
- `crosscohort_bootstrap_ci/` — bootstrap outputs
- `crosscohort_matched_rerun/` — matched-feature rerun outputs
- `crosscohort_publication_tables/` — publication-ready tables
- `paper_tables_and_figures/` — manuscript support files

## Reproducibility notes

This repository contains code and aggregate outputs only.
No participant-level ADNI, OASIS3, or NACC data are redistributed.

Users must independently obtain dataset access through the relevant data use agreements:
- ADNI
- OASIS3
- NACC

## Manuscript context

This repository supports the manuscript:

**Feature harmonization reveals that apparent cross-cohort differences in tau-positivity prediction are driven by clinical severity and variable availability**

## Author

Avinay Bhat  
University of Chicago
