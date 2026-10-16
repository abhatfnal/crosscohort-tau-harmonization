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
  Builds Table 1 demographics and the CDR-SB = 0 prevalence figure (Fig. 3).

- `harmonization_gap_table.py`  
  Demo-only AUCs before and after feature harmonization (Table 2).

- `summarize_crosscohort_severity_profiles.py`  
  Summarizes cohort-level severity profiles, including the NACC anchor row (Table 4).

- `sensitivity_cdrsb0.py`  
  Sensitivity analysis for cognitively unimpaired / CDR-SB=0 subsets.

- `oasis3_threshold_sensitivity.py`  
  OASIS3 tau-threshold sweep and the published OASIS-3 cutoff (Fig. S1, Table S5).

- `logreg_feature_importance.py`  
  Standardized coefficients of the reference-combined model (Table S6).

- `dx_stratified_auc.py`  
  Diagnosis-stratified feature-block AUCs (Fig. S2).

- `oasis3_continuous_suvr_sensitivity.py`  
  Within-OASIS3 continuous tauopathy SUVR analysis (Table S7, Fig. S3).

- `nacc_at_strict_ablation_v1.py`  
  NACC strict A/T feature-block ablation (all classifiers). Tables use its
  `logreg_balanced` rows, not `best_model_per_experiment.csv`.

- `nacc_additional_file4.py`  
  NACC sensitivity table, main cohort and flortaucipir-only subset (Additional file 4).

All tables and figures report balanced logistic regression, the prespecified
primary classifier; random forest and extra-trees results are robustness checks only.

## Repository structure

- `figures/` — manuscript Figs. 2 and 3 (copied from the script outputs) and an early workflow schematic
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

**Feature harmonization reveals that apparent cross-cohort differences in tau-positivity prediction reflect variable availability and cohort stage structure**

## Author

Avinay Bhat  
University of Chicago
