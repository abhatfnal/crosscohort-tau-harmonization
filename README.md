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

- `build_additional_files.py`  
  Builds Additional files 1–8 (one .xlsx each, with a README sheet) from the outputs above.

- `check_submission_package.py`  
  Pre-submission checks: figure fonts and resolution, a privacy scan of every file,
  a cross-check of Additional-file values against the manuscript PDF, and zipping.

All tables and figures report balanced logistic regression, the prespecified
primary classifier; random forest and extra-trees results are robustness checks only
(Additional file 1).

## Reproducing the manuscript

The scripts read controlled-access, participant-level tables whose locations are set
at the top of each script; those tables are not in this repository. Results were
produced with Python 3.10, scikit-learn 1.7.0, pandas 2.3, numpy 2.2, matplotlib 3.10
and openpyxl 3.1. All cross-validation uses five stratified folds with seed 42.

Run in this order (outputs go to the folders named, inside this repository unless noted):

| Step | Command | Produces |
|---|---|---|
| 1 | `python crosscohort_tau_severity_strip.py --dataset <adni_tau90 \| adni_tau180 \| oasis3_tau90 \| oasis3_tau180>` | Subject-level analysis tables and the all-available ablation, written next to the source data (Additional file 3 source) |
| 2 | `python audit_oasis_feature_availability.py` | `crosscohort_matched_rerun/oasis_feature_audit/` (Additional file 2) |
| 3 | `python crosscohort_tau_matched_rerun.py` | `crosscohort_matched_rerun/` (Tables 3–4 AUCs, Additional file 1) |
| 4 | `python bootstrap_crosscohort_auc_ci.py` | `crosscohort_bootstrap_ci/` (1000 resamples; CIs in the text, Fig. 2) |
| 5 | `python plot_crosscohort_severity_figure.py` | Fig. 2: `crosscohort_bootstrap_ci/figures/figure_primary_auc_with_ci.pdf` |
| 6 | `python build_table1_and_figure3.py` | Table 1 and Fig. 3: `paper_tables_and_figures/` |
| 7 | `python harmonization_gap_table.py` | Table 2: `paper_tables_and_figures/table2_harmonization_gap.csv` |
| 8 | `python nacc_at_strict_ablation_v1.py` | NACC ablation, written next to the NACC source data |
| 9 | `python summarize_crosscohort_severity_profiles.py` then `python rebuild_crosscohort_publication_tables.py` | `crosscohort_severity_summary/`, `crosscohort_publication_tables/` (Tables 3–4) |
| 10 | `python sensitivity_cdrsb0.py` | `sensitivity_cdrsb0/` (CDR-SB = 0 counts, Section 3.4) |
| 11 | `python oasis3_threshold_sensitivity.py` | Fig. S1, Table S5 and the published-cutoff comparison: `oasis3_threshold_sensitivity/` |
| 12 | `python logreg_feature_importance.py` | Table S6: `logreg_feature_importance/` |
| 13 | `python dx_stratified_auc.py` | Fig. S2: `dx_stratified_auc/` |
| 14 | `python oasis3_continuous_suvr_sensitivity.py` | Table S7, Fig. S3: `oasis3_continuous_suvr_sensitivity/` |
| 15 | `python nacc_additional_file4.py` | NACC sensitivity table: `paper_tables_and_figures/additional_file4_nacc_sensitivity.csv` |
| 16 | `python build_additional_files.py --outdir <folder>` | Additional files 1–8 (.xlsx) |
| 17 | `python check_submission_package.py --package <folder> --manuscript <pdf> [--zip <file>]` | Check report; zip only if every check passes |

Fig. 1 (study workflow) is not generated by code in this repository;
`figures/figure1_workflow_schematic.*` (from `image_creation.py`) is an earlier design,
not the manuscript figure.

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
