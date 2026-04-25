#!/usr/bin/env python3
"""
Tier 1-2 Tabular Baselines for Tau Positivity Prediction
- Tier 1: Plasma only
- Tier 2: Plasma + Amyloid Centiloids

Fast to run, establishes comparison for deep learning.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Optional: XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed, using GradientBoosting instead")


def load_data():
    """Load and prepare data."""
    ARC = "/project/aereditato/abhat/ADNI/ARC_06Mar2026/output"
    
    trainval = pd.read_csv(f"{ARC}/trainval_taupos_90d.csv")
    test     = pd.read_csv(f"{ARC}/test_taupos_90d.csv")
    folds    = pd.read_csv(f"{ARC}/folds_taupos_90d_trainval_5fold.csv")
    
    trainval = trainval.merge(folds[["RID", "fold"]], on="RID", how="left")
    
    return trainval, test


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {}
    metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    metrics["auc_pr"] = average_precision_score(y_true, y_prob)
    metrics["brier"] = brier_score_loss(y_true, y_prob)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return metrics


def find_optimal_threshold(y_true, y_prob):
    """Find threshold maximizing Youden's J."""
    best_thresh, best_j = 0.5, 0
    for t in np.arange(0.1, 0.9, 0.02):
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_thresh = t
    return best_thresh


def run_cv(trainval, feature_cols, model_class, model_params, model_name):
    """Run 5-fold CV and return metrics."""
    
    all_y_true = []
    all_y_prob = []
    fold_metrics = []
    
    for fold in range(5):
        train_df = trainval[trainval["fold"] != fold]
        val_df = trainval[trainval["fold"] == fold]
        
        X_train = train_df[feature_cols].values
        y_train = train_df["tau_pos"].values
        X_val = val_df[feature_cols].values
        y_val = val_df["tau_pos"].values
        
        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Train
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # Predict
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
        else:
            y_prob = model.predict(X_val)
        
        # Metrics
        metrics = compute_metrics(y_val, y_prob)
        fold_metrics.append(metrics)
        
        all_y_true.extend(y_val)
        all_y_prob.extend(y_prob)
    
    # Aggregate
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    
    pooled_metrics = compute_metrics(all_y_true, all_y_prob)
    opt_thresh = find_optimal_threshold(all_y_true, all_y_prob)
    pooled_metrics_opt = compute_metrics(all_y_true, all_y_prob, threshold=opt_thresh)
    
    # Summary stats
    summary = {
        "model": model_name,
        "n_features": len(feature_cols),
        "features": feature_cols,
    }
    
    for key in fold_metrics[0].keys():
        values = [m[key] for m in fold_metrics]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    
    summary["pooled_auc_roc"] = pooled_metrics["auc_roc"]
    summary["pooled_auc_pr"] = pooled_metrics["auc_pr"]
    summary["optimal_threshold"] = opt_thresh
    summary["f1_at_optimal"] = pooled_metrics_opt["f1"]
    summary["sens_at_optimal"] = pooled_metrics_opt["sensitivity"]
    summary["spec_at_optimal"] = pooled_metrics_opt["specificity"]
    
    return summary, all_y_true, all_y_prob


def evaluate_test(trainval, test, feature_cols, model_class, model_params, threshold):
    """Train on full trainval, evaluate on test."""
    
    X_train = trainval[feature_cols].values
    y_train = trainval["tau_pos"].values
    X_test = test[feature_cols].values
    y_test = test["tau_pos"].values
    
    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train
    model = model_class(**model_params)
    model.fit(X_train, y_train)
    
    # Predict
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test)
    
    metrics_05 = compute_metrics(y_test, y_prob, threshold=0.5)
    metrics_opt = compute_metrics(y_test, y_prob, threshold=threshold)
    
    return {
        "threshold_0.5": metrics_05,
        "threshold_cv": {"threshold": threshold, **metrics_opt}
    }, y_test, y_prob


def main():
    print("=" * 70)
    print("TABULAR BASELINES FOR TAU POSITIVITY PREDICTION")
    print("=" * 70)
    
    trainval, test = load_data()
    
    print(f"\nTrainval: {len(trainval)} subjects, {trainval['tau_pos'].sum()} tau+")
    print(f"Test: {len(test)} subjects, {test['tau_pos'].sum()} tau+")
    
    # Define feature sets (Tiers)
    PLASMA_COLS = [
        "PLASMA_pT217_F",
        "PLASMA_AB42_AB40_F",
        "PLASMA_NfL_F",
        "PLASMA_GFAP_F",
    ]
    
    AMYLOID_COL = ["AMY6MM_CENTILOIDS"]
    
    tiers = {
        "Tier1_Plasma": PLASMA_COLS,
        "Tier2_Plasma_Amyloid": PLASMA_COLS + AMYLOID_COL,
    }
    
    # Define models
    models = {
        "LogisticRegression": (LogisticRegression, {"max_iter": 1000, "class_weight": "balanced"}),
        "RandomForest": (RandomForestClassifier, {"n_estimators": 100, "max_depth": 5, "class_weight": "balanced", "random_state": 42}),
    }
    
    if HAS_XGB:
        # Compute scale_pos_weight
        pos_rate = trainval["tau_pos"].mean()
        scale_pos = (1 - pos_rate) / pos_rate
        models["XGBoost"] = (XGBClassifier, {
            "n_estimators": 100, 
            "max_depth": 4, 
            "learning_rate": 0.1,
            "scale_pos_weight": scale_pos,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        })
    else:
        models["GradientBoosting"] = (GradientBoostingClassifier, {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": 42
        })
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"outputs/tabular_baselines_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    
    all_results = []
    
    # Run experiments
    for tier_name, feature_cols in tiers.items():
        print(f"\n{'='*60}")
        print(f"{tier_name}: {len(feature_cols)} features")
        print(f"{'='*60}")
        
        for model_name, (model_class, model_params) in models.items():
            print(f"\n--- {model_name} ---")
            
            # CV
            summary, y_true_cv, y_prob_cv = run_cv(
                trainval, feature_cols, model_class, model_params,
                f"{tier_name}_{model_name}"
            )
            
            print(f"CV AUC-ROC: {summary['auc_roc_mean']:.4f} ± {summary['auc_roc_std']:.4f}")
            print(f"CV AUC-PR:  {summary['auc_pr_mean']:.4f} ± {summary['auc_pr_std']:.4f}")
            print(f"Optimal threshold: {summary['optimal_threshold']:.2f}")
            print(f"F1 @ optimal: {summary['f1_at_optimal']:.4f}")
            
            # Test
            test_results, y_true_test, y_prob_test = evaluate_test(
                trainval, test, feature_cols, model_class, model_params,
                summary["optimal_threshold"]
            )
            
            summary["test"] = test_results
            print(f"Test AUC-ROC: {test_results['threshold_0.5']['auc_roc']:.4f}")
            print(f"Test AUC-PR:  {test_results['threshold_0.5']['auc_pr']:.4f}")
            
            all_results.append(summary)
            
            # Save predictions
            pred_df = pd.DataFrame({
                "RID": test["RID"].values,
                "y_true": y_true_test,
                "y_prob": y_prob_test,
            })
            assert len(pred_df) == len(test)
            pred_df.to_csv(f"{out_dir}/{tier_name}_{model_name}_test_preds.csv", index=False)
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'CV AUC-ROC':>12} {'CV AUC-PR':>12} {'Test AUC-ROC':>12}")
    print("-" * 70)
    
    for r in all_results:
        print(f"{r['model']:<35} {r['auc_roc_mean']:>10.4f} ± {r['auc_roc_std']:.2f} "
              f"{r['auc_pr_mean']:>10.4f} {r['test']['threshold_0.5']['auc_roc']:>12.4f}")
    
    # Compute ΔAUC
    print(f"\n{'='*70}")
    print("INCREMENTAL VALUE (ΔAUC)")
    print(f"{'='*70}")
    
    best_model = "XGBoost" if HAS_XGB else "GradientBoosting"
    tier1 = [r for r in all_results if "Tier1" in r["model"] and best_model in r["model"]]
    tier2 = [r for r in all_results if "Tier2" in r["model"] and best_model in r["model"]]
    
    if tier1 and tier2:
        delta_cv = tier2[0]["auc_roc_mean"] - tier1[0]["auc_roc_mean"]
        delta_test = tier2[0]["test"]["threshold_0.5"]["auc_roc"] - tier1[0]["test"]["threshold_0.5"]["auc_roc"]
        print(f"Adding Amyloid CL to Plasma ({best_model}):")
        print(f"  ΔAUC (CV):   {delta_cv:+.4f}")
        print(f"  ΔAUC (Test): {delta_test:+.4f}")
    
    # Save all results
    with open(f"{out_dir}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
