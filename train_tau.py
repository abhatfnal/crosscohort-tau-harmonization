#!/usr/bin/env python3
"""
Training script for tau positivity prediction.
Supports 5-fold CV on trainval, then final eval on held-out test.

Fixes:
- Deep copy for best model state
- Test set evaluation
- Dynamic class weights per fold
"""

import os
import sys
import copy
import argparse
import yaml
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, confusion_matrix, roc_curve, precision_recall_curve
)

from data.tau_dataset import TauPositivityDataset
from models import ResNet18Tabular, ResNet18MRIOnly

# Optional: import augmentation if you want to use it
try:
    from data.augmentation import build_augmentation
    HAS_AUGMENTATION = True
except ImportError:
    HAS_AUGMENTATION = False
    print("Warning: data/augmentation.py not found, no augmentation will be applied")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model(cfg):
    model_name = cfg["model"]["name"]
    model_cfg = cfg["model"]
    
    if model_name == "ResNet18Tabular":
        return ResNet18Tabular(model_cfg)
    elif model_name == "ResNet18MRIOnly":
        return ResNet18MRIOnly(model_cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute all evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {}
    
    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        metrics["auc_roc"] = 0.5
        metrics["auc_pr"] = float(np.mean(y_true))
    else:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auc_pr"] = float(average_precision_score(y_true, y_prob))
    
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["ppv"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    metrics["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    
    metrics["tp"] = int(tp)
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    
    return metrics


def find_optimal_threshold(y_true, y_prob, metric="f1"):
    """Find threshold that maximizes the given metric."""
    best_thresh = 0.5
    best_score = 0
    
    for thresh in np.arange(0.1, 0.9, 0.02):
        y_pred = (y_prob >= thresh).astype(int)
        
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "youden":
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sens + spec - 1
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh, best_score


def train_epoch(model, loader, criterion, optimizer, device, use_tabular=True):
    model.train()
    total_loss = 0
    all_probs = []
    all_labels = []
    
    for batch in loader:
        mri = batch["mri"].to(device)
        labels = batch["label"].to(device)
        
        if use_tabular:
            tab = batch["tabular"].to(device)
            logits = model(mri, tab)
        else:
            logits = model(mri, None)
        
        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    metrics["loss"] = avg_loss
    
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5, use_tabular=True):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    all_rids = []
    
    for batch in loader:
        mri = batch["mri"].to(device)
        labels = batch["label"].to(device)
        
        if use_tabular:
            tab = batch["tabular"].to(device)
            logits = model(mri, tab)
        else:
            logits = model(mri, None)
        
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * len(labels)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        all_rids.extend(batch["rid"])
    
    avg_loss = total_loss / len(loader.dataset)
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = avg_loss
    
    return metrics, y_true, y_prob, all_rids


def train_fold(cfg, fold, output_dir, device):
    """Train one fold and return validation metrics + best model state."""
    print(f"\n{'='*60}")
    print(f"FOLD {fold}")
    print(f"{'='*60}")
    
    # Merge folds info into trainval
    trainval_df = pd.read_csv(cfg["data"]["trainval_csv"])
    folds_df = pd.read_csv(cfg["data"]["folds_csv"])
    trainval_df = trainval_df.merge(folds_df[["RID", "fold"]], on="RID", how="left")
    
    # Save merged for dataset loading
    merged_path = os.path.join(output_dir, "trainval_with_folds.csv")
    trainval_df.to_csv(merged_path, index=False)
    
    # Build augmentation for training (optional)
    train_transform = None
    if HAS_AUGMENTATION and cfg.get("augmentation"):
        train_transform = build_augmentation(cfg["augmentation"])
        print(f"Using augmentation: {list(cfg['augmentation'].keys())}")
    
    # Create datasets
    tabular_cols = cfg["data"]["tabular_cols"]
    use_tabular = cfg["model"].get("tabular_dim", 0) > 0
    
    train_ds = TauPositivityDataset(
        merged_path, tabular_cols=tabular_cols,
        fold=fold, is_val=False, 
        transform=train_transform,
        normalize_tabular=True
    )
    val_ds = TauPositivityDataset(
        merged_path, tabular_cols=tabular_cols,
        fold=fold, is_val=True, 
        transform=None,  # No augmentation for validation
        normalize_tabular=True,
        tabular_stats=train_ds.get_tabular_stats()
    )
    
    # Compute class weights from training data
    train_pos_rate = train_ds.df["tau_pos"].mean()
    pos_weight = (1 - train_pos_rate) / max(train_pos_rate, 1e-6)
    
    
    pos_weight = float(pos_weight)
    print(f"Train: {len(train_ds)} (tau+: {train_ds.df['tau_pos'].sum()}, rate: {train_pos_rate:.3f})")
    print(f"Val: {len(val_ds)} (tau+: {val_ds.df['tau_pos'].sum()})")
    print(f"Computed pos_weight: {pos_weight:.2f}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=cfg["data"].get("pin_memory", True),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=cfg["data"].get("pin_memory", True),
    )
    
    # Model
    model = get_model(cfg).to(device)
    
    # Loss with dynamic class weights
    if cfg["training"].get("use_class_weights", True):
        weight = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    
    # Scheduler
    if cfg["training"].get("scheduler") == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["epochs"]
        )
    else:
        scheduler = None
    
    # Training loop
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(cfg["training"]["epochs"]):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, use_tabular)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device, use_tabular=use_tabular)
        
        if scheduler:
            scheduler.step()
        
        # Check for improvement
        if val_metrics["auc_roc"] > best_val_auc + cfg["training"].get("min_delta", 0.001):
            best_val_auc = val_metrics["auc_roc"]
            patience_counter = 0
            # FIX: Deep copy of state dict
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {train_metrics['loss']:.4f} AUC: {train_metrics['auc_roc']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} AUC: {val_metrics['auc_roc']:.4f} F1: {val_metrics['f1']:.4f}")
        
        # Early stopping
        if patience_counter >= cfg["training"].get("patience", 10):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Final validation evaluation
    val_metrics, y_true, y_prob, rids = evaluate(model, val_loader, criterion, device, use_tabular=use_tabular)
    
    # Optimize threshold on validation
    opt_thresh, opt_score = find_optimal_threshold(y_true, y_prob, 
                                        cfg["evaluation"].get("threshold_metric", "f1"))
    val_metrics_opt = compute_metrics(y_true, y_prob, threshold=opt_thresh)
    val_metrics["optimal_threshold"] = opt_thresh
    val_metrics["f1_optimized"] = val_metrics_opt["f1"]
    val_metrics["sens_optimized"] = val_metrics_opt["sensitivity"]
    val_metrics["spec_optimized"] = val_metrics_opt["specificity"]
    
    print(f"Best val AUC: {best_val_auc:.4f}")
    print(f"Optimal threshold: {opt_thresh:.2f} -> F1: {val_metrics_opt['f1']:.4f}, "
          f"Sens: {val_metrics_opt['sensitivity']:.4f}, Spec: {val_metrics_opt['specificity']:.4f}")
    
    # Save fold model
    fold_model_path = os.path.join(output_dir, f"model_fold{fold}.pt")
    torch.save(best_model_state, fold_model_path)
    
    # Save predictions
    pred_df = pd.DataFrame({
        "RID": rids,
        "y_true": y_true,
        "y_prob": y_prob,
        "fold": fold,
    })
    pred_df.to_csv(os.path.join(output_dir, f"val_preds_fold{fold}.csv"), index=False)
    
    return val_metrics, train_ds.get_tabular_stats(), best_model_state, opt_thresh


def evaluate_test(cfg, model_state, tabular_stats, threshold, output_dir, device):
    """Evaluate on held-out test set."""
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION")
    print(f"{'='*60}")
    
    tabular_cols = cfg["data"]["tabular_cols"]
    use_tabular = cfg["model"].get("tabular_dim", 0) > 0
    
    test_ds = TauPositivityDataset(
        cfg["data"]["test_csv"],
        tabular_cols=tabular_cols,
        transform=None,
        normalize_tabular=True,
        tabular_stats=tabular_stats,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=cfg["data"].get("pin_memory", True),
    )
    
    print(f"Test: {len(test_ds)} (tau+: {test_ds.df['tau_pos'].sum()})")
    
    # Load model
    model = get_model(cfg).to(device)
    model.load_state_dict(model_state)
    
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate at default threshold (0.5)
    test_metrics_05, y_true, y_prob, rids = evaluate(
        model, test_loader, criterion, device, threshold=0.5, use_tabular=use_tabular
    )
    
    # Evaluate at optimized threshold from CV
    test_metrics_opt = compute_metrics(y_true, y_prob, threshold=threshold)
    
    print(f"\nTest Results (threshold=0.5):")
    print(f"  AUC-ROC: {test_metrics_05['auc_roc']:.4f}")
    print(f"  AUC-PR:  {test_metrics_05['auc_pr']:.4f}")
    print(f"  F1:      {test_metrics_05['f1']:.4f}")
    print(f"  Sens:    {test_metrics_05['sensitivity']:.4f}")
    print(f"  Spec:    {test_metrics_05['specificity']:.4f}")
    
    print(f"\nTest Results (threshold={threshold:.2f} from CV):")
    print(f"  F1:      {test_metrics_opt['f1']:.4f}")
    print(f"  Sens:    {test_metrics_opt['sensitivity']:.4f}")
    print(f"  Spec:    {test_metrics_opt['specificity']:.4f}")
    
    # Save test predictions
    pred_df = pd.DataFrame({
        "RID": rids,
        "y_true": y_true,
        "y_prob": y_prob,
    })
    pred_df.to_csv(os.path.join(output_dir, "test_preds.csv"), index=False)
    
    # Combine metrics
    test_metrics = {
        "threshold_0.5": test_metrics_05,
        "threshold_cv": {
            "threshold": threshold,
            **test_metrics_opt
        }
    }
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fold", type=int, default=None, help="Single fold to run (0-4), or None for all")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip_test", action="store_true", help="Skip test evaluation")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    
    # Setup output dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/{cfg['experiment']['name']}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)
    
    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Output dir: {output_dir}")
    
    # Run folds
    folds = [args.fold] if args.fold is not None else range(5)
    all_metrics = []
    all_thresholds = []
    last_tabular_stats = None
    last_model_state = None
    
    for fold in folds:
        metrics, tab_stats, model_state, opt_thresh = train_fold(cfg, fold, output_dir, device)
        all_metrics.append(metrics)
        all_thresholds.append(opt_thresh)
        last_tabular_stats = tab_stats
        last_model_state = model_state
    
    # Aggregate CV results
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    summary = {}
    for key in all_metrics[0].keys():
        if isinstance(all_metrics[0][key], (int, float)):
            values = [m[key] for m in all_metrics]
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
            print(f"{key:20s}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")
    
    avg_threshold = float(np.mean(all_thresholds))
    print(f"{'avg_threshold':20s}: {avg_threshold:.3f}")
    summary["avg_threshold"] = avg_threshold
    
    # Save CV summary
    with open(os.path.join(output_dir, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Test evaluation (use last fold's model and tabular stats)
    if not args.skip_test and len(folds) == 5:
        test_metrics = evaluate_test(
            cfg, last_model_state, last_tabular_stats, 
            avg_threshold, output_dir, device
        )
        
        # Save test results
        with open(os.path.join(output_dir, "test_results.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
