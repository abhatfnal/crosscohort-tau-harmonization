"""
Dataset for tau positivity prediction.
Loads MRI volumes + tabular features (plasma, amyloid, demographics).
Compatible with TorchIO transforms.
"""

import os
import copy
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import torchio as tio


class TauPositivityDataset(Dataset):
    """
    Dataset for tau positivity prediction.
    
    Args:
        csv_path: Path to cohort CSV (must have: RID, tau_pos, mri_path, and tabular columns)
        tabular_cols: List of column names for tabular features
        fold: If provided, filter to this fold (requires 'fold' column)
        is_val: If True and fold is provided, use this fold as validation; else use as training
        transform: TorchIO transform (expects 4D tensor: C, D, H, W)
        normalize_tabular: Whether to z-score normalize tabular features
        tabular_stats: Dict with 'mean' and 'std' for normalization (if None, compute from data)
    """
    
    TABULAR_COLS_DEFAULT = [
        "PLASMA_pT217_F",
        "PLASMA_AB42_AB40_F", 
        "PLASMA_NfL_F",
        "PLASMA_GFAP_F",
        "AMY6MM_CENTILOIDS",
    ]
    
    def __init__(
        self,
        csv_path: str,
        tabular_cols: list = None,
        fold: int = None,
        is_val: bool = False,
        transform=None,
        normalize_tabular: bool = True,
        tabular_stats: dict = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.tabular_cols = tabular_cols or self.TABULAR_COLS_DEFAULT
        self.transform = transform
        self.normalize_tabular = normalize_tabular
        
        # Filter by fold if specified
        if fold is not None and "fold" in self.df.columns:
            if is_val:
                self.df = self.df[self.df["fold"] == fold].reset_index(drop=True)
            else:
                self.df = self.df[self.df["fold"] != fold].reset_index(drop=True)
        
        # Verify required columns
        assert "tau_pos" in self.df.columns, "Missing tau_pos column"
        assert "mri_path" in self.df.columns, "Missing mri_path column"
        
        # Check tabular columns exist
        missing = [c for c in self.tabular_cols if c not in self.df.columns]
        if missing:
            print(f"Warning: Missing tabular columns: {missing}")
            self.tabular_cols = [c for c in self.tabular_cols if c in self.df.columns]
        
        # Compute or use provided tabular stats
        if normalize_tabular and self.tabular_cols:
            if tabular_stats is not None:
                self.tab_mean = np.array(tabular_stats["mean"])
                self.tab_std = np.array(tabular_stats["std"])
            else:
                tab_data = self.df[self.tabular_cols].values.astype(np.float32)
                self.tab_mean = np.nanmean(tab_data, axis=0)
                self.tab_std = np.nanstd(tab_data, axis=0) + 1e-8
        else:
            self.tab_mean = None
            self.tab_std = None
    
    def get_tabular_stats(self):
        """Return tabular normalization stats (for passing to val/test sets)."""
        if self.tab_mean is not None:
            return {"mean": self.tab_mean.tolist(), "std": self.tab_std.tolist()}
        return None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load MRI
        mri_path = row["mri_path"]
        try:
            img = nib.load(mri_path)
            mri = img.get_fdata().astype(np.float32)
        except Exception as e:
            print(f"Error loading {mri_path}: {e}")
            # Return zeros as fallback (will be obvious in training)
            mri = np.zeros((91, 109, 91), dtype=np.float32)
        
        # Normalize MRI (z-score per volume)
        mri_mean = mri.mean()
        mri_std = mri.std() + 1e-8
        mri = (mri - mri_mean) / mri_std
        
        # Convert to torch tensor with channel dim: (D, H, W) -> (1, D, H, W)
        mri = torch.from_numpy(mri).unsqueeze(0).float()
        
        # Apply TorchIO transform (expects 4D tensor: C, D, H, W)
        if self.transform is not None:
            mri = self.transform(mri)
            mri = mri.float()
        
        # Load tabular features
        if self.tabular_cols:
            tab = row[self.tabular_cols].values.astype(np.float32)
            # Handle NaN
            tab = np.nan_to_num(tab, nan=0.0)
            # Normalize
            if self.tab_mean is not None:
                tab = (tab - self.tab_mean) / self.tab_std
            tab = torch.from_numpy(tab).float()
        else:
            tab = torch.zeros(1).float()
        
        # Label
        label = int(row["tau_pos"])
        
        return {
            "mri": mri,
            "tabular": tab,
            "label": label,
            "rid": int(row["RID"]),
        }


class TauTabularOnlyDataset(Dataset):
    """
    Tabular-only dataset for Tier 1-2 baselines (sklearn-compatible).
    """
    
    def __init__(
        self,
        csv_path: str,
        tabular_cols: list,
        fold: int = None,
        is_val: bool = False,
        normalize: bool = True,
        stats: dict = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.tabular_cols = tabular_cols
        
        # Filter by fold
        if fold is not None and "fold" in self.df.columns:
            if is_val:
                self.df = self.df[self.df["fold"] == fold].reset_index(drop=True)
            else:
                self.df = self.df[self.df["fold"] != fold].reset_index(drop=True)
        
        # Get features and labels
        self.X = self.df[tabular_cols].values.astype(np.float32)
        self.y = self.df["tau_pos"].values.astype(np.int64)
        self.rids = self.df["RID"].values
        
        # Handle NaN
        self.X = np.nan_to_num(self.X, nan=0.0)
        
        # Normalize
        if normalize:
            if stats is not None:
                self.mean = np.array(stats["mean"])
                self.std = np.array(stats["std"])
            else:
                self.mean = np.mean(self.X, axis=0)
                self.std = np.std(self.X, axis=0) + 1e-8
            self.X = (self.X - self.mean) / self.std
        else:
            self.mean = None
            self.std = None
    
    def get_stats(self):
        if self.mean is not None:
            return {"mean": self.mean.tolist(), "std": self.std.tolist()}
        return None
    
    def get_numpy(self):
        """Return raw numpy arrays for sklearn."""
        return self.X, self.y, self.rids
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {
            "features": torch.from_numpy(self.X[idx]).float(),
            "label": self.y[idx],
            "rid": self.rids[idx],
        }
