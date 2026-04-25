"""
3D ResNet18 with optional tabular feature fusion for tau positivity prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


class SAM3d(nn.Module):
    """Spatial Attention Module for 3D."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        channel_max = torch.max(x, 1, keepdim=True)[0]
        channel_avg = torch.mean(x, 1, keepdim=True)
        m = torch.cat((channel_avg, channel_max), dim=1)
        m = torch.sigmoid(self.conv(m))
        return m * x


class BasicBlock1(nn.Module):
    """Residual block without downsampling."""
    def __init__(self, channel_in, dropout=0.3):
        super().__init__()
        self.residual1 = nn.Sequential(
            nn.Conv3d(channel_in, channel_in, 3, padding=1),
            nn.BatchNorm3d(channel_in),
            nn.ReLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channel_in, channel_in, 3, padding=1),
            nn.BatchNorm3d(channel_in),
            SAM3d(),
        )
        self.residual2 = nn.Sequential(
            nn.Conv3d(channel_in, channel_in, 3, padding=1),
            nn.BatchNorm3d(channel_in),
            nn.ReLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channel_in, channel_in, 3, padding=1),
            nn.BatchNorm3d(channel_in),
            SAM3d(),
        )

    def forward(self, x):
        x = F.relu(self.residual1(x) + x)
        x = F.relu(self.residual2(x) + x)
        return x


class BasicBlock2(nn.Module):
    """Residual block with 2x downsampling."""
    def __init__(self, channel_in, dropout=0.3):
        super().__init__()
        channel_out = 2 * channel_in
        self.residual1 = nn.Sequential(
            nn.Conv3d(channel_in, channel_out, 3, stride=2, padding=1),
            nn.BatchNorm3d(channel_out),
            nn.ReLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channel_out, channel_out, 3, padding=1),
            nn.BatchNorm3d(channel_out),
            SAM3d(),
        )
        self.skip1 = nn.Conv3d(channel_in, channel_out, 1, stride=2)
        self.residual2 = nn.Sequential(
            nn.Conv3d(channel_out, channel_out, 3, padding=1),
            nn.BatchNorm3d(channel_out),
            nn.ReLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channel_out, channel_out, 3, padding=1),
            nn.BatchNorm3d(channel_out),
            SAM3d(),
        )

    def forward(self, x):
        x = F.relu(self.residual1(x) + self.skip1(x))
        x = F.relu(self.residual2(x) + x)
        return x


class ResNet18Tabular(BaseModel):
    """
    3D ResNet18 with optional tabular feature fusion.
    
    For tau positivity prediction:
      - MRI volume → CNN features (64-dim with init_filters=8)
      - Tabular features → MLP (16-dim)
      - Concat → classifier (2 classes)
    """

    def _build(self):
        self.num_classes = int(self.cfg.get("num_classes", 2))
        self.initial_filters = int(self.cfg.get("init_filters", 8))
        self.in_channels = int(self.cfg.get("in_channels", 1))
        self.dropout_rate = float(self.cfg.get("dropout", 0.3))
        self.tabular_dim = int(self.cfg.get("tabular_dim", 0))
        self.tabular_hidden = int(self.cfg.get("tabular_hidden", 16))

        # CNN backbone
        self.initial_block = nn.Sequential(
            nn.Conv3d(self.in_channels, self.initial_filters, kernel_size=7, padding=3),
            nn.BatchNorm3d(self.initial_filters),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool3d(2)
        
        self.bb_1 = BasicBlock1(self.initial_filters, dropout=self.dropout_rate)
        self.bb_2 = BasicBlock2(self.initial_filters, dropout=self.dropout_rate)
        self.bb_3 = BasicBlock2(self.initial_filters * 2, dropout=self.dropout_rate)
        self.bb_4 = BasicBlock2(self.initial_filters * 4, dropout=self.dropout_rate)

        self.cnn_out_ch = self.initial_filters * 8  # 64 with init_filters=8
        self.norm = nn.BatchNorm3d(self.cnn_out_ch)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Tabular encoder
        if self.tabular_dim > 0:
            self.tabular_encoder = nn.Sequential(
                nn.Linear(self.tabular_dim, 32),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(32, self.tabular_hidden),
                nn.ReLU(),
            )
            fusion_dim = self.cnn_out_ch + self.tabular_hidden
        else:
            self.tabular_encoder = None
            fusion_dim = self.cnn_out_ch

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, self.num_classes),
        )

    def forward_cnn(self, x):
        """Extract CNN features from MRI volume."""
        x = self.pool1(self.initial_block(x))
        x = self.bb_1(x)
        x = self.bb_2(x)
        x = self.bb_3(x)
        x = self.bb_4(x)
        x = self.avgpool(self.norm(x))
        x = x.flatten(1)  # (B, cnn_out_ch)
        return x

    def forward(self, x_mri, x_tab=None):
        """
        Args:
            x_mri: (B, 1, D, H, W) MRI volume
            x_tab: (B, tabular_dim) tabular features (optional)
        """
        cnn_feat = self.forward_cnn(x_mri)
        
        if self.tabular_encoder is not None and x_tab is not None:
            tab_feat = self.tabular_encoder(x_tab)
            feat = torch.cat([cnn_feat, tab_feat], dim=1)
        else:
            feat = cnn_feat
        
        return self.classifier(feat)


class ResNet18MRIOnly(ResNet18Tabular):
    """ResNet18 for MRI-only (no tabular fusion)."""
    
    def _build(self):
        self.cfg["tabular_dim"] = 0
        super()._build()
    
    def forward(self, x_mri, x_tab=None):
        # Ignore tabular input
        return super().forward(x_mri, None)
