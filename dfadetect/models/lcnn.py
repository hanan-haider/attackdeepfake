"""
MobileNetV3-inspired Audio Deepfake Detection Model
===================================================
Memory-Safe Edition with Gradient Checkpointing, Frequency Pooling, 
and Native In-Place Activations.

Expected input shape : (batch, channels, num_coefficients, time_frames)
                       e.g.  (12, 3, 80, 404)
Output shape         : (batch, 1)  — raw logit (apply sigmoid for score)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# ─────────────────────────────────────────────────────────────
# Squeeze-and-Excitation (SE) Block
# ─────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Channel-wise Squeeze-and-Excitation attention."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        squeezed = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, squeezed, bias=False),
            nn.ReLU(inplace=True), # Memory Fix: inplace=True
            nn.Linear(squeezed, channels, bias=False),
            nn.Hardsigmoid(inplace=True), # Memory Fix: Native In-Place
        )

    def forward(self, x):
        scale = self.se(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * scale


# ─────────────────────────────────────────────────────────────
# MobileNetV3 Inverted-Residual Bottleneck (IRB)
# ─────────────────────────────────────────────────────────────

class IRBlock(nn.Module):
    """
    Inverted-Residual Bottleneck with optional SE and hard-swish.
    expand → depthwise (+ optional SE) → project
    """
    def __init__(self, in_ch: int, exp_ch: int, out_ch: int,
                 kernel: int = 3, stride: int = 1,
                 use_se: bool = True, act=None):
        super().__init__()
        # Memory Fix: Default to Native In-Place Hardswish
        act = act or nn.Hardswish(inplace=True) 
        padding = (kernel - 1) // 2
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []
        # Expand (pointwise)
        if exp_ch != in_ch:
            layers += [
                nn.Conv2d(in_ch, exp_ch, 1, bias=False),
                nn.BatchNorm2d(exp_ch),
                act,
            ]
        # Depthwise
        layers += [
            nn.Conv2d(exp_ch, exp_ch, kernel, stride=stride,
                      padding=padding, groups=exp_ch, bias=False),
            nn.BatchNorm2d(exp_ch),
            act,
        ]
        # Squeeze-and-Excitation
        if use_se:
            layers.append(SEBlock(exp_ch))
        # Project (pointwise, no activation)
        layers += [
            nn.Conv2d(exp_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out


# ─────────────────────────────────────────────────────────────
# Main Model: MobileNetV3-AA (Attack-Agnostic)
# ─────────────────────────────────────────────────────────────

class LCNN(nn.Module):
    """
    Memory-Safe MobileNetV3 Audio Deepfake Detector.
    Acts as a drop-in replacement for the legacy LCNN.
    """
    def __init__(self, **kwargs):
        super().__init__()
        input_channels  = kwargs.get("input_channels", 3)
        self.num_coefficients = kwargs.get("num_coefficients", 80)
        self.use_ckpt = kwargs.get("use_checkpoint", True) # Default to True
        self.v_emd_dim = 1

        # ── Stem ──
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True), # Memory Fix: Native In-Place
        )

        # ── Backbone Configuration ──
        # (in_ch, exp_ch, out_ch, kernel, stride, se)
        cfg = [
            (16,  16,  16, 3, 1, False),
            (16,  64,  24, 3, 2, False),
            (24,  72,  24, 3, 1, False),
            (24,  72,  40, 5, 2, True),
            (40, 120,  40, 5, 1, True),
            (40, 120,  40, 5, 1, True),
            (40, 240,  80, 3, 2, True),
            (80, 200,  80, 3, 1, True),
            (80, 184,  80, 3, 1, True),
            (80, 184,  80, 3, 1, True),
            (80, 480, 112, 3, 1, True),
            (112, 672, 112, 3, 1, True),
            (112, 672, 160, 5, 2, True),
            (160, 960, 160, 5, 1, True),
            (160, 960, 160, 5, 1, True),
        ]

        # ── Grouping into Stages for Gradient Checkpointing ──
        def build_stage(start_idx, end_idx):
            blocks = []
            for i in range(start_idx, end_idx):
                ic, ec, oc, k, s, se = cfg[i]
                blocks.append(IRBlock(ic, ec, oc, kernel=k, stride=s, use_se=se))
            return nn.Sequential(*blocks)

        self.stage1 = build_stage(0, 4)
        self.stage2 = build_stage(4, 8)
        self.stage3 = build_stage(8, 12)
        self.stage4 = build_stage(12, 15)

        # ── Conv Head ──
        self.conv_head = nn.Sequential(
            nn.Conv2d(160, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.Hardswish(inplace=True), # Memory Fix: Native In-Place
            nn.Dropout2d(0.1),
        )

        # ── Temporal Bi-LSTM ──
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, 
                            num_layers=2, bidirectional=True, batch_first=True)

        # ── Output Head ──
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, self.v_emd_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _compute_embedding(self, x):
        """
        x : (B, C, freq, time)
        returns logit : (B, 1)
        """
        x = x.permute(0, 1, 3, 2) 

        # Stem
        x = self.stem(x)

        # Backbone (With Checkpointing to prevent CUDA OOM)
        if self.training and self.use_ckpt:
            x = checkpoint.checkpoint(self.stage1, x, use_reentrant=False)
            x = checkpoint.checkpoint(self.stage2, x, use_reentrant=False)
            x = checkpoint.checkpoint(self.stage3, x, use_reentrant=False)
            x = checkpoint.checkpoint(self.stage4, x, use_reentrant=False)
        else:
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)

        # Conv Head
        x = self.conv_head(x)  # Shape: (B, 256, T//16, F//16)

        # Frequency Pooling 
        x = x.mean(dim=3)      # Shape: (B, 256, T//16)
        
        # Prepare for LSTM (Batch First)
        x = x.permute(0, 2, 1) # Shape: (B, T//16, 256)

        # Temporal Bi-LSTM
        lstm_out, _ = self.lstm(x) # Shape: (B, T//16, 256)
        
        # Temporal Mean Pool
        fused = lstm_out.mean(dim=1) # Shape: (B, 256)

        # Classifier
        emb = self.classifier(fused) # Shape: (B, 1)
        return emb

    def _compute_score(self, feature_vec):
        return torch.sigmoid(feature_vec).squeeze(1)

    def forward(self, x):
        return self._compute_embedding(x)


if __name__ == "__main__":
    print("Definition of model")
    
    model = LCNN(input_channels=3, num_coefficients=80, use_checkpoint=True)
    
    batch_size = 12
    mock_input = torch.rand((batch_size, 3, 80, 404))
    
    output = model(mock_input)
    print(output.shape)

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters : {total:,}")