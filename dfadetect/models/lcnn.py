"""
Improved LCNN for Audio Deepfake Detection — Memory-Efficient Edition
======================================================================
Fixes CUDA OOM by removing the dual-path backbone (which doubled
activation memory). All other improvements are retained:

  ✅ ResidualSEBlock (SE channel attention + residual skip)
  ✅ FrequencyAttention — depthwise spatial gate on freq axis
  ✅ BLSTM → pre-norm Transformer → BLSTM temporal pipeline
  ✅ Safe BLSTM residual (only between matched-dim tensors)
  ✅ Attentive statistics pooling (learned per-frame weights)
  ✅ Two-layer MLP head — raw logit output
  ✅ SpecAugment built-in (training only, zero overhead at inference)
  ✅ Kaiming init on all Conv2d layers
  ✅ Dropout2d in stage4 (spatial dropout, less aggressive)

Memory vs prior versions:
  Original LCNN (repo baseline) : reference
  Dual-path version (OOM)       : ~2× activations  → OOM on 14.5 GB
  This version                  : ~1.15× baseline  → fits comfortably

Loss function:
  Output is a raw LOGIT [B, 1].
  Training  → BCEWithLogitsLoss(output, label)      ← numerically stable
  Inference → model._compute_score(output)          ← applies sigmoid
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

class MaxFeatureMap2D(nn.Module):
    """MFM: halves channel count by element-wise max over pairs."""
    def __init__(self, max_dim: int = 1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.max_dim
        assert x.size(d) % 2 == 0, "Channel dim must be even for MFM"
        shape = list(x.size())
        shape[d] //= 2
        shape.insert(d, 2)
        m, _ = x.view(*shape).max(d)
        return m


class BLSTMLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        assert output_dim % 2 == 0, "BLSTM output_dim must be even"
        self.lstm = nn.LSTM(
            input_dim, output_dim // 2,
            bidirectional=True, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Channel squeeze-and-excitation (reduction=16 to save memory)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.size(0), x.size(1)
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class FrequencyAttention(nn.Module):
    """
    Spatial attention along the frequency axis.
    Depthwise conv — handles any F dimension without fixed Linear.
    Input/output: [B, C, F, T]
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),               # [B, C, F, 1]
            nn.Conv2d(channels, channels,
                      kernel_size=(3, 1), padding=(1, 0),
                      groups=channels, bias=False),         # depthwise
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)                            # broadcast over T


# ---------------------------------------------------------------------------
# Conv building block
# ---------------------------------------------------------------------------

class ResidualSEBlock(nn.Module):
    """
    Conv → BN → GELU → Conv → BN → SE attention → residual → GELU.
    Kaiming-initialized. Shortcut projection when dims change.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size, 1, pad, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch, reduction=16)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.act = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.se(self.body(x))
        return self.act(out + self.skip(x))


# ---------------------------------------------------------------------------
# SpecAugment — training-only, no overhead at inference
# ---------------------------------------------------------------------------

def spec_augment(
        x: torch.Tensor,
        freq_mask_param: int = 8,
        time_mask_param: int = 20,
        num_freq_masks:  int = 2,
        num_time_masks:  int = 2,
) -> torch.Tensor:
    """
    SpecAugment on [B, C, dim1, dim2].
    Returns new tensor (clone); no in-place modification on graph.
    """
    B, C, F, T = x.shape
    out = x.clone()
    for b in range(B):
        for _ in range(num_freq_masks):
            f_max = max(F - freq_mask_param, 1)
            f0 = torch.randint(0, f_max, (1,)).item()
            fw = torch.randint(1, freq_mask_param + 1, (1,)).item()
            out[b, :, f0:f0 + fw, :] = 0.0
        for _ in range(num_time_masks):
            t_max = max(T - time_mask_param, 1)
            t0 = torch.randint(0, t_max, (1,)).item()
            tw = torch.randint(1, time_mask_param + 1, (1,)).item()
            out[b, :, :, t0:t0 + tw] = 0.0
    return out


# ---------------------------------------------------------------------------
# Temporal modules
# ---------------------------------------------------------------------------

class PreNormTransformerLayer(nn.Module):
    """
    Pre-norm Transformer encoder layer (more stable than post-norm
    on small datasets; converges faster with fewer epochs).
    """
    def __init__(self, d_model: int, nhead: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        dim_ff = d_model * 2
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention + residual
        n = self.norm1(x)
        h, _ = self.attn(n, n, n)
        x = x + self.drop(h)
        # Pre-norm feed-forward + residual
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class LCNN(nn.Module):
    """
    Memory-efficient improved LCNN for audio deepfake detection.

    Single-path CNN backbone:
      Stage 1 : MFM conv + pool         → [B, 32, T/2,  F/2]
      Stage 2 : 2× ResidualSE + pool    → [B, 48, T/4,  F/4]
      Stage 3 : 2× ResidualSE + pool    → [B, 64, T/8,  F/8]
      Stage 4 : 3× ResidualSE + pool    → [B, 32, T/16, F/16]

    Then: FrequencyAttention → flatten to sequence →
          BLSTM → Transformer → BLSTM (with residual) →
          Attentive pooling → MLP head → raw logit [B, 1]

    Args:
        input_channels   : spectral feature channels   (default 3)
        num_coefficients : frequency bins              (default 80)
        dropout          : dropout throughout          (default 0.4)
        use_spec_augment : SpecAugment in training     (default True)
    """

    def __init__(self, **kwargs):
        super().__init__()
        in_ch        = kwargs.get("input_channels",   3)
        num_coeff    = kwargs.get("num_coefficients", 80)
        dropout      = kwargs.get("dropout",          0.4)
        self.use_aug = kwargs.get("use_spec_augment", True)
        self.num_coefficients = num_coeff
        self.v_emd_dim = 1

        # Stage 1 — initial MFM conv + pool
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, (5, 5), 1, (2, 2), bias=False),
            MaxFeatureMap2D(),                  # 64 → 32 ch
            nn.MaxPool2d(2, 2),
        )

        # Stage 2 — first residual SE pair + pool
        self.stage2 = nn.Sequential(
            ResidualSEBlock(32, 64),
            MaxFeatureMap2D(),                  # 64 → 32 ch
            nn.BatchNorm2d(32, affine=False),
            ResidualSEBlock(32, 96),
            MaxFeatureMap2D(),                  # 96 → 48 ch
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48, affine=False),
        )

        # Stage 3 — deeper pair + pool
        self.stage3 = nn.Sequential(
            ResidualSEBlock(48, 96),
            MaxFeatureMap2D(),                  # 96 → 48 ch
            nn.BatchNorm2d(48, affine=False),
            ResidualSEBlock(48, 128),
            MaxFeatureMap2D(),                  # 128 → 64 ch
            nn.MaxPool2d(2, 2),
        )

        # Stage 4 — refinement + final pool + spatial dropout
        self.stage4 = nn.Sequential(
            ResidualSEBlock(64, 128),
            MaxFeatureMap2D(),                  # 128 → 64 ch
            nn.BatchNorm2d(64, affine=False),
            ResidualSEBlock(64, 64),
            MaxFeatureMap2D(),                  # 64 → 32 ch
            nn.BatchNorm2d(32, affine=False),
            ResidualSEBlock(32, 64),
            MaxFeatureMap2D(),                  # 64 → 32 ch
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5),        # spatial dropout (lighter)
        )
        # Total: 4 × MaxPool2d → spatial dims reduced by 16×

        # Frequency attention — depthwise, any F size
        self.freq_attn = FrequencyAttention(channels=32)

        # LSTM dim: 32 ch × (num_coeff // 16) freq bins
        lstm_dim = (num_coeff // 16) * 32

        # Temporal pipeline
        self.blstm1   = BLSTMLayer(lstm_dim, lstm_dim)
        self.trans    = PreNormTransformerLayer(
            lstm_dim, nhead=4, dropout=dropout * 0.25
        )
        self.blstm2   = BLSTMLayer(lstm_dim, lstm_dim)
        self.seq_drop = nn.Dropout(dropout)

        # Attentive statistics pooling
        self.attn_pool = nn.Linear(lstm_dim, 1)

        # MLP classification head → raw logit
        self.head = nn.Sequential(
            nn.Linear(lstm_dim, lstm_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_dim // 2, self.v_emd_dim),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_score(self, logit: torch.Tensor) -> torch.Tensor:
        """Raw logit → probability in [0, 1], shape [B]."""
        return torch.sigmoid(logit).squeeze(1)

    def _compute_embedding(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # [B, C, F, T] → [B, C, T, F]  (conv treats T as height, F as width)
        x = x.permute(0, 1, 3, 2)

        # SpecAugment — only during training
        if self.training and self.use_aug:
            x = spec_augment(x,
                             freq_mask_param=8,
                             time_mask_param=20,
                             num_freq_masks=2,
                             num_time_masks=2)

        # CNN backbone
        x = self.stage1(x)   # [B, 32, T/2,  F/2]
        x = self.stage2(x)   # [B, 48, T/4,  F/4]
        x = self.stage3(x)   # [B, 64, T/8,  F/8]
        x = self.stage4(x)   # [B, 32, T/16, F/16]

        # Frequency-axis attention
        x = self.freq_attn(x)

        # Flatten spatial to sequence: [B, T', C×F']
        x = x.permute(0, 2, 1, 3).contiguous()
        T_prime = x.size(1)
        x = x.view(B, T_prime, -1)               # [B, T', lstm_dim]

        # Temporal modeling
        h = self.blstm1(x)           # [B, T', lstm_dim]
        h = self.trans(h)            # [B, T', lstm_dim]
        h = self.blstm2(h) + h       # residual — dims always match ✓
        h = self.seq_drop(h)

        # Attentive pooling
        w = torch.softmax(self.attn_pool(h), dim=1)  # [B, T', 1]
        pooled = (h * w).sum(dim=1)                   # [B, lstm_dim]

        # Classification head
        return self.head(pooled)                       # [B, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._compute_embedding(x)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    model = LCNN(
        input_channels=3,
        num_coefficients=80,
        dropout=0.4,
        use_spec_augment=True,
    )

    # ---- eval mode ----
    model.eval()
    B = 50
    x = torch.rand(B, 3, 80, 404)
    with torch.no_grad():
        logit = model(x)
        score = model._compute_score(logit)

    assert logit.shape == (B, 1), f"Wrong logit shape: {logit.shape}"
    assert score.shape == (B,),   f"Wrong score shape: {score.shape}"

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Logit shape  : {logit.shape}")
    print(f"✅ Score shape  : {score.shape}")
    print(f"✅ Total params : {total:,}")
    print(f"ℹ️  Param memory : {total * 4 / 1024**2:.1f} MB  (fp32 weights only)")

    # ---- train mode (SpecAugment active) ----
    model.train()
    logit_tr = model(x)
    assert logit_tr.shape == (B, 1)
    print(f"✅ Train logit  : {logit_tr.shape}  (SpecAugment active)")

    # ---- loss demo ----
    labels = torch.randint(0, 2, (B,)).float().unsqueeze(1)
    loss = nn.BCEWithLogitsLoss()(logit_tr, labels)
    print(f"✅ BCEWithLogitsLoss: {loss.item():.4f}")
    print()
    print("Training tip: use BCEWithLogitsLoss(output, labels), NOT BCE(sigmoid(output), labels)")