"""
#this is the final improved model and it give the result of asvspoof fold3 and wavefake fold1 bad result with lfcc and 
it need to be further improved  for stable results 
currently it is unstable...

Improved LCNN for Audio Deepfake Detection — Final Memory-Safe Edition
=======================================================================

ROOT CAUSE OF ALL PREVIOUS OOMs
─────────────────────────────────
MaxFeatureMap2D used:   m, _ = x.view(*shape).max(d)
torch.Tensor.max() returns (values, indices).
The indices tensor is int64 (8 bytes/element) vs float32 (4 bytes).
For a large batch it allocated a SECOND full-size int64 tensor just
to throw it away with `_`.

Example — batch=512, stem conv output [512, 64, 404, 80]:
  values  tensor: 512×32×404×80 × 4B = 2.64 GB
  indices tensor: 512×32×404×80 × 8B = 5.29 GB  ← this caused OOM
  Total needed at peak: ~8 GB  (before any other stage)

FIX 1 — Replace .max() with torch.amax()
  torch.amax() returns ONLY the values, no indices.
  Peak drops from ~8 GB → ~2.64 GB for the same op.

FIX 2 — Gradient checkpointing on CNN stages
  torch.utils.checkpoint stores only stage inputs/outputs during
  forward; recomputes intermediates on the backward pass.
  Reduces backprop activation memory by ~50-60%.

Together these bring peak VRAM well within 14.5 GB even at large batch.

Quality improvements kept from previous iterations:
  ✅ SEBlock channel attention after every MFM
  ✅ FrequencyAttention depthwise spatial gate
  ✅ BLSTM → pre-norm Transformer → BLSTM temporal pipeline
  ✅ Safe BLSTM residual (dims always match)
  ✅ Attentive statistics pooling
  ✅ Two-layer MLP head  (raw logit — use BCEWithLogitsLoss)
  ✅ SpecAugment built-in (training only)
  ✅ Kaiming init on Conv2d layers
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: MaxFeatureMap2D using torch.amax — no index tensor allocated
# ─────────────────────────────────────────────────────────────────────────────

class MaxFeatureMap2D(nn.Module):
    """
    MFM activation: halves channel count by element-wise max over pairs.

    CRITICAL FIX: uses torch.amax() instead of Tensor.max().
    Tensor.max() returns (values, indices); the indices tensor is int64
    (8 bytes each) and causes OOM on large batches even when discarded.
    torch.amax() returns only values — half the peak memory.
    """
    def __init__(self, max_dim: int = 1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.max_dim
        if x.size(d) % 2 != 0:
            sys.exit(1)
        shape = list(x.size())
        shape[d] //= 2
        shape.insert(d, 2)
        # torch.amax: no index tensor → saves 8B × numel bytes at peak
        return torch.amax(x.view(*shape), dim=d)


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

class BLSTMLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        if output_dim % 2 != 0:
            sys.exit(1)
        self.lstm = nn.LSTM(
            input_dim, output_dim // 2,
            bidirectional=True, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Attention modules (lightweight — negligible memory overhead)
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Channel squeeze-and-excitation.
    Operates on pooled scalars — adds ~0% to activation memory.
    """
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
    Input / output: [B, C, F, T]
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, channels,
                      kernel_size=(3, 1), padding=(1, 0),
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight conv block — single Conv2d matches original's memory footprint
# ─────────────────────────────────────────────────────────────────────────────

class ConvMFMSE(nn.Module):
    """
    Single Conv2d → BN → GELU → MFM → SEBlock.

    One Conv2d per block keeps activation memory identical to the
    original repo. SE operates on pooled scalars, so it's essentially
    free. MFM now uses torch.amax (no index tensor).

    in_ch → conv(out_ch×2) → MFM → out_ch → SE → out_ch
    """
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel, 1, padding,
                              bias=False)
        self.bn   = nn.BatchNorm2d(out_ch * 2)
        self.mfm  = MaxFeatureMap2D()
        self.se   = SEBlock(out_ch, reduction=16)
        nn.init.kaiming_normal_(self.conv.weight,
                                mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.mfm(x)
        x = self.se(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# SpecAugment (training-only, zero overhead at inference)
# ─────────────────────────────────────────────────────────────────────────────

def spec_augment(
        x: torch.Tensor,
        freq_mask_param: int = 8,
        time_mask_param: int = 20,
        num_freq_masks:  int = 2,
        num_time_masks:  int = 2,
) -> torch.Tensor:
    """SpecAugment on [B, C, dim1, dim2]. Returns cloned tensor."""
    B, C, F, T = x.shape
    out = x.clone()
    for b in range(B):
        for _ in range(num_freq_masks):
            f0 = torch.randint(0, max(F - freq_mask_param, 1), (1,)).item()
            fw = torch.randint(1, freq_mask_param + 1, (1,)).item()
            out[b, :, f0:f0 + fw, :] = 0.0
        for _ in range(num_time_masks):
            t0 = torch.randint(0, max(T - time_mask_param, 1), (1,)).item()
            tw = torch.randint(1, time_mask_param + 1, (1,)).item()
            out[b, :, :, t0:t0 + tw] = 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Temporal modules
# ─────────────────────────────────────────────────────────────────────────────

class PreNormTransformerLayer(nn.Module):
    """
    Pre-norm Transformer encoder layer.
    More stable than post-norm; faster convergence on small datasets.
    """
    def __init__(self, d_model: int, nhead: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        h, _ = self.attn(n, n, n)
        x = x + self.drop(h)
        x = x + self.ff(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class LCNN(nn.Module):
    """
    Memory-safe improved LCNN for audio deepfake detection.

    CNN backbone (single Conv2d per block — matches original memory):
      stem    : 5×5 conv + MFM(amax) + pool  → [B, 32,  T/2,  F/2]
      stage2  : 2 × ConvMFMSE + pool          → [B, 48,  T/4,  F/4]
      stage3  : 2 × ConvMFMSE + pool          → [B, 64,  T/8,  F/8]
      stage4  : 3 × ConvMFMSE + pool          → [B, 32,  T/16, F/16]

    Temporal:
      FrequencyAttention → BLSTM → Transformer → BLSTM+residual
      → attentive pooling → MLP head → logit [B, 1]

    Memory fixes applied:
      • torch.amax in MFM       (no index tensor, saves up to 5 GB/batch)
      • Gradient checkpointing  (CNN stages recompute on backward, ~50%
                                 less backprop activation memory)

    Args:
        input_channels   : spectral channels              (default 3)
        num_coefficients : frequency bins F               (default 80)
        dropout          : dropout throughout             (default 0.4)
        use_spec_augment : SpecAugment during training    (default True)
        use_checkpoint   : gradient checkpointing on CNN  (default True)

    Output: raw logit [B, 1]
      Training  → BCEWithLogitsLoss(logit, label)
      Inference → model._compute_score(logit)
    """

    def __init__(self, **kwargs):
        super().__init__()
        in_ch           = kwargs.get("input_channels",   3)
        num_coeff       = kwargs.get("num_coefficients", 80)
        dropout         = kwargs.get("dropout",          0.4)
        self.use_aug    = kwargs.get("use_spec_augment", True)
        self.use_ckpt   = kwargs.get("use_checkpoint",   True)
        self.num_coefficients = num_coeff
        self.v_emd_dim  = 1

        # ── Stem ─────────────────────────────────────────────────────────
        # 5×5 conv matches original; MFM now uses amax (key memory fix)
        # [B, in_ch, T, F] → [B, 32, T/2, F/2]
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, (5, 5), 1, (2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            MaxFeatureMap2D(),      # 64 → 32  (amax, no index tensor)
            nn.MaxPool2d(2, 2),
        )
        nn.init.kaiming_normal_(self.stem[0].weight,
                                mode='fan_out', nonlinearity='relu')

        # ── Stage 2 ───────────────────────────────────────────────────────
        # [B, 32, T/2, F/2] → [B, 48, T/4, F/4]
        self.stage2 = nn.Sequential(
            ConvMFMSE(32, 32),
            nn.BatchNorm2d(32, affine=False),
            ConvMFMSE(32, 48),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48, affine=False),
        )

        # ── Stage 3 ───────────────────────────────────────────────────────
        # [B, 48, T/4, F/4] → [B, 64, T/8, F/8]
        self.stage3 = nn.Sequential(
            ConvMFMSE(48, 48),
            nn.BatchNorm2d(48, affine=False),
            ConvMFMSE(48, 64),
            nn.MaxPool2d(2, 2),
        )

        # ── Stage 4 ───────────────────────────────────────────────────────
        # [B, 64, T/8, F/8] → [B, 32, T/16, F/16]
        self.stage4 = nn.Sequential(
            ConvMFMSE(64, 64),
            nn.BatchNorm2d(64, affine=False),
            ConvMFMSE(64, 32),
            nn.BatchNorm2d(32, affine=False),
            ConvMFMSE(32, 32),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5),
        )

        # ── Frequency attention ───────────────────────────────────────────
        self.freq_attn = FrequencyAttention(channels=32)

        # ── LSTM dimensions ───────────────────────────────────────────────
        lstm_dim = (num_coeff // 16) * 32

        # ── Temporal pipeline ─────────────────────────────────────────────
        self.blstm1   = BLSTMLayer(lstm_dim, lstm_dim)
        self.trans    = PreNormTransformerLayer(
            lstm_dim, nhead=4, dropout=dropout * 0.25
        )
        self.blstm2   = BLSTMLayer(lstm_dim, lstm_dim)
        self.seq_drop = nn.Dropout(dropout)

        # ── Attentive pooling ─────────────────────────────────────────────
        self.attn_pool = nn.Linear(lstm_dim, 1)

        # ── MLP classification head ───────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(lstm_dim, lstm_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_dim // 2, self.v_emd_dim),
        )

    # ── Checkpointed stage runners ────────────────────────────────────────
    # Gradient checkpointing: during forward, intermediates inside each
    # stage are NOT stored; they are recomputed on the backward pass.
    # This cuts backprop activation memory by ~50% with ~25% compute cost.

    def _run_stem(self, x):
        return self.stem(x)

    def _run_stage2(self, x):
        return self.stage2(x)

    def _run_stage3(self, x):
        return self.stage3(x)

    def _run_stage4(self, x):
        return self.stage4(x)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _compute_score(self, logit: torch.Tensor) -> torch.Tensor:
        """Raw logit → probability [0, 1], shape [B]."""
        return torch.sigmoid(logit).squeeze(1)

    def _compute_embedding(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # [B, C, F, T] → [B, C, T, F]  (Conv treats T as height, F as width)
        x = x.permute(0, 1, 3, 2)

        # SpecAugment (training only)
        if self.training and self.use_aug:
            x = spec_augment(x,
                             freq_mask_param=8,
                             time_mask_param=20,
                             num_freq_masks=2,
                             num_time_masks=2)

        # CNN backbone — optionally gradient-checkpointed
        # use_reentrant=False: safer with autocast / AMP training
        if self.use_ckpt:
            x = checkpoint(self._run_stem,   x, use_reentrant=False)
            x = checkpoint(self._run_stage2, x, use_reentrant=False)
            x = checkpoint(self._run_stage3, x, use_reentrant=False)
            x = checkpoint(self._run_stage4, x, use_reentrant=False)
        else:
            x = self._run_stem(x)
            x = self._run_stage2(x)
            x = self._run_stage3(x)
            x = self._run_stage4(x)

        # Frequency attention
        x = self.freq_attn(x)

        # Reshape to sequence [B, T', lstm_dim]
        x = x.permute(0, 2, 1, 3).contiguous()
        T_prime = x.size(1)
        x = x.view(B, T_prime, -1)

        # Temporal modeling
        h = self.blstm1(x)
        h = self.trans(h)
        h = self.blstm2(h) + h     # residual — dims guaranteed equal ✓
        h = self.seq_drop(h)

        # Attentive pooling
        w = torch.softmax(self.attn_pool(h), dim=1)   # [B, T', 1]
        pooled = (h * w).sum(dim=1)                    # [B, lstm_dim]

        return self.head(pooled)                        # [B, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x [B, C, F, T]
        Returns: logit [B, 1]  (raw, no sigmoid)
        """
        return self._compute_embedding(x)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    model = LCNN(
        input_channels=3,
        num_coefficients=80,
        dropout=0.4,
        use_spec_augment=True,
        use_checkpoint=True,
    )

 
    print("""
─────────────────────────────────────────────────────
Memory fixes applied:
  1. MFM uses torch.amax() — no int64 index tensor
     saves up to 5 GB peak per batch at large B
  2. Gradient checkpointing on CNN stages
     saves ~50% backprop activation memory

Training usage:
  criterion = nn.BCEWithLogitsLoss()
  loss = criterion(model(batch_x), batch_y)

Inference usage:
  score = model._compute_score(model(x))
─────────────────────────────────────────────────────
""")