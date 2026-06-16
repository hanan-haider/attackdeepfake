"""
lcnn_stabilized.py
==================
Stabilized Improved LCNN for Audio Deepfake Detection
Drop-in replacement for dfadetect/models/lcnn.py

ALL 7 EER-INSTABILITY FIXES APPLIED:
  FIX 1  AttentiveStatisticsPooling (mean+std) — primary EER fix
  FIX 2  pos_weight per fold via compute_pos_weight()
  FIX 3  LR warmup + CosineAnnealingWarmRestarts via build_scheduler()
  FIX 4  No Dropout2d in CNN; dropout only in temporal pipeline
  FIX 5  Transformer nhead=2 (head_dim=80), FFN 1.5x
  FIX 6  SpecAugment before checkpoint() boundary
  FIX 7  SmoothedBCEWithLogitsLoss (label_smoothing=0.05)
  KEPT   torch.amax in MFM (no int64 index tensor)
  KEPT   Gradient checkpointing on CNN stages
  KEPT   SEBlock + FrequencyAttention
  KEPT   BLSTM -> Transformer -> BLSTM + residual pipeline
  KEPT   Kaiming init on all Conv2d
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ======================================================================
#  LOSS — FIX 7: Label smoothing + FIX 2: pos_weight support
# ======================================================================

class SmoothedBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss with label smoothing and optional pos_weight.

    Converts hard {0,1} -> soft {eps/2, 1-eps/2} targets.
    Prevents overconfidence on minority-class samples in imbalanced folds.

    Args:
        smoothing (float): Label smoothing factor. 0.05 recommended.

    Usage:
        criterion  = SmoothedBCEWithLogitsLoss(smoothing=0.05)
        pos_weight = compute_pos_weight(train_dataset, device)
        loss = criterion(batch_out, batch_y, pos_weight=pos_weight)
    """
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        if not 0.0 <= smoothing < 0.5:
            raise ValueError(f"smoothing must be in [0, 0.5), got {smoothing}")
        self.smoothing = smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pos_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        targets_smooth = (
            targets.float() * (1.0 - self.smoothing)
            + 0.5 * self.smoothing
        )
        return F.binary_cross_entropy_with_logits(
            logits,
            targets_smooth,
            pos_weight=pos_weight,
        )


# ======================================================================
#  TRAINER HELPERS
# ======================================================================

def compute_pos_weight(dataset, device: str) -> torch.Tensor:
    """
    Compute pos_weight = n_fake / n_real for BCEWithLogitsLoss.
    FIX 2: Corrects per-fold class imbalance in AAD.

    Call ONCE per fold before the epoch loop.

    Args:
        dataset : AttackAgnosticDataset — __getitem__ returns (x, sr, label)
                  label=1 bonafide/real, label=0 fake/spoof.
        device  : 'cuda' or 'cpu'

    Returns:
        pos_weight tensor shape [1] on device.
    """
    labels = [int(dataset[i][2]) for i in range(len(dataset))]
    n_real = sum(1 for y in labels if y == 1)
    n_fake = sum(1 for y in labels if y == 0)
    ratio  = n_fake / max(n_real, 1)
    return torch.tensor([ratio], dtype=torch.float32, device=device)


def build_scheduler(
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        cosine_T0: int    = 10,
        cosine_T_mult: int = 2,
        eta_min: float    = 1e-6,
) -> torch.optim.lr_scheduler.SequentialLR:
    """
    FIX 3: Linear warmup + CosineAnnealingWarmRestarts.

    Epochs [0, warmup_epochs):  lr scales linearly 0.1x -> 1.0x.
    Epochs [warmup_epochs, ...): CosineAnnealingWarmRestarts(T0, T_mult).

    Prevents fold-dependent saddle points caused by large early
    gradient updates on the BLSTM+Transformer stack.

    Call scheduler.step() once per EPOCH (not per batch).

    Args:
        optimizer     : Adam / AdamW optimizer instance.
        warmup_epochs : Linear warmup length in epochs (default 5).
        cosine_T0     : Cosine restart period in epochs (default 10).
        cosine_T_mult : Period multiplier after restart (default 2).
        eta_min       : Minimum LR at cosine trough (default 1e-6).

    Returns:
        SequentialLR scheduler.
    """
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cosine_T0,
        T_mult=cosine_T_mult,
        eta_min=eta_min,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


# ======================================================================
#  MFM — torch.amax (no int64 index tensor)
#  old .max(): values(4B) + indices(8B) ~ 8 GB peak at batch=512
#  new .amax(): values(4B) only          ~ 2.6 GB peak
# ======================================================================

class MaxFeatureMap2D(nn.Module):
    """
    MFM activation: halves channel count by element-wise max over pairs.
    Uses torch.amax() — no int64 index tensor allocated.
    """
    def __init__(self, max_dim: int = 1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.max_dim
        if x.size(d) % 2 != 0:
            raise ValueError(
                f"MaxFeatureMap2D: dim {d} must be even, got {x.size(d)}"
            )
        shape    = list(x.size())
        shape[d] //= 2
        shape.insert(d, 2)
        return torch.amax(x.view(*shape), dim=d)


# ======================================================================
#  BLSTMLayer
# ======================================================================

class BLSTMLayer(nn.Module):
    """Bidirectional LSTM. batch_first=True → [B, T, D]."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        if output_dim % 2 != 0:
            raise ValueError(
                f"BLSTMLayer output_dim must be even, got {output_dim}"
            )
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=output_dim // 2,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


# ======================================================================
#  SEBlock — Channel Squeeze-and-Excitation
# ======================================================================

class SEBlock(nn.Module):
    """
    Channel SE block [Hu et al., CVPR 2018].
    Operates on pooled scalars — negligible memory overhead.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.size(0), x.size(1)
        w    = self.pool(x).view(b, c)
        w    = self.fc(w).view(b, c, 1, 1)
        return x * w


# ======================================================================
#  FrequencyAttention — depthwise spatial gate along frequency axis
# ======================================================================

class FrequencyAttention(nn.Module):
    """
    Depthwise conv gate along frequency axis.
    Handles any F dimension — no fixed Linear.
    Input/output: [B, C, T, F]
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(
                channels, channels,
                kernel_size=(3, 1), padding=(1, 0),
                groups=channels, bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


# ======================================================================
#  ConvMFMSE — Conv → BN → GELU → MFM(amax) → SE
#  FIX 4: NO Dropout2d — removed entirely from CNN
# ======================================================================

class ConvMFMSE(nn.Module):
    """
    Conv2d → BN → GELU → MaxFeatureMap2D(amax) → SEBlock.
    Single conv per block. NO Dropout2d (FIX 4).
    in_ch → conv(out_ch×2) → MFM → out_ch → SE → out_ch
    """
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch * 2, kernel, stride=1, padding=padding, bias=False
        )
        self.bn   = nn.BatchNorm2d(out_ch * 2)
        self.mfm  = MaxFeatureMap2D()
        self.se   = SEBlock(out_ch, reduction=16)
        nn.init.kaiming_normal_(
            self.conv.weight, mode='fan_out', nonlinearity='relu'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.gelu(x)
        x = self.mfm(x)
        x = self.se(x)
        return x


# ======================================================================
#  SpecAugment
#  FIX 6: Must be called BEFORE checkpoint() regions.
#  checkpoint() recomputes forward on backward — if augment were inside,
#  torch.randint generates different masks on recomputation → gradients
#  computed on a different input than the forward pass used.
# ======================================================================

def spec_augment(
        x: torch.Tensor,
        freq_mask_param: int = 10,
        time_mask_param: int = 20,
        num_freq_masks:  int = 2,
        num_time_masks:  int = 2,
) -> torch.Tensor:
    """
    SpecAugment on [B, C, T, F]. Called BEFORE checkpoint() — see FIX 6.
    Returns cloned tensor with frequency and time bands zeroed.
    """
    B, C, T, F_dim = x.shape
    out = x.clone()
    for b in range(B):
        for _ in range(num_freq_masks):
            f0 = torch.randint(0, max(F_dim - freq_mask_param, 1), (1,)).item()
            fw = torch.randint(1, freq_mask_param + 1, (1,)).item()
            out[b, :, :, f0: f0 + fw] = 0.0
        for _ in range(num_time_masks):
            t0 = torch.randint(0, max(T - time_mask_param, 1), (1,)).item()
            tw = torch.randint(1, time_mask_param + 1, (1,)).item()
            out[b, :, t0: t0 + tw, :] = 0.0
    return out


# ======================================================================
#  PreNormTransformerLayer
#  FIX 5: nhead=2 → head_dim=80 (was nhead=4 → head_dim=40)
#         FFN 1.5x expansion (was 2x)
# ======================================================================

class PreNormTransformerLayer(nn.Module):
    """
    Pre-norm Transformer encoder layer.
    FIX 5: nhead=2 gives head_dim=80 for lstm_dim=160.
           nhead=4 collapses attention on ~25-frame sequences.
    FFN expansion 1.5x (was 2x) — reduces parameter count / fold variance.
    """
    def __init__(self, d_model: int, nhead: int = 2, dropout: float = 0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by nhead={nhead}"
            )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        ff_dim = max(int(d_model * 1.5), d_model + 32)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n        = self.norm1(x)
        attn_out, _ = self.attn(n, n, n)
        x        = x + self.drop(attn_out)
        x        = x + self.ff(self.norm2(x))
        return x


# ======================================================================
#  AttentiveStatisticsPooling — FIX 1 (PRIMARY EER STABILITY FIX)
#
#  Previous: pooled = (h * w).sum(dim=1)  → [B, lstm_dim]  mean only
#  This:     concat(mean, std)             → [B, 2*lstm_dim]
#
#  WaveFake fold 1 + ASVspoof fold 3 attacks (MelGAN, DiffWave, HiFiGAN)
#  manifest as TEMPORAL VARIANCE artifacts. Mean-only pooling cannot
#  capture this signal — the model is literally blind to it.
#  std gives direct access to variance-based discriminative features.
#
#  Reference: Okabe et al., Interspeech 2018.
# ======================================================================

class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling [Okabe et al., Interspeech 2018].

    Learns per-feature softmax weights over the time axis, then:
      mean   = Σ(x  · w)            [B, D]
      std    = sqrt(Σ(x² · w) - μ²) [B, D]
      output = cat([mean, std])      [B, 2D]

    Args:
        input_dim  : Feature dimension D (= lstm_dim).
        bottleneck : Attention MLP hidden size (default 64).
    """
    def __init__(self, input_dim: int, bottleneck: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, bottleneck),
            nn.Tanh(),
            nn.Linear(bottleneck, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        scores  = self.attn(x)                          # [B, T, D]
        weights = torch.softmax(scores, dim=1)          # [B, T, D]
        mean    = (x * weights).sum(dim=1)              # [B, D]
        sq_mean = (x ** 2 * weights).sum(dim=1)         # [B, D]
        std     = torch.sqrt(
            (sq_mean - mean ** 2).clamp(min=1e-8)
        )                                               # [B, D]
        return torch.cat([mean, std], dim=-1)           # [B, 2D]


# ======================================================================
#  MAIN MODEL
# ======================================================================

class LCNN(nn.Module):
    """
    Stabilized Improved LCNN for Audio Deepfake Detection.
    Drop-in replacement for dfadetect/models/lcnn.py.
    All 7 EER-instability fixes applied.

    CNN backbone (no Dropout2d — FIX 4):
      stem    [B, C, T, F] → [B, 32, T/2,  F/2]
      stage2               → [B, 48, T/4,  F/4]
      stage3               → [B, 64, T/8,  F/8]
      stage4               → [B, 32, T/16, F/16]

    Temporal pipeline:
      FrequencyAttention
      reshape     → [B, T/16, lstm_dim]
      BLSTM       → [B, T/16, lstm_dim]
      Dropout(0.2)
      Transformer (nhead=2) → [B, T/16, lstm_dim]       FIX 5
      BLSTM + residual      → [B, T/16, lstm_dim]
      Dropout(0.4)
      ASP (mean+std)        → [B, 2·lstm_dim]           FIX 1
      LayerNorm → Linear → GELU → Dropout → Linear → [B, 1]

    Args:
        input_channels   (int):   Spectral channels. Default 1 (LFCC).
        num_coefficients (int):   Frequency bins F. Default 80.
        dropout          (float): Dropout in temporal pipeline. Default 0.4.
        use_spec_augment (bool):  SpecAugment during training. Default True.
        use_checkpoint   (bool):  Gradient checkpoint CNN. Default True.

    Returns:
        Raw logit [B, 1]. No sigmoid.
        Training:  SmoothedBCEWithLogitsLoss()(logit, label, pos_weight)
        Inference: model._compute_score(model(x))

    Dimension note:
        lstm_dim   = (num_coefficients // 16) * 32  = 160 (default)
        head input = 2 * lstm_dim                   = 320 (default)
    """

    def __init__(self, **kwargs):
        super().__init__()
        in_ch         = kwargs.get("input_channels",   1)
        num_coeff     = kwargs.get("num_coefficients", 80)
        dropout       = kwargs.get("dropout",          0.4)
        self.use_aug  = kwargs.get("use_spec_augment", True)
        self.use_ckpt = kwargs.get("use_checkpoint",   True)

        self.num_coefficients = num_coeff
        self.v_emd_dim        = 1
        self.lstm_dim         = (num_coeff // 16) * 32   # 160 default

        # ── Stem ──────────────────────────────────────────────────────
        # [B, C, T, F] → [B, 32, T/2, F/2]
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=(5, 5),
                      stride=1, padding=(2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            MaxFeatureMap2D(),          # 64 → 32, amax (no index tensor)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        nn.init.kaiming_normal_(
            self.stem[0].weight, mode='fan_out', nonlinearity='relu'
        )

        # ── Stage 2 ── [B,32,T/2,F/2] → [B,48,T/4,F/4]
        self.stage2 = nn.Sequential(
            ConvMFMSE(32, 32),
            nn.BatchNorm2d(32, affine=False),
            ConvMFMSE(32, 48),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48, affine=False),
        )

        # ── Stage 3 ── [B,48,T/4,F/4] → [B,64,T/8,F/8]
        self.stage3 = nn.Sequential(
            ConvMFMSE(48, 48),
            nn.BatchNorm2d(48, affine=False),
            ConvMFMSE(48, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Stage 4 ── [B,64,T/8,F/8] → [B,32,T/16,F/16]
        # FIX 4: NO Dropout2d anywhere in CNN
        self.stage4 = nn.Sequential(
            ConvMFMSE(64, 64),
            nn.BatchNorm2d(64, affine=False),
            ConvMFMSE(64, 32),
            nn.BatchNorm2d(32, affine=False),
            ConvMFMSE(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Frequency attention ──
        self.freq_attn = FrequencyAttention(channels=32)

        # ── Temporal pipeline ──
        self.blstm1 = BLSTMLayer(self.lstm_dim, self.lstm_dim)
        self.drop1  = nn.Dropout(dropout * 0.5)         # FIX 4: here only
        # FIX 5: nhead=2 → head_dim=80
        self.trans  = PreNormTransformerLayer(
            d_model=self.lstm_dim,
            nhead=2,
            dropout=dropout * 0.25,
        )
        self.blstm2 = BLSTMLayer(self.lstm_dim, self.lstm_dim)
        self.drop2  = nn.Dropout(dropout)

        # FIX 1: Attentive Statistics Pooling → [B, 2*lstm_dim]
        self.asp = AttentiveStatisticsPooling(
            input_dim=self.lstm_dim, bottleneck=64
        )

        # ── MLP head ── input: 2*lstm_dim (320 default)
        self.head = nn.Sequential(
            nn.LayerNorm(self.lstm_dim * 2),
            nn.Linear(self.lstm_dim * 2, self.lstm_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(self.lstm_dim, self.v_emd_dim),
        )

    # ── Checkpoint runners ────────────────────────────────────────────
    def _run_stem(self, x):   return self.stem(x)
    def _run_stage2(self, x): return self.stage2(x)
    def _run_stage3(self, x): return self.stage3(x)
    def _run_stage4(self, x): return self.stage4(x)

    def _compute_score(self, logit: torch.Tensor) -> torch.Tensor:
        """Raw logit [B,1] → probability [B] ∈ [0,1]."""
        return torch.sigmoid(logit).squeeze(1)

    def _compute_embedding(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # [B, C, F, T] → [B, C, T, F]  (T as spatial height for Conv2d)
        x = x.permute(0, 1, 3, 2)

        # FIX 6: SpecAugment BEFORE checkpoint() — see module docstring
        if self.training and self.use_aug:
            x = spec_augment(
                x,
                freq_mask_param=10,
                time_mask_param=20,
                num_freq_masks=2,
                num_time_masks=2,
            )

        # CNN backbone — optionally gradient-checkpointed
        # use_reentrant=False required for AMP/autocast compatibility
        if self.use_ckpt and self.training:
            x = checkpoint(self._run_stem,   x, use_reentrant=False)
            x = checkpoint(self._run_stage2, x, use_reentrant=False)
            x = checkpoint(self._run_stage3, x, use_reentrant=False)
            x = checkpoint(self._run_stage4, x, use_reentrant=False)
        else:
            x = self._run_stem(x)
            x = self._run_stage2(x)
            x = self._run_stage3(x)
            x = self._run_stage4(x)

        x = self.freq_attn(x)                   # [B, 32, T', F']

        # Reshape → sequence [B, T', lstm_dim]
        x       = x.permute(0, 2, 1, 3).contiguous()
        T_prime = x.size(1)
        x       = x.view(B, T_prime, -1)

        # Temporal modeling
        h = self.blstm1(x)                      # [B, T', lstm_dim]
        h = self.drop1(h)
        h = self.trans(h)                       # [B, T', lstm_dim]
        h = self.blstm2(h) + h                  # residual — dims match
        h = self.drop2(h)

        # FIX 1: ASP → [B, 2*lstm_dim]
        pooled = self.asp(h)

        return self.head(pooled)                # [B, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F, T] — AAD default LFCC: [B, 1, 80, T]
        Returns:
            Raw logit [B, 1]. No sigmoid applied.
        """
        return self._compute_embedding(x)


# ======================================================================
#  SANITY CHECK
# ======================================================================

if __name__ == "__main__":
    print("Definition of model")
    model = LCNN(input_channels=3, num_coefficients=80)
    batch_size = 12
    mock_input = torch.rand((batch_size, 3, 80, 404))
    output = model(mock_input)
    print(output.shape)

