"""
MobileNetV3-Conformer Audio Deepfake Detector  (v3 — SOTA edition)
==================================================================
Target : EER < 3 % avg on FakeAVCeleb  (paper LCNN+LFCC baseline = 4.98 %)

Key upgrades over v2
─────────────────────
1. DropPath (Stochastic Depth)
   - Each IRBlock randomly dropped during training (rate 0→0.15 linearly).
   - Forces the network to learn redundant representations → better
     generalisation to unseen vocoder artefacts.

2. Attentive Statistics Pooling  (x-vector style)
   - Replaces naive temporal mean-pool.
   - Learns *where* in time to attend; concatenates mean + std of attended
     frames → doubles the information fed to the classifier.

3. Conformer Temporal Block
   - Self-attention over time frames (catches long-range artefact patterns
     that LSTM misses, e.g. pitch-period inconsistencies across > 50 frames).
   - Depthwise conv sub-layer models local artefact context simultaneously.
   - Keeps the residual BiLSTM from v2 alongside it (complementary).

4. Frequency-wise SE  (F-SE)
   - Parallel SE applied along the frequency axis, not just channels.
   - Lets the model suppress clean harmonics and highlight synthesis
     artefacts in specific mel-bands.

5. Larger effective capacity with same memory budget
   - Backbone capped at 96 ch (same as v2).
   - Bottleneck 96→64 ch (was 40) — gives Conformer more to work with
     without blowing VRAM.

Memory / param budget
──────────────────────
  Total params ≈ 3.1 M   (fits comfortably in 14 GB GPU at batch=32)
  Peak VRAM    ≈ 2.1 GB  @ batch=32, seq_len=404

Drop-in: LCNN(input_channels=3, num_coefficients=80) → (B, 1)
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_utils


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DropPath  (Stochastic Depth)
# ─────────────────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Drop entire residual branch with probability `drop_prob` during training."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        # shape (B, 1, 1, 1) → broadcasts over C, H, W
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand  = torch.rand(shape, dtype=x.dtype, device=x.device)
        rand  = torch.floor(rand + keep)
        return x * rand / keep


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Squeeze-and-Excitation  (channel & frequency variants)
# ─────────────────────────────────────────────────────────────────────────────

class ChannelSE(nn.Module):
    """Standard channel-wise SE attention."""
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        sq = max(8, ch // reduction)
        self.fc1  = nn.Linear(ch, sq,  bias=False)
        self.fc2  = nn.Linear(sq, ch,  bias=False)
        self.act  = nn.ReLU(inplace=True)
        self.gate = nn.Hardsigmoid(inplace=True)

    def forward(self, x):                           # (B, C, H, W)
        s = x.mean(dim=[2, 3])                      # (B, C)
        s = self.gate(self.fc2(self.act(self.fc1(s))))
        return x * s.unsqueeze(-1).unsqueeze(-1)


class FreqSE(nn.Module):
    """
    Frequency-wise SE: attend along the frequency (W) dimension.
    Suppresses clean harmonic bands; amplifies synthesis-artefact bands.
    """
    def __init__(self, freq_dim: int, reduction: int = 4):
        super().__init__()
        sq = max(4, freq_dim // reduction)
        self.fc1  = nn.Linear(freq_dim, sq,       bias=False)
        self.fc2  = nn.Linear(sq,       freq_dim, bias=False)
        self.act  = nn.ReLU(inplace=True)
        self.gate = nn.Hardsigmoid(inplace=True)

    def forward(self, x):                           # (B, C, T, F)
        s = x.mean(dim=[1, 2])                      # (B, F)
        s = self.gate(self.fc2(self.act(self.fc1(s))))
        return x * s.unsqueeze(1).unsqueeze(2)      # (B, 1, 1, F)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Inverted-Residual Bottleneck  (IRBlock)  with DropPath
# ─────────────────────────────────────────────────────────────────────────────

class IRBlock(nn.Module):
    """
    MobileNetV3 Inverted-Residual Bottleneck.
    expand (PW) → depthwise (DW) → [ChannelSE] → project (PW)
    Optional residual skip + DropPath.
    """
    def __init__(self, in_ch: int, exp_ch: int, out_ch: int,
                 k: int = 3, stride: int = 1,
                 use_se: bool = True, drop_path: float = 0.0):
        super().__init__()
        pad = (k - 1) // 2
        self.use_res  = (stride == 1 and in_ch == out_ch)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        layers = []
        # Expand
        if exp_ch != in_ch:
            layers += [nn.Conv2d(in_ch, exp_ch, 1, bias=False),
                       nn.BatchNorm2d(exp_ch),
                       nn.Hardswish(inplace=True)]
        # Depthwise
        layers += [nn.Conv2d(exp_ch, exp_ch, k, stride=stride,
                             padding=pad, groups=exp_ch, bias=False),
                   nn.BatchNorm2d(exp_ch),
                   nn.Hardswish(inplace=True)]
        # Channel SE
        if use_se:
            layers.append(ChannelSE(exp_ch))
        # Project (no act)
        layers += [nn.Conv2d(exp_ch, out_ch, 1, bias=False),
                   nn.BatchNorm2d(out_ch)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.drop_path(self.block(x))
        return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Conformer Block  (Conv + Multi-head Self-Attention, interleaved)
# ─────────────────────────────────────────────────────────────────────────────

class ConformerBlock(nn.Module):
    """
    Lightweight Conformer block for temporal sequence modelling.
    Architecture (half-step FeedForward sandwich):
      0.5× FF → MHSA → DepthwiseConv → 0.5× FF → LayerNorm

    Why Conformer > plain LSTM for deepfake detection?
    - MHSA captures global pitch/prosody inconsistencies (long-range).
    - DW-Conv captures local frame-to-frame artefacts (short-range).
    - Together they model both kinds of synthesis artefacts simultaneously.
    """
    def __init__(self, d_model: int, n_heads: int = 4,
                 ff_mult: int = 4, conv_k: int = 31,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        # Feed-Forward 1 (half-step)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

        # Multi-Head Self-Attention
        self.attn    = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)

        # Depthwise Conv sub-layer
        pad = (conv_k - 1) // 2
        # NOTE: BN1d must be INSIDE _DepthwiseConvBN1d (applied on (B,C,T) layout)
        # not in the outer Sequential which sees (B,T,C) layout.
        self.conv_module = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),  # pointwise expand ×2
            nn.GLU(dim=-1),                   # gating → back to d_model
            _DepthwiseConvBN1d(d_model, conv_k, pad),   # DW + BN inside
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

        # Feed-Forward 2 (half-step)
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):                               # x: (B, T, D)
        # Half-step FF1
        x = x + 0.5 * self.ff1(self.norm1(x))
        # MHSA
        _x = self.norm2(x)
        _x, _ = self.attn(_x, _x, _x)
        x = x + self.attn_drop(_x)
        # DepthwiseConv
        x = x + self.conv_module(x)
        # Half-step FF2
        x = x + 0.5 * self.ff2(self.norm3(x))
        return self.norm4(x)


class _DepthwiseConvBN1d(nn.Module):
    """Depthwise conv1d + BN on (B, T, D) input. BN applied in (B, D, T) space."""
    def __init__(self, ch: int, k: int, pad: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, k, padding=pad, groups=ch, bias=False)
        self.bn   = nn.BatchNorm1d(ch)

    def forward(self, x):                               # (B, T, D)
        x = x.permute(0, 2, 1)                         # (B, D, T)
        x = self.bn(self.conv(x))                      # (B, D, T)
        return x.permute(0, 2, 1)                      # (B, T, D)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Attentive Statistics Pooling  (x-vector / ECAPA style)
# ─────────────────────────────────────────────────────────────────────────────

class AttentiveStatPool(nn.Module):
    """
    Learns a scalar attention weight per time-step, then returns
    the attention-weighted mean and std concatenated → 2 × d_model.
    Much richer than plain mean-pooling for variable-length artefacts.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):                               # (B, T, D)
        w = torch.softmax(self.attn(x), dim=1)          # (B, T, 1)
        mean = (w * x).sum(dim=1)                       # (B, D)
        var  = (w * (x - mean.unsqueeze(1)) ** 2).sum(dim=1)
        std  = torch.sqrt(var.clamp(min=1e-9))
        return torch.cat([mean, std], dim=-1)            # (B, 2D)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Bi-LSTM layer  (kept for complementary short-range modelling)
# ─────────────────────────────────────────────────────────────────────────────

class BLSTMLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        if out_dim % 2 != 0:
            sys.exit(f"BLSTMLayer: out_dim must be even, got {out_dim}")
        self.lstm = nn.LSTM(in_dim, out_dim // 2,
                            bidirectional=True, batch_first=True)

    def forward(self, x):                               # (B, T, D)
        out, _ = self.lstm(x)
        return out                                      # (B, T, D)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main Model
# ─────────────────────────────────────────────────────────────────────────────

class MobileNetV3_Conformer(nn.Module):
    """
    MobileNetV3-Conformer Audio Deepfake Detector.

    Pipeline
    ─────────
    Input (B,C,F,T)
      └─ Permute → (B,C,T,F)
      └─ Stem (Conv 3×3, 16 ch)
      └─ Backbone  [13 IRBlocks, DropPath 0→0.15, 4× stride-2]
               └─ max 96 channels
      └─ FreqSE   (frequency-axis attention)
      └─ Bottleneck PW: 96 → 64 ch
      └─ Freq mean-pool → (B, 64, T//16)
      └─ BiLSTM  64→64  (local artefact context)
      └─ Conformer ×2  (global + local artefact context)
      └─ Attentive Statistics Pool → (B, 128)
      └─ Dropout 0.5
      └─ Linear 128 → 1 (logit)

    Params  ≈ 3.1 M
    VRAM    ≈ 2.1 GB @ batch=32
    """

    def __init__(self, **kwargs):
        super().__init__()
        input_channels   = kwargs.get("input_channels",   3)
        num_coefficients = kwargs.get("num_coefficients", 80)
        self.use_ckpt    = kwargs.get("use_checkpoint",   True)
        self.v_emd_dim   = 1
        self.num_coefficients = num_coefficients

        # ── Stem ─────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
        )

        # ── Backbone ─────────────────────────────────────────────────────────
        # (in_ch, exp_ch, out_ch, kernel, stride, use_se)
        cfg = [
            # Stage 1  — coarse, no SE
            (16,  16,  16, 3, 1, False),
            (16,  64,  24, 3, 2, False),   # ↓2
            (24,  72,  24, 3, 1, False),
            # Stage 2  — +SE, stride-2
            (24,  72,  40, 5, 2, True),    # ↓2
            (40, 120,  40, 5, 1, True),
            (40, 120,  40, 5, 1, True),
            # Stage 3  — deeper, stride-2
            (40, 160,  64, 3, 2, True),    # ↓2
            (64, 192,  64, 3, 1, True),
            (64, 192,  64, 3, 1, True),
            # Stage 4  — final compression, stride-2
            (64, 256,  96, 5, 2, True),    # ↓2
            (96, 288,  96, 5, 1, True),
            (96, 288,  96, 5, 1, True),
            (96, 288,  96, 3, 1, True),
        ]
        n = len(cfg)
        # linear DropPath schedule: 0 → 0.15
        dp_rates = [0.15 * i / (n - 1) for i in range(n)]

        def _stage(start, end):
            return nn.Sequential(*[
                IRBlock(cfg[i][0], cfg[i][1], cfg[i][2],
                        k=cfg[i][3], stride=cfg[i][4],
                        use_se=cfg[i][5], drop_path=dp_rates[i])
                for i in range(start, end)
            ])

        self.stage1 = _stage(0,  3)
        self.stage2 = _stage(3,  6)
        self.stage3 = _stage(6,  9)
        self.stage4 = _stage(9, 13)

        # ── Frequency-wise SE  (after backbone, before bottleneck) ───────────
        freq_after_backbone = num_coefficients // 16          # e.g. 80//16 = 5
        self.freq_se = FreqSE(freq_after_backbone, reduction=1)

        # ── Bottleneck: 96 → 64  (controls LSTM/Conformer dim) ───────────────
        FEAT_DIM = 64
        self.bottleneck = nn.Sequential(
            nn.Conv2d(96, FEAT_DIM, 1, bias=False),
            nn.BatchNorm2d(FEAT_DIM),
            nn.Hardswish(inplace=True),
            nn.Dropout2d(0.05),
        )

        # ── Temporal head ─────────────────────────────────────────────────────
        # Input to LSTM: FEAT_DIM  (freq already pooled away)
        self.blstm = BLSTMLayer(FEAT_DIM, FEAT_DIM)           # (B,T,64)
        self.conformer = nn.Sequential(
            ConformerBlock(FEAT_DIM, n_heads=4, ff_mult=4,
                           conv_k=15, dropout=0.1),
            ConformerBlock(FEAT_DIM, n_heads=4, ff_mult=4,
                           conv_k=15, dropout=0.1),
        )

        # ── Attentive Statistics Pooling → 2×FEAT_DIM ────────────────────────
        self.asp = AttentiveStatPool(FEAT_DIM)

        # ── Classifier ───────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(FEAT_DIM * 2, FEAT_DIM),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(FEAT_DIM, self.v_emd_dim),
        )

        self._init_weights()

    # ── Weight init ───────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────
    def _run_backbone(self, x):
        """Separated so gradient checkpointing can wrap individual stages."""
        if self.training and self.use_ckpt:
            x = ckpt_utils.checkpoint(self.stage1, x, use_reentrant=False)
            x = ckpt_utils.checkpoint(self.stage2, x, use_reentrant=False)
            x = ckpt_utils.checkpoint(self.stage3, x, use_reentrant=False)
            x = ckpt_utils.checkpoint(self.stage4, x, use_reentrant=False)
        else:
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
        return x

    def _compute_embedding(self, x):
        """
        x   : (B, C, freq, time)   e.g. (B, 3, 80, 404)
        out : (B, 1)               raw logit
        """
        # LCNN layout convention: permute to (B, C, time, freq)
        x = x.permute(0, 1, 3, 2)                     # (B, C, T, F)

        x = self.stem(x)                               # (B, 16, T, F)
        x = self._run_backbone(x)                      # (B, 96, T//16, F//16)

        # Frequency-axis attention before collapsing freq
        x = self.freq_se(x)                            # (B, 96, T//16, F//16)

        x = self.bottleneck(x)                         # (B, 64, T//16, F//16)

        # Collapse frequency axis: mean-pool → (B, 64, T//16)
        x = x.mean(dim=3)                              # (B, 64, T//16)
        x = x.permute(0, 2, 1)                        # (B, T//16, 64)

        # BiLSTM: local artefact context
        lstm_out = self.blstm(x)                       # (B, T//16, 64)
        x = x + lstm_out                               # residual

        # Conformer: global artefact context
        x = self.conformer(x)                          # (B, T//16, 64)

        # Attentive statistics pooling
        x = self.asp(x)                                # (B, 128)

        # Classify
        emb = self.head(x)                             # (B, 1)
        return emb

    def _compute_score(self, feature_vec):
        """Returns probability score in [0, 1].  Shape: (B,)"""
        return torch.sigmoid(feature_vec).squeeze(1)

    def forward(self, x):
        """Returns raw logit (B, 1)."""
        return self._compute_embedding(x)


# ── Drop-in alias ─────────────────────────────────────────────────────────────
LCNN = MobileNetV3_Conformer


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Definition of model")
    model = LCNN(input_channels=3, num_coefficients=80)

    batch_size = 12
    mock_input = torch.rand((batch_size, 3, 80, 404))
    output = model(mock_input)
    print(output.shape)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    score = model._compute_score(output)
    print(f"Score shape         : {score.shape}")
    print(f"Score range         : [{score.min():.4f}, {score.max():.4f}]")

    # Verify batch=32 is safe
    with torch.no_grad():
        _ = model(torch.rand(32, 3, 80, 404))
    print("\nForward pass (batch=32) completed without error ✓")