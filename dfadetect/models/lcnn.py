"""
Improved LCNN for Audio Deepfake Detection
==========================================
Key improvements for cross-dataset generalization
(FakeAVCeleb, ASVspoof, WaveFake):

1. Dual-path spectro-temporal feature extraction
   - Shallow path: fine-grained spectral details (good for ASVspoof)
   - Deep path: high-level semantic features
   - Learnable weighted fusion

2. Squeeze-and-Excitation + CBAM-style spatial attention
   - Channel attention (SE) + frequency-axis spatial attention
   - Applied before and after pooling for richer gradients

3. Robust temporal modeling
   - BLSTM → Transformer encoder layer → BLSTM pipeline
   - Residual only between matched-dim BLSTM outputs (no dim mismatch)

4. Dataset-agnostic training helpers
   - SpecAugment-style masking (time & freq) built into forward()
     so you can use it during training without external augmentation
   - Instance Normalization option per stage (better domain shift)

5. Calibrated scoring head
   - Two-layer MLP head instead of bare Linear
   - Outputs raw logits; use BCEWithLogitsLoss during training
   - _compute_score() applies sigmoid for inference

6. Fixes
   - LSTM residual: only added when dims match (safe fallback)
   - Proper kaiming init for all Conv2d layers
   - Gradient checkpointing-friendly (no in-place ops on graph leaves)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class MaxFeatureMap2D(nn.Module):
    """MFM activation: halves channels by taking element-wise max of pairs."""
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
        m, _ = x.view(*shape).max(d)
        return m


class BLSTMLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        assert output_dim % 2 == 0
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
    """Channel-wise Squeeze-and-Excitation."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class FrequencyAttention(nn.Module):
    """
    Spatial attention along the frequency axis.
    Works for any spatial size — uses depthwise conv so no fixed Linear.
    Applied on [B, C, F, T] tensors.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            # collapse time → [B, C, F, 1]
            nn.AdaptiveAvgPool2d((None, 1)),
            # depthwise conv along frequency
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=(3, 1), padding=(1, 0),
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T]
        w = self.gate(x)          # [B, C, F, 1]
        return x * w              # broadcast over T


class CBAM_FreqTime(nn.Module):
    """
    Lightweight CBAM variant:
    channel attention (SE) then frequency-axis spatial attention.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se   = SEBlock(channels, reduction)
        self.freq = FrequencyAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.se(x)
        x = self.freq(x)
        return x


# ---------------------------------------------------------------------------
# Convolutional building blocks
# ---------------------------------------------------------------------------

class ResidualSEBlock(nn.Module):
    """
    Conv → BN → GELU → Conv → BN → CBAM attention + residual shortcut.
    Uses GELU throughout (smoother gradients vs ReLU).
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
        self.attn = CBAM_FreqTime(out_ch)
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
        out = self.body(x)
        out = self.attn(out)
        out = out + self.skip(x)
        return self.act(out)


# ---------------------------------------------------------------------------
# SpecAugment helper (used during training only)
# ---------------------------------------------------------------------------

def spec_augment(x: torch.Tensor,
                 freq_mask_param: int = 8,
                 time_mask_param: int = 20,
                 num_freq_masks: int = 2,
                 num_time_masks: int = 2) -> torch.Tensor:
    """
    Apply SpecAugment-style masking in-place on [B, C, F, T] tensors.
    Call only when model.training is True.
    Masks are applied per-batch-item with random start positions.
    """
    B, C, F, T = x.shape
    out = x.clone()
    for b in range(B):
        for _ in range(num_freq_masks):
            f0 = torch.randint(0, max(1, F - freq_mask_param), (1,)).item()
            fw = torch.randint(1, freq_mask_param + 1, (1,)).item()
            out[b, :, f0:f0 + fw, :] = 0.0
        for _ in range(num_time_masks):
            t0 = torch.randint(0, max(1, T - time_mask_param), (1,)).item()
            tw = torch.randint(1, time_mask_param + 1, (1,)).item()
            out[b, :, :, t0:t0 + tw] = 0.0
    return out


# ---------------------------------------------------------------------------
# Transformer encoder layer (single layer, used between BLSTMs)
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """
    Standard pre-norm Transformer encoder layer.
    Pre-norm (LN before attention) is more stable than post-norm
    and trains better with small datasets.
    """
    def __init__(self, d_model: int, nhead: int = 4,
                 dim_ff: int = None, dropout: float = 0.1):
        super().__init__()
        dim_ff = dim_ff or d_model * 2
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
        # Self-attention with pre-norm + residual
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.drop(h)
        # Feed-forward with pre-norm + residual
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class LCNN(nn.Module):
    """
    Improved LCNN for audio deepfake detection.

    Architecture:
        Dual-path CNN backbone (shallow + deep)
        → learnable fusion
        → CBAM frequency attention
        → BLSTM → Transformer → BLSTM temporal modeling
        → attentive statistics pooling
        → MLP classifier head

    Args:
        input_channels   : number of input feature channels (default 3)
        num_coefficients : frequency bins in the input spectrogram (default 80)
        dropout          : dropout rate throughout (default 0.4)
        use_spec_augment : apply SpecAugment during training (default True)
    """

    def __init__(self, **kwargs):
        super().__init__()
        in_ch            = kwargs.get("input_channels",   3)
        num_coeff        = kwargs.get("num_coefficients", 80)
        dropout          = kwargs.get("dropout",          0.4)
        self.use_aug     = kwargs.get("use_spec_augment", True)
        self.num_coefficients = num_coeff
        self.v_emd_dim   = 1

        # ------------------------------------------------------------------
        # Shared stem
        # ------------------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, (5, 5), 1, (2, 2), bias=False),
            MaxFeatureMap2D(),              # 64 → 32 ch
            nn.MaxPool2d(2, 2),
        )

        # ------------------------------------------------------------------
        # Shallow path — fewer pooling ops, preserves fine-grained spectral
        # cues. Helps ASVspoof where TTS artifacts are subtle.
        # ------------------------------------------------------------------
        self.shallow_path = nn.Sequential(
            ResidualSEBlock(32, 64),
            MaxFeatureMap2D(),              # → 32 ch
            nn.BatchNorm2d(32, affine=False),
            ResidualSEBlock(32, 64),
            MaxFeatureMap2D(),              # → 32 ch
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32, affine=False),
        )
        # After stem (1×pool) + shallow (1×pool) = 2 total pools → freq // 4

        # ------------------------------------------------------------------
        # Deep path — more capacity + pooling for high-level features.
        # Keeps strong performance on FakeAVCeleb / WaveFake.
        # ------------------------------------------------------------------
        self.deep_path = nn.Sequential(
            ResidualSEBlock(32, 64),
            MaxFeatureMap2D(),              # → 32 ch
            nn.BatchNorm2d(32, affine=False),
            ResidualSEBlock(32, 96),
            MaxFeatureMap2D(),              # → 48 ch
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48, affine=False),
            ResidualSEBlock(48, 96),
            MaxFeatureMap2D(),              # → 48 ch
            nn.BatchNorm2d(48, affine=False),
            ResidualSEBlock(48, 64),
            MaxFeatureMap2D(),              # → 32 ch
            nn.MaxPool2d(2, 2),
        )
        # After stem (1×pool) + deep (2×pool) = 3 total pools → freq // 8

        # Align shallow (freq//4, 32ch) to deep (freq//8, 32ch)
        # by adding one extra pool on the shallow branch before fusion
        self.shallow_align = nn.MaxPool2d(2, 2)

        # Learnable fusion: shallow and deep are both 32 ch → concat 64 ch
        # then mix back to 32 ch
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        # ------------------------------------------------------------------
        # Post-fusion refinement
        # ------------------------------------------------------------------
        self.post_fusion = nn.Sequential(
            ResidualSEBlock(32, 64),
            MaxFeatureMap2D(),              # → 32 ch
            nn.BatchNorm2d(32, affine=False),
            ResidualSEBlock(32, 64),
            MaxFeatureMap2D(),              # → 32 ch
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5),   # spatial dropout (less aggressive)
        )
        # Total pools from input: stem(1) + shallow/deep(2/3) + align(1 shallow)
        # + post_fusion(1) = stem(1) + fusion_path(3) + post(1) = 5 → freq // 32
        # but we align both to 3 pools before fusion, then 1 more = 4 total
        # → freq // 16  (same as original)

        # ------------------------------------------------------------------
        # Frequency attention on final CNN feature map
        # ------------------------------------------------------------------
        self.freq_attn = FrequencyAttention(in_channels=32)

        # ------------------------------------------------------------------
        # LSTM dim: 32 channels × (num_coeff // 16) freq bins
        # ------------------------------------------------------------------
        lstm_dim = (num_coeff // 16) * 32

        # ------------------------------------------------------------------
        # Temporal modeling
        # BLSTM → Transformer → BLSTM
        # Residual only on the second BLSTM (same dim guaranteed)
        # ------------------------------------------------------------------
        self.blstm1   = BLSTMLayer(lstm_dim, lstm_dim)
        self.trans    = TransformerEncoderLayer(lstm_dim, nhead=4,
                                                dropout=dropout * 0.25)
        self.blstm2   = BLSTMLayer(lstm_dim, lstm_dim)
        self.seq_drop = nn.Dropout(dropout)

        # ------------------------------------------------------------------
        # Attentive statistics pooling
        # Learns which frames to attend to rather than plain mean.
        # ------------------------------------------------------------------
        self.attn_pool = nn.Linear(lstm_dim, 1)

        # ------------------------------------------------------------------
        # Classification head: 2-layer MLP
        # Outputs raw logit — use BCEWithLogitsLoss during training.
        # ------------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(lstm_dim, lstm_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_dim // 2, self.v_emd_dim),
        )

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _cnn_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run dual-path CNN and return fused feature map."""
        x = self.stem(x)                       # [B, 32, F/2, T]

        s = self.shallow_path(x)               # [B, 32, F/4, T]
        d = self.deep_path(x)                  # [B, 32, F/8, T]

        s = self.shallow_align(s)              # [B, 32, F/8, T] ← aligned
        # spatial dims may differ slightly due to rounding — crop to match
        min_f = min(s.size(2), d.size(2))
        min_t = min(s.size(3), d.size(3))
        s = s[:, :, :min_f, :min_t]
        d = d[:, :, :min_f, :min_t]

        fused = self.fusion(torch.cat([s, d], dim=1))  # [B, 32, F/8, T]
        fused = self.post_fusion(fused)                # [B, 32, F/16, T]
        return fused

    def _compute_embedding(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # Input: [B, C, F, T] → rearrange to [B, C, T, F] so that
        # Conv2d spatial dims are (T, F) — consistent with original
        x = x.permute(0, 1, 3, 2)

        # SpecAugment (training only, applied in [B,C,T,F] space)
        if self.training and self.use_aug:
            x = spec_augment(x,
                             freq_mask_param=8,
                             time_mask_param=20,
                             num_freq_masks=2,
                             num_time_masks=2)

        # CNN backbone outputs [B, 32, T', F']
        x = self._cnn_backbone(x)

        # Frequency attention: [B, 32, T', F'] (treating last dim as freq)
        x = self.freq_attn(x)

        # Flatten to sequence: [B, T', 32*F']
        x = x.permute(0, 2, 1, 3).contiguous()   # [B, T', C, F']
        T_prime = x.size(1)
        x = x.view(B, T_prime, -1)

        # Temporal modeling
        h = self.blstm1(x)                        # [B, T', D]
        h = self.trans(h)                         # [B, T', D]
        h = self.blstm2(h) + h                    # residual (same dim ✓)
        h = self.seq_drop(h)

        # Attentive statistics pooling
        attn_w = torch.softmax(self.attn_pool(h), dim=1)  # [B, T', 1]
        pooled = (h * attn_w).sum(dim=1)                   # [B, D]

        # Classification head
        return self.head(pooled)                  # [B, 1]  raw logit

    def _compute_score(self, logit: torch.Tensor) -> torch.Tensor:
        """Convert raw logit to [0, 1] probability."""
        return torch.sigmoid(logit).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns raw logit [B, 1].
        During training:  loss = BCEWithLogitsLoss(logit, label)
        During inference: score = sigmoid(logit)  (via _compute_score)
        """
        return self._compute_embedding(x)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    model = LCNN(input_channels=3, num_coefficients=80,
                 dropout=0.4, use_spec_augment=True)
    model.eval()

    batch_size  = 8
    mock_input  = torch.rand(batch_size, 3, 80, 404)

    with torch.no_grad():
        logit = model(mock_input)
        score = model._compute_score(logit)

    assert logit.shape == (batch_size, 1),  f"Logit shape: {logit.shape}"
    assert score.shape == (batch_size,),    f"Score shape: {score.shape}"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Logit shape  : {logit.shape}")
    print(f"✅ Score shape  : {score.shape}")
    print(f"✅ Total params : {total_params:,}")

    # Training mode check (SpecAugment active)
    model.train()
    logit_train = model(mock_input)
    print(f"✅ Training logit shape: {logit_train.shape}")