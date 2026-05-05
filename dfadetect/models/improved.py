"""
Improved LCNN for Audio Deepfake Detection
Based on ASVSpoof2021 baseline LCNN, with the following enhancements:

1. Residual (skip) connections in CNN blocks to ease gradient flow
2. Self-attention pooling instead of naive mean pooling
3. Multi-head attention before the output layer
4. Label smoothing + mixup-ready architecture (loss handled externally)
5. Squeeze-and-Excitation (SE) channel attention in CNN blocks
6. Reduced dropout (0.4) + spatial dropout for better regularization
7. Layer normalization before LSTM for training stability
8. Learnable temperature scaling on the final sigmoid for calibration
9. Auxiliary classification head (can be used with deep supervision)
10. Weight initialization following best practices for ReLU/MFM activations
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class BLSTMLayer(nn.Module):
    """Bidirectional LSTM wrapper.
    Input:  (batch, length, dim_in)
    Output: (batch, length, dim_out)
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        assert output_dim % 2 == 0, "output_dim must be even"
        self.l_blstm = nn.LSTM(
            input_dim,
            output_dim // 2,
            bidirectional=True,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out, _ = self.l_blstm(x)
        return self.drop(out)


class MaxFeatureMap2D(nn.Module):
    """Max-Feature-Map activation (halves channel dim)."""
    def __init__(self, max_dim: int = 1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, x):
        shape = list(x.size())
        assert shape[self.max_dim] % 2 == 0, "Channel dim must be even for MFM"
        shape[self.max_dim] //= 2
        shape.insert(self.max_dim, 2)
        m, _ = x.view(*shape).max(self.max_dim)
        return m


class SqueezeExcitation(nn.Module):
    """Channel-wise Squeeze-and-Excitation block."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).view(x.size(0), -1, 1, 1)
        return x * w


class ResidualCNNBlock(nn.Module):
    """
    Conv → MFM → BN → Conv → MFM → BN  with optional 1×1 skip projection.
    SE attention is applied on the output before adding residual.
    """
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int,
                 kernel: int = 3, se_reduction: int = 4):
        super().__init__()
        pad = kernel // 2
        self.conv1   = nn.Conv2d(in_ch,   mid_ch * 2, (1, 1), bias=False)
        self.mfm1    = MaxFeatureMap2D()
        self.bn1     = nn.BatchNorm2d(mid_ch, affine=False)
        self.conv2   = nn.Conv2d(mid_ch, out_ch * 2, (kernel, kernel), padding=pad, bias=False)
        self.mfm2    = MaxFeatureMap2D()
        self.bn2     = nn.BatchNorm2d(out_ch, affine=False)
        self.se      = SqueezeExcitation(out_ch, se_reduction)
        # skip projection if channel dims differ
        self.skip    = nn.Conv2d(in_ch, out_ch, (1, 1), bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.bn1(self.mfm1(self.conv1(x)))
        out = self.bn2(self.mfm2(self.conv2(out)))
        out = self.se(out)
        return out + residual


class SpatialDropout2d(nn.Module):
    """Drops entire feature maps (channels) rather than individual values."""
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # mask shape: (batch, channels, 1, 1)
        mask = torch.bernoulli(torch.full((x.size(0), x.size(1), 1, 1), 1 - self.p, device=x.device))
        return x * mask / (1 - self.p)


class MultiHeadSelfAttention(nn.Module):
    """
    Lightweight multi-head self-attention over the time dimension.
    Input/output: (batch, time, dim)
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


class AttentiveStatisticsPooling(nn.Module):
    """
    Weighted mean + std pooling where weights are learned from content.
    Produces a (batch, 2*dim) vector from a (batch, time, dim) sequence.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x):
        # x: (batch, time, dim)
        w = torch.softmax(self.attention(x), dim=1)   # (batch, time, 1)
        mean = (w * x).sum(dim=1)                      # (batch, dim)
        std  = (w * (x - mean.unsqueeze(1)) ** 2).sum(dim=1).clamp(min=1e-8).sqrt()
        return torch.cat([mean, std], dim=-1)           # (batch, 2*dim)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ImprovedLCNN(nn.Module):
    """
    Improved LCNN for audio anti-spoofing / deepfake detection.

    Key improvements over the baseline:
    - Residual CNN blocks with SE attention
    - Spatial dropout instead of aggressive scalar dropout
    - LayerNorm before sequence modelling
    - Multi-head self-attention after BLSTM stack
    - Attentive statistics pooling (mean + std)
    - Learnable temperature on sigmoid output for calibration
    - Optional auxiliary head for deep supervision
    """

    def __init__(self, **kwargs):
        super().__init__()
        input_channels   = kwargs.get("input_channels", 3)
        num_coefficients = kwargs.get("num_coefficients", 80)
        lstm_dropout      = kwargs.get("lstm_dropout", 0.1)
        attn_heads        = kwargs.get("attn_heads", 4)
        spatial_drop_p    = kwargs.get("spatial_drop_p", 0.2)
        use_aux_head      = kwargs.get("use_aux_head", False)

        self.num_coefficients = num_coefficients
        self.use_aux_head     = use_aux_head
        self.v_emd_dim        = 1

        # ------------------------------------------------------------------
        # Stage 1 – initial conv (same as original to keep pretrain compat.)
        # ------------------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, (5, 5), 1, padding=(2, 2)),
            MaxFeatureMap2D(),                   # → 32 ch
            nn.MaxPool2d((2, 2)),
        )

        # ------------------------------------------------------------------
        # Stage 2 – residual blocks with SE
        # in_ch, mid_ch (before MFM), out_ch (after MFM)
        # ------------------------------------------------------------------
        self.stage2 = nn.Sequential(
            ResidualCNNBlock(32,  48, 48),       # 32 → 48
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(48, affine=False),
        )

        self.stage3 = nn.Sequential(
            ResidualCNNBlock(48,  64, 64),       # 48 → 64
            nn.MaxPool2d((2, 2)),
        )

        self.stage4 = nn.Sequential(
            ResidualCNNBlock(64,  64, 32),       # 64 → 32
            ResidualCNNBlock(32,  32, 32),       # 32 → 32
            nn.MaxPool2d((2, 2)),
        )

        self.spatial_drop = SpatialDropout2d(spatial_drop_p)

        # Compute seq_dim dynamically via a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 100, num_coefficients)
            dummy = dummy.permute(0, 1, 3, 2)   # (1, C, freq, time) → wait, input is (B,C,time,freq)
            # Actually input arrives as (B, C, time, freq), permute to (B, C, freq, time)
            # so dummy should be (1, C, time=100, freq=num_coefficients) then permute
            dummy = torch.zeros(1, input_channels, 100, num_coefficients)
            dummy = dummy.permute(0, 1, 3, 2)   # → (1, C, freq=num_coefficients, time=100)
            dummy = self.stem(dummy)
            dummy = self.stage2(dummy)
            dummy = self.stage3(dummy)
            dummy = self.stage4(dummy)
            _, C, freq_out, _ = dummy.shape
            seq_dim = C * freq_out

        # ------------------------------------------------------------------
        # Sequence model
        # ------------------------------------------------------------------
        self.pre_norm = nn.LayerNorm(seq_dim)

        self.blstm1 = BLSTMLayer(seq_dim,  seq_dim, dropout=lstm_dropout)
        self.blstm2 = BLSTMLayer(seq_dim,  seq_dim, dropout=lstm_dropout)

        self.self_attn = MultiHeadSelfAttention(seq_dim, num_heads=attn_heads, dropout=0.1)

        # ------------------------------------------------------------------
        # Pooling → output
        # ------------------------------------------------------------------
        self.asp = AttentiveStatisticsPooling(seq_dim)
        # ASP gives 2*seq_dim; project back to seq_dim then to output
        self.fc  = nn.Sequential(
            nn.Linear(2 * seq_dim, seq_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(seq_dim, self.v_emd_dim),
        )

        # Learnable temperature (starts at 1 → no change)
        self.log_temperature = nn.Parameter(torch.zeros(1))

        # Optional auxiliary classification head (used with deep supervision)
        if use_aux_head:
            self.aux_head = nn.Linear(seq_dim, self.v_emd_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def _extract_features(self, x):
        """x: (batch, C, time, freq) — as provided by the trainer."""
        # Match original: permute to (batch, C, freq, time) for 2D CNN
        x = x.permute(0, 1, 3, 2)              # → (B, C, freq, time)

        x = self.stem(x)                        # → (B, 32, freq/2, time/2)
        x = self.stage2(x)                      # → (B, 48, freq/4, time/4)
        x = self.stage3(x)                      # → (B, 64, freq/8, time/8)
        x = self.stage4(x)                      # → (B, 32, freq/16, time/16)
        x = self.spatial_drop(x)

        # Reshape to sequence: treat time as sequence axis
        # After CNN: (B, C, time_frames, freq_bins)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous() # → (B, T, C, F)
        x = x.view(B, T, C * F)                # → (B, T, C*F)

        # Sequence modelling
        x = self.pre_norm(x)
        residual = x
        x = self.blstm1(x)
        x = self.blstm2(x) + residual          # residual around BLSTM stack
        x = self.self_attn(x)
        return x

    # ------------------------------------------------------------------
    def _compute_embedding(self, x):
        seq = self._extract_features(x)         # (B, T, D)
        pooled = self.asp(seq)                  # (B, 2D)
        emb = self.fc(pooled)                   # (B, 1)

        if self.use_aux_head and self.training:
            aux = self.aux_head(seq.mean(1))    # (B, 1)
            return emb, aux
        return emb

    def _compute_score(self, emb):
        temp = self.log_temperature.exp().clamp(0.1, 10.0)
        return torch.sigmoid(emb / temp).squeeze(1)

    def forward(self, x):
        if self.use_aux_head and self.training:
            emb, aux = self._compute_embedding(x)
            return emb, aux
        emb = self._compute_embedding(x)
        return emb


# ---------------------------------------------------------------------------
# Label-smoothed BCE loss (drop-in replacement for BCEWithLogitsLoss)
# ---------------------------------------------------------------------------

class LabelSmoothingBCELoss(nn.Module):
    """
    Binary cross-entropy with label smoothing.
    targets are 0/1; epsilon smooths them to [eps/2, 1 - eps/2].
    """
    def __init__(self, epsilon: float = 0.05, reduction: str = "mean"):
        super().__init__()
        self.epsilon   = epsilon
        self.reduction = reduction

    def forward(self, logits, targets):
        smooth = targets.float() * (1 - self.epsilon) + (1 - targets.float()) * (self.epsilon / 2)
        loss = F.binary_cross_entropy_with_logits(logits, smooth, reduction=self.reduction)
        return loss


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = ImprovedLCNN(
        input_channels=3,
        num_coefficients=80,
        use_aux_head=True,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    model.train()
    mock_input = torch.rand((4, 3, 80, 404))   # same shape as original
    emb, aux = model(mock_input)
    print(f"Embedding shape : {emb.shape}")     # (4, 1)
    print(f"Aux head shape  : {aux.shape}")     # (4, 1)

    model.eval()
    with torch.no_grad():
        emb = model(mock_input)
        scores = model._compute_score(emb)
        print(f"Score shape     : {scores.shape}")  # (4,)
        print(f"Score range     : [{scores.min():.4f}, {scores.max():.4f}]")

    # Test loss
    criterion = LabelSmoothingBCELoss(epsilon=0.05)
    labels = torch.randint(0, 2, (4,)).float()
    loss = criterion(emb.squeeze(1), labels)
    print(f"Label-smoothed loss: {loss.item():.4f}")