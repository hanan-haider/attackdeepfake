"""
Robust LCNN for Audio Deepfake Detection
- Designed for per-dataset training but better cross-dataset generalization
- Keeps MFM, SE, residual convs, BLSTM + self-attn, frequency gate
- More conservative capacity, stronger normalization and regularization
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------
# Basic building blocks
# ----------------------

class BLSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        input_dim: dim of input feature
        hidden_dim: total output dim (bidirectional)
        """
        super().__init__()
        if hidden_dim % 2 != 0:
            sys.exit("BLSTMLayer: hidden_dim must be even.")
        self.blstm = nn.LSTM(
            input_dim,
            hidden_dim // 2,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        # x: [B, T, D]
        out, _ = self.blstm(x)
        return out  # [B, T, hidden_dim]


class MaxFeatureMap2D(nn.Module):
    """
    Max-Feature-Map across channel dim (like original LCNN).
    """
    def __init__(self, max_dim=1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, inputs):
        shape = list(inputs.size())
        if self.max_dim >= len(shape):
            sys.exit("MaxFeatureMap2D: max_dim out of range.")
        if shape[self.max_dim] % 2 != 0:
            sys.exit("MaxFeatureMap2D: channel dim must be even.")
        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)
        m, _ = inputs.view(*shape).max(self.max_dim)
        return m


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class ResidualSEBlock(nn.Module):
    """
    Conv block with residual skip + SE attention + MFM.
    Uses small kernels and consistent BatchNorm for stability.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, affine=True)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True)

        self.se = SEBlock(out_channels)

        # skip path
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, affine=True),
            )
        else:
            self.shortcut = nn.Identity()

        # Apply MFM after SE (reduces to out_channels/2)
        self.mfm = MaxFeatureMap2D()
        self.out_channels = out_channels // 2

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out = out + self.shortcut(x)
        out = self.mfm(out)
        return out


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention (temporal)."""
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        attn_out, _ = self.attn(x, x, x)
        x = x + self.drop(attn_out)
        return self.norm(x)


class FrequencyAttentionGate(nn.Module):
    """
    Dynamic frequency-wise gate, independent of exact F dimension.
    Uses depthwise conv on pooled freq maps.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((None, 1))  # keep F, collapse T
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=in_channels,
            bias=True,
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, F, T]
        w = self.pool(x)             # [B, C, F, 1]
        w = self.depthwise_conv(w)   # [B, C, F, 1]
        w = self.act(w)
        return x * w                 # broadcast over T


# ----------------------
# Robust LCNN backbone
# ----------------------

class RobustLCNN(nn.Module):
    """
    LCNN variant tuned for cross-dataset robustness.
    Expected input: [B, C_in, F= num_coefficients, T]
    """

    def __init__(self, input_channels=3, num_coefficients=80,
                 lstm_hidden_factor=1.0, num_heads=4, dropout=0.5):
        super().__init__()

        self.num_coefficients = num_coefficients
        self.v_emd_dim = 1
        self.dropout_p = dropout

        # Stage 1: initial conv + MFM + pool
        self.stage1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(5, 5),
                      stride=1, padding=(2, 2), bias=False),
            nn.BatchNorm2d(64, affine=True),
            MaxFeatureMap2D(),                 # -> 32 ch
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )  # F: /2, T: /2

        # Stage 2
        self.res2a = ResidualSEBlock(32, 64)   # -> 32 ch
        self.bn2a = nn.BatchNorm2d(32, affine=True)
        self.res2b = ResidualSEBlock(32, 96)   # -> 48 ch
        self.bn2b = nn.BatchNorm2d(48, affine=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # F: /4, T:/4

        # Stage 3
        self.res3a = ResidualSEBlock(48, 96)   # -> 48 ch
        self.bn3a = nn.BatchNorm2d(48, affine=True)
        self.res3b = ResidualSEBlock(48, 128)  # -> 64 ch
        self.bn3b = nn.BatchNorm2d(64, affine=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # F: /8, T:/8

        # Stage 4
        self.res4a = ResidualSEBlock(64, 128)  # -> 64 ch
        self.bn4a = nn.BatchNorm2d(64, affine=True)
        self.res4b = ResidualSEBlock(64, 64)   # -> 32 ch
        self.bn4b = nn.BatchNorm2d(32, affine=True)
        self.res4c = ResidualSEBlock(32, 64)   # -> 32 ch
        self.bn4c = nn.BatchNorm2d(32, affine=True)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.drop4 = nn.Dropout(self.dropout_p)
        # After stage4: channels = 32, F: /16, T:/16

        # Frequency attention
        self.freq_attn = FrequencyAttentionGate(in_channels=32)

        # LSTM input dim: channels * freq_bins
        # num_coefficients is original F; after 4 pools (stride 2) -> F/16
        lstm_input_dim = int((num_coefficients // 16) * 32)
        lstm_hidden_dim = int(lstm_input_dim * lstm_hidden_factor)

        # Temporal modeling: one BLSTM + self-attn
        self.blstm = BLSTMLayer(lstm_input_dim, lstm_hidden_dim)
        self.self_attn = MultiHeadSelfAttention(
            embed_dim=lstm_hidden_dim, num_heads=num_heads,
            dropout=self.dropout_p,
        )

        # Final pooling and classifier
        self.drop_seq = nn.Dropout(self.dropout_p)
        # use both mean and max over time
        self.fc = nn.Linear(2 * lstm_hidden_dim, self.v_emd_dim)

    # ----------------------
    # Forward helpers
    # ----------------------

    def _conv_backbone(self, x):
        # x: [B, C, F, T]
        # many LCNN variants prefer [B, C, T, F] but we keep F last in conv for compatibility
        # Your original model permuted; here we follow the same convention as you had.

        # [B, C, F, T] -> [B, C, T, F]
        x = x.permute(0, 1, 3, 2)

        x = self.stage1(x)

        x = self.res2a(x)
        x = F.gelu(self.bn2a(x), approximate='tanh')
        x = self.res2b(x)
        x = F.gelu(self.bn2b(x), approximate='tanh')
        x = self.pool2(x)

        x = self.res3a(x)
        x = F.gelu(self.bn3a(x), approximate='tanh')
        x = self.res3b(x)
        x = F.gelu(self.bn3b(x), approximate='tanh')
        x = self.pool3(x)

        x = self.res4a(x)
        x = F.gelu(self.bn4a(x), approximate='tanh')
        x = self.res4b(x)
        x = F.gelu(self.bn4b(x), approximate='tanh')
        x = self.res4c(x)
        x = F.gelu(self.bn4c(x), approximate='tanh')
        x = self.pool4(x)
        x = self.drop4(x)

        # Frequency attention: x is [B, C, T', F'] now, need [B, C, F', T']
        x = x.permute(0, 1, 3, 2)   # [B, C, F', T']
        x = self.freq_attn(x)

        # Prepare sequence: [B, C, F', T'] -> [B, T', C*F']
        B, C, Freq, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T', C, F']
        x = x.view(B, T, C * Freq)              # [B, T', D]
        return x

    def _compute_embedding(self, x):
        # x: [B, C_in, F, T]
        seq = self._conv_backbone(x)           # [B, T', D]
        h = self.blstm(seq)                    # [B, T', H]
        h = self.self_attn(h)                  # [B, T', H]
        h = self.drop_seq(h)

        # Global temporal pooling: mean + max
        h_mean = h.mean(dim=1)
        h_max, _ = h.max(dim=1)
        h_cat = torch.cat([h_mean, h_max], dim=-1)  # [B, 2H]

        return self.fc(h_cat)                  # [B, 1]

    def forward(self, x):
        # only embedding/logit; apply sigmoid in loss/eval code
        return self._compute_embedding(x)

    def predict_score(self, x):
        """Convenience: return scores in [0,1]."""
        logits = self.forward(x).squeeze(1)
        return torch.sigmoid(logits)


if __name__ == "__main__":
    model = RobustLCNN(input_channels=3, num_coefficients=80)
    batch_size = 8
    mock_input = torch.rand((batch_size, 3, 80, 404))
    output = model(mock_input)
    assert output.shape == (batch_size, 1), f"Unexpected shape: {output.shape}"
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Output shape : {output.shape}")
    print(f"✅ Total params : {total_params:,}")