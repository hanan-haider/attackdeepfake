"""
Improved LCNN for Audio Deepfake Detection (Fixed)
"""
import sys
import torch
import torch.nn as torch_nn
import torch.nn.functional as F


class BLSTMLayer(torch_nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        if output_dim % 2 != 0:
            sys.exit(1)
        self.l_blstm = torch_nn.LSTM(
            input_dim, output_dim // 2,
            bidirectional=True, batch_first=True
        )

    def forward(self, x):
        blstm_data, _ = self.l_blstm(x)
        return blstm_data


class MaxFeatureMap2D(torch_nn.Module):
    def __init__(self, max_dim=1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, inputs):
        shape = list(inputs.size())
        if self.max_dim >= len(shape):
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)
        m, _ = inputs.view(*shape).max(self.max_dim)
        return m


class SEBlock(torch_nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = torch_nn.Sequential(
            torch_nn.AdaptiveAvgPool2d(1),
            torch_nn.Flatten(),
            torch_nn.Linear(channels, mid),
            torch_nn.ReLU(inplace=True),
            torch_nn.Linear(mid, channels),
            torch_nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class ResidualSEBlock(torch_nn.Module):
    """Conv block with residual skip + SE attention."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        pad = kernel_size // 2
        self.conv_block = torch_nn.Sequential(
            torch_nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad),
            torch_nn.BatchNorm2d(out_channels),
            torch_nn.GELU(),
            torch_nn.Conv2d(out_channels, out_channels, kernel_size, 1, pad),
            torch_nn.BatchNorm2d(out_channels),
        )
        self.se = SEBlock(out_channels)
        self.shortcut = (
            torch_nn.Sequential(
                torch_nn.Conv2d(in_channels, out_channels, 1, stride),
                torch_nn.BatchNorm2d(out_channels)
            )
            if in_channels != out_channels or stride != 1
            else torch_nn.Identity()
        )
        self.act = torch_nn.GELU()

    def forward(self, x):
        out = self.conv_block(x)
        out = self.se(out)
        out = out + self.shortcut(x)
        return self.act(out)


class MultiHeadSelfAttention(torch_nn.Module):
    """Multi-head self-attention for temporal sequence."""
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = torch_nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = torch_nn.LayerNorm(embed_dim)
        self.drop = torch_nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.drop(attn_out))


class FrequencyAttentionGate(torch_nn.Module):
    """
    Dynamic frequency-wise attention gate.
    Uses depthwise conv — no fixed Linear layer,
    so it handles any spatial freq dimension.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.gate = torch_nn.Sequential(
            torch_nn.AdaptiveAvgPool2d((None, 1)),          # [B, C, F, 1]
            torch_nn.Conv2d(in_channels, in_channels,
                            kernel_size=(3, 1), padding=(1, 0),
                            groups=in_channels),             # depthwise
            torch_nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, F, T]
        w = self.gate(x)    # [B, C, F, 1]
        return x * w        # broadcast over T


class LCNN(torch_nn.Module):
    """
    Improved LCNN for audio deepfake detection.
    Improvements: SE blocks, residual connections,
    multi-head self-attention, frequency attention gate,
    GELU activations, reduced dropout (0.4).
    """
    def __init__(self, **kwargs):
        super().__init__()
        input_channels   = kwargs.get("input_channels", 3)
        num_coefficients = kwargs.get("num_coefficients", 80)
        self.num_coefficients = num_coefficients
        self.v_emd_dim = 1

        # Stage 1: Initial MFM conv
        self.stage1 = torch_nn.Sequential(
            torch_nn.Conv2d(input_channels, 64, (5, 5), 1, padding=(2, 2)),
            MaxFeatureMap2D(),                  # → 32 ch
            torch_nn.MaxPool2d((2, 2), (2, 2)),
        )

        # Stage 2: Residual SE blocks
        self.stage2 = torch_nn.Sequential(
            ResidualSEBlock(32, 64),            # → 64 ch
            MaxFeatureMap2D(),                  # → 32 ch
            torch_nn.BatchNorm2d(32, affine=False),
            ResidualSEBlock(32, 96),            # → 96 ch
            MaxFeatureMap2D(),                  # → 48 ch
            torch_nn.MaxPool2d((2, 2), (2, 2)),
            torch_nn.BatchNorm2d(48, affine=False),
        )

        # Stage 3: Deeper residual SE blocks
        self.stage3 = torch_nn.Sequential(
            ResidualSEBlock(48, 96),            # → 96 ch
            MaxFeatureMap2D(),                  # → 48 ch
            torch_nn.BatchNorm2d(48, affine=False),
            ResidualSEBlock(48, 128),           # → 128 ch
            MaxFeatureMap2D(),                  # → 64 ch
            torch_nn.MaxPool2d((2, 2), (2, 2)),
        )

        # Stage 4: Final conv stages
        self.stage4 = torch_nn.Sequential(
            ResidualSEBlock(64, 128),           # → 128 ch
            MaxFeatureMap2D(),                  # → 64 ch
            torch_nn.BatchNorm2d(64, affine=False),
            ResidualSEBlock(64, 64),            # → 64 ch
            MaxFeatureMap2D(),                  # → 32 ch
            torch_nn.BatchNorm2d(32, affine=False),
            ResidualSEBlock(32, 64),            # → 64 ch
            MaxFeatureMap2D(),                  # → 32 ch
            torch_nn.MaxPool2d((2, 2), (2, 2)),
            torch_nn.Dropout(0.4),
        )

        # Frequency attention — in_channels=32 after stage4
        self.freq_attn = FrequencyAttentionGate(in_channels=32)

        # LSTM input dim: channels * freq_bins after 4 MaxPool2d
        lstm_input_dim = (num_coefficients // 16) * 32

        # Temporal modeling
        self.blstm1    = BLSTMLayer(lstm_input_dim, lstm_input_dim)
        self.self_attn = MultiHeadSelfAttention(lstm_input_dim, num_heads=4)
        self.blstm2    = BLSTMLayer(lstm_input_dim, lstm_input_dim)

        self.m_output_act = torch_nn.Linear(lstm_input_dim, self.v_emd_dim)

    def _compute_embedding(self, x):
        batch_size = x.shape[0]

        # [B, C, F, T] → permute to [B, C, T, F]
        x = x.permute(0, 1, 3, 2)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Frequency attention
        x = self.freq_attn(x)

        # [B, C, T', F'] → [B, T', C*F']
        x = x.permute(0, 2, 1, 3).contiguous()
        frame_num = x.shape[1]
        x = x.view(batch_size, frame_num, -1)

        # Temporal modeling with residual
        h1 = self.blstm1(x)
        h1 = self.self_attn(h1)
        h2 = self.blstm2(h1)

        pooled = (h2 + x).mean(dim=1)
        return self.m_output_act(pooled)

    def _compute_score(self, feature_vec):
        return torch.sigmoid(feature_vec).squeeze(1)

    def forward(self, x):
        return self._compute_embedding(x)


if __name__ == "__main__":
    model = ImprovedLCNN(input_channels=3, num_coefficients=80)
    batch_size = 50
    mock_input = torch.rand((batch_size, 3, 80, 404))
    output = model(mock_input)
    assert output.shape == (batch_size, 1), f"Unexpected shape: {output.shape}"
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Output shape : {output.shape}")
    print(f"✅ Total params : {total_params:,}")