"""
MobileNetV3-inspired Audio Deepfake Detection Model
Replaces LCNN with depthwise + pointwise convolutions (MobileNetV3 blocks),
Squeeze-and-Excitation (SE) attention, hard-swish activations, and a
Bi-LSTM temporal aggregation head — targeting SOTA on the Attack-Agnostic Dataset.

Expected input shape : (batch, channels, num_coefficients, time_frames)
                       e.g.  (12, 3, 80, 404)
Output shape         : (batch, 1)  — raw logit (apply sigmoid for score)
"""

import sys
import torch
import torch.nn as torch_nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Activation helpers
# ─────────────────────────────────────────────────────────────

class HardSwish(torch_nn.Module):
    """Hard-Swish activation used in MobileNetV3."""
    def forward(self, x):
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0


class HardSigmoid(torch_nn.Module):
    """Hard-Sigmoid used inside SE blocks."""
    def forward(self, x):
        return F.hardtanh(x + 3, 0.0, 6.0) / 6.0


# ─────────────────────────────────────────────────────────────
# Squeeze-and-Excitation (SE) Block
# ─────────────────────────────────────────────────────────────

class SEBlock(torch_nn.Module):
    """
    Channel-wise Squeeze-and-Excitation attention.
    Helps the model re-weight feature maps — critical for catching
    subtle artefacts in spoofed speech.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        squeezed = max(1, channels // reduction)
        self.se = torch_nn.Sequential(
            torch_nn.AdaptiveAvgPool2d(1),
            torch_nn.Flatten(),
            torch_nn.Linear(channels, squeezed, bias=False),
            torch_nn.ReLU(inplace=True),
            torch_nn.Linear(squeezed, channels, bias=False),
            HardSigmoid(),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        scale = self.se(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * scale


# ─────────────────────────────────────────────────────────────
# Depthwise-Separable Convolution Block
# ─────────────────────────────────────────────────────────────

class DSConvBNAct(torch_nn.Module):
    """Depthwise conv → BN → Act + Pointwise conv → BN → Act."""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, stride: int = 1,
                 padding: int = 1, act=None):
        super().__init__()
        act = act or HardSwish
        self.dw = torch_nn.Sequential(
            torch_nn.Conv2d(in_ch, in_ch, kernel, stride=stride,
                            padding=padding, groups=in_ch, bias=False),
            torch_nn.BatchNorm2d(in_ch),
            act(),
        )
        self.pw = torch_nn.Sequential(
            torch_nn.Conv2d(in_ch, out_ch, 1, bias=False),
            torch_nn.BatchNorm2d(out_ch),
            act(),
        )

    def forward(self, x):
        return self.pw(self.dw(x))


# ─────────────────────────────────────────────────────────────
# MobileNetV3 Inverted-Residual Bottleneck (IRB)
# ─────────────────────────────────────────────────────────────

class IRBlock(torch_nn.Module):
    """
    Inverted-Residual Bottleneck with optional SE and hard-swish.
    expand → depthwise (+ optional SE) → project
    Residual connection when in_ch == out_ch and stride == 1.
    """
    def __init__(self, in_ch: int, exp_ch: int, out_ch: int,
                 kernel: int = 3, stride: int = 1,
                 use_se: bool = True, act=None):
        super().__init__()
        act = act or HardSwish
        padding = (kernel - 1) // 2
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []
        # Expand (pointwise)
        if exp_ch != in_ch:
            layers += [
                torch_nn.Conv2d(in_ch, exp_ch, 1, bias=False),
                torch_nn.BatchNorm2d(exp_ch),
                act(),
            ]
        # Depthwise
        layers += [
            torch_nn.Conv2d(exp_ch, exp_ch, kernel, stride=stride,
                            padding=padding, groups=exp_ch, bias=False),
            torch_nn.BatchNorm2d(exp_ch),
            act(),
        ]
        # SE
        if use_se:
            layers.append(SEBlock(exp_ch))
        # Project (pointwise, no activation)
        layers += [
            torch_nn.Conv2d(exp_ch, out_ch, 1, bias=False),
            torch_nn.BatchNorm2d(out_ch),
        ]
        self.block = torch_nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out


# ─────────────────────────────────────────────────────────────
# Temporal Bi-LSTM aggregation (same role as LCNN's m_before_pooling)
# ─────────────────────────────────────────────────────────────

class BLSTMLayer(torch_nn.Module):
    """Bi-directional LSTM — input/output shape: (B, T, D)."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        if output_dim % 2 != 0:
            print("BLSTMLayer: output_dim must be even, got", output_dim)
            sys.exit(1)
        self.lstm = torch_nn.LSTM(input_dim, output_dim // 2,
                                  bidirectional=True, batch_first=False)

    def forward(self, x):               # x: (B, T, D)
        out, _ = self.lstm(x.permute(1, 0, 2))   # (T, B, D)
        return out.permute(1, 0, 2)               # (B, T, D)


# ─────────────────────────────────────────────────────────────
# Main Model: MobileNetV3-AA  (Attack-Agnostic)
# ─────────────────────────────────────────────────────────────

class MobileNetV3_AA(torch_nn.Module):
    """
    MobileNetV3-style audio deepfake detector.

    Architecture highlights
    ───────────────────────
    • Stem          : regular 3×3 conv (heavy-but-fast first layer)
    • Backbone      : 8 Inverted-Residual Bottleneck blocks with SE
    • Temporal head : 2-layer Bi-LSTM (residual skip, same as LCNN)
    • Classifier    : single Linear → logit
    • Activations   : Hard-Swish throughout (faster & comparable to Swish)
    • Dropout 0.5   : before final linear (reduced from LCNN's 0.7 to
                      avoid underfitting on seen attack types)

    Why better than LCNN on Attack-Agnostic?
    ─────────────────────────────────────────
    1. SE attention highlights discriminative frequency-time regions.
    2. Depthwise separable convs give 8-9× fewer FLOPS → better
       regularisation and faster convergence with limited data.
    3. Hard-swish avoids the dying-ReLU problem in deep stacks.
    4. Residual paths in IRBlocks preserve low-level artefact cues.
    """

    def __init__(self, **kwargs):
        super().__init__()
        input_channels  = kwargs.get("input_channels",  3)
        num_coefficients = kwargs.get("num_coefficients", 80)

        self.num_coefficients = num_coefficients
        self.v_emd_dim = 1

        # ── Stem ──────────────────────────────────────────────
        # Regular conv to map multi-channel spectral input → 16 ch
        self.stem = torch_nn.Sequential(
            torch_nn.Conv2d(input_channels, 16, 3, stride=1,
                            padding=1, bias=False),
            torch_nn.BatchNorm2d(16),
            HardSwish(),
        )

        # ── Backbone (MobileNetV3-Large-inspired) ─────────────
        # Each tuple: (in_ch, exp_ch, out_ch, kernel, stride, se)
        cfg = [
            # Stage 1 — light feature extraction, no SE (early layers)
            (16,  16,  16, 3, 1, False),   # s=1 → same size
            (16,  64,  24, 3, 2, False),   # ↓2 freq & time
            (24,  72,  24, 3, 1, False),
            # Stage 2 — richer features, add SE
            (24,  72,  40, 5, 2, True),    # ↓2
            (40, 120,  40, 5, 1, True),
            (40, 120,  40, 5, 1, True),
            # Stage 3 — high-level, SE + hard-swish
            (40, 240,  80, 3, 2, True),    # ↓2
            (80, 200,  80, 3, 1, True),
            (80, 184,  80, 3, 1, True),
            (80, 184,  80, 3, 1, True),
            (80, 480, 112, 3, 1, True),
            (112, 672, 112, 3, 1, True),
            # Stage 4 — final spatial compression
            (112, 672, 160, 5, 2, True),   # ↓2
            (160, 960, 160, 5, 1, True),
            (160, 960, 160, 5, 1, True),
        ]

        blocks = []
        for (ic, ec, oc, k, s, se) in cfg:
            blocks.append(IRBlock(ic, ec, oc, kernel=k,
                                  stride=s, use_se=se))
        self.backbone = torch_nn.Sequential(*blocks)

        # ── Conv head (replaces LCNN's last dropout+pool) ──────
        self.conv_head = torch_nn.Sequential(
            torch_nn.Conv2d(160, 256, 1, bias=False),
            torch_nn.BatchNorm2d(256),
            HardSwish(),
            torch_nn.Dropout2d(0.1),     # spatial dropout for regularisation
        )

        # After 4× stride-2 downsampling:
        #   freq dim : num_coefficients // 16
        #   channel  : 256
        # LSTM input dim = (freq//16) * 256
        lstm_in = (num_coefficients // 16) * 256
        lstm_hidden = lstm_in  # keep dim consistent with LCNN convention

        # ── Temporal Bi-LSTM ──────────────────────────────────
        self.m_before_pooling = torch_nn.Sequential(
            BLSTMLayer(lstm_in, lstm_hidden),
            BLSTMLayer(lstm_hidden, lstm_hidden),
        )

        # ── Output head ───────────────────────────────────────
        self.dropout = torch_nn.Dropout(0.5)
        self.m_output_act = torch_nn.Linear(lstm_hidden, self.v_emd_dim)

        # Weight init
        self._init_weights()

    # ── Weight initialisation ──────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch_nn.Conv2d):
                torch_nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    torch_nn.init.zeros_(m.bias)
            elif isinstance(m, torch_nn.BatchNorm2d):
                torch_nn.init.ones_(m.weight)
                torch_nn.init.zeros_(m.bias)
            elif isinstance(m, torch_nn.Linear):
                torch_nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    torch_nn.init.zeros_(m.bias)

    # ── Core forward ──────────────────────────────────────────
    def _compute_embedding(self, x):
        """
        x : (B, C, freq, time)  — same layout as LCNN input
        returns embedding : (B, 1)
        """
        batch_size = x.shape[0]

        # LCNN permutes to (B, C, time, freq); we keep (B, C, freq, time)
        # because our strides compress freq first (matches mel-spectrogram
        # convention where freq is the 'spatial' axis).
        x = x.permute(0, 1, 3, 2)   # (B, C, time, freq) — match LCNN layout

        # CNN feature extraction
        x = self.stem(x)             # (B, 16, T, F)
        x = self.backbone(x)         # (B, 160, T//16, F//16)
        x = self.conv_head(x)        # (B, 256, T//16, F//16)

        # Reshape for LSTM: (B, T//16, 256 * F//16)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()   # (B, T, C, F)
        x = x.view(B, T, C * F)                   # (B, T, feat)

        # Bi-LSTM temporal modelling (residual, same as LCNN)
        lstm_out = self.m_before_pooling(x)        # (B, T, feat)
        fused = (lstm_out + x).mean(dim=1)         # (B, feat)  — temporal mean pool

        # Classifier
        fused = self.dropout(fused)
        emb = self.m_output_act(fused)             # (B, 1)
        return emb

    def _compute_score(self, feature_vec):
        """Sigmoid squash to [0, 1] score."""
        return torch.sigmoid(feature_vec).squeeze(1)

    def forward(self, x):
        """Returns raw logit (B, 1). Use _compute_score for probabilities."""
        return self._compute_embedding(x)


# ─────────────────────────────────────────────────────────────
# Alias so existing code calling LCNN(...) still works
# ─────────────────────────────────────────────────────────────
LCNN = MobileNetV3_AA


# ─────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Definition of model")
    model = LCNN(input_channels=3, num_coefficients=80)

    batch_size = 12
    mock_input = torch.rand((batch_size, 3, 80, 404))
    output = model(mock_input)
    print(output.shape)

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Score output
    score = model._compute_score(output)
    print(f"Score shape         : {score.shape}")
    print(f"Score range         : [{score.min():.4f}, {score.max():.4f}]")