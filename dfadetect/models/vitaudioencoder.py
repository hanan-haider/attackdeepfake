"""
ViT-based Audio Deepfake Detector
Drop-in replacement for LCNN baseline.
Input:  (batch, channels, num_coefficients, time_frames)  — same as LCNN
Output: (batch, 1) feature vector  → pass through sigmoid for score
"""

import torch
import torch.nn as nn
import math


class PatchEmbed(nn.Module):
    """Split spectrogram into patches and embed them."""
    def __init__(self, in_channels=3, patch_size=(4, 4), embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) → (B, embed_dim, H/ph, W/pw) → (B, N, embed_dim)
        x = self.proj(x)                      # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)      # (B, N, D)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.drop(attn_out))


class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ff   = FeedForward(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = self.attn(x)
        x = self.ff(x)
        return x


class ViTAudioEncoder(nn.Module):
    """
    ViT-based audio deepfake detector.

    Matches LCNN interface:
      Input:  (batch, channels, num_coefficients, time_frames)
      Output: (batch, 1)   — raw logit, apply sigmoid for probability

    Args:
        input_channels   : number of input channels (default 3, same as LCNN)
        num_coefficients : frequency bins (default 80, same as LCNN)
        patch_size       : (freq_patch, time_patch) — controls token count
        embed_dim        : transformer hidden dimension
        num_heads        : attention heads (must divide embed_dim)
        depth            : number of transformer blocks
        mlp_ratio        : FFN hidden dim multiplier
        dropout          : dropout probability
    """
    def __init__(
        self,
        input_channels=3,
        num_coefficients=80,
        patch_size=(4, 4),
        embed_dim=256,
        num_heads=8,
        depth=6,
        mlp_ratio=4.0,
        dropout=0.1,
        **kwargs           # absorbs extra kwargs for drop-in compatibility
    ):
        super().__init__()
        self.v_emd_dim = 1   # keep same attribute as LCNN

        # 1. Patch embedding — same role as LCNN's first Conv2d stack
        self.patch_embed = PatchEmbed(input_channels, patch_size, embed_dim)

        # 2. Learnable [CLS] token + positional embedding (registered as buffer)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embedding will be sized dynamically on first forward pass
        self.pos_embed = None
        self._pos_embed_shape = None
        self.embed_dim = embed_dim

        # 3. Transformer encoder blocks — replaces LCNN conv stack + BLSTM
        self.blocks = nn.Sequential(*[
            ViTBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # 4. Classification head — same role as LCNN's m_output_act Linear
        self.head = nn.Linear(embed_dim, 1)

        # Weight init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _build_pos_embed(self, num_patches, device):
        """Build sinusoidal positional embeddings dynamically."""
        N = num_patches + 1   # +1 for CLS token
        pe = torch.zeros(1, N, self.embed_dim, device=device)
        pos = torch.arange(N, dtype=torch.float, device=device).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.float, device=device)
            * -(math.log(10000.0) / self.embed_dim)
        )
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe

    def _compute_embedding(self, x):
        """Mirrors LCNN's _compute_embedding — same name for compatibility."""
        B = x.shape[0]

        # Patch embedding
        tokens = self.patch_embed(x)          # (B, N, D)
        N = tokens.shape[1]

        # CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)   # (B, N+1, D)

        # Positional embedding (build once, cache shape)
        if self._pos_embed_shape != N:
            self.pos_embed = self._build_pos_embed(N, x.device)
            self._pos_embed_shape = N
        tokens = tokens + self.pos_embed

        # Transformer blocks
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)

        # CLS token → classification head
        cls_out = tokens[:, 0]                # (B, D)
        return self.head(cls_out)             # (B, 1)

    def _compute_score(self, feature_vec):
        """Mirrors LCNN's _compute_score."""
        return torch.sigmoid(feature_vec).squeeze(1)

    def forward(self, x):
        return self._compute_embedding(x)


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ViTAudioEncoder(input_channels=3, num_coefficients=80)
    batch_size = 12
    mock_input = torch.rand((batch_size, 3, 80, 404))   # identical to LCNN test
    output = model(mock_input)
    print("Output shape:", output.shape)   # expect: torch.Size([12, 1])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")