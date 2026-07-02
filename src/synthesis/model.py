"""
model.py — Class-conditional U-Net for DDPM on CIFAR-10 (32×32).

Architecture overview:
    - Sinusoidal time embedding  → MLP → time_emb
    - Class embedding (nn.Embedding) added to time_emb
    - Encoder: 3 down-blocks  [128, 256, 256]  with residual convolutions
    - Self-attention at 8×8 resolution
    - Decoder: 3 up-blocks with skip connections
    - ~11M parameters — fits in 6 GB VRAM with batch 128
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Create sinusoidal positional embeddings for diffusion timesteps.

    Args:
        timesteps: (B,) int tensor of timestep indices
        dim:       embedding dimension (must be even)

    Returns:
        (B, dim) float tensor
    """
    assert dim % 2 == 0, "Embedding dimension must be even"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SelfAttention(nn.Module):
    """Single-head self-attention for spatial feature maps."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)          # (B, C, HW)
        qkv = self.qkv(h).chunk(3, dim=1)            # 3 × (B, C, HW)
        q, k, v = qkv
        attn = (q.transpose(1, 2) @ k) * self.scale  # (B, HW, HW)
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(1, 2))              # (B, C, HW)
        out = self.proj(out).view(B, C, H, W)
        return x + out


# ═══════════════════════════════════════════════════════════════════════════
# Residual Block
# ═══════════════════════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    """
    Residual block with time + class conditioning.

    Conv → GroupNorm → SiLU → Conv → GroupNorm → SiLU + skip
    Time/class embedding is projected and added after the first norm.
    """

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Project embedding to match out_ch
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch),
        )

        # Skip connection (1×1 conv if channel mismatch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # Add conditioning: emb is (B, out_ch), broadcast to (B, out_ch, 1, 1)
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# ═══════════════════════════════════════════════════════════════════════════
# Down / Up blocks
# ═══════════════════════════════════════════════════════════════════════════
class DownBlock(nn.Module):
    """Two residual blocks + optional attention + 2× downsample."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, has_attn: bool = False):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, emb_dim)
        self.attn = SelfAttention(out_ch) if has_attn else nn.Identity()
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        h = self.res1(x, emb)
        h = self.res2(h, emb)
        h = self.attn(h)
        return self.downsample(h), h   # return both downsampled and skip


class UpBlock(nn.Module):
    """2× upsample + concatenate skip + two residual blocks + optional attention."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, has_attn: bool = False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        # After concat with skip, channels = in_ch + out_ch
        self.res1 = ResBlock(in_ch + out_ch, out_ch, emb_dim)
        self.res2 = ResBlock(out_ch, out_ch, emb_dim)
        self.attn = SelfAttention(out_ch) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, emb: torch.Tensor):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        h = self.res1(x, emb)
        h = self.res2(h, emb)
        h = self.attn(h)
        return h


# ═══════════════════════════════════════════════════════════════════════════
# Bottleneck
# ═══════════════════════════════════════════════════════════════════════════
class MidBlock(nn.Module):
    """ResBlock → SelfAttention → ResBlock at the bottom of the U-Net."""

    def __init__(self, channels: int, emb_dim: int):
        super().__init__()
        self.res1 = ResBlock(channels, channels, emb_dim)
        self.attn = SelfAttention(channels)
        self.res2 = ResBlock(channels, channels, emb_dim)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        x = self.res1(x, emb)
        x = self.attn(x)
        x = self.res2(x, emb)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Full U-Net
# ═══════════════════════════════════════════════════════════════════════════
class UNet(nn.Module):
    """
    Class-conditional U-Net for denoising diffusion on 32×32 images.

    Args:
        in_channels:    input image channels (3 for RGB)
        base_channels:  channel width at the first level
        channel_mults:  multipliers for each encoder level
        num_classes:    number of conditioning classes
        time_emb_dim:   dimension of the time+class embedding
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: tuple = (1, 2, 2),
        num_classes: int = 10,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        # --- Time embedding MLP ---
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # --- Class embedding ---
        self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        # --- Initial convolution ---
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # --- Encoder ---
        ch_sizes = [base_channels * m for m in channel_mults]  # [128, 256, 256]
        self.downs = nn.ModuleList()
        in_ch = base_channels
        for i, out_ch in enumerate(ch_sizes):
            # Attention at the lowest resolution (last down block)
            has_attn = (i == len(ch_sizes) - 1)
            self.downs.append(DownBlock(in_ch, out_ch, time_emb_dim, has_attn))
            in_ch = out_ch

        # --- Bottleneck ---
        self.mid = MidBlock(ch_sizes[-1], time_emb_dim)

        # --- Decoder ---
        self.ups = nn.ModuleList()
        for i, out_ch in enumerate(reversed(ch_sizes)):
            up_in = ch_sizes[-1] if i == 0 else ch_sizes[len(ch_sizes) - i]
            has_attn = (i == 0)  # attention at lowest resolution
            self.ups.append(UpBlock(up_in, out_ch, time_emb_dim, has_attn))

        # --- Output projection ---
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:            (B, 3, 32, 32) noisy images
            t:            (B,) int timestep indices
            class_labels: (B,) int class labels [0..9]

        Returns:
            (B, 3, 32, 32) predicted noise
        """
        # Compute conditioning embedding
        t_emb = sinusoidal_embedding(t, self.init_conv.out_channels)  # (B, base_ch)
        emb = self.time_mlp(t_emb) + self.class_emb(class_labels)    # (B, emb_dim)

        # Initial convolution
        h = self.init_conv(x)  # (B, 128, 32, 32)

        # Encoder
        skips = []
        for down in self.downs:
            h, skip = down(h, emb)
            skips.append(skip)

        # Bottleneck
        h = self.mid(h, emb)

        # Decoder
        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip, emb)

        # Output
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# ═══════════════════════════════════════════════════════════════════════════
# Quick sanity check
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model = UNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet parameter count: {n_params:,}")

    # Smoke test
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    c = torch.randint(0, 10, (2,))
    out = model(x, t, c)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Output shape mismatch!"
    print("Smoke test passed.")
