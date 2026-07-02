"""
diffusion.py — Gaussian Diffusion process for DDPM.

Implements the forward (noising) and reverse (denoising) processes:
    - Linear β schedule from β₁=1e-4 to β_T=0.02, T=1000
    - Forward:  q(x_t | x_0) = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    - Reverse:  p_θ(x_{t-1} | x_t) via learned noise prediction
    - Training: simple MSE loss  L = ||ε - ε_θ(x_t, t, c)||²
"""

import torch
import torch.nn as nn
import numpy as np


class GaussianDiffusion:
    """
    Manages the diffusion noising/denoising schedule and sampling.

    Args:
        num_timesteps: total number of diffusion steps (T)
        beta_start:    β₁
        beta_end:      β_T
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.T = num_timesteps

        # Linear beta schedule
        betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        # Pre-compute all constants as float32 tensors
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas = torch.tensor(alphas, dtype=torch.float32)
        self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32)
        self.alphas_cumprod_prev = torch.tensor(alphas_cumprod_prev, dtype=torch.float32)

        # Coefficients for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Coefficients for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = torch.tensor(self.posterior_variance, dtype=torch.float32)

        # Coefficient to extract x_0 from noise prediction
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.beta_over_sqrt_one_minus_alpha_cumprod = (
            self.betas / self.sqrt_one_minus_alphas_cumprod
        )

    def _extract(self, tensor: torch.Tensor, t: torch.Tensor, shape):
        """Gather values from `tensor` at indices `t` and reshape for broadcasting."""
        out = tensor.to(t.device).gather(0, t)
        return out.view(-1, *([1] * (len(shape) - 1)))

    # -------------------------------------------------------------------
    # Forward process
    # -------------------------------------------------------------------
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Sample x_t from q(x_t | x_0).

        Args:
            x_0:   (B, C, H, W) clean images
            t:     (B,) timestep indices
            noise: (B, C, H, W) optional pre-generated noise

        Returns:
            x_t: (B, C, H, W) noised images
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    # -------------------------------------------------------------------
    # Training loss
    # -------------------------------------------------------------------
    def training_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the simple MSE denoising loss.

        L = E[ ||ε - ε_θ(x_t, t, c)||² ]

        Args:
            model:        the U-Net noise predictor
            x_0:          (B, C, H, W) clean images
            t:            (B,) random timestep indices
            class_labels: (B,) class labels

        Returns:
            scalar loss
        """
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = model(x_t, t, class_labels)
        return nn.functional.mse_loss(predicted_noise, noise)

    # -------------------------------------------------------------------
    # Reverse process  (sampling)
    # -------------------------------------------------------------------
    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single reverse diffusion step: sample x_{t-1} from p_θ(x_{t-1}|x_t).

        Uses the DDPM reverse formula:
            μ_θ = (1/√α_t) · (x_t - (β_t / √(1-ᾱ_t)) · ε_θ(x_t, t, c))
            x_{t-1} = μ_θ + √(posterior_var) · z   (z=0 if t=0)
        """
        predicted_noise = model(x_t, t, class_labels)

        # Compute mean
        coeff1 = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        coeff2 = self._extract(
            self.beta_over_sqrt_one_minus_alpha_cumprod, t, x_t.shape
        )
        mean = coeff1 * (x_t - coeff2 * predicted_noise)

        # Add noise (except at t=0)
        if t[0].item() > 0:
            variance = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple,
        class_labels: torch.Tensor,
        device: torch.device = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """
        Full reverse diffusion: generate images from pure noise.

        Args:
            model:        trained U-Net
            shape:        (B, C, H, W) desired output shape
            class_labels: (B,) class labels for conditional generation
            device:       target device
            progress:     if True, show a tqdm progress bar

        Returns:
            (B, C, H, W) generated images
        """
        if device is None:
            device = next(model.parameters()).device

        # Start from pure noise
        x = torch.randn(shape, device=device)
        class_labels = class_labels.to(device)

        timesteps = list(reversed(range(self.T)))
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="  Sampling", leave=False)

        for t_val in timesteps:
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, class_labels)

        return x


# ═══════════════════════════════════════════════════════════════════════════
# Quick sanity check
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    diff = GaussianDiffusion(num_timesteps=1000)
    print(f"Diffusion timesteps: {diff.T}")
    print(f"Beta range: [{diff.betas[0]:.6f}, {diff.betas[-1]:.6f}]")
    print(f"Alpha_bar range: [{diff.alphas_cumprod[-1]:.6f}, {diff.alphas_cumprod[0]:.6f}]")

    # Smoke test forward
    x0 = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 1000, (4,))
    xt = diff.q_sample(x0, t)
    print(f"q_sample: {x0.shape} -> {xt.shape}")
    print("Diffusion smoke test passed.")
