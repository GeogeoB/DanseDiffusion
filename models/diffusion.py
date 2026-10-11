"""Diffusion Forcing utilities: noise schedule, training loss and sampling.

Conventions:
    - the model predicts x0 (clean frames) from noisy frames and one timestep per frame
    - timesteps are integers in [0, T); t = -1 denotes a fully clean frame (alpha_bar = 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineScheduler:
    def __init__(self, diffusion_steps: int, s: float = 0.008, device="cpu"):
        """
        Cosine schedule from Improved DDPM (Nichol & Dhariwal)
        """
        self.T = diffusion_steps

        timesteps = torch.linspace(0, diffusion_steps, diffusion_steps + 1)
        alphas_bar = torch.cos(((timesteps / diffusion_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]

        betas = torch.clip(1 - (alphas_bar[1:] / alphas_bar[:-1]), 1e-8, 0.999)
        alphas = 1.0 - betas

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_bar = torch.cumprod(alphas, dim=0).to(device)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """alpha_bar for integer timesteps, with alpha_bar(-1) = 1 (clean)."""
        return torch.where(
            t >= 0, self.alphas_bar[t.clamp(min=0)], torch.ones_like(t, dtype=self.alphas_bar.dtype)
        )

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bar(t)
        return alpha_bar / (1 - alpha_bar).clamp(min=1e-12)


def q_sample(x0, t, scheduler: CosineScheduler, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    alpha_bar = scheduler.alpha_bar(t).unsqueeze(-1)
    return alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise


def sample_noise_levels(
    B: int,
    L: int,
    T: int,
    device,
    p_shared: float = 0.1,
    p_clean_prefix: float = 0.2,
):
    """
    Diffusion Forcing noise levels: an independent timestep per frame, plus
        - with prob p_shared: the same timestep for the whole window (full-sequence diffusion,
          needed to generate the first window from scratch)
        - with prob p_clean_prefix: a clean prefix of random length (autoregressive setting with
          clean context, exactly what is seen at sampling time)

    Returns:
        t: (B, L) timesteps, -1 for clean context frames
        context_mask: (B, L) True on clean context frames
    """
    t = torch.randint(0, T, (B, L), device=device)

    shared = torch.rand(B, 1, device=device) < p_shared
    t = torch.where(shared, t[:, :1], t)

    prefix = torch.rand(B, 1, device=device) < p_clean_prefix
    prefix_len = torch.randint(1, L, (B, 1), device=device)
    context_mask = prefix & (torch.arange(L, device=device)[None] < prefix_len)
    t = torch.where(context_mask, -1, t)

    return t, context_mask


def diffusion_loss(
    model: nn.Module,
    x0: torch.Tensor,
    t: torch.Tensor,
    context_mask: torch.Tensor,
    scheduler: CosineScheduler,
    min_snr_gamma: float | None = 5.0,
    velocity_weight: float = 0.0,
):
    """
    x0-prediction loss with truncated Min-SNR weighting: w = clamp(SNR, 1, gamma).
        - the upper bound (Min-SNR, https://arxiv.org/abs/2303.09556) stops nearly clean frames
          from dominating the loss
        - the lower bound keeps a plain x0 regression for very noisy frames, which is what the
          model uses to predict the next frames from the context
    Clean context frames are not supervised.
    """
    x_t = q_sample(x0, t, scheduler)
    x0_pred = model(x_t, t.clamp(min=0))

    target_mask = (~context_mask).float()

    if min_snr_gamma is not None:
        weight = scheduler.snr(t).clamp(1.0, min_snr_gamma)
    else:
        weight = torch.ones_like(t, dtype=x0.dtype)
    weight = weight * target_mask

    per_frame = ((x0_pred.float() - x0) ** 2).mean(-1)  # (B, L)
    loss_x0 = (weight * per_frame).sum() / target_mask.sum().clamp(min=1)

    losses = {"x0": loss_x0}
    loss = loss_x0

    if velocity_weight > 0:
        v_pred = x0_pred[:, 1:].float() - x0_pred[:, :-1].float()
        v_true = x0[:, 1:] - x0[:, :-1]
        loss_v = F.mse_loss(v_pred, v_true)
        losses["velocity"] = loss_v
        loss = loss + velocity_weight * loss_v

    losses["total"] = loss
    return losses


def ddim_step(x, x0_hat, t_cur, t_next, scheduler: CosineScheduler, eta: float = 0.0):
    """
    DDIM update with one timestep per frame. Frames with t_next == t_cur are left unchanged,
    frames with t_next == -1 become x0_hat.
    """
    ab_cur = scheduler.alpha_bar(t_cur).unsqueeze(-1)
    ab_next = scheduler.alpha_bar(t_next).unsqueeze(-1)

    eps = (x - ab_cur.sqrt() * x0_hat) / (1 - ab_cur).clamp(min=1e-12).sqrt()

    sigma = eta * (
        (1 - ab_next) / (1 - ab_cur).clamp(min=1e-12) * (1 - ab_cur / ab_next)
    ).clamp(min=0).sqrt()

    x_next = (
        ab_next.sqrt() * x0_hat
        + (1 - ab_next - sigma**2).clamp(min=0).sqrt() * eps
        + sigma * torch.randn_like(x)
    )

    moving = (t_next < t_cur).unsqueeze(-1)
    return torch.where(moving, x_next, x)


@torch.no_grad()
def denoise_chunk(
    model: nn.Module,
    scheduler: CosineScheduler,
    n_new: int,
    context: torch.Tensor | None,
    shape: tuple[int, int],
    n_steps: int = 50,
    pyramid: int = 0,
    eta: float = 0.0,
    context_t: int = -1,
    clip_x0: float | None = None,
    guidance: float = 1.0,
):
    """
    Generate n_new frames following the (clean) context frames.

    Noise levels follow the Diffusion Forcing pyramid scheduling matrix: frame j of the chunk
    lags `pyramid` denoising steps behind frame j-1 (pyramid=0: all frames denoised together).

    Args:
        context: (B, C, D) clean frames or None
        shape: (B, D)
        context_t: noise level given to the context (-1 = clean, >0 = noised context, which
            makes the model robust to its own errors at the cost of some fidelity)
        guidance: history guidance weight w (https://arxiv.org/abs/2502.06764). The x0
            prediction with context is mixed with the one where the context is replaced by pure
            noise (t = T-1, i.e. no history): x0 = x0_free + w * (x0_ctx - x0_free).
            w = 1: plain conditional sampling, w < 1: follows the history less (more diversity),
            w > 1: follows it more (more coherence). Costs a second model pass (same batch).
    """
    B, D = shape
    device = scheduler.alphas_bar.device

    # level 0 = pure noise (t = T-1), level n_steps = clean (t = -1)
    ddim_timesteps = torch.linspace(scheduler.T - 1, 0, n_steps, device=device).round().long()
    level_to_t = torch.cat([ddim_timesteps, torch.tensor([-1], device=device)])

    x = torch.randn(B, n_new, D, device=device)
    levels = torch.zeros(n_new, dtype=torch.long, device=device)
    frame_idx = torch.arange(n_new, device=device)

    if context is not None:
        n_ctx = context.shape[1]
        t_ctx = torch.full((B, n_ctx), context_t, device=device)
        ctx_in = q_sample(context, t_ctx, scheduler) if context_t >= 0 else context
    else:
        n_ctx = 0
    use_guidance = n_ctx > 0 and guidance != 1.0
    if use_guidance:
        ctx_free = torch.randn_like(context)  # history fully masked by noise
        t_free = torch.full((B, n_ctx), scheduler.T - 1, device=device)

    for k in range(n_steps + (n_new - 1) * pyramid):
        next_levels = (k + 1 - frame_idx * pyramid).clamp(0, n_steps)
        next_levels = torch.maximum(next_levels, levels)

        t_cur = level_to_t[levels].expand(B, -1)
        t_next = level_to_t[next_levels].expand(B, -1)

        if n_ctx > 0:
            x_in = torch.cat([ctx_in, x], dim=1)
            t_in = torch.cat([t_ctx, t_cur], dim=1)
        else:
            x_in, t_in = x, t_cur

        if use_guidance:
            x_in = torch.cat([x_in, torch.cat([ctx_free, x], dim=1)])
            t_in = torch.cat([t_in, torch.cat([t_free, t_cur], dim=1)])
            x0_ctx, x0_free = model(x_in, t_in.clamp(min=0))[:, n_ctx:].float().chunk(2)
            x0_hat = x0_free + guidance * (x0_ctx - x0_free)
        else:
            x0_hat = model(x_in, t_in.clamp(min=0))[:, n_ctx:].float()
        if clip_x0 is not None:
            x0_hat = x0_hat.clamp(-clip_x0, clip_x0)

        x = ddim_step(x, x0_hat, t_cur, t_next, scheduler, eta)
        levels = next_levels

    return x


@torch.no_grad()
def generate(
    model: nn.Module,
    scheduler: CosineScheduler,
    n_samples: int,
    n_frames: int,
    dim: int,
    window: int,
    n_context: int,
    prefix: torch.Tensor | None = None,
    **chunk_kwargs,
):
    """
    Autoregressive Diffusion Forcing generation with a sliding window: each chunk generates
    `window - n_context` new frames conditioned on the last `n_context` generated frames.
    The first window is generated from scratch (or after `prefix` if given).

    Returns: (n_samples, n_frames, dim)
    """
    if not 0 < n_context < window:
        raise ValueError(f"need 0 < n_context < window, got {n_context=} {window=}")

    model.eval()
    shape = (n_samples, dim)

    if prefix is None:
        frames = denoise_chunk(model, scheduler, window, None, shape, **chunk_kwargs)
    else:
        frames = prefix

    while frames.shape[1] < n_frames:
        context = frames[:, -n_context:]
        new = denoise_chunk(model, scheduler, window - n_context, context, shape, **chunk_kwargs)
        frames = torch.cat([frames, new], dim=1)

    return frames[:, :n_frames]
