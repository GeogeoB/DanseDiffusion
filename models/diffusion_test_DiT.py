import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
from DiT import DiT
from dataset import SequenceDataset


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, mlp_dim: int = 256):
        super().__init__()

        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, t: torch.Tensor):
        """
        t: (B, L)  -> scalar timestep per token
        return: (B, L, dim)
        """

        B, L = t.size()

        t_emb = self._sinusoidal_embedding(t, self.dim)  # (B, L, dim)
        t_emb = self.mlp(t_emb)  # (B, L, dim)

        return t_emb

    def _sinusoidal_embedding(self, t, dim):
        """
        t: (B, )
        """
        device = t.device

        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )

        args = t.to(freqs.dtype) * freqs  # (B, L, dim/2)

        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)

        return emb


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int,
        conditioning_dim: int,
        mlp_dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        max_len: int = 512,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, conditioning_dim)

        self.timestep_embedding = TimestepEmbedding(
            dim=conditioning_dim, mlp_dim=mlp_dim
        )

        # Positional embedding (learned)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len + 1, conditioning_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conditioning_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(conditioning_dim, input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x : (B, L, D)
        t : (B,)
        """

        B, L, _ = x.shape

        # 1. projection
        x_proj = self.input_proj(x)  # (B, L, C)

        # 2. timestep embedding
        t_emb = self.timestep_embedding(t)  # (B, C)
        t_emb = t_emb.unsqueeze(1)  # (B, 1, C)

        # 3. concat (timestep as first token)
        x_cat = torch.cat([t_emb, x_proj], dim=1)  # (B, L+1, C)

        # 4. add positional embedding
        x_cat = x_cat + self.pos_emb[:, : L + 1, :]

        # 5. transformer
        h = self.transformer(x_cat)  # (B, L+1, C)

        # 6. remove timestep token
        h_x = h[:, 1:, :]  # (B, L, C)

        # 7. output projection
        out = self.output_proj(h_x)  # (B, L, D)

        return out


class CosineScheduler:
    def __init__(self, diffusion_steps: int, s: float = 0.008, device="cpu"):
        """
        Cosine schedule from Improved DDPM (Nichol & Dhariwal)
        """
        self.T = diffusion_steps
        self.device = device

        timesteps = torch.linspace(
            0, diffusion_steps, diffusion_steps + 1, device=device
        )

        # cosine alpha_bar
        def alpha_bar_fn(t):
            return (
                torch.cos(((t / diffusion_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            )

        alphas_bar = alpha_bar_fn(timesteps)
        alphas_bar = alphas_bar / alphas_bar[0]  # normalize so alpha_bar(0)=1

        # betas from alpha_bar
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = torch.clip(betas, 1e-8, 0.999)

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.timesteps = timesteps
        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar


def q_sample(x0, t, scheduler, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    alpha_bar = scheduler.alphas_bar[t]
    x_t = (
        torch.sqrt(alpha_bar).unsqueeze(-1) * x0
        + torch.sqrt(1 - alpha_bar).unsqueeze(-1) * noise
    )

    return x_t, noise


@torch.no_grad()
def sample(model, scheduler, T, shape, device):
    model.eval()
    x_t = torch.randn(shape, device=device)

    for t in reversed(range(T)):
        t_tensor = torch.full((shape[0], shape[1]), t, device=device, dtype=torch.long)

        beta_t = scheduler.betas[t]
        alpha_t = scheduler.alphas[t]
        alpha_bar_t = scheduler.alphas_bar[t]

        if t > 0:
            alpha_bar_prev = scheduler.alphas_bar[t - 1]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        # 1. predict x0
        x0_hat = model(x_t, t_tensor)

        # (optionnel mais très courant)
        # x0_hat = torch.clamp(x0_hat, -1, 1)

        # 2. compute coefficients
        coef1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t)
        coef2 = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)

        # 3. posterior mean
        mu = coef1 * x0_hat + coef2 * x_t

        # 4. sample
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(beta_t)
            x_t = mu + sigma * noise
        else:
            x_t = mu

    return x_t


def loss(x0_pred: torch.Tensor, x0: torch.Tensor, step: int):
    v_pred = x0_pred[:, 1:, :] - x0_pred[:, :-1, :]
    v_true = x0[:, 1:, :] - x0[:, :-1, :]

    l1 = F.mse_loss(x0_pred, x0)  # TODO add log SNR regulatization to this loss
    # l2 = F.mse_loss(x0_pred[:, 1:, :], x0_pred[:, :-1, :])
    l3 = F.mse_loss(v_pred, v_true)

    writer.add_scalar("loss/l1", l1.item(), step)
    # writer.add_scalar("loss/l2", l2.item(), step)
    writer.add_scalar("loss/l3", l3.item(), step)

    return l1 + 0.3 * l3


if __name__ == "__main__":
    dataset = SequenceDataset("dataset/latent", sequence_size=20)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DiT(
        n_layers=4,
        hidden_size=136,
        num_heads=2,
        conditioning_size=256,
        mlp_dim=512,
        mlp_ratio=4,
    ).to(device)

    T = 1000
    scheduler = CosineScheduler(T, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)

    warmup_steps = 10_000
    total_steps = 200_000

    scheduler_lr = SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(opt, T_max=total_steps - warmup_steps),
        ],
        milestones=[warmup_steps],
    )

    writer = SummaryWriter()

    data = next(iter(dataset))
    basic_skeleton_transformation = data["mhr"][:, 136:]
    basic_shape = data["shape"]
    basic_expr = data["expr"]

    step = 0
    for epoch in range(1000):
        for x0 in tqdm(dataloader):
            opt.zero_grad()

            x0 = x0["mhr"].to(device)
            x0 = x0[:, :, :136]

            B, L, D = x0.size()
            t = torch.randint(0, T, (B, L), device=device)

            x_t, noise = q_sample(x0, t, scheduler)

            x0_pred = model(x_t, t)

            l = loss(x0_pred, x0, step)

            l.backward()
            opt.step()

            step += 1
            writer.add_scalar("loss", l.item(), step)
            writer.add_scalar("lr", scheduler_lr.get_last_lr()[0], step)
            scheduler_lr.step()

        samples = sample(model, scheduler, T, (25, *x0.size()[1:]), device)

        for i, sample_i in enumerate(samples):
            sample_i = sample_i.unsqueeze(0)
            basic_skeleton_transformation_torch = torch.from_numpy(
                basic_skeleton_transformation
            ).to(device)
            basic_expanded = basic_skeleton_transformation_torch.unsqueeze(0)
            basic_expanded = basic_expanded
            x_sample = torch.cat([sample_i, basic_expanded], dim=2)

            mhr = x_sample.detach().cpu().squeeze().numpy()

            np.savez(
                f"output/output_{step}_{i}.npz",
                sequence_shape=basic_shape,
                sequence_mhr_model_params_latent=mhr,
                sequence_expr_params=basic_expr,
            )

        torch.save(model.state_dict(), f"output/model_{step}.pth")
