import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter


from DiT import DiT
from dataset import SequenceDataset

import math


class CosineScheduling:
    def __init__(self, diffusion_steps: int, s: float = 0.008, device="cpu"):
        self.diffusion_steps = diffusion_steps
        self.s = s
        self.device = device

        # précompute alpha_bar
        self.alpha_bar = self._compute_alpha_bar()  # (T+1,)

        # alpha_t = alpha_bar_t / alpha_bar_{t-1}
        self.alphas = self.alpha_bar[1:] / self.alpha_bar[:-1]  # (T,)
        self.betas = 1 - self.alphas  # (T,)

    def _compute_alpha_bar(self):
        t = torch.linspace(0, self.diffusion_steps, self.diffusion_steps + 1, device=self.device)
        f = (t / self.diffusion_steps + self.s) / (1 + self.s)

        alpha_bar = torch.cos(f * math.pi / 2) ** 2
        alpha_bar = (
            alpha_bar / alpha_bar[0]
        )  # normalisation pour avoir alpha_bar[0] = 1

        return alpha_bar  # (T+1,)

    def apply(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x: (B, L, D)
            t: (B, L)  timestep par token
        """
        noise = torch.randn_like(x)

        # récupérer alpha_bar pour chaque token
        alpha_bar_t = self.alpha_bar[t]  # (B, L)

        # ajouter dimension pour matcher D
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)  # (B, L, 1)

        x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

        return x_t, noise, alpha_bar_t

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DiT(
        n_layers=8,
        hidden_size=204,
        num_heads=6,
        conditioning_size=256,
        mlp_dim=256,
        mlp_ratio=4
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)

    warmup_steps = 1000
    total_steps = 10_000

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
            ),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps),
        ],
        milestones=[warmup_steps],
    )

    writer = SummaryWriter("runs/dit_diffusion")

    dataset = SequenceDataset("dataset/latent", sequence_size=25)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4
    )

    diffusion_steps = 1000
    cosine_scheduling = CosineScheduling(diffusion_steps, device=device)

    criterion = torch.nn.MSELoss()

    global_step = 0
    model.train()
    for epoch in range(100):
        for batch in dataloader:
            x0 = batch["mhr"].to(device)
            B, L, _ = x0.size()

            t = torch.randint(
                low=0,
                high=diffusion_steps,
                size=(B, L),
                device=x0.device,
                dtype=torch.long
            )

            mhr_noised, noise, alpha_bar_t = cosine_scheduling.apply(x0, t)

            predicted_mhr = model(mhr_noised, t) #x0 prediction

            loss = criterion(mhr_noised - predicted_mhr, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            writer.add_scalar("loss/train", loss.item(), global_step)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)

            if global_step % 100 == 0:
                print(f"step {global_step} | loss {loss.item():.4f}")

            global_step += 1

    writer.close()