import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# -----------------------
# Dataset
# -----------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, N: int):
        self.x, _ = make_moons(n_samples=N, noise=0.05, random_state=42)
        self.x = torch.tensor(self.x, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


# -----------------------
# Scheduler (corrigé + propre)
# -----------------------
class LinearScheduler:
    def __init__(self, diffusion_steps: int, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = diffusion_steps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, device=device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar

    def extract(self, t, tensor):
        """
        t: (B,)
        tensor: (T,)
        """
        return tensor.gather(0, t)
    
import torch

class CosineScheduler:
    def __init__(self, diffusion_steps: int, s: float = 0.008, device="cpu"):
        """
        Cosine schedule from Improved DDPM (Nichol & Dhariwal)
        """
        self.T = diffusion_steps
        self.device = device

        timesteps = torch.linspace(0, diffusion_steps, diffusion_steps + 1, device=device)
        
        # cosine alpha_bar
        def alpha_bar_fn(t):
            return torch.cos(((t / diffusion_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2

        alphas_bar = alpha_bar_fn(timesteps)
        alphas_bar = alphas_bar / alphas_bar[0]  # normalize so alpha_bar(0)=1

        # betas from alpha_bar
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = torch.clip(betas, 1e-8, 0.999)

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar

    def extract(self, t, tensor):
        """
        t: (B,)
        tensor: (T,)
        """
        return tensor.gather(0, t)

# -----------------------
# Model (corrigé normalisation t)
# -----------------------
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, t):
        # IMPORTANT: normalisation cohérente avec code 2
        t = t.float().unsqueeze(1) / 1000.0
        x = torch.cat([x, t], dim=1)
        return self.net(x)


# -----------------------
# Forward diffusion (corrigé stabilité)
# -----------------------
def q_sample(x0, t, scheduler, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    alpha_bar = scheduler.alphas_bar[t].unsqueeze(1)
    x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

    return x_t, noise


# -----------------------
# Sampling (corrigé cohérence device)
# -----------------------
@torch.no_grad()
def sample(model, scheduler, T, shape, device):
    model.eval()
    x = torch.randn(shape, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        beta_t = scheduler.betas[t]
        alpha_t = scheduler.alphas[t]
        alpha_bar_t = scheduler.alphas_bar[t]

        v = model(x, t_batch)

        alpha_bar_t = scheduler.alphas_bar[t]
        alpha_bar_t = alpha_bar_t.view(1, 1)

        eps = torch.sqrt(1 - alpha_bar_t) * x + torch.sqrt(alpha_bar_t) * v

        x = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps
        )

        if t > 0:
            x += torch.sqrt(beta_t) * torch.randn_like(x)

    return x

@torch.no_grad()
def ddim_sample(model, scheduler, T, shape, device, eta=0.0):
    """
    DDIM sampler
    eta = 0 -> deterministic
    eta > 0 -> stochastic
    """
    model.eval()

    x = torch.randn(shape, device=device)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        alpha_bar_t = scheduler.alphas_bar[t]
        alpha_bar_prev = scheduler.alphas_bar[t - 1] if t > 0 else torch.tensor(1.0, device=device)

        alpha_t = scheduler.alphas[t]

        # -----------------------
        # model predicts v
        # -----------------------
        v = model(x, t_batch)

        # v -> epsilon
        alpha_bar_t_ = alpha_bar_t.view(1, 1)

        eps = torch.sqrt(1 - alpha_bar_t_) * x + torch.sqrt(alpha_bar_t_) * v

        # -----------------------
        # x0 estimate
        # -----------------------
        x0 = (x - torch.sqrt(1 - alpha_bar_t_) * eps) / torch.sqrt(alpha_bar_t_)

        # -----------------------
        # DDIM direction
        # -----------------------
        if t > 0:
            alpha_bar_prev_ = alpha_bar_prev.view(1, 1)

            sigma_t = eta * torch.sqrt(
                (1 - alpha_bar_prev_) / (1 - alpha_bar_t_) *
                (1 - alpha_bar_t_ / alpha_bar_prev_)
            )

            noise = torch.randn_like(x) if eta > 0 else 0.0

            x = (
                torch.sqrt(alpha_bar_prev_) * x0 +
                torch.sqrt(1 - alpha_bar_prev_ - sigma_t**2) * eps +
                sigma_t * noise
            )
        else:
            x = x0

    return x

def plot_samples(x):
    x = x.detach().cpu().numpy()

    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], s=5)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.title("DDPM samples")
    return plt.gcf()


# -----------------------
# Training (corrigé)
# -----------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    T = 200          # plus proche code 2 → stabilité
    dataset = Dataset(5000)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    scheduler = CosineScheduler(T, device=device)
    model = Model().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter()

    step = 0

    for epoch in range(1000):
        for x0 in loader:
            x0 = x0.to(device)

            B = x0.size(0)
            t = torch.randint(0, T, (B,), device=device)

            x_t, noise = q_sample(x0, t, scheduler)

            alpha_bar = scheduler.alphas_bar[t].unsqueeze(1)
            v_target = torch.sqrt(alpha_bar) * noise - torch.sqrt(1 - alpha_bar) * x0

            v_pred = model(x_t, t)

            loss = F.mse_loss(v_pred, v_target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            writer.add_scalar("loss", loss.item(), step)
            step += 1

        # sampling
        samples = ddim_sample(
            model,
            scheduler,
            T,
            shape=(1024, 2),
            device=device
        )

        fig = plot_samples(samples)
        writer.add_figure("samples", fig, global_step=step)
        plt.close(fig)

    writer.close()


if __name__ == "__main__":
    train()