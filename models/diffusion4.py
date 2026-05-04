import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

from dataset import SequenceDataset

# -----------------------
# Scheduler (identique)
# -----------------------
class CosineScheduler:
	def __init__(self, diffusion_steps: int, s: float = 0.008, device="cpu"):
		self.T = diffusion_steps
		self.device = device

		timesteps = torch.linspace(0, diffusion_steps, diffusion_steps + 1, device=device)

		def alpha_bar_fn(t):
			return torch.cos(((t / diffusion_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2

		alphas_bar = alpha_bar_fn(timesteps)
		alphas_bar = alphas_bar / alphas_bar[0]

		betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
		betas = torch.clip(betas, 1e-8, 0.999)

		alphas = 1.0 - betas
		alphas_bar = torch.cumprod(alphas, dim=0)

		self.betas = betas
		self.alphas = alphas
		self.alphas_bar = alphas_bar


# -----------------------
# Model (MLP simple)
# -----------------------
class Model(nn.Module):
	def __init__(self, dim):
		super().__init__()

		self.net = nn.Sequential(
			nn.Linear(dim + 1, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, dim)
		)

	def forward(self, x, t):
		B, L, D = x.shape

		t = t.float().unsqueeze(-1) / 1000.0
		t = t.unsqueeze(1).repeat(1, L, 1)

		x = torch.cat([x, t], dim=-1)
		return self.net(x)


# -----------------------
# Forward diffusion
# -----------------------
def q_sample(x0, t, scheduler, noise=None):
	if noise is None:
		noise = torch.randn_like(x0)

	alpha_bar = scheduler.alphas_bar[t].view(-1, 1, 1)

	x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
	return x_t, noise


# -----------------------
# Sampling (DDIM stable)
# -----------------------
@torch.no_grad()
def ddim_sample(model, scheduler, T, shape, device):
	model.eval()

	x = torch.randn(shape, device=device)
	B, L, D = shape

	for t in reversed(range(T)):
		t_batch = torch.full((B,), t, device=device, dtype=torch.long)

		alpha_bar_t = scheduler.alphas_bar[t]
		alpha_bar_prev = scheduler.alphas_bar[t - 1] if t > 0 else torch.tensor(1.0, device=device)

		v = model(x, t_batch)

		alpha_bar_t_ = alpha_bar_t.view(1, 1, 1)

		eps = torch.sqrt(1 - alpha_bar_t_) * x + torch.sqrt(alpha_bar_t_) * v

		x0 = (x - torch.sqrt(1 - alpha_bar_t_) * eps) / torch.sqrt(alpha_bar_t_)

		if t > 0:
			alpha_bar_prev_ = alpha_bar_prev.view(1, 1, 1)

			x = (
				torch.sqrt(alpha_bar_prev_) * x0 +
				torch.sqrt(1 - alpha_bar_prev_) * eps
			)
		else:
			x = x0

	return x


# -----------------------
# Training
# -----------------------
def train():
	device = "cuda" if torch.cuda.is_available() else "cpu"

	T = 200

	dataset = SequenceDataset("dataset/latent", sequence_size=1)
	loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

	# récupérer une batch pour connaître D
	sample = next(iter(loader))
	x0 = sample["mhr"]
	D = x0.shape[-1]

	scheduler = CosineScheduler(T, device=device)
	model = Model(D).to(device)

	opt = torch.optim.Adam(model.parameters(), lr=1e-3)
	writer = SummaryWriter()

	step = 0

	os.makedirs("output", exist_ok=True)

	# récupérer une référence du dataset
	data = next(iter(dataset))
	basic_skeleton_transformation = data["mhr"][:, D:][0]  # partie non générée
	basic_shape = data["shape"][0]
	basic_expr = data["expr"][0]

	for epoch in range(200):
		for batch in tqdm(loader):
			x0 = batch["mhr"].to(device)

			# NORMALISATION IMPORTANTE
			x0 = (x0 - x0.mean()) / (x0.std() + 1e-6)

			B, L, D = x0.shape
			t = torch.randint(0, T, (B,), device=device)

			x_t, noise = q_sample(x0, t, scheduler)

			alpha_bar = scheduler.alphas_bar[t].view(-1, 1, 1)
			v_target = torch.sqrt(alpha_bar) * noise - torch.sqrt(1 - alpha_bar) * x0

			v_pred = model(x_t, t)

			loss = F.mse_loss(v_pred, v_target)

			opt.zero_grad()
			loss.backward()
			opt.step()

			writer.add_scalar("loss", loss.item(), step)
			step += 1

		samples = ddim_sample(
			model,
			scheduler,
			T,
			shape=(1, L, D),
			device=device
		)

		samples = ddim_sample(
			model,
			scheduler,
			T,
			shape=(1, L, D),
			device=device
		)

		# -> numpy
		samples_np = samples.detach().cpu().numpy()

		# -----------------------
		# reconstruction mhr complet
		# -----------------------
		basic_skeleton_transformation_torch = torch.from_numpy(
			basic_skeleton_transformation
		).to(device)

		basic_expanded = basic_skeleton_transformation_torch.unsqueeze(0).unsqueeze(1)
		basic_expanded = basic_expanded.repeat(1, L, 1)

		x_sample = torch.cat([samples, basic_expanded], dim=2)

		mhr = x_sample.detach().cpu().squeeze().numpy()

		# -----------------------
		# sauvegarde format identique
		# -----------------------
		np.savez(
			f"output/output_{step}.npz",
			shape=np.tile(basic_shape, (L, 1)),
			mhr=mhr,
			expr=np.tile(basic_expr, (L, 1))
		)

		writer.close()


if __name__ == "__main__":
	train()