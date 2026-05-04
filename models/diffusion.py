import time
from typing import Tuple

import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from DiT import DiT
from dataset import SequenceDataset

import math
import numpy as np

from dataset import ToyDataset

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
		t = torch.linspace(0, self.diffusion_steps + 1, self.diffusion_steps + 1, device=self.device)

		print(f"{t.size() = }")
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
		# alpha_bar_t = self.alpha_bar[t]  # (B, L)
		alpha_bar_t = self.alpha_bar[t]

		# ajouter dimension pour matcher D
		alpha_bar_t = alpha_bar_t.unsqueeze(-1)  # (B, L, 1)

		x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

		return x_t, noise, alpha_bar_t

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


# def sample(
# 	model: torch.nn.Module,
# 	diffusion_steps: int,
# 	sampling_steps: int,
# 	cosine_scheduling: CosineScheduling,
# 	shape,
# 	eta: float = 0.0,
# 	device="cuda",
# ):
# 	model.eval()

# 	alpha_bars = cosine_scheduling.alpha_bar.to(device)

# 	# 🔷 timesteps réguliers
# 	step_indices = torch.linspace(diffusion_steps - 1, 0, sampling_steps).to(torch.long)
# 	step_indices = torch.unique_consecutive(step_indices)

# 	x = torch.randn(shape, device=device)
# 	print(f"{x.norm().item()}")

# 	with torch.no_grad():
# 		for i in range(len(step_indices) - 1):
# 			t = step_indices[i]
# 			t_prev = step_indices[i + 1]

# 			alpha_bar_t = alpha_bars[t].view(1, 1, 1)
# 			alpha_bar_prev = alpha_bars[t_prev].view(1, 1, 1)

# 			eps = 1e-8
# 			alpha_bar_t = alpha_bar_t.clamp(min=eps)
# 			alpha_bar_prev = alpha_bar_prev.clamp(min=eps)

# 			B, L, _ = x.shape
# 			t_tensor = torch.full((B, L), t.item(), device=device, dtype=torch.long)

# 			eps = model(x, t_tensor)

# 			x0_pred = (
# 				x - torch.sqrt(torch.clamp(1 - alpha_bar_t, min=1e-8)) * eps
# 			) / torch.sqrt(torch.clamp(alpha_bar_t, min=1e-8))

# 			x0_pred = x0_pred.clamp(-4, 4)

# 			pred_dir = torch.sqrt(torch.clamp(1 - alpha_bar_prev, min=1e-8)) * eps

# 			sigma = eta * torch.sqrt(
# 				torch.clamp(
# 					(1 - alpha_bar_prev) / (1 - alpha_bar_t)
# 					* (1 - alpha_bar_t / alpha_bar_prev),
# 					min=1e-8
# 				)
# 			)

# 			noise = torch.randn_like(x)

# 			# x = (
# 			#     torch.sqrt(alpha_bar_prev) * x0_pred +
# 			#     torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma**2, min=1e-8)) * eps +
# 			#     sigma * noise
# 			# )

# 			x = (
# 				torch.sqrt(alpha_bar_prev) * x0_pred +
# 				torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps +
# 				sigma * noise
# 			)

# 			print("eps norm:", eps.norm().item())
# 			print("x0_pred norm:", x0_pred.norm().item())

# 			print("NaN:", torch.isnan(x).any().item())
# 			print("Inf:", torch.isinf(x).any().item())

# 			print("mean:", x.mean().item())
# 			print("std:", x.std().item())
# 			print("min:", x.min().item())
# 			print("max:", x.max().item())

# 			print("norm:", x.norm().item())
# 			print()
# 		return x

def sample(
	model: torch.nn.Module,
	diffusion_steps: int,
	sampling_steps: int,
	cosine_scheduling,
	shape,
	eta: float = 0.0,
	device="cuda",
):
	model.eval()

	alpha_bars = cosine_scheduling.alpha_bar.to(device)

	# timesteps
	step_indices = torch.linspace(
		diffusion_steps - 1, 0, sampling_steps, device=device
	).long()
	step_indices = torch.unique_consecutive(step_indices)

	x = torch.randn(shape, device=device)

	with torch.no_grad():
		for i in range(len(step_indices) - 1):
			t = step_indices[i]
			t_prev = step_indices[i + 1]

			alpha_bar_t = alpha_bars[t]
			alpha_bar_prev = alpha_bars[t_prev]

			# sécurité numérique
			alpha_bar_t = alpha_bar_t.clamp(min=1e-8)
			alpha_bar_prev = alpha_bar_prev.clamp(min=1e-8)

			B, L, _ = x.shape

			# 🔴 timestep GLOBAL (important)
			t_tensor = torch.full((B,), t.item(), device=device, dtype=torch.long)
			t_tensor = t_tensor[:, None].repeat(1, L)

			# prédiction du bruit
			eps = model(x, t_tensor)

			# 🔷 prédiction x0
			sqrt_alpha_t = torch.sqrt(alpha_bar_t)
			sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_bar_t)

			x0_pred = (x - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t
			x0_pred = x0_pred.clamp(-4, 4)

			# 🔷 DDIM sigma (correct)
			sigma = eta * torch.sqrt(
				((1 - alpha_bar_prev) / (1 - alpha_bar_t))
				* (1 - alpha_bar_t / alpha_bar_prev)
			)

			# 🔴 clamp pour éviter NaN
			sigma = sigma.clamp(min=0.0)

			noise = torch.randn_like(x)

			# 🔷 direction vers x_t-1
			dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps

			# 🔷 update final
			x = (
				torch.sqrt(alpha_bar_prev) * x0_pred
				+ dir_xt
				+ sigma * noise
			)

			print("eps norm:", eps.norm().item())
			print("x0_pred norm:", x0_pred.norm().item())

			print("NaN:", torch.isnan(x).any().item())
			print("Inf:", torch.isinf(x).any().item())

			print("mean:", x.mean().item())
			print("std:", x.std().item())
			print("min:", x.min().item())
			print("max:", x.max().item())

			print("norm:", x.norm().item())
			print()

	return x

# @torch.no_grad()
# def ddim_sample(
#     model,
#     diffusion_steps,
#     cosine_scheduling,
#     shape,
#     sampling_steps=100,
#     eta=0.0,
#     device="cuda",
# ):
#     model.eval()

#     alpha_bars = cosine_scheduling.alpha_bar.to(device)

#     B, L, D = shape
#     x = torch.randn(shape, device=device)

#     # timesteps espacés
#     step_indices = torch.linspace(diffusion_steps - 1, 0, sampling_steps, device=device).long()
#     step_indices = torch.unique_consecutive(step_indices)

#     for i in range(len(step_indices) - 1):
#         t = step_indices[i]
#         t_prev = step_indices[i + 1]

#         alpha_bar_t = alpha_bars[t].clamp(min=1e-8)
#         alpha_bar_prev = alpha_bars[t_prev].clamp(min=1e-8)

#         t_tensor = torch.full((B, L), t, device=device, dtype=torch.long)

#         eps = model(x, t_tensor)

#         # x0 prediction
#         x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
#         x0 = x0.clamp(-4, 4)

#         # DDIM sigma
#         sigma = eta * torch.sqrt(
#             (1 - alpha_bar_prev) / (1 - alpha_bar_t)
#             * (1 - alpha_bar_t / alpha_bar_prev)
#         )
#         sigma = sigma.clamp(min=0.0)

#         # direction toward x_{t-1}
#         dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps

#         noise = torch.randn_like(x)

#         x = (
#             torch.sqrt(alpha_bar_prev) * x0
#             + dir_xt
#             + sigma * noise
#         )

#     return x

@torch.no_grad()
def ddpm_sample(model, diffusion_steps, cosine_scheduling, shape, device="cuda"):
	model.eval()

	alphas = cosine_scheduling.alphas.to(device)
	print(f"{alphas.size() = }")
	alpha_bars = cosine_scheduling.alpha_bar.to(device)

	B, L, D = shape
	x = torch.randn(shape, device=device)

	with torch.no_grad():
		for t in tqdm(reversed(range(diffusion_steps))):
			t_tensor = torch.full((B, L), t, device=device, dtype=torch.long)

			eps = model(x, t_tensor)

			alpha_t = alphas[t].clamp(min=1e-8)
			alpha_bar_t = alpha_bars[t].clamp(min=1e-8)
			alpha_bar_prev = alpha_bars[t - 1] if t > 0 else torch.tensor(1.0, device=device)

			beta_t = 1 - alpha_t

			# posterior variance (IMPORTANT)
			beta_tilde = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
			beta_tilde = beta_tilde.clamp(min=1e-8)

			mean = (1 / torch.sqrt(alpha_t)) * (
				x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps
			)

			if t > 0:
				z = torch.randn_like(x)
				x = mean + torch.sqrt(beta_tilde) * z
			else:
				x = mean

	return x

@torch.no_grad()
def ddim_sample(
	model,
	diffusion_steps,
	cosine_scheduling,
	shape,
	sampling_steps=100,
	eta=0.0,
	device="cuda",
):
	# model.eval()

	alpha_bars = cosine_scheduling.alpha_bar.to(device)

	B, L, D = shape
	x = torch.randn(shape, device=device)

	step_indices = torch.linspace(
		diffusion_steps - 1, 0, sampling_steps, device=device
	).long()
	step_indices = torch.unique_consecutive(step_indices)

	for i in range(len(step_indices) - 1):
		t = step_indices[i].item()
		t_prev = step_indices[i + 1].item()

		alpha_bar_t = alpha_bars[t].clamp(min=1e-8).view(1, 1, 1)
		alpha_bar_prev = alpha_bars[t_prev].clamp(min=1e-8).view(1, 1, 1)

		t_tensor = torch.full((B, L), t, device=device, dtype=torch.long)

		eps = model(x, t_tensor)

		# x0 prediction
		sqrt_alpha_t = torch.sqrt(alpha_bar_t)
		sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_bar_t)

		x0 = (x - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t
		x0 = x0.clamp(-1., 1.)

		# sigma
		sigma = eta * torch.sqrt(
			torch.clamp(
				(1 - alpha_bar_prev) / (1 - alpha_bar_t)
				* (1 - alpha_bar_t / alpha_bar_prev),
				min=1e-8,
			)
		).view(1, 1, 1)

		c = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma**2, min=1e-8))

		noise = torch.randn_like(x)

		x = (
			torch.sqrt(alpha_bar_prev) * x0
			+ c * eps
			+ sigma * noise
		)

		# debug sécurité
		if torch.isnan(x).any():
			print("NaN detected → stopping sampling")
			break

	return x


@torch.no_grad()
def test_ddim_perfect_model():
	device = "cuda"

	diffusion_steps = 1000
	sampling_steps = 50

	cosine = CosineScheduling(diffusion_steps, device=device)

	B, L, D = 1, 100, 1

	# 🔷 signal simple (sinusoïde)
	t_lin = torch.linspace(0, 2 * math.pi, L, device=device)
	x0 = torch.sin(t_lin)[None, :, None]  # (1, L, 1)

	# 🔷 oracle model (retourne le vrai bruit)
	def perfect_model(x, t):
		alpha_bar_t = cosine.alpha_bar[t[0, 0].item()].view(1,1,1).to(x.device)
		return (x - torch.sqrt(alpha_bar_t) * x0) / torch.sqrt(1 - alpha_bar_t) + torch.randn_like(alpha_bar_t) * 0.01

	x = ddim_sample(
		model=perfect_model,
		diffusion_steps=diffusion_steps,
		cosine_scheduling=cosine,
		shape=(B, L, D),
		sampling_steps=sampling_steps,
	)

	print("MSE final:", ((x - x0) ** 2).mean().item())


if __name__ == "__main__":

	device = "cuda" if torch.cuda.is_available() else "cpu"

	MEAN = torch.tensor([ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0505e+00,  3.3544e-01,
		 6.6190e-01,  6.7721e-03, -6.4809e-03,  6.1592e-03, -1.7795e-03,
		 3.7200e-02,  2.8523e-01, -1.9838e-02,  7.6051e-03,  6.5601e-03,
		-8.6135e-03, -1.0634e-02,  1.6396e-01,  1.0494e-02, -2.6813e-03,
		-4.4289e-02,  1.0476e-02,  1.4616e-03,  7.1394e-02,  6.3291e-02,
		-1.0401e-02, -4.1463e-03, -2.1539e-02,  1.0898e-03, -7.4362e-03,
		 1.0477e-02,  3.6607e-02,  1.3801e-01,  4.4252e-01, -1.8955e-01,
		 7.7999e-03,  3.4630e-01, -3.0305e-01,  5.3365e-02, -2.9065e-01,
		-7.4983e-02,  5.0038e-02,  1.4465e-01,  4.9524e-01, -2.0342e-01,
		-7.2436e-02,  3.6113e-01, -2.6576e-01,  3.5327e-02, -2.7933e-01,
		-1.4017e-01,  2.6628e-02, -2.9697e-01,  6.0447e-01, -8.5820e-02,
		 1.0006e-02,  5.5567e-02,  5.3308e-02, -5.0374e-02, -1.4982e-01,
		-1.0763e-03, -3.0005e-01,  5.8130e-01, -1.1141e-01,  2.6263e-02,
		 3.4591e-02,  2.4764e-02, -9.4851e-03,  2.4628e-01,  1.2017e-01,
		-1.8732e-01, -3.9886e-01,  1.1953e-01,  5.5164e-01,  1.4197e-01,
		-8.5941e-02,  3.1193e-03, -2.1241e-01, -7.2925e-02,  4.0159e-01,
		 5.9032e-01,  3.2326e-01,  6.4021e-01,  6.6051e-01,  3.7986e-01,
		 5.5939e-01,  6.9674e-01,  2.6388e-01,  5.3117e-01,  6.1804e-01,
		 3.8840e-01, -8.1391e-02,  7.8748e-02, -1.4697e-01, -1.0787e-01,
		 2.1961e-01,  7.2860e-02, -2.1655e-01, -3.5252e-01,  1.3866e-01,
		 5.3763e-01,  9.6820e-02, -8.3092e-02, -6.7039e-03, -1.5968e-01,
		-2.4306e-02,  3.2801e-01,  5.7934e-01,  3.2775e-01,  5.7164e-01,
		 6.4338e-01,  3.6861e-01,  5.7666e-01,  6.6002e-01,  2.5883e-01,
		 4.6862e-01,  5.8243e-01,  3.5060e-01,  1.3109e-02,  1.5888e-03,
		-1.2927e-01, -6.4834e-02, -5.9986e-02, -1.0806e-01,  1.1944e-01,
		-1.4699e-02, -1.0973e-01, -9.0893e-02,  5.0985e-02, -3.9744e-02,
		 9.0748e-02, -3.2941e-02, -1.4059e-01, -1.2432e-01, -7.3411e-03,
		 1.9627e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00,  8.4141e-02,
		 1.7399e-01, -1.0308e-01,  1.6721e-01, -4.1608e-02, -9.6569e-02,
		-9.6105e-02, -2.0799e-02, -1.5603e-02, -1.8965e-03, -1.9828e-01,
		-5.4649e-01,  3.1543e-05,  3.0282e-10,  5.7673e-03,  2.0795e-01,
		-7.1731e-02,  1.1658e-01,  1.5579e-01,  2.1247e-01,  1.4378e-01,
		-2.1127e-01, -1.8942e-01, -1.8839e-01, -1.1492e-01, -1.3125e-01,
		-9.0198e-02, -3.4552e-02,  1.9468e-02,  2.1192e-02, -8.8129e-02,
		 1.1466e-01, -2.9452e-02,  3.8239e-02, -1.1867e-01,  5.5055e-02,
		 1.4353e-01,  9.8106e-02,  1.2466e-01,  6.7154e-03,  2.0795e-01,
		-7.1731e-02,  1.1658e-01,  1.5579e-01,  2.1247e-01,  1.4378e-01,
		-2.1127e-01, -1.8942e-01, -1.8839e-01, -1.1492e-01, -1.3125e-01,
		-9.0198e-02, -3.4552e-02,  1.9468e-02,  2.1192e-02, -8.8129e-02,
		 1.1466e-01, -2.9452e-02,  3.8239e-02, -1.1867e-01,  5.5055e-02,
		 1.4353e-01,  9.8106e-02,  1.2466e-01,  6.7154e-03]).unsqueeze(0).unsqueeze(0).to(device)
	STD = torch.tensor([0.0000e+00, 0.0000e+00, 0.0000e+00, 1.6823e+00, 7.2401e-01, 1.9111e+00,
		5.9620e-02, 5.2321e-02, 3.6013e-02, 1.1730e-02, 6.4394e-02, 6.7550e-02,
		2.0859e-02, 1.4213e-02, 3.5358e-02, 2.8572e-02, 4.2721e-02, 1.1493e-01,
		2.1981e-02, 5.5716e-02, 5.8635e-02, 3.7927e-02, 3.4226e-02, 4.7435e-02,
		1.8459e-01, 1.0666e-01, 1.4006e-01, 1.2212e-01, 7.0305e-02, 1.5381e-01,
		1.0208e-01, 9.3419e-02, 9.7160e-02, 3.8201e-01, 2.9995e-01, 4.0906e-01,
		5.9867e-01, 1.8939e-01, 1.6639e-01, 2.3014e-01, 9.8754e-02, 9.0760e-02,
		1.0529e-01, 4.0355e-01, 3.0787e-01, 3.8457e-01, 6.0110e-01, 1.6819e-01,
		1.6245e-01, 2.4389e-01, 1.6255e-01, 1.6999e-01, 3.4404e-01, 4.7359e-01,
		9.5306e-02, 1.4515e-01, 8.4608e-02, 6.5928e-02, 7.4273e-02, 1.5888e-01,
		1.5738e-01, 3.2938e-01, 4.6013e-01, 1.1634e-01, 1.0618e-01, 5.0738e-02,
		5.2053e-02, 1.1256e-01, 5.7357e-02, 7.3678e-02, 3.4851e-02, 5.6357e-02,
		5.4808e-02, 5.4171e-02, 6.5162e-02, 3.3867e-02, 3.3668e-02, 9.5011e-02,
		3.8920e-02, 9.0184e-02, 1.5055e-01, 7.4528e-02, 1.2570e-01, 1.7213e-01,
		7.6443e-02, 1.1966e-01, 1.7563e-01, 8.4701e-02, 1.4958e-01, 1.6166e-01,
		1.0116e-01, 3.3645e-02, 5.6518e-02, 3.9779e-02, 3.4836e-02, 4.6758e-02,
		7.3043e-02, 4.1249e-02, 5.0275e-02, 4.8021e-02, 5.0334e-02, 5.8109e-02,
		2.9625e-02, 3.6823e-02, 8.3103e-02, 2.8786e-02, 9.3145e-02, 1.4561e-01,
		6.8021e-02, 1.1954e-01, 1.6783e-01, 7.5729e-02, 1.2190e-01, 1.6590e-01,
		8.8066e-02, 1.4448e-01, 1.4863e-01, 1.0776e-01, 3.0665e-02, 3.3267e-02,
		3.5394e-02, 2.5350e-02, 4.9253e-02, 1.0216e-01, 9.1710e-02, 3.8074e-02,
		6.1391e-02, 9.8703e-02, 8.3658e-02, 2.2714e-02, 1.1958e-01, 8.6164e-02,
		8.1507e-02, 1.7749e-01, 4.7319e-02, 1.9498e-01, 0.0000e+00, 0.0000e+00,
		0.0000e+00, 1.2317e-01, 2.9261e-02, 2.2481e-02, 6.8223e-02, 8.2353e-02,
		3.4835e-02, 3.3324e-02, 4.5623e-02, 2.6149e-02, 2.3518e-03, 5.0412e-02,
		8.7262e-02, 3.7061e-03, 1.6095e-10, 1.8511e-02, 2.1288e-02, 1.7306e-02,
		2.4427e-02, 2.8233e-02, 4.1965e-02, 5.2591e-02, 3.1125e-02, 2.6019e-02,
		1.8085e-02, 2.3293e-02, 2.2310e-02, 2.6808e-02, 1.0146e-02, 9.8146e-03,
		1.2332e-02, 1.1366e-02, 8.0832e-03, 1.6680e-02, 1.2275e-02, 1.7036e-02,
		2.3465e-02, 9.8275e-03, 2.4788e-02, 1.9181e-02, 1.9516e-02, 2.1288e-02,
		1.7306e-02, 2.4427e-02, 2.8233e-02, 4.1965e-02, 5.2591e-02, 3.1125e-02,
		2.6019e-02, 1.8085e-02, 2.3293e-02, 2.2310e-02, 2.6808e-02, 1.0146e-02,
		9.8146e-03, 1.2332e-02, 1.1366e-02, 8.0832e-03, 1.6680e-02, 1.2275e-02,
		1.7036e-02, 2.3465e-02, 9.8275e-03, 2.4788e-02, 1.9181e-02, 1.9516e-02]).unsqueeze(0).unsqueeze(0).to(device)

	model = DiT(
		n_layers=8,
		hidden_size=136,
		num_heads=4,
		conditioning_size=128,
		mlp_dim=128,
		mlp_ratio=4
	).to(device)

	optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

	warmup_steps = 1_000
	total_steps = 50_000

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

	writer = SummaryWriter(f"runs/dit_diffusion_{time.time()}")

	# dataset = SequenceDataset("dataset/latent", sequence_size=25)
	dataset = ToyDataset()

	dataloader = torch.utils.data.DataLoader(
		dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4
	)

	diffusion_steps = 1000
	cosine_scheduling = CosineScheduling(diffusion_steps, device=device)

	criterion = torch.nn.MSELoss()

	data = next(iter(dataset))
	# basic_skeleton_transformation = data["mhr"][:, 136:]
	# basic_shape = data["shape"]
	# basic_expr = data["expr"]

	# skeleton_transformation0 = basic_skeleton_transformation[0]
	# shape0 = basic_shape[0]
	# expr0 = basic_expr[0] 

	# basic_shape = np.tile(shape0, (25, 1))
	# basic_expr = np.tile(expr0, (25, 1))     # (25, 72)
	# basic_skeleton_transformation = np.tile(skeleton_transformation0, (25, 1))

	# print(f"{basic_shape.shape = }")
	# print(f"{basic_expr.shape = }")
	# print(f"{basic_skeleton_transformation.shape = }")

	batch = next(iter(dataloader))
	global_step = 0
	for epoch in range(100):
		model.train()
		# for batch in dataloader:
		for _ in range(1000):
			# x0 = (batch["mhr"].to(device) - MEAN) / (STD + 1e-8)
			x0 = batch["mhr"].to(device)
			# x0 = x0[:, :, :136]

			B, L, _ = x0.size()

			# t = torch.randint(
			# 	low=0,
			# 	high=diffusion_steps,
			# 	size=(B, L),
			# 	device=x0.device,
			# 	dtype=torch.long
			# )

			t = torch.randint(0, diffusion_steps, (B,), device=device)
			t = t.unsqueeze(1).repeat(1, L)

			mhr_noised, noise, alpha_bar_t = cosine_scheduling.apply(x0, t)

			predicted_noise = model(mhr_noised, t) # noise prediction
			loss = criterion(predicted_noise , noise)

			alpha_bar_t_safe = alpha_bar_t.clamp(min=1e-8)
			sqrt_alpha = torch.sqrt(alpha_bar_t_safe).clamp(min=1e-8)
			sqrt_one_minus_alpha = torch.sqrt((1 - alpha_bar_t_safe).clamp(min=1e-8))

			x0_hat = (mhr_noised - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha

			# print()
			# print(f"{sqrt_one_minus_alpha.min().item() = }, {sqrt_one_minus_alpha.max().item() = }")
			# print(f"{sqrt_alpha.min().item() = }, {sqrt_alpha.max().item() = }")
			# print(f"{predicted_noise.norm().item() = }")
			# print(f"{mhr_noised.norm().item() = }")
			# print(f"{x0_hat.norm().item() = }")
			# print(f"{x0.norm().item() = }")
			# print()
			# recon_loss = ((x0_hat - x0) ** 2)[(alpha_bar_t > 0.5).squeeze()].mean()

			optimizer.zero_grad()
			loss.backward()
			# torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			scheduler.step()

			writer.add_scalar("loss/train", loss.item(), global_step)
			writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
			# writer.add_scalar("loss/reconstruction", recon_loss.item(), global_step)
			writer.add_scalar("stats/x0_hat_std", x0_hat.std().item(), global_step)
			writer.add_scalar("stats/x0_std", x0.std().item(), global_step)

			if global_step % 100 == 0:
				print(f"step {global_step} | loss {loss.item():.4f}")

			global_step += 1

		# x_sample = ddpm_sample(
		# 	model=model,
		# 	diffusion_steps=diffusion_steps,
		# 	cosine_scheduling=cosine_scheduling,
		# 	shape=(1, *x0.shape[1:]),
		# )

		x_sample = ddim_sample(
			model=model,
			diffusion_steps=diffusion_steps,
			cosine_scheduling=cosine_scheduling,
			shape=(1, *x0.shape[1:]),
			sampling_steps=50,
			eta=0
		)

		import matplotlib.pyplot as plt

		plt.plot(x_sample[0, :, 0].cpu(), label="generated")
		plt.plot(x0[0, :, 0].cpu(), label="real")
		plt.legend()
		plt.savefig(f"output/output_{global_step}.png")
		plt.close()

		# x_sample = sample(
		# 	model=model,
		# 	diffusion_steps=diffusion_steps,
		# 	sampling_steps=100,
		# 	cosine_scheduling=cosine_scheduling,
		# 	shape=(1, *x0.shape[1:]),
		# 	eta=0.0
		# )

		# basic_skeleton_transformation_torch = torch.from_numpy(basic_skeleton_transformation).unsqueeze(0).to(device)
		# x_sample = STD[:, :, :136] * x_sample + MEAN[:, :, :136]

		# x_sample = torch.cat([x_sample, basic_skeleton_transformation_torch], dim=2)

		# mhr = x_sample.detach().cpu().squeeze().numpy()
		# print(f"{basic_shape.shape = }, {mhr.shape = }, {basic_expr.shape = }")
		# np.savez(f"output/output_{global_step}.npz", shape=basic_shape, mhr=mhr, expr=basic_expr)

	writer.close()