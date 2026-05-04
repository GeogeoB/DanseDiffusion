import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from dataset import SequenceDataset
from DiT import DiT
import numpy as np

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
		t = t.float().unsqueeze(1) / 1000.0
		x = torch.cat([x, t], dim=1)
		return self.net(x)


def q_sample(x0, t, scheduler, noise=None):
	if noise is None:
		noise = torch.randn_like(x0)

	alpha_bar = scheduler.alphas_bar[t].view(-1, 1, 1)
	x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

	return x_t, noise


@torch.no_grad()
def sample(model, scheduler, T, shape, device):
	model.eval()
	x = torch.randn(shape, device=device)
	B, L, D = shape

	for t in reversed(range(T)):
		# t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
		t_batch = torch.full((B, L), t, device=device, dtype=torch.long)

		beta_t = scheduler.betas[t]
		alpha_t = scheduler.alphas[t]
		alpha_bar_t = scheduler.alphas_bar[t]

		eps = model(x, t_batch)

		x = (1 / torch.sqrt(alpha_t)) * (
			x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps
		)

		if t > 0:
			x += torch.sqrt(beta_t) * torch.randn_like(x)

		# x = torch.clamp(x, -1, 1)

	return x


def train():
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

	T = 20 # plus proche code 2 → stabilité
	dataset = SequenceDataset("dataset/latent", sequence_size=1)
	loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

	scheduler = LinearScheduler(T, device=device)
	model = DiT(
		n_layers=2,
		hidden_size=136,
		num_heads=2,
		conditioning_size=128,
		mlp_dim=128,
		mlp_ratio=4
	).to(device)

	warmup_steps = 1_000
	total_steps = 50_000

	opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
	# scheduler_lr = SequentialLR(
	# 	opt,
	# 	schedulers=[
	# 		LinearLR(
	# 			opt, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
	# 		),
	# 		CosineAnnealingLR(opt, T_max=total_steps - warmup_steps),
	# 	],
	# 	milestones=[warmup_steps],
	# )

	writer = SummaryWriter()

	step = 0

	data = next(iter(dataset))
	basic_skeleton_transformation = data["mhr"][:, 136:][0]
	basic_shape = data["shape"][0]
	basic_expr = data["expr"][0]

	x0 = next(iter(loader))
	x0 = x0["mhr"].to(device)
	x0 = (x0 - MEAN) / (STD + 1e-8)
	x0 = x0[:1, :, :136]
	print(f"{x0.shape}")

	for epoch in range(1000):
		for x0 in tqdm(loader):
		# for _ in tqdm(range(1000)):
			x0 = x0["mhr"].to(device)
			x0 = (x0 - MEAN) / (STD + 1e-8)
			x0 = x0[:1, :, :136]

			B,L,D = x0.size()
			t = torch.randint(0, T, (B,), device=device)

			x_t, noise = q_sample(x0, t, scheduler)

			noise_pred = model(x_t, t.unsqueeze(1).repeat(1, L))

			loss = F.mse_loss(noise_pred, noise)

			opt.zero_grad()
			loss.backward()
			opt.step()
			# scheduler_lr.step()

			writer.add_scalar("loss", loss.item(), step)
			step += 1

		with torch.inference_mode():
			samples = sample(
				model,
				scheduler,
				T,
				shape=(1, *(x0.size()[1:])),
				device=device
			)

			writer.add_scalar("loss/reconstruction", F.mse_loss(samples, x0), step)

			samples = STD[:, :, :136] * samples + MEAN[:, :, :136]

			basic_skeleton_transformation_torch = torch.from_numpy(basic_skeleton_transformation).to(device)
			basic_expanded = basic_skeleton_transformation_torch.unsqueeze(0).unsqueeze(1)
			basic_expanded = basic_expanded.repeat(1, 25, 1)

			print(f"{basic_skeleton_transformation_torch.size() = }")
			print(f"{samples.size() = }")
			x_sample = torch.cat([samples, basic_expanded], dim=2)
			print(f"{x_sample.size() = }")
			print(f"{basic_shape.shape = }, {basic_expr.shape = }")

			mhr = x_sample.detach().cpu().squeeze().numpy()

			print(f"{mhr.shape = }, {basic_shape.shape = }, {basic_expr.shape = }")
			np.savez(f"output/output_{step}.npz", shape=np.tile(basic_shape, (25, 1)), mhr=mhr, expr=np.tile(basic_expr, (25, 1)))

	writer.close()


if __name__ == "__main__":
	train()