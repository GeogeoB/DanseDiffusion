import torch
import torch.nn as nn
import math
import numpy as np
from DiT import DiT
from dataset import SequenceDataset
from diffusion_test_DiT import CosineScheduler
from tqdm import tqdm

# @torch.no_grad()
# def sample(model, scheduler, T, shape, device):
#     model.eval()
#     x_t = torch.randn(shape, device=device)

#     x_final = []

#     B, L, D = shape

#     for _ in tqdm(range(2)):
#         for t in reversed(range(T)):
#             t_tensor = torch.full((B, L), t, device=device, dtype=torch.long)

#             beta_t = scheduler.betas[t]
#             alpha_t = scheduler.alphas[t]
#             alpha_bar_t = scheduler.alphas_bar[t]

#             if t > 0:
#                 alpha_bar_prev = scheduler.alphas_bar[t - 1]
#             else:
#                 alpha_bar_prev = torch.tensor(1.0, device=device)

#             # 1. prédiction de x0
#             x0_hat = model(x_t, t_tensor)
#             x0_hat = torch.clamp(x0_hat, -1, 1)

#             # 2. coefficients
#             coef1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t)
#             coef2 = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)

#             # 3. moyenne du posterior
#             mu = coef1 * x0_hat + coef2 * x_t

#             # 4. sampling
#             if t > 0:
#                 noise = torch.randn_like(x_t)
#                 sigma = torch.sqrt(beta_t)
#                 x_t = mu + sigma * noise
#             else:
#                 x_t = mu

#         # on garde la moitié droite
#         x_final.append(x_t[:, L // 2:, :])

#         # on réinitialise la moitié droite avec du bruit
#         x_t[:, L // 2:, :] = torch.randn_like(x_t[:, L // 2:, :])

#     x_final = torch.cat(x_final, dim=1)
#     return x_final

@torch.no_grad()
def sample(model, scheduler, T, shape, device):
    model.eval()
    x_t = torch.randn(shape, device=device)
    right = None

    x_final = []

    B,L, D = shape

    for i in tqdm(range(10)):
        for t in reversed(range(T)):
            if i == 0:
                t_tensor = torch.full((B, L), t, device=device, dtype=torch.long)
            else:
                t_tensor = torch.full((B, L), t, device=device, dtype=torch.long)
                t_tensor[:, :L//2] = 0

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
            x0_hat = torch.clamp(x0_hat, -1, 1)

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

            if right is not None:
                x_t[:, :L // 2, :] = right

        x_final.append(x_t[:, L // 2:, :].clone())

        right = x_t[:, L // 2:, :].clone()
        x_t[:, :L // 2, :] = right
        x_t[:, L // 2:, :] = torch.randn_like(right)

    return torch.cat(x_final, dim=1)

@torch.no_grad()
def sample_ddim(model, scheduler, T, shape, device):
    model.eval()
    x_t = torch.randn(shape, device=device)
    right = None
    
    eta = 0.0
    s = 3

    x_final = []

    B,L, D = shape

    N = 10

    for i in tqdm(range(10)):
        for t in reversed(range(T)):
            t_tensor = torch.full((B, L), t, device=device, dtype=torch.long)
            t_tensor_m1 = torch.clamp(t_tensor - 1, min=0)
            
            if i > 0:
                t_tensor[:, :N] = 0
                t_tensor_m1[:, :N] = 0

            alpha_bar_t = scheduler.alphas_bar[t_tensor]

            alpha_bar_prev = scheduler.alphas_bar[t_tensor_m1]
            alpha_bar_prev = torch.where(
                t_tensor == 0,
                torch.ones_like(alpha_bar_prev),
                alpha_bar_prev
            )

            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            alpha_bar_prev = alpha_bar_prev.unsqueeze(-1)

            # 1. predict x0
            x0_hat = model(x_t, t_tensor)
            x0_hat = torch.clamp(x0_hat, -1, 1)

            # 1.1 predict x0 free guidance
            if i > 0:
                x_t_free_guidance = x_t.clone()
                x_t_free_guidance[:, :N, :] = torch.randn_like(x_t_free_guidance[:, :N, :], device=device)
                t_tensor_free_guidance = t_tensor.clone()
                t_tensor_free_guidance[:, :N] = T - 1

                x0_hat_free_guidance = model(x_t_free_guidance, t_tensor_free_guidance)
                x0_hat_free_guidance = torch.clamp(x0_hat_free_guidance, -1, 1)

                x0_hat = x0_hat_free_guidance + s * (x0_hat - x0_hat_free_guidance)

            # 2. compute epsilon
            eps = (x_t - torch.sqrt(alpha_bar_t) * x0_hat) / torch.sqrt(1 - alpha_bar_t)

            # 3. compute sigma (DDIM)
            if t > 0:
                sigma = eta * torch.sqrt(
                    (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                ) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
            else:
                sigma = 0.0

            # 4. direction term
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps

            # 5. compute x_{t-1}
            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = torch.sqrt(alpha_bar_prev) * x0_hat + dir_xt + sigma * noise
            else:
                x_t = torch.sqrt(alpha_bar_prev) * x0_hat + dir_xt

            if right is not None:
                x_t[:, :N, :] = right

        if i == 0:
            x_final.append(x_t.clone())
        else:
            x_final.append(x_t[:, N:, :].clone())

        right = x_t[:, L-N:, :].clone()
        x_t[:, :N, :] = right
        x_t[:, N:, :] = torch.randn_like(x_t[:, N:, :])

    return torch.cat(x_final, dim=1)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DiT(
        n_layers=4,
        hidden_size=136,
        num_heads=2,
        conditioning_size=256,
        mlp_dim=512,
        mlp_ratio=4,
    ).to(device)

    model.load_state_dict(torch.load("output/model_151424.pth"))
    model.eval() 

    dataset = SequenceDataset("dataset/latent", sequence_size=20)

    T = 1000
    scheduler = CosineScheduler(T, device=device)

    data = next(iter(dataset))
    x0 = data["mhr"][:, :136]
    basic_skeleton_transformation = data["mhr"][:, 136:]
    basic_shape = np.repeat(data["shape"][0:1, :], 204, axis=0)
    basic_expr = np.repeat(data["expr"][0:1, :], 204, axis=0)

    samples = sample_ddim(model, scheduler, T, (25, *x0.shape), device)
    print(samples.size())

    for i, sample_i in enumerate(samples):
        sample_i = sample_i.unsqueeze(0)
        basic_skeleton_transformation_torch = torch.from_numpy(
            basic_skeleton_transformation
        ).to(device)
        basic_expanded = basic_skeleton_transformation_torch.unsqueeze(0)
        basic_expanded = basic_expanded[:, 0, :].unsqueeze(1).repeat(1, sample_i.size()[1], 1)

        x_sample = torch.cat([sample_i, basic_expanded], dim=2)

        mhr = x_sample.detach().cpu().squeeze().numpy()

        np.savez(
            f"output/sample_{i}.npz",
            sequence_shape=basic_shape,
            sequence_mhr_model_params_latent=mhr,
            sequence_expr_params=basic_expr,
        )