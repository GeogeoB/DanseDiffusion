from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from mhr.mhr import MHR
import plotly.graph_objects as go
import plotly.graph_objects as go
from IPython.display import display, clear_output


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))


class Encoder1D(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128, n_res_blocks=3):
        super().__init__()

        self.conv_in = nn.Conv1d(in_channels, hidden_dim, 4, stride=2, padding=1)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock1D(hidden_dim) for _ in range(n_res_blocks)]
        )

        self.conv_out = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_blocks(x)
        x = self.conv_out(x)
        return x


class Decoder1D(nn.Module):
    def __init__(self, out_channels=1, hidden_dim=128, n_res_blocks=3):
        super().__init__()

        self.conv_in = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock1D(hidden_dim) for _ in range(n_res_blocks)]
        )

        self.deconv = nn.ConvTranspose1d(
            hidden_dim, out_channels, 4, stride=2, padding=1
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_blocks(x)
        x = self.deconv(x)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    def forward(self, z):
        # z: (B, C, T)
        z = z.permute(0, 2, 1).contiguous()  # (B, T, C)
        flat_z = z.view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_z**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(z.shape)

        # losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # straight-through estimator
        quantized = z + (quantized - z).detach()

        return quantized.permute(0, 2, 1), loss


class VQVAE1D(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128, num_embeddings=512):
        super().__init__()

        self.encoder = Encoder1D(in_channels, hidden_dim)
        self.quantizer = VectorQuantizer(num_embeddings, hidden_dim)
        self.decoder = Decoder1D(in_channels, hidden_dim)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss = self.quantizer(z)

        x_recon = self.decoder(z_q)

        recon_loss = F.mse_loss(x_recon, x)
        loss = recon_loss + vq_loss

        return x_recon, loss, recon_loss, vq_loss


class MHRDataset(Dataset):
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.files_path = list(self.folder_path.rglob("*.npz"))

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, index):
        file_path = self.files_path[index]

        data = np.load(file_path, allow_pickle=True)["arr_0"].item()
        return data["shape_params"], data["mhr_model_params"], data["expr_params"]


class MLPEncoder(nn.Module):
    def __init__(self, input_dim=204, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class VectorQuantizerMLP(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1 / num_embeddings, 1 / num_embeddings
        )

    def forward(self, z):
        # z: (B, D)
        flat_z = z

        distances = (
            torch.sum(flat_z**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices)

        # losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # straight-through
        quantized = z + (quantized - z).detach()

        return quantized, loss
    

class VQVAE_MLP(nn.Module):
    def __init__(self, input_dim=204, hidden_dim=256, latent_dim=128, num_embeddings=512):
        super().__init__()

        self.encoder = MLPEncoder(input_dim, hidden_dim, latent_dim)
        self.quantizer = VectorQuantizerMLP(num_embeddings, latent_dim)
        self.decoder = MLPDecoder(input_dim, hidden_dim, latent_dim)

    def forward(self, x):
        z = self.encoder(x)                 # (B, latent_dim)
        z_q, vq_loss = self.quantizer(z)   # (B, latent_dim)
        x_recon = self.decoder(z_q)        # (B, input_dim)

        recon_loss = F.mse_loss(x_recon, x)
        loss = recon_loss + vq_loss

        return x_recon, loss, recon_loss, vq_loss

class MLPDecoder(nn.Module):
    def __init__(self, output_dim=204, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.net(z)

if __name__ == "__main__":
    dataset = MHRDataset("dataset/data")
    dataloader = DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VQVAE1D(in_channels=1, hidden_dim=10, num_embeddings=1024).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    writer = SummaryWriter("runs/vqvae_training")

    mhr_model = MHR.from_files(device=torch.device("cpu"), lod=1)

    epochs = 20
    global_step = 0

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{epochs}]")

        for shape_params, mhr_pose, expr_params in loop:
            # mhr_pose: (B, 204) -> (B, 1, 204)
            x = mhr_pose.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            x_recon, loss, recon_loss, vq_loss = model(x)
            loss.backward()
            optimizer.step()

            # TensorBoard logging
            writer.add_scalar("Loss/Total", loss.item(), global_step)
            writer.add_scalar("Loss/Reconstruction", recon_loss.item(), global_step)
            writer.add_scalar("Loss/VQ", vq_loss.item(), global_step)

            loop.set_postfix(
                total_loss=loss.item(),
                recon_loss=recon_loss.item(),
                vq_loss=vq_loss.item(),
            )

            global_step += 1

            # Optional: log reconstructions of first batch
            if global_step % 5 == 0:
                with torch.no_grad():
                    recon_sample = x_recon[0].cpu()
                    writer.add_histogram(
                        "Reconstruction/FirstSample", recon_sample, epoch
                    )

            if global_step % 2000 == 0:
                vertices_initial, skeleton_state = mhr_model(
                    shape_params, mhr_pose, expr_params
                )
                vertices_reconstructed, _ = mhr_model(
                    shape_params, x_recon.squeeze().cpu(), expr_params
                )

                print(f"{vertices_initial.shape = }")
                print(f"{vertices_reconstructed.shape = }")

                verts_init = vertices_initial[0].detach()         # shape [18439, 3]
                verts_recon = vertices_reconstructed[0].detach()  # shape [18439, 3]

                fig = go.Figure()

                # Original en bleu
                fig.add_trace(go.Scatter3d(
                    x=verts_init[:, 0],
                    y=verts_init[:, 1],
                    z=verts_init[:, 2],
                    mode='markers',
                    marker=dict(size=1, color='blue'),
                    name='Vertices Initiaux'
                ))

                # Reconstruit en rouge
                fig.add_trace(go.Scatter3d(
                    x=verts_recon[:, 0],
                    y=verts_recon[:, 1],
                    z=verts_recon[:, 2],
                    mode='markers',
                    marker=dict(size=1, color='red'),
                    name='Vertices Reconstitués'
                ))

                # Mise en page
                fig.update_layout(
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z'
                    ),
                    width=900,
                    height=700,
                    title=f'Comparaison Vertices - Step {global_step}',
                    showlegend=True
                )

                fig.show()

        save_path = f"weights/vqvae/vqvae_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Poids du VQVAE enregistrés : {save_path}")

    writer.close()
    print("Training terminé ! TensorBoard logs sont dans runs/vqvae_training")
