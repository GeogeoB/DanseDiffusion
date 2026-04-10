import io
from time import sleep

from mhr.mhr import MHR
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import torch
import random
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


class SequenceDataset(Dataset):
    def __init__(self, sequence_folder: str, sequence_size=10):
        self.sequence_folder = Path(sequence_folder)
        self.sequence_files = sorted(self.sequence_folder.glob("*.npz"))
        self.sequence_size = sequence_size

        self.file_lengths = []
        self.n_sequence = 0

        for file in self.sequence_files:
            data = np.load(file, allow_pickle=True)
            length = (
                data["sequence_mhr_model_params_latent"].shape[0] - self.sequence_size
            )
            length = max(0, length)
            self.file_lengths.append(length)
            self.n_sequence += length

    def __len__(self):
        return self.n_sequence

    def __getitem__(self, index):
        # Trouver le bon fichier
        for file, length in zip(self.sequence_files, self.file_lengths):
            if index < length:
                data = np.load(file, allow_pickle=True)

                start = index
                end = start + self.sequence_size

                sequence_shape = data["sequence_shape"][start:end]
                sequence_mhr = data["sequence_mhr_model_params_latent"][start:end]
                sequence_expr = data["sequence_expr_params"][start:end]

                return {
                    "shape": sequence_shape,
                    "mhr": sequence_mhr,
                    "expr": sequence_expr,
                }
            else:
                index -= length

        raise IndexError("Index out of range")


class DiffusionTransformer(nn.Module):
    def __init__(self, dim, num_heads, num_layers):
        super().__init__()

        self.embedding = nn.Linear(dim, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x : données bruitées
        t : timestep de diffusion
        """
        # Encodage du temps (simplifié)
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.output(x)

        return x


class FlowMatching:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ):
        self.model = model
        self.dataloader = dataloader
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optimizer

        self.global_step = 0

        self.writer = SummaryWriter("runs/flow_matching")
        self.sequence_shape = None

    def _loss(self, prediction, data, noise):
        return self.criterion(prediction, data - noise)

    def _log(self, loss: float):
        self.writer.add_scalar("Loss/Total", loss.item(), self.global_step)

    def train_one_epoch(self):
        for data in tqdm(self.dataloader):
            sequence_mhr = data["mhr"].to("cuda")

            if self.sequence_shape is None:
                self.sequence_shape = sequence_mhr.shape

            B, n_item_per_sequence, _ = sequence_mhr.shape

            # idea extract from https://arxiv.org/pdf/2307.15042
            t = torch.linspace(0, 1, n_item_per_sequence).unsqueeze(0).repeat(B, 1)
            t = t + 0.05 * torch.randn_like(t)
            t = t[:, :, None].to("cuda")

            noise = torch.randn(sequence_mhr.shape).to("cuda")
            x = t * sequence_mhr + (1 - t) * noise

            prediction = self.model(x)

            loss = self._loss(prediction, sequence_mhr, noise)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.global_step += 1

            self._log(loss)

    def train(self, nb_epoch: int):
        for epoch in range(nb_epoch):
            print(f"{epoch = }")
            self.train_one_epoch()

    def forward(self):
        x = torch.randn((1, *self.sequence_shape[1:])).to("cuda")

        n = 30

        for _ in range(100):
            with torch.no_grad():
                u = self.model(x)

                x += 1/n * u

                # pop first element
                fist_element = x[0, 0, :]

                x = x[:, 1:, :]

                noise = torch.randn_like(x[:, -1:, :])
                x = torch.cat([x, noise], dim=1)
                
                yield fist_element


if __name__ == "__main__":
    dataset = SequenceDataset(sequence_folder="dataset/latent", sequence_size=10)
    dataloader = DataLoader(
        dataset=dataset, shuffle=True, num_workers=4, pin_memory=True, batch_size=128
    )

    model = DiffusionTransformer(dim=204, num_heads=4, num_layers=4).to("cuda")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,  # learning rate
        betas=(0.9, 0.999),  # moyennes exponentielles
        eps=1e-8,
    )

    flow_matching = FlowMatching(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
    )

    flow_matching.train(nb_epoch=3)

    mhr_model = MHR.from_files(device=torch.device("cpu"), lod=1)
    data = next(iter(dataset))

    print(f"{data = }")
    shape_params = data["shape"][0][None, :]
    expr_params = data["expr"][0][None, :]

    print(f"{shape_params.shape = }")
    print(f"{expr_params.shape = }")

    xs = []
    for x in flow_matching.forward():
        x = x[None, :].detach().cpu()

        vertices_initial, skeleton_state = mhr_model(
            torch.from_numpy(shape_params), x, torch.from_numpy(expr_params)
        )
        xs.append(vertices_initial)

        if len(xs) == 60:
            break

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    images = []

    for i, x in enumerate(flow_matching.forward()):
        x = x[None, :].detach().cpu()

        vertices_initial, skeleton_state = mhr_model(
            torch.from_numpy(shape_params), x, torch.from_numpy(expr_params)
        )

        verts = vertices_initial.squeeze().numpy()

        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            s=1
        )

        # enlever axes
        ax.set_axis_off()

        # caméra fixe (IMPORTANT pour cohérence visuelle)
        ax.view_init(elev=20, azim=45)

        # convertir en image
        fig.canvas.draw()
        
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape((h, w, 4))

        # Convert ARGB → RGBA
        img = img[:, :, [1, 2, 3, 0]]

        # Drop alpha channel → RGB
        img = img[:, :, :3]

        images.append(Image.fromarray(img))

        plt.close(fig)

        if len(images) == 60:
            break

    cols = 10
    rows = 6
    w, h = images[0].size

    grid = Image.new("RGB", (cols * w, rows * h))

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        grid.paste(img, (col * w, row * h))

    grid.save("grid.png")
    grid.show()
