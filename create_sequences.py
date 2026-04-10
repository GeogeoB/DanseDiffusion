from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from dataset import VQVAE1D

folder_path = Path("dataset/data")
latent_path = Path("dataset/latent")

subfolders = list(folder_path.iterdir())

for subfolder in tqdm(subfolders):
    if not subfolder.is_dir():
        continue

    sequence_shape = []
    sequence_mhr_model_params_latent = []
    sequence_expr_params = []

    frame_files = sorted(subfolder.glob("frame_*.npz"))
    for frame_file in frame_files:
        data = np.load(frame_file, allow_pickle=True)["arr_0"].item()

        shape_params, mhr_model_params, expr_params = data["shape_params"], data["mhr_model_params"], data["expr_params"]

        sequence_shape.append(shape_params)
        sequence_mhr_model_params_latent.append(mhr_model_params)
        sequence_expr_params.append(expr_params)

    if len(sequence_shape) == 0:
        continue

    sequence_shape = np.vstack(sequence_shape)
    sequence_mhr_model_params_latent = np.vstack(sequence_mhr_model_params_latent)
    sequence_expr_params = np.vstack(sequence_expr_params)

    output_file = latent_path / f"{subfolder.name}.npz"

    np.savez(
        output_file,
        sequence_shape=sequence_shape,
        sequence_mhr_model_params_latent=sequence_mhr_model_params_latent,
        sequence_expr_params=sequence_expr_params,
    )