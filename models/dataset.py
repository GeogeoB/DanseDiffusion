import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from mhr.mhr import MHR, LOD
from typing import Optional
from tqdm import tqdm
import trimesh
import pyvista as pv
import time
import torch.nn.functional as F


class LODConverter:
    """Converts an MHR mesh from one LOD to another using precomputed barycentric mappings."""

    def __init__(
        self,
        source_lod: LOD = 1,
        mapping_dir: Optional[Path] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self._source_lod = source_lod
        self._device = device
        self._mapping_dir = Path(mapping_dir or Path(__file__).parent)
        self._models: dict[LOD, MHR] = {}
        self._mappings: dict[LOD, dict[str, np.ndarray]] = {}

        self._models[source_lod] = MHR.from_files(lod=source_lod, device=device)

    def _ensure_model(self, target_lod: LOD) -> MHR:
        if target_lod not in self._models:
            self._models[target_lod] = MHR.from_files(
                lod=target_lod, device=self._device
            )
        return self._models[target_lod]

    def _ensure_mapping(self, target_lod: LOD) -> dict[str, np.ndarray]:
        if target_lod not in self._mappings:
            path = (
                self._mapping_dir
                / f"lod{self._source_lod}_to_lod{target_lod}_mapping.npz"
            )
            self._mappings[target_lod] = dict(np.load(path))
        return self._mappings[target_lod]

    def convert(self, src_verts: np.ndarray, target_lod: LOD) -> trimesh.Trimesh:
        """Map vertices from the source LOD onto the target LOD topology.

        Args:
            src_verts: Source mesh vertices, shape ``(num_source_verts, 3)``.
            target_lod: The target LOD to convert to.
        """
        source = self._models[self._source_lod]
        expected_n = len(source.character.mesh.vertices)

        if src_verts.ndim != 2 or src_verts.shape[1] != 3:
            raise ValueError(f"src_verts must have shape (N, 3), got {src_verts.shape}")
        if src_verts.shape[0] != expected_n:
            raise ValueError(
                f"src_verts has {src_verts.shape[0]} vertices, "
                f"expected {expected_n} for LOD {self._source_lod}"
            )

        target = self._ensure_model(target_lod)
        mapping = self._ensure_mapping(target_lod)

        src_faces = source.character.mesh.faces
        tri_verts = src_verts[src_faces[mapping["triangle_ids"]]]
        new_verts = np.einsum("ijk,ij->ik", tri_verts, mapping["baryc_coords"])

        return trimesh.Trimesh(new_verts, target.character.mesh.faces, process=False)


class MHRSequenceViewer:
    def __init__(self, fps=30):
        self.fps = fps
        self.delay = 1.0 / fps

        # MHR + converter
        self.mhr_model = MHR.from_files(
            folder=Path("MHR/assets"), device=torch.device("cpu"), lod=1
        )

        self.converter = LODConverter(
            mapping_dir=Path("MHR/tools/mhr_LOD_conversion"), source_lod=1
        )

        # PyVista viewer
        self.plotter = pv.Plotter()
        self.mesh = None
        self.faces = None

        self.meshes = []
        self.faces_list = []

    # -------------------------
    # 🔹 Compute ONE mesh
    # -------------------------
    def _compute_mesh(self, shape, mhr, expr):
        vertices_initial, _ = self.mhr_model(
            shape.unsqueeze(0),
            mhr.unsqueeze(0),
            expr.unsqueeze(0),
        )

        mesh = self.converter.convert(vertices_initial.squeeze(0), target_lod=2)

        return np.asarray(mesh.vertices), np.asarray(mesh.faces)

    # -------------------------
    # 🔹 Precompute sequence
    # -------------------------
    def precompute(self, sequence_shape, sequence_mhr, sequence_expr):
        vertices_seq = []

        for shape, mhr, expr in zip(sequence_shape, sequence_mhr, sequence_expr):
            vertices, faces = self._compute_mesh(shape, mhr, expr)

            if self.faces is None:
                self.faces = faces  # topology fixe

            vertices_seq.append(vertices)

        return vertices_seq

    # -------------------------
    # 🔹 Init mesh PyVista
    # -------------------------
    def _init_mesh(self, vertices):
        # PyVista attend un format spécial pour les faces
        faces_pv = np.hstack(
            [
                np.full((self.faces.shape[0], 1), 3),  # triangles
                self.faces,
            ]
        ).astype(np.int32)

        self.mesh = pv.PolyData(vertices, faces_pv)
        self.plotter.add_mesh(self.mesh)

        self.plotter.show(auto_close=False, interactive_update=True)

    # -------------------------
    # 🔹 Animation
    # -------------------------
    def play(self, vertices_seq, loop=False):
        while True:
            for vertices in vertices_seq:
                if self.mesh is None:
                    self._init_mesh(vertices)
                else:
                    # update vertices ONLY
                    self.mesh.points = vertices
                    self.plotter.update()

                time.sleep(self.delay)

            if not loop:
                break

    # -------------------------
    # 🔹 Full pipeline
    # -------------------------
    def display_sequence(self, sequence_shape, sequence_mhr, sequence_expr, loop=False):
        vertices_seq = self.precompute(sequence_shape, sequence_mhr, sequence_expr)
        self.play(vertices_seq, loop=loop)

    # -------------------------
    def close(self):
        self.plotter.close()

    def precompute_multiple(self, sequences_shape, sequences_mhr, sequences_expr):
        all_vertices_seq = []

        for seq_shape, seq_mhr, seq_expr in zip(sequences_shape, sequences_mhr, sequences_expr):
            vertices_seq = []

            for shape, mhr, expr in zip(seq_shape, seq_mhr, seq_expr):
                vertices, faces = self._compute_mesh(shape, mhr, expr)

                if len(self.faces_list) < len(all_vertices_seq) + 1:
                    self.faces_list.append(faces)

                vertices_seq.append(vertices)

            all_vertices_seq.append(vertices_seq)

        return all_vertices_seq
    
    def _init_meshes(self, vertices_list):
        spacing = 100.0

        for i, vertices in enumerate(vertices_list):
            faces = self.faces_list[i]

            offset = np.array([i * spacing, 0, 0])
            vertices_translated = vertices + offset

            faces_pv = np.hstack([
                np.full((faces.shape[0], 1), 3),
                faces,
            ]).astype(np.int32)

            mesh = pv.PolyData(vertices_translated, faces_pv)

            # Optionnel : couleurs différentes
            self.plotter.add_mesh(mesh)

            self.meshes.append(mesh)

        self.plotter.show(auto_close=False, interactive_update=True)

    def play_multiple(self, all_vertices_seq, loop=False):
        spacing = 100.0
        min_len = min(len(seq) for seq in all_vertices_seq)

        while True:
            for t in range(min_len):
                current_vertices = [seq[t] for seq in all_vertices_seq]

                if not self.meshes:
                    self._init_meshes(current_vertices)
                else:
                    for i, (mesh, vertices) in enumerate(zip(self.meshes, current_vertices)):
                        offset = np.array([i * spacing, 0, 0])
                        mesh.points = vertices + offset

                    self.plotter.update()

                time.sleep(self.delay)

            if not loop:
                break

    def display_multiple_sequences(self, sequences_shape, sequences_mhr, sequences_expr, loop=False):
        all_vertices_seq = self.precompute_multiple(
            sequences_shape, sequences_mhr, sequences_expr
        )
        self.play_multiple(all_vertices_seq, loop=loop)


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


def interpolate_1d(x, M):
    """
    x: [N, d]
    return: [M, d]
    """
    N, d = x.shape

    # reshape pour interpolation 1D: [batch=1, channels=d, length=N]
    x = x.T.unsqueeze(0)  # [1, d, N]

    x_interp = F.interpolate(x, size=M, mode="linear", align_corners=True)

    return x_interp.squeeze(0).T  # [M, d]

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=1000, seq_len=100, dim=136):
        self.data = torch.randn(n_samples, seq_len, dim) * 0.01

        # structure simple (sinusoïde → pattern apprenable)
        t = torch.linspace(0, 2 * torch.pi, seq_len)
        pattern = torch.sin(t)[None, :, None]

        self.data += pattern

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "mhr": self.data[idx]
        }

if __name__ == "__main__":
    dataset = MHRDataset("dataset/data")
    dataset = SequenceDataset("dataset/latent", sequence_size=25)

    data = next(iter(dataset))

    shape, mhr, expr = data

    viewer = MHRSequenceViewer(fps=40)

    i = 0
    for sequence in iter(dataset):
        i += 1
        if i < 510:
            continue

        shape = torch.from_numpy(sequence["shape"])
        mhr = torch.from_numpy(sequence["mhr"])
        expr = torch.from_numpy(sequence["expr"])
        
        print(f"{mhr.size() = }")

        cols_zero = (mhr == 0).all(dim=0)
        print(torch.where(cols_zero)[0])
        input()

        # viewer.display_sequence(
        #     interpolate_1d(shape, 100), interpolate_1d(mhr, 100), interpolate_1d(expr, 100), loop=True
        # )

    viewer.close()

    # Compute std and mean

    # def compute_mean_std(dataset):
    #     n = 0
    #     mean = None
    #     m2 = None  # pour variance (Welford)

    #     for data in tqdm(dataset):
    #         _, mhr, _ = data

    #         mhr = torch.from_numpy(mhr).float()  # (L, D) ou (D,)

    #         # aplatir si besoin (optionnel selon ton format)
    #         mhr = mhr.reshape(-1)

    #         if mean is None:
    #             mean = torch.zeros_like(mhr)
    #             m2 = torch.zeros_like(mhr)

    #         n += 1
    #         delta = mhr - mean
    #         mean += delta / n
    #         delta2 = mhr - mean
    #         m2 += delta * delta2

    #     std = torch.sqrt(m2 / (n - 1 + 1e-8))

    #     return mean, std

    # dataset = MHRDataset("dataset/data")
    # print(compute_mean_std(dataset))
