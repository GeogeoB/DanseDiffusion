import argparse
import torch
from pathlib import Path
import numpy as np
from mhr.mhr import MHR, LOD
from typing import Optional
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

            N = 500
            seq_shape, seq_mhr, seq_expr = interpolate_1d(seq_shape, N), interpolate_1d(seq_mhr, N), interpolate_1d(seq_expr, N)

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

if __name__ == "__main__":
    # Play sequence files (dataset/latent/*.npz or generated samples) side by side
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    shapes, mhrs, exprs = [], [], []
    for path in args.files:
        data = np.load(path)
        shapes.append(torch.from_numpy(data["sequence_shape"]).float())
        mhrs.append(torch.from_numpy(data["sequence_mhr_model_params_latent"]).float())
        exprs.append(torch.from_numpy(data["sequence_expr_params"]).float())

    viewer = MHRSequenceViewer(fps=args.fps)
    viewer.display_multiple_sequences(shapes, mhrs, exprs, loop=True)
    viewer.close()
