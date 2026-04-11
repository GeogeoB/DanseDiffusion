from mhr.mhr import MHR
import torch
from MHR.tools.mhr_LOD_conversion.example import LODConverter


class DisplayMHRMesh:
    def __init__(self):
        self.mhr_model = MHR.from_files(device=torch.device("cpu"), lod=1)

    def display(self, shape: torch.Tensor, mhr: torch.Tensor, expr: torch.Tensor):
        vertices_initial, skeleton_state = self.mhr_model(shape, mhr, expr)

        converter = LODConverter(source_lod=1)
        src_verts = converter._models[1].character.mesh.vertices
        mesh = converter.convert(src_verts, target_lod=6)

        mesh.show()


