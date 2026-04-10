import torch
from mhr.mhr import MHR

# Load MHR model (LOD 1, on CPU)
mhr_model = MHR.from_files(device=torch.device("cpu"), lod=1)

# Define parameters
batch_size = 2
identity_coeffs = 0.8 * torch.randn(batch_size, 45)      # Identity
model_parameters = 0.2 * (torch.rand(batch_size, 204) - 0.5)  # Pose
face_expr_coeffs = 0.3 * torch.randn(batch_size, 72)     # Facial expression

# Generate mesh vertices and skeleton information (joint orientation and positions).
vertices, skeleton_state = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)