# 💃 DanceDiffuser

Deep learning project for generating human dance motion sequences using diffusion models.

![alt text](media/video.gif)

> ⚠️ **Warning**: the code is not cleaned yet !!!

## 📌 Overview

DanceDiffuser explores generative modeling of 3D human motion using diffusion-based architectures. The goal is to generate coherent long-term dance sequences from structured pose representations.

The project combines:
- Diffusion models for sequence generation
- Structured human body representations (MHR)
- Modern sequence modeling techniques (Transformers, attention, positional encodings)


## 🧠 Motivation

Recent advances in diffusion models and human motion synthesis show strong performance in:
- long-horizon motion generation
- controllable human animation
- representation learning for structured 3D bodies

This project explores how these ideas can be combined for dance generation.


## 📊 Dataset

- **AIST Dance Database**
- Pose extraction using **SAM 3D Body**
- Data stored in `.npz` format

### Extracted signals:
- 2D / 3D keypoints
- SMPL-like parameters
- Mesh vertices
- Body shape, pose, expression parameters
- Camera and global motion parameters


## 🧍 Representation

Uses **MHR (Modular Human Representation)**:
- Identity parameters (body shape)
- Model parameters (pose + motion)
- Expression parameters (unused in this project)

Main learning target: **model parameters**


## 🧩 Method

### 1. Diffusion Forcing Model
- Predicts motion in latent or parameter space
- Tested parameterizations:
  - ε-prediction
  - x₀-prediction (main choice)
  - v-prediction
- **DiT** 

### 2. Temporal Modeling
- Sequence-aware diffusion
- Progressive noise injection over time
- Inspired by TEDi and Diffusion Forcing


## ⚙️ Training Setup

- Cosine noise schedule (1000 steps), x₀-prediction
- Diffusion Forcing noise levels: one independent timestep per frame, plus shared timesteps
  (full-sequence diffusion) and clean prefixes (autoregressive context)
- Truncated Min-SNR weighting: `w = clamp(SNR, 1, γ)`
- DiT backbone with per-frame adaLN-Zero conditioning, RoPE and QK-norm
- Per-feature normalization, EMA weights, bf16, AdamW + warmup / cosine LR

## 🛠️ Usage

```bash
# environment (PyTorch CUDA 12.8)
uv sync

# stack per-frame SAM 3D Body outputs (dataset/data/<seq>/frame_*.npz) into dataset/latent/<seq>.npz
uv run python utils/create_sequences.py [--sequences <seq> ...]

# train (logs, checkpoints and samples in runs/<run-name>)
uv run python models/train.py --run-name my_run [--sequences <seq> ...] [--val-sequences <seq> ...]
uv run tensorboard --logdir runs

# sample with a checkpoint
uv run python models/sample.py runs/my_run/checkpoints/last.pt --n 4 --frames 300 [--pyramid 2]

# long generation shown in the dashboard as runs/long_5min, with more diversity:
#   --eta 1            stochastic sampling (0 = deterministic DDIM)
#   --context-t 200    slightly noised context: follows the past less literally
#   --guidance 0.7     history guidance (<1: more diversity, >1: more coherence)
uv run python models/sample.py runs/my_run/checkpoints/last.pt --n 10 --frames 3000 --context 16 \
    --eta 1 --context-t 200 --guidance 0.7 --run long_5min

# dashboard (curves + 3D samples of every evaluation) on http://localhost:8000
uv run --project dashboard python dashboard/server.py
```

Samples are saved in the `dataset/latent` format and can be played with `models/viewer.py`
(requires the MHR assets, `pyvista` and `trimesh`).


## 🎯 Sampling

- **DDIM** with one timestep per frame (`eta` > 0 for stochastic sampling)
- **Diffusion Forcing** autoregressive generation with a sliding window: each chunk is
  generated conditioned on the last frames, optionally with the pyramid scheduling matrix
  (each frame lags `pyramid` denoising steps behind the previous one)

## 🚀 Future Work

- Full VQ-VAE + diffusion pipeline
- Long-horizon motion synthesis
- Multi-agent interaction modeling
- Conditioning on music / style
- Real-time avatar animation (codec avatars)

## 📚 References

- Human Motion Diffusion Model: https://arxiv.org/abs/2209.14916
- History-Guided Video Diffusion: https://arxiv.org/pdf/2502.06764
- An Introduction to Flow Matching and Diffusion Models: https://arxiv.org/pdf/2506.02070
- Diffusion Forcing for Multi-Agent Interaction Sequence Modeling: https://arxiv.org/pdf/2512.17900
- SAM 3D Body: Robust Full-Body Human Mesh
Recovery: https://arxiv.org/pdf/2602.15989
- Diffusion Forcing: Next-token Prediction
Meets Full-Sequence Diffusion: https://arxiv.org/pdf/2407.01392
- Denoising Diffusion Probabilistic Models: https://arxiv.org/pdf/2006.11239
- DENOISING DIFFUSION IMPLICIT MODELS: https://arxiv.org/pdf/2010.02502
- The DDIM Sampling Algorithm: https://apxml.com/courses/intro-diffusion-models/chapter-5-sampling-generation-process/ddim-sampling-algorithm
- A practical guide to Diffusion models: https://selflein.github.io/diffusion_practical_guide
- Diffusion Meets Flow Matching: Two Sides of the Same Coin: https://diffusionflow.github.io/
- Scalable Diffusion Models with Transformers: https://arxiv.org/pdf/2212.09748

## 🧭 Status

Experimental research project — active development.