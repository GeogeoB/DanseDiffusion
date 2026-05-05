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

- Cosine / shifted cosine noise schedule
- Min-SNR weighting strategy
- Transformer-based backbone (planned / partial)
- Positional encoding via RoPE

Training follows a progressive difficulty schedule:
1. Short sequences first
2. Gradual extension to longer sequences


## 🎯 Sampling

Two sampling strategies:
- **DDPM**: stochastic, stable
- **DDIM**: deterministic, faster inference
- **Diffusion forcing**

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