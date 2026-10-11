"""Generate motion with a trained Diffusion Forcing model.

Usage:
    uv run python models/sample.py runs/<run>/checkpoints/last.pt --n 4 --frames 300
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from DiT import DiT
from dataset import MotionData, Normalizer
from diffusion import CosineScheduler, generate


def build_model(config: dict) -> DiT:
    return DiT(
        in_dim=config["n_features"],
        hidden_size=config["hidden"],
        n_layers=config["depth"],
        num_heads=config["heads"],
        mlp_ratio=config["mlp_ratio"],
        causal=config["causal"],
    )


def load_checkpoint(path: str | Path, device, use_ema: bool = True):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = build_model(config).to(device)
    model.load_state_dict(ckpt["ema" if use_ema else "model"])
    model.eval()

    normalizer = Normalizer.from_state_dict(ckpt["normalizer"]).to(device)
    scheduler = CosineScheduler(config["diffusion_steps"], device=device)

    return model, scheduler, normalizer, ckpt["reference"], config


def to_mhr_arrays(motion: np.ndarray, reference: dict) -> dict[str, np.ndarray]:
    """motion: (L, n_features) denormalized -> arrays in the dataset/latent format"""
    L = len(motion)
    mhr = np.concatenate([motion, np.repeat(reference["tail"][None], L, axis=0)], axis=1)
    return {
        "sequence_shape": np.repeat(reference["shape"][None], L, axis=0),
        "sequence_mhr_model_params_latent": mhr.astype(np.float32),
        "sequence_expr_params": np.repeat(reference["expr"][None], L, axis=0),
    }


def save_samples(samples: torch.Tensor, reference: dict, out_dir: Path, prefix: str = "sample"):
    """samples: (N, L, n_features) denormalized"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, motion in enumerate(samples.cpu().numpy()):
        np.savez(out_dir / f"{prefix}_{i}.npz", **to_mhr_arrays(motion, reference))


def plot_samples(
    samples: torch.Tensor,
    real: torch.Tensor,
    path: Path,
    features: list[int],
    title: str = "",
):
    """Plot a few features over time: real sequence vs generated samples (normalized space)."""
    samples, real = samples.cpu().numpy(), real.cpu().numpy()

    fig, axes = plt.subplots(len(features), 1, figsize=(12, 2 * len(features)), sharex=True)
    for ax, f in zip(np.atleast_1d(axes), features):
        ax.plot(real[:, f], color="black", lw=2, alpha=0.5, label="data")
        for i, s in enumerate(samples[:4]):
            ax.plot(s[:, f], lw=1, label=f"sample {i}")
        ax.set_ylabel(f"feat {f}")
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=7, ncol=5)
    np.atleast_1d(axes)[-1].set_xlabel("frame")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


@torch.no_grad()
def motion_metrics(samples: torch.Tensor, train_windows: torch.Tensor, max_ref: int = 20_000):
    """
    Metrics in normalized space:
        - nn_mse: mean distance of each generated window to its nearest training window
          (-> 0 when the model memorizes the data)
        - nn_mse_first: same for the first window only (generated from scratch, before any
          autoregressive drift)
        - jerk_ratio: mean |second difference| of samples / of data (>1: jittery, <1: too smooth)
    """
    W = train_windows.shape[1]
    ref = train_windows
    if len(ref) > max_ref:
        ref = ref[torch.randperm(len(ref), device=ref.device)[:max_ref]]
    ref = ref.flatten(1)

    gen = samples.unfold(1, W, 1).permute(0, 1, 3, 2).flatten(0, 1).flatten(1)
    nn_mse = torch.cdist(gen.float(), ref.float()).min(1).values.pow(2) / ref.shape[1]

    def jerk(x):
        return (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]).abs().mean()

    nn_mse = nn_mse.view(len(samples), -1)

    return {
        "nn_mse": nn_mse.mean().item(),
        "nn_mse_first": nn_mse[:, 0].mean().item(),
        "jerk_ratio": (jerk(samples) / jerk(train_windows)).item(),
    }


@torch.no_grad()
def long_motion_metrics(
    samples: torch.Tensor,
    real_sequences: list[torch.Tensor],
    segment: int = 100,
    window: int = 32,
    stride: int = 8,
):
    """
    Diversity of long generations (normalized space), each compared with real sequences:
        - frozen: fraction of `segment`-frame segments that barely move (temporal std < 10% of
          the real one)
        - repetition: distance of each window to the closest earlier window of the same sequence
          (at least `window` frames before). -> 0 when the dance loops on itself
        - spread: std across samples at the same time step (do the dances differ from each other)
    """

    segment = min(segment, samples.shape[1], *(len(r) for r in real_sequences))

    def activity(x):  # (N, F, D) -> (N, n_segments) temporal std per segment
        n = x.shape[1] // segment
        seg = x[:, : n * segment].reshape(x.shape[0], n, segment, x.shape[-1])
        return seg.std(2).mean(-1)

    def repetition(seq):  # (F, D)
        w = seq.unfold(0, window, stride).flatten(1)  # (n, window * D)
        d = torch.cdist(w.float(), w.float()).pow(2) / w.shape[1]
        i = torch.arange(len(w), device=w.device)
        earlier = i[None, :] <= i[:, None] - window // stride  # j at least `window` frames before
        d = d.masked_fill(~earlier, float("inf")).min(1).values
        return d[torch.isfinite(d)]

    real_activity = torch.cat([activity(r[None]).flatten() for r in real_sequences]).median()
    real_rep = torch.cat([repetition(r) for r in real_sequences]).mean()
    frozen = (activity(samples) < 0.1 * real_activity).float().mean()
    rep = torch.cat([repetition(s) for s in samples]).mean()

    return {
        "frozen": frozen.item(),
        "repetition": rep.item(),
        "repetition_real": real_rep.item(),
        "spread": samples.std(0).mean().item() if len(samples) > 1 else 0.0,
    }


if __name__ == "__main__":
    import json

    from dataset import WindowSampler

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--run", default=None, help="save as runs/<run> (visible in the dashboard) instead of --out"
    )
    parser.add_argument("--fps", type=float, default=10, help="playback rate stored with --run")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--context", type=int, default=None, help="context frames")
    parser.add_argument("--pyramid", type=int, default=0)
    parser.add_argument("--eta", type=float, default=0.0, help="0: DDIM, 1: DDPM-like noise")
    parser.add_argument("--context-t", type=int, default=-1, help="noise level of the context")
    parser.add_argument("--guidance", type=float, default=1.0, help="history guidance weight")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    model, scheduler, normalizer, reference, config = load_checkpoint(
        args.checkpoint, device, use_ema=not args.no_ema
    )
    step = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["step"]
    sampling = {
        k: getattr(args, k)
        for k in ["n", "frames", "steps", "context", "pyramid", "eta", "context_t", "guidance", "seed"]
    }
    sampling["context"] = args.context or config["context"]

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        samples = generate(
            model,
            scheduler,
            n_samples=args.n,
            n_frames=args.frames,
            dim=config["n_features"],
            window=config["window"],
            n_context=sampling["context"],
            n_steps=args.steps,
            pyramid=args.pyramid,
            eta=args.eta,
            context_t=args.context_t,
            guidance=args.guidance,
        )

    if args.run:
        run_dir = Path("runs") / args.run
        out = run_dir / "samples" / f"step_{step:07d}"
    else:
        run_dir = None
        out = args.out or args.checkpoint.parent.parent / "samples"
    save_samples(normalizer.denormalize(samples), reference, out)

    # metrics and plot against the training data
    data = MotionData(config["data"], config["n_features"], config["sequences"])
    windows = WindowSampler(data.motions, normalizer, config["window"], device).subset(20_000, seed=1)
    long_real = [  # the longest real sequences, as a reference for repetition / activity
        normalizer.normalize(torch.from_numpy(m).to(device))
        for m in sorted(data.motions, key=len, reverse=True)[:50]
    ]
    metrics = motion_metrics(samples, windows) | long_motion_metrics(samples, long_real)
    (out / "metrics.json").write_text(json.dumps(metrics | {"sampling": sampling}, indent=2))

    real = normalizer.normalize(torch.from_numpy(data.motions[0]).to(device))
    features = torch.from_numpy(np.concatenate(data.motions).std(0)).argsort(descending=True)[:6]
    title = f"{args.run or args.checkpoint} | " + " ".join(f"{k}={v}" for k, v in sampling.items())
    plot_samples(samples, real, out / "features.png", features.tolist(), title=title)

    if run_dir is not None:
        note = (
            f"Génération de {args.frames} frames depuis {args.checkpoint} (pas {step}) : "
            + ", ".join(f"{k}={v}" for k, v in sampling.items())
        )
        run_config = config | {
            "run_dir": str(run_dir), "fps": args.fps, "sampling": sampling, "note": note,
            "steps": None, "eval_every": None,
        }
        (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    print(f"Saved {args.n} samples of {args.frames} frames to {out}")
    print(json.dumps(metrics, indent=2))
