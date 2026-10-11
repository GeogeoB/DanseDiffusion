"""Train a Diffusion Forcing DiT on MHR motion sequences.

Usage (overfit a single sequence):
    uv run python models/train.py --sequences gBR_sBM_c01_d04_mBR3_ch03 --run-name overfit
"""

import argparse
import copy
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MotionData, Normalizer, WindowSampler
from diffusion import (
    CosineScheduler,
    diffusion_loss,
    generate,
    q_sample,
    sample_noise_levels,
)
from sample import build_model, motion_metrics, plot_samples, save_samples


def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data", default="dataset/latent")
    p.add_argument("--sequences", nargs="*", help="train on these sequences only")
    p.add_argument("--val-sequences", nargs="*", help="held-out sequences")
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="hold out ~this fraction of the frames (whole choreographies, seeded) "
        "if --val-sequences is not given",
    )
    p.add_argument("--n-features", type=int, default=136)
    p.add_argument("--window", type=int, default=20)
    p.add_argument(
        "--balance-power",
        type=float,
        default=0.5,
        help="sampling weight of a data source ~ size**power (1 = uniform, 0.5 = sqrt balancing)",
    )
    # model
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--causal", action="store_true")
    # diffusion
    p.add_argument("--diffusion-steps", type=int, default=1000)
    p.add_argument("--min-snr-gamma", type=float, default=5.0, help="<= 0 to disable")
    p.add_argument("--velocity-weight", type=float, default=0.3)
    p.add_argument("--p-shared", type=float, default=0.1)
    p.add_argument("--p-clean-prefix", type=float, default=0.2)
    # optimization
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--no-amp", action="store_true")
    # evaluation / sampling
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--n-eval-samples", type=int, default=8)
    p.add_argument("--sample-frames", type=int, default=200)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--context", type=int, default=10)
    p.add_argument("--pyramid", type=int, default=0)
    # misc
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def data_source(name: str) -> str:
    """Genre + situation for AIST names (gBR_sFM_...), video type otherwise (sCY_...)."""
    m = re.match(r"(g[A-Z]{2}_s[A-Z]{2}|s[A-Z]{2})", name)
    return m.group(1) if m else name


def choreography(name: str) -> str:
    """Sequence name without camera (_cXX) and track (_pK): views / dancers of one choreography."""
    return re.sub(r"_p\d+$", "", re.sub(r"_c\d+", "", name))


def split_by_choreography(data_dir, names, fraction: float, seed: int):
    """Hold out whole choreographies (all cameras and dancers) until ~fraction of the frames."""
    paths = sorted(Path(data_dir).glob("*.npz"))
    if names:
        paths = [p for p in paths if p.stem in set(names)]
    lengths = {p.stem: len(np.load(p)["sequence_shape"]) for p in paths}

    groups: dict[str, list[str]] = {}
    for name in lengths:
        groups.setdefault(choreography(name), []).append(name)
    group_frames = {g: sum(lengths[n] for n in members) for g, members in groups.items()}
    total = sum(group_frames.values())

    rng = np.random.default_rng(seed)
    val, val_frames = set(), 0
    for g in rng.permutation(sorted(groups)):
        if val_frames >= fraction * total:
            break
        if group_frames[g] > 2 * fraction * total:  # never hold out a huge chunk of the data
            continue
        val.add(g)
        val_frames += group_frames[g]

    train_names = sorted(n for g, m in groups.items() if g not in val for n in m)
    val_names = sorted(n for g, m in groups.items() if g in val for n in m)
    return train_names, val_names


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate_loss(model, windows, scheduler, autocast, seed=0):
    """x0 MSE (normalized space) on fixed windows, at fixed noise levels."""
    g = torch.Generator(device=windows.device).manual_seed(seed)
    noise = torch.randn(windows.shape, device=windows.device, generator=g)
    B, L, _ = windows.shape
    T = scheduler.T
    metrics = {}

    for t_val in [50, 250, 500, 750, 950]:
        t = torch.full((B, L), t_val, device=windows.device)
        with autocast():
            pred = model(q_sample(windows, t, scheduler, noise), t)
        metrics[f"x0_mse/t{t_val}"] = ((pred.float() - windows) ** 2).mean().item()

    # next-frame prediction: clean context, last frame pure noise
    t = torch.full((B, L), -1, device=windows.device)
    t[:, -1] = T - 1
    with autocast():
        pred = model(q_sample(windows, t, scheduler, noise), t.clamp(min=0))
    metrics["x0_mse/next_frame"] = ((pred[:, -1].float() - windows[:, -1]) ** 2).mean().item()

    return metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.val_fraction > 0 and not args.val_sequences:
        args.sequences, args.val_sequences = split_by_choreography(
            args.data, args.sequences, args.val_fraction, args.seed
        )

    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"run_dir": str(run_dir)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # ---------------- data ----------------
    train_data = MotionData(args.data, args.n_features, args.sequences)
    normalizer = Normalizer.fit(train_data.motions)
    train_sampler = WindowSampler(
        train_data.motions,
        normalizer,
        args.window,
        device,
        groups=[data_source(n) for n in train_data.names],
        balance_power=args.balance_power,
    )

    val_windows = None
    if args.val_sequences:
        val_data = MotionData(args.data, args.n_features, args.val_sequences)
        val_windows = WindowSampler(val_data.motions, normalizer, args.window, device).subset(512)

    # evaluation subsets (fixed): denoising losses and nearest-neighbour reference windows
    train_eval_windows = train_sampler.subset(512)
    reference_windows = train_sampler.subset(20_000, seed=1)
    reference = train_data.reference(0)
    real_sequence = normalizer.normalize(torch.from_numpy(train_data.motions[0]).to(device))
    # plot the features that move the most
    plot_features = torch.from_numpy(np.concatenate(train_data.motions).std(0)).argsort(
        descending=True
    )[:6].tolist()

    print(
        f"train: {len(train_data)} sequences, {train_data.n_frames} frames, "
        f"{len(train_sampler)} windows of {args.window}"
    )
    if val_windows is not None:
        print(f"val: {len(args.val_sequences)} sequences, {len(val_windows)} windows")

    # ---------------- model ----------------
    model = build_model(config).to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    scheduler = CosineScheduler(args.diffusion_steps, device=device)
    gamma = args.min_snr_gamma if args.min_snr_gamma > 0 else None

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim < 2 or "embedder" in name else decay).append(param)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.99),
        fused=device == "cuda",
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, args.warmup, args.steps)
    )

    use_amp = device == "cuda" and not args.no_amp

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp)

    writer = SummaryWriter(run_dir)

    def save_checkpoint(name):
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "config": config,
                "normalizer": normalizer.state_dict(),
                "reference": reference,
            },
            run_dir / "checkpoints" / name,
        )

    # ---------------- training ----------------
    step = 0
    pbar = tqdm(total=args.steps, dynamic_ncols=True)
    while step < args.steps:
        model.train()
        x0 = train_sampler.sample(args.batch_size)
        t, context_mask = sample_noise_levels(
            *x0.shape[:2],
            args.diffusion_steps,
            device,
            p_shared=args.p_shared,
            p_clean_prefix=args.p_clean_prefix,
        )

        with autocast():
            losses = diffusion_loss(
                model, x0, t, context_mask, scheduler, gamma, args.velocity_weight
            )

        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        lr_scheduler.step()

        with torch.no_grad():
            ema_decay = min(args.ema, (1 + step) / (10 + step))
            for p_ema, p in zip(ema.parameters(), model.parameters()):
                p_ema.lerp_(p, 1 - ema_decay)

        step += 1
        pbar.update(1)

        if step % 50 == 0:
            for k, v in losses.items():
                writer.add_scalar(f"train/{k}", v.item(), step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), step)
            writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], step)
            pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

        if step % args.eval_every == 0 or step == args.steps:
            ema.eval()
            metrics = {
                f"train_{k}": v
                for k, v in evaluate_loss(ema, train_eval_windows, scheduler, autocast).items()
            }
            if val_windows is not None:
                metrics |= {
                    f"val_{k}": v
                    for k, v in evaluate_loss(ema, val_windows, scheduler, autocast).items()
                }

            with autocast():
                samples = generate(
                    ema,
                    scheduler,
                    n_samples=args.n_eval_samples,
                    n_frames=args.sample_frames,
                    dim=args.n_features,
                    window=args.window,
                    n_context=args.context,
                    n_steps=args.sample_steps,
                    pyramid=args.pyramid,
                )
            metrics |= {
                f"sample/{k}": v
                for k, v in motion_metrics(samples, reference_windows).items()
            }

            for k, v in metrics.items():
                writer.add_scalar(k, v, step)
            tqdm.write(f"[{step}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

            sample_dir = run_dir / "samples" / f"step_{step:07d}"
            save_samples(normalizer.denormalize(samples), reference, sample_dir)
            plot_samples(
                samples,
                real_sequence,
                sample_dir / "features.png",
                plot_features,
                title=f"{run_name} step {step}",
            )
            save_checkpoint("last.pt")

    pbar.close()
    writer.close()


if __name__ == "__main__":
    main()
