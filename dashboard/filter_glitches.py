"""Remove generated dances with glitches (discontinuous jumps) from a dashboard run.

A frame is a glitch when the body moves more between two frames than in almost all real
frames: displacement = 95th percentile over the mesh vertices of |v_t - v_t-1| (cm), compared
with the `--percentile` of the same measure on real training sequences. Measured on the mesh,
not on the parameters (an angle wrapping from +pi to -pi is a jump in parameters only).

Rejected samples (and their cached meshes) are moved to <step dir>/rejected/, nothing is deleted.

Usage (from the repository root):
    uv run --project dashboard python dashboard/filter_glitches.py 100x1min_ctx400_hg0.5 [--dry-run]
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from server import ROOT, MeshBuilder


def body_displacement(mesh: MeshBuilder, npz: Path) -> np.ndarray:
    data = np.load(npz)
    vertices = mesh.vertices(
        data["sequence_shape"], data["sequence_mhr_model_params_latent"], data["sequence_expr_params"]
    )
    step = np.linalg.norm(np.diff(vertices, axis=0), axis=-1)  # (frames - 1, vertices) cm
    return np.percentile(step, 95, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--step", type=int, default=None, help="evaluation step (default: last)")
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--max-glitches", type=int, default=0, help="glitch frames allowed")
    parser.add_argument("--n-real", type=int, default=80, help="real sequences for the threshold")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = ROOT / "runs" / args.run
    config = json.loads((run_dir / "config.json").read_text())
    steps = sorted((run_dir / "samples").glob("step_*"))
    step_dir = run_dir / "samples" / f"step_{args.step:07d}" if args.step else steps[-1]

    mesh = MeshBuilder(ROOT / "MHR" / "assets", lod=6)

    # threshold from real training sequences
    rng = np.random.default_rng(0)
    names = config["sequences"]
    names = rng.choice(names, min(args.n_real, len(names)), replace=False)
    real = np.concatenate([body_displacement(mesh, ROOT / config["data"] / f"{n}.npz") for n in names])
    threshold = float(np.percentile(real, args.percentile))
    print(f"threshold: p{args.percentile} of real frames = {threshold:.1f} cm/frame")

    samples = sorted(step_dir.glob("sample_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    report = {}
    for npz in samples:
        d = body_displacement(mesh, npz)
        glitches = np.flatnonzero(d > threshold) + 1  # frame index after the jump
        report[npz.stem] = {"glitch_frames": glitches.tolist(), "max_cm": float(d.max())}

    rejected = [k for k, v in report.items() if len(v["glitch_frames"]) > args.max_glitches]
    print(f"{len(rejected)}/{len(samples)} samples with glitches, {len(samples) - len(rejected)} kept")

    if args.dry_run:
        return

    out = step_dir / "rejected"
    out.mkdir(exist_ok=True)
    for name in rejected:
        index = name.split("_")[1]
        for f in [step_dir / f"{name}.npz", *step_dir.glob(f"mesh_lod*_{index}.bin")]:
            shutil.move(str(f), out / f.name)
    (out / "report.json").write_text(
        json.dumps({"threshold_cm": threshold, "percentile": args.percentile, "samples": report}, indent=2)
    )
    print(f"moved to {out}")


if __name__ == "__main__":
    main()
