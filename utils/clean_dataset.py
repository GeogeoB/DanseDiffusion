"""Keep a single camera per choreography: the same performance filmed by several cameras
(_c01 ... _c09) is the same motion seen from another viewpoint, which duplicates the data.

For each choreography (sequence name without camera _cXX and track _pK), the camera with the most
tracked frames is kept, with all its tracks. The source folder is left untouched.

Usage:
    python utils/clean_dataset.py [--latent dataset/latent] [--out dataset/latent_cleaned]
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def camera(name: str) -> str:
    m = re.search(r"_c(\d+)", name)
    return m.group(1) if m else ""


def choreography(name: str) -> str:
    return re.sub(r"_p\d+$", "", re.sub(r"_c\d+", "", name))


def source(name: str) -> str:
    m = re.match(r"(g[A-Z]{2}_s[A-Z]{2}|s[A-Z]{2})", name)
    return m.group(1) if m else name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent", type=Path, default=Path("dataset/latent"))
    parser.add_argument("--out", type=Path, default=Path("dataset/latent_cleaned"))
    args = parser.parse_args()

    lengths = {p.stem: len(np.load(p)["sequence_shape"]) for p in sorted(args.latent.glob("*.npz"))}

    # choreography -> camera -> sequence names
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for name in lengths:
        groups[choreography(name)][camera(name)].append(name)

    kept, report = [], {}
    for chor, cameras in sorted(groups.items()):
        best = max(sorted(cameras), key=lambda c: sum(lengths[n] for n in cameras[c]))
        kept.extend(cameras[best])
        report[chor] = {
            "kept_camera": best,
            "cameras": {c: sum(lengths[n] for n in names) for c, names in sorted(cameras.items())},
        }

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    for name in kept:
        shutil.copy2(args.latent / f"{name}.npz", args.out / f"{name}.npz")
    (args.out / "cleaning_report.json").write_text(json.dumps(report, indent=2))

    def stats(names):
        per_source = defaultdict(int)
        for n in names:
            per_source[source(n)] += lengths[n]
        return sum(lengths[n] for n in names), per_source

    total_before, src_before = stats(list(lengths))
    total_after, src_after = stats(kept)
    print(f"choreographies: {len(groups)}")
    print(f"sequences: {len(lengths)} -> {len(kept)}")
    print(f"frames: {total_before} -> {total_after}")
    for s, n in sorted(src_after.items(), key=lambda kv: -kv[1]):
        print(f"  {s:10s} {src_before[s]:7d} -> {n:7d}  ({100 * n / total_after:4.1f}%)")
    print(f"written to {args.out}")
