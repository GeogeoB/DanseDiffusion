"""Stack per-frame SAM 3D Body outputs (dataset/data/<seq>/frame_*.npz) into sequence files
(dataset/latent/<seq>.npz, or <seq>_p<k>.npz with one file per tracked person).

A frame file holds one entry per detected person (arr_0, arr_1, ...), ordered left to right, so
the index of a dancer changes whenever dancers cross. People are therefore tracked from frame to
frame with their 3D position (pred_cam_t); short gaps (missed detections) are interpolated and
each track becomes its own sequence.

Usage:
    python utils/create_sequences.py                      # all sequences
    python utils/create_sequences.py --sequences gBR_sBM_c01_d04_mBR3_ch03
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

KEYS = {
    "sequence_shape": "shape_params",
    "sequence_mhr_model_params_latent": "mhr_model_params",
    "sequence_expr_params": "expr_params",
}


def load_frame(frame_file: Path) -> list[dict]:
    data = np.load(frame_file, allow_pickle=True)
    people = []
    for name in data.files:
        person = data[name].item()
        people.append(
            {
                "position": np.asarray(person["pred_cam_t"], dtype=np.float32),
                "height": float(person["bbox"][3] - person["bbox"][1]),
                **{key: np.asarray(person[key], dtype=np.float32) for key in KEYS.values()},
            }
        )
    return people


# pred_cam_t depth is noisy (up to ~0.7 m between consecutive frames) while x / y are stable
# (< 0.2 m): depth is down-weighted in the association distance
POSITION_WEIGHTS = np.array([1.0, 1.0, 0.2], dtype=np.float32)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b) * POSITION_WEIGHTS))


def track_single(frames: list[list[dict]]) -> dict[int, dict]:
    """One dancer: keep the detection closest to the previous one (the largest one first)."""
    track, last = {}, None
    for t, people in enumerate(frames):
        if not people:
            continue
        if last is None:
            person = max(people, key=lambda p: p["height"])
        else:
            person = min(people, key=lambda p: distance(p["position"], last))
        track[t] = person
        last = person["position"]
    return track


def track_people(frames: list[list[dict]], max_distance: float, max_gap: int) -> list[dict]:
    """
    Greedy frame-to-frame association on 3D positions (several dancers).
    Returns tracks as {frame_index: person} dicts.
    """
    tracks = []  # each: {"frames": {t: person}, "last_t": int, "last_pos": array}

    for t, people in enumerate(frames):
        alive = [k for k, tr in enumerate(tracks) if t - tr["last_t"] <= max_gap + 1]
        pairs = sorted(
            (distance(tracks[k]["last_pos"], p["position"]), k, i)
            for k in alive
            for i, p in enumerate(people)
        )
        used_tracks, used_people = set(), set()
        for dist, k, i in pairs:
            if dist > max_distance:
                break
            if k in used_tracks or i in used_people:
                continue
            used_tracks.add(k)
            used_people.add(i)
            tracks[k]["frames"][t] = people[i]
            tracks[k]["last_t"] = t
            tracks[k]["last_pos"] = people[i]["position"]

        for i, person in enumerate(people):
            if i not in used_people:
                tracks.append({"frames": {t: person}, "last_t": t, "last_pos": person["position"]})

    return [tr["frames"] for tr in tracks]


def track_to_arrays(track: dict[int, dict]) -> dict[str, np.ndarray]:
    """Dense arrays from first to last frame of the track, gaps linearly interpolated."""
    times = np.array(sorted(track))
    full = np.arange(times[0], times[-1] + 1)
    arrays = {}
    for out_key, key in KEYS.items():
        values = np.stack([track[t][key] for t in times])  # (n, d)
        if len(times) == len(full):
            arrays[out_key] = values
        else:
            arrays[out_key] = np.stack(
                [np.interp(full, times, values[:, j]) for j in range(values.shape[1])], axis=1
            ).astype(np.float32)
    return arrays


def build_sequence(
    subfolder: Path,
    latent: Path,
    min_frames: int = 64,
    max_distance: float = 0.3,
    max_gap: int = 3,
    max_multi_fraction: float = 0.05,
    min_relative_height: float = 0.3,
) -> list[tuple[str, int]]:
    frame_files = sorted(subfolder.glob("frame_*.npz"))
    if not frame_files:
        return []

    frames = [load_frame(f) for f in frame_files]

    # solo dance with a few spurious detections: never split the sequence
    multi_fraction = np.mean([len(people) > 1 for people in frames])
    if multi_fraction <= max_multi_fraction:
        track = track_single(frames)
        if len(track) < min_frames:
            return []
        arrays = track_to_arrays(track)
        np.savez(latent / f"{subfolder.name}.npz", **arrays)
        return [(subfolder.name, len(arrays["sequence_shape"]))]

    tracks = [tr for tr in track_people(frames, max_distance, max_gap) if len(tr) >= min_frames]
    if not tracks:
        return []

    # drop tiny tracks (far background people, false positives)
    heights = [np.median([p["height"] for p in tr.values()]) for tr in tracks]
    tracks = [tr for tr, h in zip(tracks, heights) if h >= min_relative_height * max(heights)]
    # largest people first
    tracks.sort(key=lambda tr: -np.median([p["height"] for p in tr.values()]))

    written = []
    for k, track in enumerate(tracks):
        name = f"{subfolder.name}_p{k}"
        arrays = track_to_arrays(track)
        np.savez(latent / f"{name}.npz", **arrays)
        written.append((name, len(arrays["sequence_shape"])))
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("dataset/data"))
    parser.add_argument("--latent", type=Path, default=Path("dataset/latent"))
    parser.add_argument(
        "--sequences", nargs="*", help="Sequence folder names (default: all)"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-frames", type=int, default=64)
    parser.add_argument(
        "--overwrite", action="store_true", help="rebuild sequences that already exist"
    )
    args = parser.parse_args()

    args.latent.mkdir(parents=True, exist_ok=True)

    if args.sequences:
        subfolders = [args.data / name for name in args.sequences]
    else:
        subfolders = sorted(p for p in args.data.iterdir() if p.is_dir())

    def done(subfolder):
        return (args.latent / f"{subfolder.name}.npz").exists() or any(
            args.latent.glob(f"{subfolder.name}_p*.npz")
        )

    jobs = [s for s in subfolders if args.overwrite or not done(s)]
    print(f"{len(jobs)} sequences to build ({len(subfolders) - len(jobs)} already done)")

    total_frames = total_tracks = 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(build_sequence, s, args.latent, args.min_frames) for s in jobs]
        for subfolder, future in tqdm(zip(jobs, futures), total=len(jobs)):
            written = future.result()
            total_tracks += len(written)
            total_frames += sum(n for _, n in written)
            tqdm.write(
                f"{subfolder.name}: {len(written)} track(s) "
                + ", ".join(str(n) for _, n in written)
            )
    print(f"{total_tracks} sequences, {total_frames} frames written to {args.latent}")
