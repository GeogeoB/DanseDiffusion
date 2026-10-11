"""Render MHR sequences (dataset/latent format) to an mp4 / gif with a small software rasterizer.

Usage (from the repository root):
    uv run --project dashboard --with imageio-ffmpeg python dashboard/render.py \
        runs/<run>/samples/step_XXXXXXX/sample_0.npz [more.npz ...] -o media/dance.mp4 \
        [--cols 3] [--seconds 10] [--fps-in 10] [--fps-out 30] [--lod 3]
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from server import ROOT, MeshBuilder

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BACKGROUND = (244, 243, 240)
GRID = (218, 217, 212)
SHADOW = (205, 204, 199)


def hex_rgb(color: str) -> np.ndarray:
    return np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)


def catmull_rom(frames: np.ndarray, fps_in: float, fps_out: float) -> np.ndarray:
    """(F, V, 3) -> resampled at fps_out with Catmull-Rom interpolation."""
    n = len(frames)
    times = np.arange(0, n - 1, fps_in / fps_out)
    k = np.floor(times).astype(int)
    t = (times - k)[:, None, None]
    p0, p1 = frames[np.maximum(k - 1, 0)], frames[k]
    p2, p3 = frames[np.minimum(k + 1, n - 1)], frames[np.minimum(k + 2, n - 1)]
    t2, t3 = t * t, t * t * t
    return (
        (-0.5 * t3 + t2 - 0.5 * t) * p0
        + (1.5 * t3 - 2.5 * t2 + 1) * p1
        + (-1.5 * t3 + 2 * t2 + 0.5 * t) * p2
        + (0.5 * t3 - 0.5 * t2) * p3
    )


class Camera:
    def __init__(self, eye, target, width, height, fov_deg=35.0):
        self.eye = np.asarray(eye, dtype=np.float32)
        forward = np.asarray(target, dtype=np.float32) - self.eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        self.rotation = np.stack([right, up, forward])  # world -> camera
        self.focal = 0.5 * height / np.tan(np.radians(fov_deg) / 2)
        self.width, self.height = width, height

    def project(self, points: np.ndarray):
        cam = (points - self.eye) @ self.rotation.T
        z = np.maximum(cam[..., 2], 1e-3)
        x = self.width / 2 + self.focal * cam[..., 0] / z
        y = self.height / 2 - self.focal * cam[..., 1] / z
        return np.stack([x, y], -1), cam[..., 2]


def render_frame(cam: Camera, actors, faces: np.ndarray, grid_lines, scale: int) -> Image.Image:
    image = Image.new("RGB", (cam.width, cam.height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    for a, b in grid_lines:  # 1 m segments, the ones behind the camera are skipped
        (pa, pb), (za, zb) = cam.project(np.stack([a, b]))
        if za > 0.1 and zb > 0.1:
            draw.line([tuple(pa), tuple(pb)], fill=GRID, width=scale)

    # soft contact shadow under each dancer
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    for vertices, _ in actors:
        cx, cz = vertices[:, 0].mean(), vertices[:, 2].mean()
        ring = np.stack([cx + 0.38 * np.cos(angles), np.zeros_like(angles), cz + 0.3 * np.sin(angles)], -1)
        screen, depth = cam.project(ring)
        if (depth > 0.1).all():
            draw.polygon([tuple(p) for p in screen], fill=SHADOW)

    light = np.array([0.4, 0.8, 0.45], dtype=np.float32)
    light /= np.linalg.norm(light)
    polygons, depths, colors = [], [], []
    for vertices, color in actors:
        tri = vertices[faces]  # (T, 3, 3)
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9
        center = tri.mean(1)
        visible = np.einsum("ij,ij->i", normal, cam.eye - center) > 0
        screen, depth = cam.project(tri[visible])
        shade = 0.35 + 0.65 * np.clip(normal[visible] @ light, 0, 1)
        polygons.append(screen)
        depths.append(depth.mean(1))
        colors.append(np.clip(shade[:, None] * color, 0, 255).astype(np.uint8))

    polygons = np.concatenate(polygons)
    depths = np.concatenate(depths)
    colors = np.concatenate(colors)
    for i in np.argsort(-depths):  # painter's algorithm: far to near
        c = tuple(colors[i].tolist())
        draw.polygon([tuple(p) for p in polygons[i]], fill=c, outline=c)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True, help=".mp4 or .gif")
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=None, help="only the first seconds")
    parser.add_argument("--fps-in", type=float, default=10)
    parser.add_argument("--fps-out", type=float, default=30)
    parser.add_argument("--lod", type=int, default=2)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--supersample", type=int, default=2)
    args = parser.parse_args()

    mesh = MeshBuilder(ROOT / "MHR" / "assets", args.lod)
    n = len(args.files)
    cols = args.cols or n
    rows = int(np.ceil(n / cols))
    spacing = 1.3

    sequences = []
    for i, path in enumerate(args.files):
        data = np.load(path)
        frames = len(data["sequence_shape"])
        if args.seconds:
            frames = min(frames, int(args.seconds * args.fps_in) + 2)
        v = mesh.vertices(
            data["sequence_shape"][:frames],
            data["sequence_mhr_model_params_latent"][:frames],
            data["sequence_expr_params"][:frames],
        ) * 0.01  # cm -> m
        # centered on the first frame, lowest point on the floor, placed on the grid
        v[..., 0] -= v[0, :, 0].mean()
        v[..., 2] -= v[0, :, 2].mean()
        v[..., 1] -= v[..., 1].min()
        v[..., 0] += ((i % cols) - (cols - 1) / 2) * spacing
        v[..., 2] += ((i // cols) - (rows - 1) / 2) * spacing
        sequences.append(catmull_rom(v, args.fps_in, args.fps_out))
    n_out = min(len(s) for s in sequences)

    W, H, s = args.width * args.supersample, args.height * args.supersample, args.supersample
    extent = max(cols, rows) * spacing
    if n == 1:
        cam = Camera(eye=[0, 1.5, 4.2], target=[0, 0.85, 0], width=W, height=H)
    else:
        cam = Camera(eye=[0, 1.0 + extent * 0.8, extent * 1.05 + 1.5], target=[0, 0.5, 0], width=W, height=H)

    half = int(extent) + 6
    grid_lines = [((x, 0, z), (x, 0, z + 1)) for x in range(-half, half + 1) for z in range(-half, half)]
    grid_lines += [((x, 0, z), (x + 1, 0, z)) for z in range(-half, half + 1) for x in range(-half, half)]

    colors = [hex_rgb(COLORS[i % len(COLORS)]) for i in range(n)]
    images = []
    for f in range(n_out):
        frame = render_frame(cam, [(seq[f], c) for seq, c in zip(sequences, colors)], mesh.faces, grid_lines, s)
        images.append(np.asarray(frame.resize((args.width, args.height), Image.LANCZOS)))
        if f % 50 == 0:
            print(f"frame {f}/{n_out}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".gif":
        pil = [Image.fromarray(im).quantize(colors=128, method=Image.Quantize.MEDIANCUT) for im in images]
        pil[0].save(args.output, save_all=True, append_images=pil[1:], duration=1000 / args.fps_out, loop=0, optimize=True)
    else:
        import imageio_ffmpeg

        writer = imageio_ffmpeg.write_frames(
            str(args.output), (args.width, args.height), fps=args.fps_out, quality=8, macro_block_size=8
        )
        writer.send(None)
        for im in images:
            writer.send(np.ascontiguousarray(im))
        writer.close()
    print(f"wrote {args.output} ({n_out} frames)")


if __name__ == "__main__":
    main()
