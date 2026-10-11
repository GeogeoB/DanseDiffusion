"""Training dashboard: training curves + 3D MHR dances generated at each evaluation.

Usage (from the repository root):
    uv run --project dashboard python dashboard/server.py
    -> http://localhost:8000

Reads runs/<run>/ (tensorboard events, config.json, samples/step_*/sample_*.npz) and turns
generated MHR parameters into animated meshes on demand (cached next to the samples).
"""

import argparse
import json
import re
import struct
import threading
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
MAX_POINTS = 2000  # per curve


class MeshBuilder:
    """MHR parameters -> mesh vertices, run on CPU at a reduced LOD."""

    def __init__(self, assets: Path, lod: int):
        from mhr.mhr import MHR

        torch.set_num_threads(4)
        self.lod = lod
        self.model = MHR.from_files(folder=assets, device=torch.device("cpu"), lod=lod)
        self.faces = np.asarray(self.model.character.mesh.faces, dtype=np.uint32)
        self.lock = threading.Lock()

    @torch.no_grad()
    def vertices(self, shape, mhr, expr, batch: int = 256) -> np.ndarray:
        out = []
        with self.lock:
            for i in range(0, len(mhr), batch):
                v, _ = self.model(
                    torch.from_numpy(shape[i : i + batch]).float(),
                    torch.from_numpy(mhr[i : i + batch]).float(),
                    torch.from_numpy(expr[i : i + batch]).float(),
                )
                out.append(v.numpy())
        return np.concatenate(out)  # (frames, verts, 3) in cm, Y up


def encode_animation(vertices: np.ndarray) -> bytes:
    """
    Binary format (little endian):
        uint32 n_frames, uint32 n_verts, float32 offset[3], float32 scale[3]  (32 bytes)
        int16  data[n_frames * n_verts * 3]    vertex = data * scale + offset
    """
    n_frames, n_verts, _ = vertices.shape
    vmin = vertices.reshape(-1, 3).min(0)
    vmax = vertices.reshape(-1, 3).max(0)
    offset = (vmin + vmax) / 2
    scale = np.maximum((vmax - vmin) / 2 / 32000, 1e-6)
    q = np.round((vertices - offset) / scale).astype("<i2")

    header = struct.pack("<II3f3f", n_frames, n_verts, *offset, *scale)
    return header + q.tobytes()


ALLOWED_LODS = (1, 2, 3, 4, 5, 6)  # vertices: 18439, 10661, 4899, 2461, 971, 595


class Dashboard:
    def __init__(self, runs_dir: Path, assets: Path, default_lod: int):
        self.runs_dir = runs_dir
        self.assets = assets
        self.default_lod = default_lod
        self._meshes: dict[int, MeshBuilder] = {}
        self._mesh_lock = threading.Lock()
        self._scalars: dict[str, tuple[float, EventAccumulator]] = {}
        self._lock = threading.Lock()
        self.mesh(default_lod)

    def mesh(self, lod: int | None = None) -> MeshBuilder:
        lod = lod or self.default_lod
        if lod not in ALLOWED_LODS:
            raise FileNotFoundError(f"lod {lod} not allowed")
        with self._mesh_lock:
            if lod not in self._meshes:
                self._meshes[lod] = MeshBuilder(self.assets, lod)
            return self._meshes[lod]

    # ---------------- helpers ----------------
    def run_dir(self, run: str) -> Path:
        path = (self.runs_dir / run).resolve()
        if path.parent != self.runs_dir.resolve() or not path.is_dir():
            raise FileNotFoundError(run)
        return path

    @staticmethod
    def config(run_dir: Path) -> dict:
        path = run_dir / "config.json"
        return json.loads(path.read_text()) if path.exists() else {}

    @staticmethod
    def eval_dirs(run_dir: Path) -> list[Path]:
        return sorted((run_dir / "samples").glob("step_*"))

    # ---------------- API ----------------
    def runs(self):
        out = []
        for run_dir in sorted(self.runs_dir.iterdir(), key=lambda p: -p.stat().st_mtime):
            if not run_dir.is_dir():
                continue
            evals = self.eval_dirs(run_dir)
            events = list(run_dir.glob("events.out.tfevents.*"))
            out.append(
                {
                    "name": run_dir.name,
                    "config": self.config(run_dir),
                    "n_evals": len(evals),
                    "last_eval": int(evals[-1].name.split("_")[1]) if evals else None,
                    "updated": max([p.stat().st_mtime for p in events + evals] or [0]),
                }
            )
        return out

    def scalars(self, run: str):
        run_dir = self.run_dir(run)
        events = list(run_dir.glob("events.out.tfevents.*"))
        mtime = max([p.stat().st_mtime for p in events] or [0])

        with self._lock:
            cached = self._scalars.get(run)
            if cached is None or cached[0] < mtime:
                acc = cached[1] if cached else EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
                acc.Reload()
                self._scalars[run] = (mtime, acc)
            acc = self._scalars[run][1]

            curves = {}
            for tag in acc.Tags()["scalars"]:
                events = acc.Scalars(tag)
                stride = max(1, len(events) // MAX_POINTS)
                points = [(e.step, e.value) for e in events[::stride]]
                if events and points[-1][0] != events[-1].step:
                    points.append((events[-1].step, events[-1].value))
                curves[tag] = points
        return curves

    def evals(self, run: str):
        out = []
        for d in self.eval_dirs(self.run_dir(run)):
            samples = sorted(d.glob("sample_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
            out.append(
                {
                    "step": int(d.name.split("_")[1]),
                    "samples": [int(p.stem.split("_")[1]) for p in samples],
                    "plot": (d / "features.png").exists(),
                }
            )
        return out

    def _animation(self, npz_path: Path, cache_path: Path, mesh: MeshBuilder, max_frames: int | None = None) -> bytes:
        if cache_path.exists() and cache_path.stat().st_mtime >= npz_path.stat().st_mtime:
            return cache_path.read_bytes()

        data = np.load(npz_path)
        shape = data["sequence_shape"][:max_frames]
        mhr = data["sequence_mhr_model_params_latent"][:max_frames]
        expr = data["sequence_expr_params"][:max_frames]

        payload = encode_animation(mesh.vertices(shape, mhr, expr))
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_bytes(payload)
        tmp.replace(cache_path)
        return payload

    def sample_animation(self, run: str, step: int, index: int, lod: int | None = None) -> bytes:
        d = self.run_dir(run) / "samples" / f"step_{step:07d}"
        npz = d / f"sample_{index}.npz"
        if not npz.exists():
            raise FileNotFoundError(npz)
        mesh = self.mesh(lod)
        return self._animation(npz, d / f"mesh_lod{mesh.lod}_{index}.bin", mesh)

    def reference_animation(self, run: str, lod: int | None = None) -> bytes:
        """First training sequence of the run (what the model is supposed to imitate)."""
        run_dir = self.run_dir(run)
        config = self.config(run_dir)
        data_dir = ROOT / config.get("data", "dataset/latent")
        names = config.get("sequences") or sorted(p.stem for p in data_dir.glob("*.npz"))
        if not names:
            raise FileNotFoundError("no reference sequence")
        npz = data_dir / f"{names[0]}.npz"
        mesh = self.mesh(lod)
        return self._animation(npz, run_dir / f"reference_mesh_lod{mesh.lod}.bin", mesh)

    def plot(self, run: str, step: int) -> bytes:
        path = self.run_dir(run) / "samples" / f"step_{step:07d}" / "features.png"
        return path.read_bytes()


class Handler(SimpleHTTPRequestHandler):
    dashboard: Dashboard

    routes = [
        (re.compile(r"^/api/runs$"), "runs"),
        (re.compile(r"^/api/faces$"), "faces"),
        (re.compile(r"^/api/runs/([^/]+)/scalars$"), "scalars"),
        (re.compile(r"^/api/runs/([^/]+)/evals$"), "evals"),
        (re.compile(r"^/api/runs/([^/]+)/reference$"), "reference"),
        (re.compile(r"^/api/runs/([^/]+)/evals/(\d+)/samples/(\d+)$"), "sample"),
        (re.compile(r"^/api/runs/([^/]+)/evals/(\d+)/plot$"), "plot"),
    ]

    def log_message(self, format, *args):
        if "/api/" in self.path and " 200 " not in format % args:
            super().log_message(format, *args)

    def _send(self, body: bytes, content_type: str, cache: bool = False):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        query = parse_qs(url.query)
        lod = int(query["lod"][0]) if "lod" in query else None
        if not path.startswith("/api/"):
            return super().do_GET()

        d = self.dashboard
        try:
            for pattern, name in self.routes:
                m = pattern.match(path)
                if not m:
                    continue
                args = m.groups()
                if name == "runs":
                    return self._json(d.runs())
                if name == "faces":
                    return self._send(d.mesh(lod).faces.tobytes(), "application/octet-stream", cache=True)
                if name == "scalars":
                    return self._json(d.scalars(args[0]))
                if name == "evals":
                    return self._json(d.evals(args[0]))
                if name == "reference":
                    return self._send(d.reference_animation(args[0], lod), "application/octet-stream")
                if name == "sample":
                    body = d.sample_animation(args[0], int(args[1]), int(args[2]), lod)
                    return self._send(body, "application/octet-stream", cache=True)
                if name == "plot":
                    return self._send(d.plot(args[0], int(args[1])), "image/png", cache=True)
            self.send_error(HTTPStatus.NOT_FOUND)
        except FileNotFoundError as e:
            self.send_error(HTTPStatus.NOT_FOUND, str(e))
        except Exception as e:  # keep the server alive
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, repr(e))
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--assets", type=Path, default=ROOT / "MHR" / "assets")
    parser.add_argument("--lod", type=int, default=4, help="mesh level of detail (1 = finest)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Loading MHR (LOD {args.lod})...")
    Handler.dashboard = Dashboard(args.runs, args.assets, args.lod)

    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=str(STATIC)))
    print(f"Dashboard on http://localhost:{args.port}  (runs: {args.runs})")
    server.serve_forever()


if __name__ == "__main__":
    main()
