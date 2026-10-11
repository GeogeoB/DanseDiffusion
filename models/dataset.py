from pathlib import Path

import numpy as np
import torch


class MotionData:
    """
    Sequences of dataset/latent/*.npz (one file per video) kept in memory.

    Only the first `n_features` MHR model parameters are learned (pose / motion); the remaining
    ones (skeleton scales, nearly constant) are kept aside to rebuild full MHR parameters.
    """

    def __init__(
        self,
        folder: str | Path,
        n_features: int = 136,
        sequences: list[str] | None = None,
    ):
        paths = sorted(Path(folder).glob("*.npz"))
        if sequences:
            paths = [p for p in paths if p.stem in set(sequences)]
            missing = set(sequences) - {p.stem for p in paths}
            if missing:
                raise FileNotFoundError(f"Sequences not found in {folder}: {sorted(missing)}")
        if not paths:
            raise FileNotFoundError(f"No .npz sequence in {folder}")

        self.n_features = n_features
        self.names = []
        self.motions = []  # (frames, n_features)
        self.tails = []  # (frames, 204 - n_features)
        self.shapes = []
        self.exprs = []

        for path in paths:
            data = np.load(path)
            mhr = data["sequence_mhr_model_params_latent"].astype(np.float32)

            self.names.append(path.stem)
            self.motions.append(mhr[:, :n_features])
            self.tails.append(mhr[:, n_features:])
            self.shapes.append(data["sequence_shape"].astype(np.float32))
            self.exprs.append(data["sequence_expr_params"].astype(np.float32))

    def __len__(self):
        return len(self.motions)

    @property
    def n_frames(self):
        return sum(len(m) for m in self.motions)

    def reference(self, index: int = 0) -> dict[str, np.ndarray]:
        """Per-frame constant parameters (one frame) used to turn generated motion into MHR."""
        return {
            "tail": self.tails[index].mean(0),
            "shape": self.shapes[index][0],
            "expr": self.exprs[index][0],
        }


class Normalizer:
    """Per-feature standardization. Constant features get std = 1 (they stay at 0)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, motions: list[np.ndarray], min_std: float = 1e-5):
        data = torch.from_numpy(np.concatenate(motions, axis=0)).double()
        mean = data.mean(0)
        std = data.std(0)
        std = torch.where(std < min_std, torch.ones_like(std), std)
        return cls(mean.float(), std.float())

    def to(self, device):
        return Normalizer(self.mean.to(device), self.std.to(device))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def state_dict(self):
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["mean"], state["std"])


class WindowSampler:
    """
    All windows of `window` consecutive frames, sampled directly on the GPU.
    Sequences are concatenated; windows never cross a sequence boundary.
    """

    def __init__(
        self,
        motions: list[np.ndarray],
        normalizer: Normalizer,
        window: int,
        device,
        stride: int = 1,
        groups: list[str] | None = None,
        balance_power: float = 1.0,
    ):
        """
        groups / balance_power: windows of a group (e.g. a data source) are sampled with total
        probability proportional to (group size) ** balance_power (1 = uniform over windows,
        0.5 = sqrt balancing, 0 = every group equally likely).
        """
        self.window = window
        self.device = device

        data, starts, window_groups, offset = [], [], [], 0
        for i, motion in enumerate(motions):
            n = len(motion)
            if n >= window:
                new = range(offset, offset + n - window + 1, stride)
                starts.extend(new)
                window_groups.extend([groups[i] if groups else ""] * len(new))
            data.append(torch.from_numpy(motion))
            offset += n

        if not starts:
            raise ValueError(f"No sequence is long enough for a window of {window} frames")

        self.data = normalizer.normalize(torch.cat(data).to(device))
        self.starts = torch.tensor(starts, device=device)
        self._offsets = torch.arange(window, device=device)

        self.weights = None
        if groups and balance_power != 1.0:
            counts: dict[str, int] = {}
            for g in window_groups:
                counts[g] = counts.get(g, 0) + 1
            w = [counts[g] ** (balance_power - 1.0) for g in window_groups]
            self.weights = torch.tensor(w, device=device, dtype=torch.float)

    def __len__(self):
        return len(self.starts)

    def _gather(self, starts: torch.Tensor) -> torch.Tensor:
        return self.data[starts[:, None] + self._offsets]  # (B, window, D)

    def sample(self, batch_size: int, generator: torch.Generator | None = None):
        if self.weights is not None:
            idx = torch.multinomial(self.weights, batch_size, replacement=True, generator=generator)
        else:
            idx = torch.randint(
                len(self.starts), (batch_size,), device=self.device, generator=generator
            )
        return self._gather(self.starts[idx])

    def all_windows(self) -> torch.Tensor:
        return self._gather(self.starts)

    def subset(self, n: int, seed: int = 0) -> torch.Tensor:
        """Fixed random subset of at most n windows (uniform, for evaluation)."""
        if len(self.starts) <= n:
            return self.all_windows()
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(len(self.starts), generator=g)[:n].to(self.device)
        return self._gather(self.starts[idx])
