import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (half-split convention, as in LLaMA / GPT-NeoX).
    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, head_dim: int, base: int = 10_000):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype):
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(positions, self.inv_freq)  # (L, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (L, head_dim)

        return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE and QK-norm."""

    def __init__(self, hidden_size: int, num_heads: int, causal: bool = False):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D)"""
        B, L, D = x.shape

        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, H, L, head_dim)
        q, k = self.q_norm(q.float()).type_as(v), self.k_norm(k.float()).type_as(v)

        cos, sin = self.rope(L, q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.transpose(1, 2).reshape(B, L, D)

        return self.proj(out)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    """
    DiT block with adaLN-Zero (https://arxiv.org/pdf/2212.09748).
    The conditioning is per token, so each frame is modulated by its own noise level
    (needed for Diffusion Forcing).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        causal: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = RoPEAttention(hidden_size, num_heads, causal=causal)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tokens (B, L, D)
            c (torch.Tensor): per-token conditioning (B, L, D)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )
        self.linear = nn.Linear(hidden_size, out_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding of the (per token) diffusion timestep followed by an MLP."""

    def __init__(self, hidden_size: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B, L) timestep per token
        return: (B, L, hidden_size)
        """
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(-1) * freqs  # (B, L, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        return self.mlp(emb)


class DiT(nn.Module):
    """Diffusion Transformer over a sequence of frames, one noise level per frame."""

    def __init__(
        self,
        in_dim: int,
        hidden_size: int = 256,
        n_layers: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        causal: bool = False,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_size)
        self.t_embedder = TimestepEmbedding(hidden_size)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio, causal=causal)
                for _ in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, in_dim)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # adaLN-Zero: every block starts as the identity, the output starts at 0
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, in_dim) noisy frames
        t: (B, L) timestep per frame
        return: (B, L, in_dim) prediction of the clean frames (x0)
        """
        x = self.input_proj(x)
        c = self.t_embedder(t)

        for block in self.blocks:
            x = block(x, c)

        return self.final_layer(x, c)


if __name__ == "__main__":
    model = DiT(in_dim=136, hidden_size=256, n_layers=6, num_heads=4)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    B, L, D = 8, 20, 136
    x = torch.randn(B, L, D)
    t = torch.randint(0, 1000, (B, L))
    print(model(x, t).shape)
