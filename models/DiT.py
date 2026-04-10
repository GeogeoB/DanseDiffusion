import torch.nn as nn
import torch


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding
    Reference : https://blog.eleuther.ai/rotary-embeddings/ Paper: https://huggingface.co/papers/2104.09864
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        rotary_embedding_base: int = 10_000,
    ):
        super().__init__()
        dim = hidden_size // num_attention_heads
        base = rotary_embedding_base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        # Ignore copy
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.cached_sequence_length = None
        self.cached_rotary_positional_embedding = None

    def forward(self, hidden_states):
        sequence_length = hidden_states.shape[2]

        if (
            sequence_length == self.cached_sequence_length
            and self.cached_rotary_positional_embedding is not None
        ):
            return self.cached_rotary_positional_embedding

        self.cached_sequence_length = sequence_length
        # Embeddings are computed in the dtype of the inv_freq constant
        time_stamps = torch.arange(sequence_length).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", time_stamps, self.inv_freq)
        embeddings = torch.cat((freqs, freqs), dim=-1)

        cos_embeddings = embeddings.cos()[None, None, :, :]
        sin_embeddings = embeddings.sin()[None, None, :, :]
        # Computed embeddings are cast to the dtype of the hidden state inputs
        self.cached_rotary_positional_embedding = torch.stack(
            [cos_embeddings, sin_embeddings]
        ).type_as(hidden_states)
        return self.cached_rotary_positional_embedding


class ROPE(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.rotary_positional_embedding = RotaryPositionalEmbedding(
            hidden_size=hidden_size, num_attention_heads=num_attention_heads
        )

    def _rotated_x(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): (B, num_heads, q_len, head_dim)
        """
        cos, sin = self.rotary_positional_embedding(x)
        return x * cos + self._rotated_x(x) * sin


class ROPEMHAAttention(nn.Module):
    """
    Implementation of ROPEAttention from https://github.com/huggingface/transformers/blob/e42587f596181396e1c4b63660abf0c736b10dae/src/transformers/models/llama/modeling_llama.py#L233-L368
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.positional_embedding = ROPE(
            hidden_size=hidden_size, num_attention_heads=num_attention_heads
        )

    def forward(self, hidden_states: torch.Tensor):
        bsz, q_len, _ = hidden_states.size()

        query_states: torch.Tensor = self.q_proj(
            hidden_states
        )  # (B, L, num_head*head_dim)
        key_states: torch.Tensor = self.k_proj(
            hidden_states
        )  # (B, L, num_head*head_dim)
        value_states: torch.Tensor = self.v_proj(
            hidden_states
        )  # (B, L, num_head*head_dim)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)
        key_states = key_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)
        value_states = value_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)

        query_states = self.positional_embedding(
            query_states
        )  # (bsz, num_heads, q_len, head_dim)
        key_states = self.positional_embedding(
            key_states
        )  # (bsz, num_heads, q_len, head_dim)

        scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / (
            self.head_dim**0.5
        )
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class AdaLN(nn.Module):
    def __init__(self, conditioning_size: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.mlp = nn.Linear(conditioning_size, 6)

    def forward(self, c: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): (B, q_len, dim)
        """
        b, q_len, dim = c.size()
        c = c.view((b * q_len, dim))

        c = self.mlp(self.silu(c)).view(b, q_len, 6).permute(2, 0, 1)

        return c


class PointwiseFeedforwardMLP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class DiTBlock(nn.Module):
    """
    Implementation of DITBlock from https://arxiv.org/pdf/2212.09748
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        conditioning_size: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,  # Doesn't learn gamme and Beta by itself
            eps=1.0e-6,
        )
        self.attn = ROPEMHAAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,  # Doesn't learn gamme and Beta by itself
            eps=1.0e-6,
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approw_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = PointwiseFeedforwardMLP()
        self.adaLN_modulation = AdaLN(conditioning_size=conditioning_size)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward method of the DiTBlock from https://arxiv.org/pdf/2212.09748

        Args:
            x (torch.Tensor): the input tokens (B, N, D)
            c (torch.Tensor): the conditioning tokens (B, N, D')

        Returns:
            torch.Tensor: output token
        """
        print(f"{x.size() = }")
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN_modulation(c)
        print(
            f"{gamma1.shape = }, {beta1.shape = }, {alpha1.shape = }, {gamma2.shape = }, {beta2.shape = }, {alpha2.shape = }"
        )

        x += alpha1[:, :, None] * self.attn(gamma1[:, :, None] * self.norm1(x) + beta1[:, :, None])
        # x += alpha2[:, :, None] * self.attn(gamma2[:, :, None] * self.mlp(x) + beta2[:, :, None])


if __name__ == "__main__":
    hidden_size = 1024
    num_attention_heads = 8
    q_len = 20
    batch_size = 4
    conditioning_size=256

    dit_block = DiTBlock(
        hidden_size=hidden_size,
        num_heads=num_attention_heads,
        conditioning_size=conditioning_size
    )

    x = torch.zeros((batch_size, q_len, hidden_size))
    c = torch.zeros((batch_size, q_len, conditioning_size))
    dit_block(x, c)
