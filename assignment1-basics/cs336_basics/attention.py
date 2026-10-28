import torch
import torch.nn as nn

from cs336_basics.activations import softmax
from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1)
    scores = scores / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = softmax(scores, dim=-1)
    return weights @ V


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        max_seq_len=None,
        theta=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须被 num_heads整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )
        self.k_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )
        self.v_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )
        self.output_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )

        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_head,
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            self.rope = None

    def forward(self, x, token_positions=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        head_shape = x.shape[:-1] + (self.num_heads, self.d_head)

        q = q.reshape(head_shape).transpose(-3, -2)
        k = k.reshape(head_shape).transpose(-3, -2)
        v = v.reshape(head_shape).transpose(-3, -2)

        seq_len = x.shape[-2]

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    seq_len,
                    device=x.device,
                    dtype=torch.long,
                )
            elif token_positions.ndim == x.ndim - 1:
                token_positions = token_positions.unsqueeze(-2)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = torch.ones(
            seq_len,
            seq_len,
            device=x.device,
            dtype=torch.bool,
        ).tril()

        head_outputs = scaled_dot_product_attention(q, k, v, mask)

        head_outputs = head_outputs.transpose(-3, -2)
        merged = head_outputs.reshape(x.shape)

        return self.output_proj(merged)
