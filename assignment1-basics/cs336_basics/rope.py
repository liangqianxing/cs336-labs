import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta,d_k,max_seq_len,device=None):
        super().__init__()
        
        if d_k%2!=0:
            raise ValueError("d_k must be even")

        self.d_k=d_k

        pair_dims=torch.arange(
            0,d_k,2,
            device=device,
            dtype=torch.float32,
        )

        inv_freq=theta**(-pair_dims/d_k)
        positions=torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )

        angles=torch.outer(positions,inv_freq)

        self.register_buffer(
            "cos_cached",torch.cos(angles),persistent=False
        )
        self.register_buffer(
            "sin_cached",torch.sin(angles),persistent=False
        )

    def forward(self,x,token_positions):
        cos= self.cos_cached[token_positions]
        sin= self.sin_cached[token_positions]

        x_even=x[...,0::2]
        x_odd=x[...,1::2]

        rotated_even=x_even*cos-x_odd*sin
        rotated_odd=x_even*sin+x_odd*cos

        rotate=torch.stack(
            (rotated_even,rotated_odd),
            dim=-1,
        )

        return rotate.reshape(x.shape).to(dtype=x.dtype)