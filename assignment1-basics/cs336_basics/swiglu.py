import torch
import torch.nn as nn

from cs336_basics.linear import Linear
from cs336_basics.activations import silu

class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff,device=None,dtype=None):
        super().__init__()

        self.w1=Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w3=Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w2=Linear(d_ff,d_model,device=device,dtype=dtype)
    
    def forward(self,x):
        gate=silu(self.w1(x))
        values=self.w3(x)
        combined=gate*values
        return self.w2(combined)