import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self,d_model,eps=1e-5,device=None,dtype=None):
        super().__init__()

        self.eps=eps
        self.weight=nn.Parameter(
            torch.ones(d_model,device=device,dtype=dtype)
        )

    def forward(self,x):
        original_dtype=x.dtype
        x=x.to(torch.float32)

        mean_square=(x**2).mean(dim=-1,keepdim=True)
        rms=torch.sqrt(mean_square+self.eps)

        result=(x/rms)*self.weight
        return result.to(original_dtype)