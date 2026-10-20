import torch

def silu(x):
    return x/(1+torch.exp(-x))


def softmax(x,dim=-1):
    max_value=torch.amax(x,dim=dim,keepdim=True)
    shifted=x-max_value
    exp_x=torch.exp(shifted)
    return exp_x/exp_x.sum(dim=dim,keepdim=True)