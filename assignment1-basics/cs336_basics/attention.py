import torch 
import torch.nn as nn

from cs336_basics.activations import softmax

def scaled_dot_product_attention(Q,K,V,mask=None):
    d_k = Q.shape[-1]
    scores=Q@K.transpose(-2,-1)
    scores=scores/(d_k**0.5)

    if mask is not None:
        scores=scores.masked_fill(~mask,float("-inf"))

    weights=softmax(scores,dim=-1)
    return weights@V