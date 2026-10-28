import importlib.metadata

from cs336_basics.activations import silu, softmax
from cs336_basics.attention import MultiHeadSelfAttention, scaled_dot_product_attention
from cs336_basics.bpe import train_bpe
from cs336_basics.data import get_batch
from cs336_basics.decoding import generate, top_p_filtering
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.serialization import load_checkpoint, save_checkpoint
from cs336_basics.swiglu import SwiGLU
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerBlock, TransformerLM

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass

__all__ = [
    "AdamW",
    "Embedding",
    "Linear",
    "MultiHeadSelfAttention",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SwiGLU",
    "Tokenizer",
    "TransformerBlock",
    "TransformerLM",
    "cross_entropy",
    "generate",
    "get_batch",
    "get_lr_cosine_schedule",
    "gradient_clipping",
    "load_checkpoint",
    "save_checkpoint",
    "scaled_dot_product_attention",
    "silu",
    "softmax",
    "top_p_filtering",
    "train_bpe",
]
