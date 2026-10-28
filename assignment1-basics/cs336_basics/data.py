import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length
    if max_start <= 0:
        raise ValueError("Dataset length must be strictly greater than context_length")

    start_indices = np.random.randint(0, max_start, size=(batch_size,))
    offsets = np.arange(context_length)
    window_indices = start_indices[:, None] + offsets[None, :]

    x = torch.as_tensor(dataset[window_indices], dtype=torch.long).to(device)
    y = torch.as_tensor(dataset[window_indices + 1], dtype=torch.long).to(device)
    return x, y
