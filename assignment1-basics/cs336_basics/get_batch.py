import torch
import numpy.typing as npt


def get_batch(x: npt.NDArray, batch_size, context_length, device: str):
    result = torch.randint(
        0, len(x) - context_length, (batch_size,), dtype=torch.int64, device=None
    )
    x_batch = torch.stack(
        [
            torch.tensor(x[i : i + context_length], dtype=torch.int64, device=device)
            for i in result
        ]
    )
    y_batch = torch.stack(
        [
            torch.tensor(
                x[i + 1 : i + context_length + 1], dtype=torch.int64, device=device
            )
            for i in result
        ]
    )
    return (x_batch, y_batch)
