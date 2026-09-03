import torch
import typing
import os


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):

    obj = {
        "model_dict": model.state_dict(),
        "optimizer_dict": optimizer.state_dict(),
        "iteration": iteration,
        "vocab_size": model.vocab_size,
        "context_length": model.context_length,
        "num_layers": model.num_layers,
        "d_model": model.d_model,
        "num_heads": model.num_heads,
        "d_ff": model.d_ff,
        "theta": model.theta,
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    obj = torch.load(src)
    model.load_state_dict(obj["model_dict"])
    optimizer.load_state_dict(obj["optimizer_dict"])
    iteration = obj["iteration"]
    return iteration
