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
