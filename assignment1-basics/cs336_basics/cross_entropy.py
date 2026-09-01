import torch


def cross_entropy(o: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross entropy loss between the predicted output and the true labels.

    Args:
        o (torch.Tensor): The predicted output tensor of shape (batch_size, num_classes).
        x (torch.Tensor): The true labels tensor of shape (batch_size,).

    Returns:
        torch.Tensor: The computed cross entropy loss.
    """
    # Compute the log softmax of the predicted output
    index = torch.unsqueeze(x, dim=-1)
    o_i = torch.gather(o, -1, index)
    log_softmax_o = torch.squeeze(
        (o_i - torch.logsumexp(o, dim=-1, keepdim=True)), dim=-1
    )
    loss = torch.mean(-log_softmax_o)

    return loss
