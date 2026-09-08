import numpy
import torch
from .get_batch import get_batch
import argparse
from .transformer_lm import TransformerLM
from .cross_entropy import cross_entropy
from .adam_w import AdamW
from .sgd import lr_cosine_schedule, gradient_clipping
from .checkpoints import save_checkpoint, load_checkpoint
import time
import csv


def evaluate(model, val_dataset, num_batches, batch_size, context_length, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for _ in range(num_batches):
            x_batch, y_batch = get_batch(
                val_dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )
            output = model(x_batch)
            loss = cross_entropy(output, y_batch)
            total_loss += loss.item()

    average_loss = total_loss / num_batches
    model.train()  # Set the model back to training mode
    return average_loss


parser = argparse.ArgumentParser()
parser.add_argument(
    "--filename", type=str, required=True, help="Path to the memmap file"
)
parser.add_argument(
    "--val_filename", type=str, required=True, help="Path to the memmap validation file"
)
parser.add_argument(
    "--dtype", type=str, required=True, help="Data type of the memmap file"
)
parser.add_argument(
    "--mode", type=str, default="r", help="Mode for opening the memmap file"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training"
)
parser.add_argument(
    "--context_length", type=int, default=10, help="Context length for training"
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device to use for training (e.g., 'cpu' or 'cuda')",
)
parser.add_argument(
    "--vocab_size", type=int, default=10000, help="Vocabulary size for the model"
)
parser.add_argument(
    "--num_layers", type=int, default=6, help="Number of layers in the model"
)
parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
parser.add_argument(
    "--num_heads", type=int, default=8, help="Number of attention heads"
)
parser.add_argument("--d_ff", type=int, default=2048, help="Feedforward dimension")

parser.add_argument(
    "--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer"
)
parser.add_argument(
    "--weight_decay", type=float, default=1e-2, help="Weight decay for the optimizer"
)
parser.add_argument(
    "--betas", type=float, nargs=2, default=(0.9, 0.999), help="Betas for the optimizer"
)
parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon for the optimizer")
parser.add_argument(
    "--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping"
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=1000,
    help="Number of warmup steps for learning rate scheduling",
)
parser.add_argument(
    "--total_steps",
    type=int,
    default=100000,
    help="Total number of training steps for learning rate scheduling",
)
parser.add_argument(
    "--min_lr",
    type=float,
    default=1e-5,
    help="Minimum learning rate for cosine schedule",
)
parser.add_argument(
    "--max_lr",
    type=float,
    default=1e-3,
    help="Maximum learning rate for cosine schedule",
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to the checkpoint file"
)

arg = parser.parse_args()

transformer_lm = TransformerLM(
    vocab_size=arg.vocab_size,
    context_length=arg.context_length,
    num_layers=arg.num_layers,
    d_model=arg.d_model,
    num_heads=arg.num_heads,
    d_ff=arg.d_ff,
)
adamw_optimizer = AdamW(
    params=transformer_lm.parameters(),
    lr=arg.learning_rate,
    betas=arg.betas,
    eps=arg.eps,
    weight_decay=arg.weight_decay,
)

src = arg.checkpoint

dataset = numpy.memmap(filename=arg.filename, dtype=arg.dtype, mode=arg.mode)
val_dataset = numpy.memmap(filename=arg.val_filename, dtype=arg.dtype, mode=arg.mode)
iteration = 0
history = []
# Load checkpoint if provided
if src is not None:
    iteration = load_checkpoint(
        src=src, model=transformer_lm, optimizer=adamw_optimizer
    )
    print(f"Loaded checkpoint from {src} at iteration {iteration}")
start = time.time()
while iteration < arg.total_steps:
    # 1. Get a batch of data
    x_batch, y_batch = get_batch(
        dataset,
        batch_size=arg.batch_size,
        context_length=arg.context_length,
        device=arg.device,
    )

    # 2. forward the transformer model
    output = transformer_lm(x_batch)

    # 3. clear grad
    adamw_optimizer.zero_grad()

    # 4. compute the loss (e.g., cross-entropy loss) between the model's output and the target labels
    loss = cross_entropy(output, y_batch)

    # 5. backpropagate the loss and update the model's parameters using an optimizer (e.g., Adam)
    loss.backward()
    lr = lr_cosine_schedule(
        t=iteration,
        a_max=arg.max_lr,
        a_min=arg.min_lr,
        t_w=arg.warmup_steps,
        t_c=arg.total_steps,
    )

    for param_group in adamw_optimizer.param_groups:
        param_group["lr"] = lr

    gradient_clipping(transformer_lm.parameters(), l_max=arg.max_grad_norm)

    adamw_optimizer.step()
    if iteration % 100 == 0:
        save_checkpoint(
            model=transformer_lm,
            optimizer=adamw_optimizer,
            iteration=iteration,
            out=f"checkpoint_{iteration}.pt",
        )
        print(f"Iteration: {iteration}, Loss: {loss.item()}, Learning Rate: {lr}")
        duration = time.time() - start
        average_loss = evaluate(
            transformer_lm,
            val_dataset,
            num_batches=10,
            batch_size=arg.batch_size,
            context_length=arg.context_length,
            device=arg.device,
        )
        history.append((iteration, duration, average_loss, lr))
    iteration += 1

with open("training_history.txt", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Iteration", "Duration", "Average Loss", "Learning Rate"])
    writer.writerows(history)
