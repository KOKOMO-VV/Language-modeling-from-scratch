import torch
import argparse
from .tokenizer import Tokenizer
from .transformer_lm import TransformerLM
from .adam_w import AdamW

parser = argparse.ArgumentParser(
    description="Decoding with a trained Transformer model"
)
parser.add_argument(
    "--vocab_filepath", type=str, required=True, help="Path to the vocab.json file"
)
parser.add_argument(
    "--merges_filepath", type=str, required=True, help="Path to the merges file"
)
parser.add_argument(
    "--special_tokens", type=list[str], required=True, help="List of special tokens"
)
parser.add_argument(
    "--checkpoints_src", type=str, required=True, help="Path to the checkpoint file"
)
parser.add_argument(
    "--temperature", type=float, default=1.0, help="Temperature for sampling"
)
parser.add_argument(
    "probability_limit", type=float, help="Probability limit for decoding"
)

args = parser.parse_args()
file_path = args.vocab_filepath
merges_path = args.merges_filepath
special_tokens = args.special_tokens
checkpoints_src = args.checkpoints_src
temperature = args.temperature

tokenizer = Tokenizer.from_files(file_path, merges_path, special_tokens)
checkpoint_obj = torch.load(checkpoints_src)
vocab_size = checkpoint_obj["vocab_size"]
context_length = checkpoint_obj["context_length"]
num_layers = checkpoint_obj["num_layers"]
d_model = checkpoint_obj["d_model"]
num_heads = checkpoint_obj["num_heads"]
d_ff = checkpoint_obj["d_ff"]
tehta = checkpoint_obj["theta"]

model = TransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    theta=tehta,
)


def decoding(prompt: str):
    """
    Decode a prompt using the tokenizer and return the decoded string.
    """
    # while loop to decode the token ids back to string
    decoded_string = ""
    # Encode the prompt to get token ids
    token_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    token_ids = torch.reshape(token_ids, (1, -1))  # reshape to (1, seq_len)
    max_length = model.context_length - token_ids.shape[1]
    output = ""
    iteration = 0
    while iteration < max_length:
        # Get the vocab_scores for the last token
        logits = model(token_ids)[0, -1, :]

        # normalize the logits to get probabilities
        new_logits = logits / temperature
        softmax_logits = torch.exp(new_logits) / torch.sum(
            torch.exp(new_logits), dim=-1, keepdim=True
        )
        softmax_logits_clipping = torch.where(
            softmax_logits > args.probability_limit, softmax_logits, torch.tensor(0.0)
        )
        next_token_id = torch.multinomial(softmax_logits_clipping, num_samples=1).item()
        if next_token_id == tokenizer.vocab_reverse[("<|endoftext|>".encode("utf-8"))]:
            break
        token_ids = torch.cat(
            (token_ids, torch.reshape(torch.tensor(next_token_id), (1, 1))), dim=-1
        )
        output = tokenizer.decode(token_ids.squeeze().tolist())
        iteration += 1

    return output
