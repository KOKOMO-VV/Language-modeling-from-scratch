from .tokenizer import Tokenizer
from typing import Iterable
import numpy as np
import itertools
import time


def create_token_id(vocab_path, merges_path, special_tokens, text: Iterable[str]):
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    result = tokenizer.encode_iterable(text)
    yield from result


start = time.time()
with open("./data/TinyStoriesV2-GPT4-train.txt", "r") as f:
    arr = np.fromiter(
        create_token_id(
            "./data/tiny_story_vocab_train.json",
            "./data/tiny_story_merges_train.json",
            ["<|endoftext|>"],
            f,
        ),
        dtype=np.uint16,
    )

arr.tofile("./data/TinyStoriesV2-GPT4-train-token-id.bin")
duration = time.time() - start
print(duration)
# with open("./data/TinyStoriesV2-GPT4-valid.txt", "r") as f:
#     arr = np.fromiter(
#         create_token_id(
#             "./data/tiny_story_vocab_train.json",
#             "./data/tiny_story_merges_train.json",
#             ["<|endoftext|>"],
#             f,
#         ),
#         dtype=np.uint16,
#     )

# arr.tofile("./data/TinyStoriesV2-GPT4-valid-token-id.bin")
