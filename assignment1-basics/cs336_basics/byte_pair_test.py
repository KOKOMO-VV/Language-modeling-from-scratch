import time
import resource
import json

from tests.common import gpt2_bytes_to_unicode
from cs336_basics.byte_pair_encode import train_bpe


def bytes_to_str(byte_words: bytes) -> str:
    """
    Convert bytes to str
    """
    return "".join([gpt2_bytes_to_unicode()[b] for b in byte_words])


def byte_pair_test(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    """
    Return a the result of time_taken and longest token and memory_taken
    """
    start_time = time.time()
    vocab, merge = train_bpe(input_path, vocab_size, special_tokens)
    end_time = time.time()
    time_taken = end_time - start_time
    longest_token = max(vocab.values(), key=len)
    memory_taken = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    # Convert to MB

    # create new vocab_str to convert bytes to str, so that it can be dumped to json
    vocab_str = {k: bytes_to_str(v) for k, v in vocab.items()}
    merge_str = [(bytes_to_str(pair[0]), bytes_to_str(pair[1])) for pair in merge]

    json_content = {
        "vocab": vocab_str,
        "merges": merge_str,
    }

    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(json_content, f, ensure_ascii=False, indent=4)

    return time_taken, longest_token, memory_taken


if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    time_taken, longest_token, memory_taken = byte_pair_test(
        input_path, vocab_size, special_tokens
    )
    print(f"Time taken: {time_taken:.2f} seconds")
    print(f"Longest token: {longest_token}")
    print(f"Memory taken: {memory_taken:.2f} MB")
