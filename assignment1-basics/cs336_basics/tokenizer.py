from typing import Iterable, Iterator
from tests.common import gpt2_bytes_to_unicode
import json
import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        # Create a reverse vocabulary mapping from bytes to token ids
        self.vocab_reverse = {v: k for k, v in vocab.items()}

        # create a dictionary of merges and .index
        self.merges_dict = {pair: i for i, pair in enumerate(merges)}

        # Add special tokens to the vocabulary and reverse vocabulary
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab_reverse:
                self.vocab_reverse[token_bytes] = len(self.vocab_reverse)
                self.vocab[len(self.vocab)] = token_bytes

        # save special_tokens_ends
        self.end_special_tokens = {
            token[:a] for token in self.special_tokens for a in range(1, len(token))
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        vocab = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            vocab = {int(k): str_to_byte(v) for k, v in data.items()}
        with open(merges_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            merges = [(str_to_byte(pair[0]), str_to_byte(pair[1])) for pair in data]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode text to list of token ids
        """
        # 1. split original text into chunks based on special tokens
        chunks = (
            split_string(text, self.special_tokens) if self.special_tokens else [text]
        )

        # 2.pre-tokenize each chunk
        encode_chunks = []
        for chunk in chunks:
            if chunk in self.special_tokens:
                encode_chunks.append(self.vocab_reverse[chunk.encode("utf-8")])
            else:
                pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                pre_tokens = re.finditer(pattern, chunk)
                for p in pre_tokens:
                    # first get str (p.group()), then convert it to bytes
                    p_tuple = tuple(bytes([b]) for b in p.group().encode("utf-8"))
                    encode_chunks.extend(self.pre_token_merge(p_tuple))
        return encode_chunks

    def pre_token_merge(self, word_tuple: tuple[bytes]) -> list[int]:
        """
        Merge the pre-tokenized tokens into a single list of token ids.
        """
        pairs = {(word_tuple[i], word_tuple[i + 1]) for i in range(len(word_tuple) - 1)}
        token_ids = [self.vocab_reverse[w] for w in word_tuple]

        chosen_pair = None
        pair_options = [pair for pair in pairs if pair in self.merges_dict]
        if pair_options:
            chosen_pair = min(pair_options, key=lambda k: self.merges_dict[k])

        if chosen_pair != None:
            i = 0
            change_word = word_tuple
            while i < len(change_word) - 1:
                new_word = change_word
                if new_word[i : i + 2] == chosen_pair:
                    new_word = (
                        new_word[:i]
                        + (chosen_pair[0] + chosen_pair[1],)
                        + new_word[i + 2 :]
                    )
                change_word = new_word
                i += 1
            return self.pre_token_merge(new_word)
        else:
            return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields results. This is
        required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        # set buffer size
        it = iter(iterable)
        has_next = True
        while has_next:
            buffer = ""
            # add str to buffer till certain size
            while len(buffer) <= 4096:
                try:
                    piece = next(it)
                    buffer += piece
                except StopIteration:
                    has_next = False
                    break
            # extend the buffer in case one word is split into two pieces
            if buffer == "":
                break
            while re.match(
                r"[\p{L}\p{N}]", buffer[-1]
            ) or self.check_buffer_end_with_special_tokens(buffer):
                try:
                    piece = next(it)
                    buffer += piece
                except StopIteration:
                    has_next = False
                    break
            # send the complete buffer to encode
            result = self.encode(buffer)
            yield from result

    # check if buffer ends with special tokens
    def check_buffer_end_with_special_tokens(self, buffer: str) -> bool:
        return buffer.endswith(tuple(self.end_special_tokens))

    def decode(self, ids: list[int]) -> str:
        result = b"".join(self.vocab[id] for id in ids)
        return result.decode("utf-8", errors="replace")


# --------------------------- help functions ---------------------------------


# only used for vocab and merges
def str_to_byte(words: str) -> bytes:
    """
    Convert str to bytes
    """
    gpt2_bytes_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    return bytes([gpt2_bytes_decoder[c] for c in words])


def split_string(
    chunk: str,
    split_special_token: list[str],
) -> list[str]:
    """
    Split a string into a list of substrings based on special tokens. (create chunks)
    """
    assert isinstance(
        split_special_token, list
    ), "Must represent special token as a list of strings"
    reordered_tokens = sorted(split_special_token, key=len, reverse=True)
    # Create a regex pattern that matches any of the special tokens
    escaped_tokens = [re.escape(token) for token in reordered_tokens]
    # Join the escaped tokens with the '|' operator to create a regex pattern
    if escaped_tokens:
        pattern = "|".join(escaped_tokens)
        pattern = f"({pattern})"
        # Split the chunk using the regex pattern
        split_results = re.split(pattern, chunk)
    else:
        split_results = [chunk]
    # Remove empty strings from the result
    split_results = [s for s in split_results if s]
    return split_results
