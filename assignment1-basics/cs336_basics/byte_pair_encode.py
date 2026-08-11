import regex as re


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
        # Split the chunk using the regex pattern
        split_results = re.split(pattern, chunk)
    else:
        split_results = [chunk]
    # Remove empty strings from the result
    split_results = [s for s in split_results if s]
    return split_results


def pre_tokenization(
    split_result: str,
) -> dict[str, int]:
    """
    Return a dictionary of seperate tokens for pre-tokenization. (calculate word frequency of each chunk)
    """
    pattern = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    pre_tokens = re.finditer(pattern, split_result)
    pre_token_dict = {}
    for i in pre_tokens:
        if i.group() not in pre_token_dict:
            pre_token_dict[i.group()] = 1
        else:
            pre_token_dict[i.group()] += 1
    return pre_token_dict


def merge_chunks(chunks: list[dict[str, int]]) -> dict[str, int]:
    """
    Merge the chunks into a single dictionary of word frequency. (merge the word frequency of each chunk)
    """
    merged_dict = {}
    for chunk in chunks:
        for word in chunk:
            if word not in merged_dict:
                merged_dict[word] = chunk[word]
            else:
                merged_dict[word] += chunk[word]
    return merged_dict


def word_byte_conversion(words: dict[str, int]) -> dict[bytes, int]:
    """
    Return a dictionary of pair byte frequency. (create byte tupel with frequency of each word)
    """
    # store original words in byte tupel with frequency
    byte_words = {}
    for word in words:
        # convert the word to bytes
        byte_word = word.encode("utf-8")
        # add the byte word to the dictionary with its frequency
        byte_word_new = tuple(bytes([b]) for b in byte_word)
        byte_words[byte_word_new] = words[word]
    return byte_words


def create_pair_frequent_table_and_reverse_index_table(byte_words: dict[bytes, int]):
    """
    Return a dictionary of pair byte frequency and a reverse index table. (create a pair frequent table and reverse index table)
    """
    pair_frequent_table = {}
    reverse_index_table = {}
    # split the keys of the dictionary into byte pairs
    for byte_word in byte_words:
        # split the byte word into pairs of bytes
        byte_pairs = [byte_word[i : i + 2] for i in range(len(byte_word) - 1)]
        # count the frequency of each byte pair
        for pair in byte_pairs:
            if pair not in pair_frequent_table:
                pair_frequent_table[pair] = byte_words[byte_word]
                reverse_index_table[pair] = {byte_word}
            else:
                pair_frequent_table[pair] += byte_words[byte_word]
                reverse_index_table[pair].add(byte_word)

    return pair_frequent_table, reverse_index_table


def byte_pair_merge(
    byte_words: dict[bytes, int],
    pair_frequent_table: dict[bytes, int],
    reverse_index_table: dict[bytes, set[bytes]],
) -> dict[bytes, int]:
    """
    Return a dictionary of merged byte pairs with frequency. (merge the most frequent byte pair into a single byte)
    """

    # choose the most frequent byte pair
    most_frequent_pair = max(
        pair_frequent_table, key=lambda k: (pair_frequent_table[k], k)
    )

    new_byte_words = byte_words.copy()
    # merge the most frequent byte pair into a single byte
    searched_byte_words = reverse_index_table[most_frequent_pair]

    # 这里必须复制一个出来，因为在循环中会修改reverse_index_table的内容，导致迭代器失效
    # 必须有三个变量，word保存最原始的byte tuple， change_word保存每次while loop更新后的byte tuple，new_word保存每次更新中的byte tuple
    check_zero_frequent_pair = set()
    check_zero_value_reverse = set()
    for word in list(searched_byte_words):
        del new_byte_words[word]
        # replace the most frequent byte pair with a single byte
        i = 0
        change_word = word

        while i < len(change_word) - 1:
            new_word = change_word
            if new_word[i : i + 2] == most_frequent_pair:
                new_word = (
                    new_word[:i]
                    + (most_frequent_pair[0] + most_frequent_pair[1],)
                    + new_word[i + 2 :]
                )
                if i > 0:
                    pair_frequent_table[(new_word[i - 1], new_word[i])] = (
                        pair_frequent_table.get((new_word[i - 1], new_word[i]), 0)
                    ) + byte_words[word]
                    pair_frequent_table[
                        (change_word[i - 1], change_word[i])
                    ] -= byte_words[word]
                    check_zero_frequent_pair.add((change_word[i - 1], change_word[i]))
                if i < len(new_word) - 1:
                    pair_frequent_table[(new_word[i], new_word[i + 1])] = (
                        pair_frequent_table.get((new_word[i], new_word[i + 1]), 0)
                    ) + byte_words[word]
                    pair_frequent_table[
                        (change_word[i + 1], change_word[i + 2])
                    ] -= byte_words[word]
                    check_zero_frequent_pair.add(
                        (change_word[i + 1], change_word[i + 2])
                    )
                pair_frequent_table[most_frequent_pair] -= byte_words[word]
                check_zero_frequent_pair.add(most_frequent_pair)
                # 每次更新一个pair都要更新这个word，因为一个word中可能存在多个pair
                change_word = new_word
            i += 1

        old_byte_pair = [word[i : i + 2] for i in range(len(word) - 1)]
        for pair in old_byte_pair:
            if pair in reverse_index_table and word in reverse_index_table[pair]:
                reverse_index_table[pair].remove(word)
                check_zero_value_reverse.add(pair)
        new_byte_pair = [new_word[i : i + 2] for i in range(len(new_word) - 1)]
        for pair in new_byte_pair:
            reverse_index_table.setdefault(pair, set()).add(new_word)
        new_byte_words[new_word] = byte_words[word]

    # remove the pairs with zero frequency from the pair_frequent_table
    for pair in check_zero_frequent_pair:
        if pair_frequent_table[pair] <= 0:
            del pair_frequent_table[pair]
    # remove the pairs with zero value from the reverse_index_table
    for pair in check_zero_value_reverse:
        if pair in reverse_index_table and len(reverse_index_table[pair]) == 0:
            del reverse_index_table[pair]

    byte_words = new_byte_words
    return byte_words, most_frequent_pair, pair_frequent_table, reverse_index_table


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    """
    Return a dictionary of merged byte pairs with tokenID and a list of merged byte pairs. (train the BPE model)
    """
    # read the input file
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # calculate the number of special tokens
    num_init_words = len(special_tokens) + 256

    # caculate the number of tokens to be merged
    num_tokens_to_merge = vocab_size - num_init_words

    # split the text into chunks based on special tokens
    chunks = split_string(text, special_tokens)

    # pre-tokenize the chunks and merge them into a single dictionary of word frequency
    new_chunks = []
    for chunk in chunks:
        pre_tokens = pre_tokenization(chunk)
        new_chunks.append(pre_tokens)
    merged_chunks = merge_chunks(new_chunks)

    # convert the words into byte pairs with frequency
    byte_words = word_byte_conversion(merged_chunks)

    # init vocab and special tokens
    vocab = {i: bytes([i]) for i in range(256)}
    for i in special_tokens:
        vocab[len(vocab)] = i.encode("utf-8")

    # init merge
    merge = []

    # merge the byte pairs into a single byte until the desired vocabulary size is reached
    pair_frequent_table, reverse_index_table = (
        create_pair_frequent_table_and_reverse_index_table(byte_words)
    )
    for i in range(num_tokens_to_merge):
        byte_words, most_frequent_pair, pair_frequent_table, reverse_index_table = (
            byte_pair_merge(byte_words, pair_frequent_table, reverse_index_table)
        )
        vocab[len(vocab)] = most_frequent_pair[0] + most_frequent_pair[1]
        merge.append(most_frequent_pair)

    return vocab, merge
