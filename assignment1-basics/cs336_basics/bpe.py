from __future__ import annotations

from collections import Counter, defaultdict
import os

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_COMPILED_PAT = re.compile(PAT)


def _pretokenize_text(text: str, special_tokens: list[str] | None = None) -> Counter[tuple[bytes, ...]]:
    word_counts: Counter[tuple[bytes, ...]] = Counter()
    if special_tokens:
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        split_pattern = "|".join(re.escape(tok) for tok in sorted_specials)
        segments = re.split(split_pattern, text)
    else:
        segments = [text]

    single_bytes = [bytes([b]) for b in range(256)]
    str_to_byte_tuple: dict[str, tuple[bytes, ...]] = {}

    for segment in segments:
        if not segment:
            continue
        for match in _COMPILED_PAT.finditer(segment):
            token_str = match.group(0)
            byte_tuple = str_to_byte_tuple.get(token_str)
            if byte_tuple is None:
                raw_bytes = token_str.encode("utf-8")
                byte_tuple = tuple(single_bytes[b] for b in raw_bytes)
                str_to_byte_tuple[token_str] = byte_tuple
            word_counts[byte_tuple] += 1

    return word_counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = list(special_tokens) if special_tokens else []

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    existing_vocab_bytes = set(vocab.values())
    for sp in special_tokens:
        sp_bytes = sp.encode("utf-8")
        if sp_bytes not in existing_vocab_bytes:
            vocab[len(vocab)] = sp_bytes
            existing_vocab_bytes.add(sp_bytes)

    num_merges = max(0, vocab_size - len(vocab))
    if num_merges == 0:
        return vocab, []

    with open(input_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    word_counts = _pretokenize_text(text, special_tokens)

    words: list[list[bytes]] = []
    freqs: list[int] = []
    for word_tuple, count in word_counts.items():
        if len(word_tuple) >= 2:
            words.append(list(word_tuple))
            freqs.append(count)

    pair_counts: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for w_idx, word in enumerate(words):
        freq = freqs[w_idx]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(w_idx)

    merges: list[tuple[bytes, bytes]] = []

    for _ in range(num_merges):
        if not pair_counts:
            break

        best_pair: tuple[bytes, bytes] | None = None
        best_count = 0
        for pair, count in pair_counts.items():
            if count > best_count or (count == best_count and best_pair is not None and pair > best_pair):
                best_count = count
                best_pair = pair

        if best_pair is None or best_count <= 0:
            break

        p0, p1 = best_pair
        new_token = p0 + p1
        merges.append(best_pair)
        vocab[len(vocab)] = new_token

        affected_word_indices = list(pair_to_words.pop(best_pair, ()))
        pair_counts.pop(best_pair, None)

        for w_idx in affected_word_indices:
            word = words[w_idx]
            freq = freqs[w_idx]
            n = len(word)
            if n < 2:
                continue

            # Subtract old pair counts for this word
            for i in range(n - 1):
                old_pair = (word[i], word[i + 1])
                if old_pair != best_pair:
                    new_c = pair_counts.get(old_pair, 0) - freq
                    if new_c <= 0:
                        pair_counts.pop(old_pair, None)
                        pair_to_words.pop(old_pair, None)
                    else:
                        pair_counts[old_pair] = new_c
                        pair_to_words[old_pair].discard(w_idx)

            # Merge occurrences of best_pair in word
            new_word: list[bytes] = []
            i = 0
            while i < n:
                if i < n - 1 and word[i] == p0 and word[i + 1] == p1:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            words[w_idx] = new_word
            m = len(new_word)
            for i in range(m - 1):
                new_pair = (new_word[i], new_word[i + 1])
                pair_counts[new_pair] += freq
                pair_to_words[new_pair].add(w_idx)

    return vocab, merges
