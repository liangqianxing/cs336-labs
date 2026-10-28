from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
import os

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_COMPILED_PAT = re.compile(PAT)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self.special_tokens: list[str] = list(special_tokens) if special_tokens else []

        existing_bytes = set(self.vocab.values())
        for sp in self.special_tokens:
            sp_bytes = sp.encode("utf-8")
            if sp_bytes not in existing_bytes:
                self.vocab[len(self.vocab)] = sp_bytes
                existing_bytes.add(sp_bytes)

        self.byte_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern_str = "(" + "|".join(re.escape(tok) for tok in sorted_specials) + ")"
            self._special_re = re.compile(pattern_str)
            self._special_set = set(self.special_tokens)
        else:
            self._special_re = None
            self._special_set = set()

        self._word_cache: dict[str, tuple[int, ...]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab: dict[int, bytes] = {}
        for k, v in raw_vocab.items():
            if isinstance(v, int):
                vocab[v] = k.encode("utf-8")
            else:
                vocab[int(k)] = v.encode("utf-8")

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                cleaned = line.rstrip("\r\n")
                if not cleaned:
                    continue
                parts = cleaned.split(" ")
                if len(parts) == 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _encode_pretoken(self, pretoken: str) -> tuple[int, ...]:
        cached = self._word_cache.get(pretoken)
        if cached is not None:
            return cached

        raw_bytes = pretoken.encode("utf-8")
        if not raw_bytes:
            return ()
        if len(raw_bytes) == 1:
            res = (self.byte_to_id[raw_bytes],)
            self._word_cache[pretoken] = res
            return res

        parts = [bytes([b]) for b in raw_bytes]
        while len(parts) >= 2:
            best_rank = None
            best_pair = None
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            p0, p1 = best_pair
            merged_token = p0 + p1
            new_parts: list[bytes] = []
            i = 0
            n = len(parts)
            while i < n:
                if i < n - 1 and parts[i] == p0 and parts[i + 1] == p1:
                    new_parts.append(merged_token)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        ids = tuple(self.byte_to_id[p] for p in parts)
        if len(self._word_cache) < 8192:
            self._word_cache[pretoken] = ids
        return ids

    def _encode_ordinary_segment(self, segment: str) -> list[int]:
        ids: list[int] = []
        for match in _COMPILED_PAT.finditer(segment):
            ids.extend(self._encode_pretoken(match.group(0)))
        return ids

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        if self._special_re is None:
            return self._encode_ordinary_segment(text)

        ids: list[int] = []
        for chunk in self._special_re.split(text):
            if not chunk:
                continue
            if chunk in self._special_set:
                ids.append(self.byte_to_id[chunk.encode("utf-8")])
            else:
                ids.extend(self._encode_ordinary_segment(chunk))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            if not chunk:
                continue
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        raw_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return raw_bytes.decode("utf-8", errors="replace")
