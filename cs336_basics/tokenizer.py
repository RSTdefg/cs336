PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

import json
import regex as re
from collections.abc import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab, merges: list[tuple[bytes, bytes]], special_tokens=None):     
        self.id_to_token = vocab
        self.token_to_id = {v: k for k, v in vocab.items()}

        self.merges = merges
        self.merges_order = {item: i for i, item in enumerate(merges)}
        self.special_tokens = sorted(special_tokens, key=lambda x:-len(x)) if special_tokens else []
        if special_tokens:
            for token in special_tokens:
                self.id_to_token[len(self.id_to_token)] = token
                self.token_to_id[token] = len(self.id_to_token)
        
    # def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
    #     vocab, merges = {}, []
    #     with open(vocab_filepath) as vocab_file:
    #         self.vocab = json.load(vocab_file)

    #     with open(merges_filepath) as merges_file:
    #         for line in merges_file:
    #             merges.append(tuple(b"".join(b.encode() for b in line.split())))
    #     self.vocab = vocab

    def encode(self, text: str) -> list[int]:
        tokens = self.pre_tokenize(text)
        result = []
        for token in tokens:
            if token in self.token_to_id:
                result.append(self.token_to_id[token])
                continue
            to_merge = tuple(bytes([b]) for b in token)
            while True:
                # find the highest ranked pair or merging
                lowest_rank = len(self.merges)
                token_pair_i = 0
                for i in range(len(to_merge)-1):
                    k = to_merge[i:i+2]
                    if k not in self.merges_order or self.merges_order[k] >= lowest_rank:
                        continue
                    lowest_rank = self.merges_order[k]
                    token_pair_i = i
                if lowest_rank == len(self.merges):
                    # no merges can be made, break out
                    break
                merged:tuple[bytes,...] = to_merge[:token_pair_i] +tuple([b"".join(to_merge[token_pair_i:token_pair_i+2])]) + to_merge[token_pair_i+2:]
                to_merge = merged
            
            for tok in to_merge:
                result.append(self.token_to_id[tok])
        return result

    # returns the list of tokens binaries that were broken up, includes special tokens in full form 
    def pre_tokenize(self, text) -> list[bytes]:
        pattern = "|".join(re.escape(token) for token in self.special_tokens) if self.special_tokens else "(?!)"
        pre_tokens = []
        prev_start = 0
        for it in re.finditer(pattern, text):
            start, end = it.span()
            if start == end:
                continue
            if start > prev_start:
                pre_tokens += [item.group(0).encode("utf-8") for item in re.finditer(PAT, text[prev_start:start])]
            # the special char
            pre_tokens.append(text[start:end].encode("utf-8"))
            prev_start = end
        # append the last token after the last special token
        if prev_start < len(text):
            pre_tokens += [item.group(0).encode("utf-8") for item in re.finditer(PAT, text[prev_start:])]
        
        return pre_tokens

    def encode_iterable(self, iterable: Iterable[str]) ->Iterator[int]:
        for it in iterable:
            yield from self.encode(it)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.id_to_token[id] for id in ids).decode()

