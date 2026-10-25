PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

import json
import regex as re
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):     
        self.id_to_token = vocab
        self.token_to_id = {v: k for k, v in vocab.items()}

        self.merges = merges
        self.merges_order = {item: i for i, item in enumerate(merges)}
        self.special_tokens = sorted(special_tokens, key=lambda x:-len(x)) if special_tokens else []
        for token in special_tokens:
            self.id_to_token[len(self.id_to_token)] = token
            self.token_to_id[token] = len(self.id_to_token)
        
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab, merges = {}, []
        with open(vocab_filepath) as vocab_file:
            vocab = json.load(vocab_file)

        with open(merges_filepath) as merges_file:
            for line in merges_file:
                merges.append(line.split(' '))
        pass

    def encode(self, text: str) -> list[int]:

        tokens = self.pre_tokenize(text)
        result = []
        for token in tokens:
            if token in self.vocab:
                result.append(self.vocab[token])
                continue
            
        
        pass

    # returns the list of tokens binaries that were broken up, includes special tokens in full form 
    def pre_tokenize(self, text):
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
        pass

    def decode(self, ids: list[int]) -> str:
        pass

