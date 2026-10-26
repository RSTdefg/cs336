import regex as re
import collections

from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
class BPE_Tokenizer:
    def __init__(self, vocab, merges, special_tokens: list[str]):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def train_bpe(self, input_path, vocab_size: list[str]):
        vocab = {
            i: bytes([i]) for i in range(256)
        }

        for i in range(256, 256+len(self.special_tokens)):
            vocab[i] = self.special_tokens[i-256].encode("utf-8")

        # pre-tokenize
        freq_counter = self.pre_tokenize(input_path, self.special_tokens)

        # merges
        self.merges = self.merge(freq_counter, vocab_size, vocab)
        self.vocab = vocab

    def merge(self, freq_counter: collections.Counter[tuple[bytes, ...]], vocab_size, vocab) -> list[tuple[bytes, bytes]]:
        merges: list[tuple[bytes, bytes]] = []
        vocab_i = len(vocab)
        while vocab_size > vocab_i:
            pair_count = collections.Counter()
            for key, value in freq_counter.items():
                for i in range(len(key)-1):
                    k = key[i:i+2]
                    pair_count[k] += value

            # merge the winning pair, add it to vocab
            if len(pair_count) == 0:
                # there are nothing more to merge
                break
            _, winner = max(
                (count, pair) for pair, count in pair_count.items())
            new_freq_counter = collections.Counter()

            for key, value in freq_counter.items():
                new_merged_key = []
                i = 0
                size = len(key)
                while i < size:
                    
                    if i !=size-1 and key[i:i+2] == winner:
                        k = key[i:i+2]
                        new_merged_key.append(b"".join(k))
                        i += 1
                    else:
                        new_merged_key.append(key[i])
                    i += 1
                new_freq_counter[tuple(new_merged_key)] += value
            
            merges.append(winner)
            vocab[vocab_i] = b"".join(winner)
            vocab_i += 1
            freq_counter = new_freq_counter
        
        return merges
    

    def pre_tokenize(self, input_path, special_tokens) -> collections.Counter[tuple[bytes, ...]]:
        counter = collections.Counter()
        pattern = "|".join(re.escape(token) for token in special_tokens)
        with open(input_path, "rb") as f:
            num_processes = 1
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                segments = re.split(pattern, chunk) if special_tokens else [chunk]
                for segment in segments:
                    # Run pre-tokenization on your chunk and store the counts for each pre-token
                    counter.update(
                        tuple(bytes([b]) for b in item.group(0).encode("utf-8"))
                        for item in re.finditer(PAT, segment)
                    )
        return counter