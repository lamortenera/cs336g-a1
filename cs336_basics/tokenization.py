import regex as re
from collections import defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def pretokenize(s: str, counts: defaultdict[bytes, int]):
    for part in re.finditer(PAT, s):
        counts[part.group(0).encode("utf8")] += 1
    return counts

def train_tokenizer_from_counters(counters: dict[int, bytes], num_merges: int, special_tokens: list[str]):
    vocab = {}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf8")
    for i in range(256):
        vocab[len(vocab)] = (i).to_bytes()
    

    byte_pairs = []
    parts = {tuple((b).to_bytes() for b in part): c for part, c in counters.items()}
    for _ in range(num_merges):
        pairs = defaultdict(int)
        for part, c in parts.items():
            for i in range(len(part) - 1):
                pairs[(part[i], part[i+1])] += c

        if not pairs:
            return vocab

        best_score = None
        best_pair = None
        for pair, c in pairs.items():
            score = (c, pair)
            if best_score is None or score > best_score:
                best_score = score
                best_pair = pair
        merged_best_pair = b"".join(best_pair)
        vocab[len(vocab)] = merged_best_pair
        byte_pairs.append(best_pair)

        new_parts = {}
        for part, c in parts.items():
            sub_parts = []

            i = 0
            while i < len(part) - 1:
                if (part[i], part[i+1]) == best_pair:
                    sub_parts.append(merged_best_pair)
                    i += 2
                else:
                    sub_parts.append(part[i])
                    i += 1
            if i < len(part):
                sub_parts.append(part[i])

            new_parts[tuple(sub_parts)] = c
        parts = new_parts
    return vocab, byte_pairs


def train_tokenizer(input_path: str, vocab_size: int, special_tokens: list[str]):
    input_string = open(input_path, "r").read()
    pretokenization_counts = defaultdict(int)
    split_regexp = "|".join(re.escape(s) for s in special_tokens)
    for s in re.split(split_regexp, input_string):
        pretokenize(s, pretokenization_counts)
    num_merges = vocab_size - 256 - len(special_tokens)
    assert num_merges >= 0
    return train_tokenizer_from_counters(pretokenization_counts, num_merges, special_tokens)
        
