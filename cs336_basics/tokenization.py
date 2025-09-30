import regex as re
from collections import defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def pretokenize(s: str, counts: defaultdict[bytes, int]):
    for part in re.finditer(PAT, s):
        counts[part.group(0).encode("utf8")] += 1
    return counts

def count_pair_frequencies(parts):
    pairs = defaultdict(int)
    pair_indices = defaultdict(set)
    for part_index, (part, c) in enumerate(parts):
        for i in range(len(part) - 1):
            key = (part[i], part[i+1])
            pairs[key] += c
            pair_indices[key].add(part_index)

    return pairs, pair_indices

def get_best_pair(pairs):
    best_score = None
    best_pair = None
    for pair, c in pairs.items():
        score = (c, pair)
        if best_score is None or score > best_score:
            best_score = score
            best_pair = pair
    return best_pair 

def update_parts_and_pairs(parts, pairs, pair_indices, best_pair, affected_indices):
    del pairs[best_pair]
    del pair_indices[best_pair]
    merged_best_pair = b"".join(best_pair)
    for part_index in affected_indices:
        part, c = parts[part_index]
        for i in range(len(part)-1):
            key = (part[i], part[i+1])
            if key != best_pair:
              pairs[key] -= c
              pair_indices[key].discard(part_index)

        i = 0
        j = 0
        while i < len(part) - 1:
            if (part[i], part[i+1]) == best_pair:
                part[j] = merged_best_pair
                i += 2
                j += 1
            else:
                part[j] = part[i]
                i += 1
                j += 1

        if i < len(part):
            part[j] = part[i]
            i += 1
            j += 1

        del part[j:]

        for i in range(len(part) -1):
            pairs[(part[i], part[i+1])] += c
            pair_indices[(part[i], part[i+1])].add(part_index)


def train_tokenizer_from_counters(counters: dict[int, bytes], num_merges: int, special_tokens: list[str]):
    vocab = {}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf8")
    for i in range(256):
        vocab[len(vocab)] = (i).to_bytes()
    

    byte_pairs = []
    parts = [([(b).to_bytes() for b in part], c) for part, c in counters.items()]
    pairs, pair_indices = count_pair_frequencies(parts)

    for _ in range(num_merges):

        if not pairs:
            return vocab

        best_pair = get_best_pair(pairs)
        merged_best_pair = b"".join(best_pair)
        vocab[len(vocab)] = merged_best_pair
        byte_pairs.append(best_pair)

        update_parts_and_pairs(parts, pairs, pair_indices, best_pair, pair_indices[best_pair])
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
        
