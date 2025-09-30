import regex as re
import logging
import os
from collections import defaultdict
from .pretokenization_example import find_chunk_boundaries
from .utils import stopwatch
from multiprocessing import Pool

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
def pretokenize(chunk: str, counts: defaultdict[bytes, int], special_tokens: list[str]):
    split_regexp = re.compile("|".join(re.escape(s) for s in special_tokens))

    for s in split_regexp.split(chunk):
        for part in PAT.finditer(s):
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


def pretokenize_file_single_worker(input_path: str, special_tokens: list[str], boundaries: list[int]):
    pretokenization_counts = defaultdict(int)
    if len(boundaries) < 2:
        return pretokenization_counts

    with open(input_path, "rb") as file:
        start = boundaries[0]
        file.seek(start)
        for end in boundaries[1:]:
            chunk_size = end - start
            logging.info("Reading a chunk of size: %d", chunk_size)
            chunk = file.read(chunk_size).decode("utf-8", errors="ignore")
            pretokenize(chunk, pretokenization_counts, special_tokens)
            start = end
    return pretokenization_counts


def pretokenize_file_single_worker_wrapper(args):
    return pretokenize_file_single_worker(**args)

def ceiling_division(n, d):
    return -(n // -d)

def pretokenize_file(input_path: str, special_tokens: list[str], max_chunk_bytes: int, num_workers: int):
    
    with open(input_path, "rb") as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        num_chunks = num_workers
        if max_chunk_bytes > 0:
            min_chunks = ceiling_division(file_size, max_chunk_bytes)
            num_chunks = max(min_chunks, num_workers)

        boundaries = find_chunk_boundaries(file, num_chunks, b"<|endoftext|>")
        logging.info("File size: %d, max chunk bytes: %d, boundaries len: %d", file_size, max_chunk_bytes, len(boundaries))
    
    
    if num_workers == 1:
        return pretokenize_file_single_worker(input_path, special_tokens, boundaries)
    
    chunks_per_worker = ceiling_division(len(boundaries)-1, num_workers)

    pool = Pool(num_workers)
    all_args = ({"input_path": input_path, "special_tokens": special_tokens, "boundaries": boundaries[i*chunks_per_worker:((i+1)*chunks_per_worker+1)]} for i in range(num_workers))

    pretokenization_counts = defaultdict(int)
    for counter in pool.imap_unordered(pretokenize_file_single_worker_wrapper, all_args):
        for b, c in counter.items():
            pretokenization_counts[b] += c

    return pretokenization_counts


def train_tokenizer(input_path: str, vocab_size: int, special_tokens: list[str], 
                    max_chunk_bytes: int = -1, num_workers: int = 1):

    pretokenization_counts = stopwatch(pretokenize_file)(input_path, special_tokens, max_chunk_bytes, num_workers)
        
    num_merges = vocab_size - 256 - len(special_tokens)
    assert num_merges >= 0
    vocab, merges = stopwatch(train_tokenizer_from_counters)(pretokenization_counts, num_merges, special_tokens)
    return vocab, merges

