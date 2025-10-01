import regex as re
import logging
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from .pretokenization_example import find_chunk_boundaries
from .token_utils import save_vocab_and_merges, load_vocab_and_merges
from .utils import stopwatch
from multiprocessing import Pool

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
def pretokenize(chunk: str, counts: defaultdict[str, int], special_tokens: list[str]):
    split_regexp = re.compile("|".join(re.escape(s) for s in special_tokens))

    for s in split_regexp.split(chunk):
        for part in PAT.finditer(s):
            counts[part.group(0)] += 1
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
    count_deltas = defaultdict(int)
    for part_index in affected_indices:
        part, c = parts[part_index]
        for i in range(len(part)-1):
            key = (part[i], part[i+1])
            if key != best_pair:
              count_deltas[key] -= c
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
            count_deltas[(part[i], part[i+1])] += c
            pair_indices[(part[i], part[i+1])].add(part_index)
    affected_pairs = set()
    for pair, delta in count_deltas.items():
        if delta != 0:
            affected_pairs.add(pair)
            pairs[pair] += delta
    return affected_pairs


class PairHeap:
    def __init__(self, pairs):
        self._tree = [[c, pair] for pair, c in pairs.items()]
        self._tree.sort(reverse=True)
        self._pointers = {pair: i for i, (_, pair) in enumerate(self._tree)}

    def set_count(self, pair, c):
        index = self._pointers.get(pair)
        if index is None:
            self.add_pair(pair, c)
            return
        curr_count = self._tree[index][0]
        self._tree[index][0] = c
        if c > curr_count:
            self._bubble_up(index)
        elif c < curr_count:
            self._bubble_down(index)
    
    def add_pair(self, pair, c):
        assert pair not in self._pointers.keys()
        index = len(self._tree)
        self._pointers[pair] = index
        self._tree.append([c, pair])
        self._bubble_up(index)

    def top(self):
        return tuple(self._tree[0])

    def _swap(self, i1, i2):
        p1 = self._tree[i1]
        p2 = self._tree[i2]
        self._tree[i2] = p1
        self._tree[i1] = p2
        self._pointers[p1[1]] = i2
        self._pointers[p2[1]] = i1


    def _bubble_up(self, index):
        while index > 0:
            parent_index = ((index + 1) // 2) - 1
            if self._tree[index] <= self._tree[parent_index]:
                return
            self._swap(parent_index, index)
            index = parent_index

    def _bubble_down(self, index):
        while (index+1)*2 - 1 < len(self._tree):
            left_child_index = (index + 1)*2 - 1
            child_index = left_child_index
            if child_index + 1 < len(self._tree) and self._tree[child_index + 1] > self._tree[child_index]:
                child_index += 1
            if self._tree[child_index] <= self._tree[index]:
                return
            self._swap(child_index, index)
            index = child_index


def train_tokenizer_from_counters(counters: dict[str, int], num_merges: int, special_tokens: list[str]):
    vocab = {}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf8")
    for i in range(256):
        vocab[len(vocab)] = (i).to_bytes()
    

    byte_pairs = []
    parts = [([(b).to_bytes() for b in part.encode("utf8")], c) for part, c in counters.items()]
    pairs, pair_indices = count_pair_frequencies(parts)
    pair_heap = PairHeap(pairs)

    for _ in range(num_merges):
        c, best_pair = pair_heap.top()

        if c <= 0:
            return vocab

        pair_heap.set_count(best_pair, 0)
        merged_best_pair = b"".join(best_pair)
        vocab[len(vocab)] = merged_best_pair
        byte_pairs.append(best_pair)

        affected_pairs = update_parts_and_pairs(parts, pairs, pair_indices, best_pair, pair_indices[best_pair])
        for pair in affected_pairs:
            pair_heap.set_count(pair, pairs[pair])

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


class Tokenizer(object):
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self._vocab = vocab
        self._reverse_vocab = {v: k for k, v in vocab.items()}
        self._merges = {pair: i for i, pair in enumerate(merges)}
        self._special_tokens = special_tokens if special_tokens is not None else []
        for t in special_tokens:
            assert t.encode("utf8") in self._reverse_vocab.keys()
        self._special_tokens_regexp = re.compile("|".join(re.escape(s) for s in special_tokens))

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab, merges = load_vocab_and_merges(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    def _encode_pretokenized(self, text: str):
        bytes_list = [(b).to_bytes() for b in text.encode("utf8")]
        while True:
            best_index = None
            best_score = None
            for i in range(len(bytes_list) - 1):
                pair = (bytes_list[i], bytes_list[i+1])
                merge_priority = self._merges.get(pair)
                if merge_priority is None:
                    continue
                if best_score is None or best_score > merge_priority
                    best_score = merge_priority
                    best_index = i
            if best_index is None:
                break
            bytes_list[best_index] = b"".join(bytes_list[best_index:best_index+2])
            del bytes_list[best_index + 1]
        return [self._reverse_vocab[b] for b in bytes_list]


    def _encode_no_special_tokens(self, text: str):
        result = []
        for part in PAT.finditer(text):
            result += self._encode_pretokenized(part)
        return result

    def encode(self, text: str) -> list[int]:
        result = []
        last_start = 0
        for match in self._special_tokens_regexp.finditer(text):
            result += self._encode_no_special_tokens(text[last_start:m.start()])
            result.append(self._reverse_vocab[m.group(0).encode("utf8")])
            last_start = m.end()
        if last_start != len(text):
            result += self._encode_no_special_tokens(text[last_start:])
        return result
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token in self.encode(text):
                yield token

    def decode(self,  ids: list[int]) -> str:
        return b"".join(self._vocab[token] for token in ids).decode("utf8")

