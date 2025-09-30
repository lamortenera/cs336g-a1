from cs336_basics import tokenization, token_utils
import cProfile
from tests.common import FIXTURES_PATH


if __name__ == "__main__":
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = tokenization.train_tokenizer(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab-test.json"
    merges_path = FIXTURES_PATH / "train-bpe-reference-merges-test.txt"
    token_utils.save_vocab_and_merges(vocab, merges, vocab_path=vocab_path, merges_path=merges_path)


    
