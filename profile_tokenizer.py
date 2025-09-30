from cs336_basics import tokenization
import cProfile
from tests.common import FIXTURES_PATH


if __name__ == "__main__":
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = tokenization.train_tokenizer(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )
    print(vocab)

