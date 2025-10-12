from cs336_basics import tokenization, token_utils
import logging
import argparse
import sys
import numpy as np

if __name__ == "__main__":
    print("Running command:\n" + " ".join(sys.argv[:]))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", help="Path to text file with data to be tokenized",
        required=True)
    parser.add_argument(
        "--vocab_path",
        help="Path to the previously trained tokenizer vocabulary",
        required=True)
    parser.add_argument(
        "--merges_path",
        help="Path to the previously trained tokenizer merges list",
        required=True)
    parser.add_argument(
        "--output_path",
        help="Path to a file with a numpy array of uint16 as raw bytes",
        required=True)
    parser.add_argument(
        "--max_chunk_bytes",
        help="Maximum number of bytes to load in memory at a time", type=int,
        default=20000000)
    parser.add_argument(
        "--log_level", help="Log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO")
    parser.add_argument(
        "--test_detokenize",
        help="If true, at the very end, we will detokenize everything and check if it matches the original string",
        type=bool, default=False)
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    vocab, merges = token_utils.load_vocab_and_merges(
        args.vocab_path,
        args.merges_path,
    )

    tokenizer = tokenization.Tokenizer(
        vocab, merges, special_tokens=["<|endoftext|>"])

    stats = tokenization.tokenize_file(
        tokenizer, args.input_path, args.max_chunk_bytes, args.output_path)
    print("Tokenization stats")
    for k, v in stats.items():
        print(k, ": ", v)

    if args.test_detokenize:
        all_bytes = open(args.output_path, "rb").read()
        np_array = np.frombuffer(all_bytes, dtype="uint16")
        int_array = np_array.tolist()
        print("Before detokenize")
        all_text = tokenizer.decode(int_array)
        print("Before read")
        orig_text = open(args.input_path, "r").read()
        print("I am here!!!")
        assert all_text == orig_text
        print("Assertion passed")
