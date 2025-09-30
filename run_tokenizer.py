from cs336_basics import tokenization, token_utils
import logging
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Path to text file with data for training the tokenizer", required=True)
    parser.add_argument("--output_path_prefix", help="Path where to write the tokenizer info, if absent, nothing is written")
    parser.add_argument("--vocab_size", help="Desired size of the vocabulary", type=int, required=True)
    parser.add_argument("--max_chunk_bytes", help="Maximum number of bytes to load in memory at a time", type=int, default=20000000)
    parser.add_argument("--num_workers", help="Number of workers for the pretokenization phase", type=int, default=1)
    parser.add_argument("--log_level", help="Log level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    vocab, merges = tokenization.train_tokenizer(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
        max_chunk_bytes=args.max_chunk_bytes,
        num_workers=args.num_workers
    )

    if args.output_path_prefix:
        vocab_path = args.output_path_prefix + "vocab.json"
        merges_path = args.output_path_prefix + "merges.txt"
        token_utils.save_vocab_and_merges(vocab, merges, vocab_path=vocab_path, merges_path=merges_path)



