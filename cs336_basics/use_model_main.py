from cs336_basics import tokenization, token_utils, lm_training, transformer
import argparse
import torch

def use_model(args):
    tokenizer = tokenization.Tokenizer.from_files(
        args.vocab_path, args.merges_path)
    dtype = lm_training.get_dtype(args.dtype)
    theta = args.rope_theta if args.rope_theta > 0 else None
    
    model = transformer.TransformerLM(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        d_model=args.d_model, num_heads=args.num_heads, d_ff=args.d_ff,
        max_seq_len=args.context_length, theta=theta, dtype=dtype)
    lm_training.load_checkpoint(args.checkpoint_path, model, None)

    model.eval()

    with torch.no_grad():
        while True:
            input_prompt = input("Enter prompt (enter Q to quit): ")
            if input_prompt == "Q":
                break
            tokens = tokenizer.encode(input_prompt)
            print("Max tokens: ", args.max_tokens)
            while len(tokens) <= args.max_tokens and tokens[-1] != 0:
                tokens_tensor = torch.Tensor(tokens).to(torch.int64)
                output = model(tokens_tensor)
                logits = output[-1]
                logits /= args.temperature
                probs = transformer.softmax(logits, -1)
                if args.top_p is not None:
                    probs = transformer.top_p(probs, args.top_p)
                new_token = torch.multinomial(probs, 1).item()
                print("Token ", new_token)
                tokens.append(new_token)
            output = tokenizer.decode(tokens)
            print("Output: ")
            print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--checkpoint_path",
        help="Path to checkpoint", required=True)
    parser.add_argument(
        "--vocab_size",
        help="Total number of tokens, it will be assumed the largest token is vocab_size -1",
        type=int, default=10000)
    parser.add_argument(
        "--num_layers", help="Number of layers", type=int, default=4)
    parser.add_argument(
        "--num_heads", help="Number of heads", type=int, default=16)
    parser.add_argument(
        "--d_model", help="Model embedding size", type=int, default=512)
    parser.add_argument(
        "--d_ff", help="Dimension of feed-forward layer", type=int,
        default=1344)
    parser.add_argument("--context_length",
                        help="Context length", type=int, default=256)
    parser.add_argument(
        "--rope_theta",
        help="Theta param for rotary position embeddings, if zero or negative, no position embeddings",
        type=float, default=10000)
    parser.add_argument("--dtype", help="The dtype to use", default="float32")
    parser.add_argument(
        "--vocab_path", help="Path to vocabularty", required=True)
    parser.add_argument(
        "--merges_path",
        help="Path to the merges for the vocabulary",
        required=True)
    parser.add_argument(
        "--temperature", help="Softmax temperature", type=float, default=1)
    parser.add_argument(
        "--top_p", help="If provided, will do nucleus sampling", type=float)
    parser.add_argument(
        "--max_tokens", help="Maximum number of tokens to generate", default=256
    )
    
    
    args = parser.parse_args()
    use_model(args)
