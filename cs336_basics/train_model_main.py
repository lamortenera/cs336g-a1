from cs336_basics import tokenization, token_utils, lm_training, transformer
from typing import Optional
import logging
import argparse
import sys
import numpy as np
import wandb
import json
import datetime
import torch
import time
import gc
import psutil
import random


class Logger(object):
    def __init__(self, output_file: Optional[str], run_name: str, run_desc: str, config: dict, use_wandb: bool = True):
        print("Use wandb: ", use_wandb)
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.login()
        self.init_args = {"project": "cs336_train_lm", "name": run_name, "notes": run_desc, "config": config}
        self.initialized = False
        self.closed = False
        self.output_file = output_file

    def _dict_to_string(self, d, prefix=""):
        parts = []
        for k, v in d.items():
            if type(v) == dict:
                parts.append(self._dict_to_string(v, prefix=f"{k}."))
            else:
                key = prefix + str(k)
                value = str(v) if type(v) != float else f"{v:.3f}"
                parts.append(f"{key}:{value}")
        return "\t".join(parts)
        
    def __enter__(self):
        assert not self.closed
        self.initialized = True
        if self.use_wandb:
            self.run = wandb.init(**self.init_args)
        if self.output_file is not None:
            self.output_stream = open(self.output_file, "w")
            self.output_stream.write(json.dumps(self.init_args) + "\n")
        else:
            self.output_stream = None
        print(self._dict_to_string(self.init_args))
        return self

    def __exit__(self, type, value, traceback):
        if self.use_wandb:
            self.run.finish()
        if self.output_stream is not None:
            self.output_stream.close()
        self.closed = True

    def log(self, data: dict):
        assert self.initialized, "The logger must be used as a context manager"
        if self.use_wandb:
            self.run.log(data)
        if self.output_stream is not None:
            self.output_stream.write(json.dumps(data) + "\n")
        print(self._dict_to_string(data))


def resource_estimate(args):
    num_params = 2*args.vocab_size*args.d_model
    num_params += args.num_layers* (args.d_model*args.d_model*4 + 2*args.d_model + 3*args.d_model*args.d_ff)
    num_params += args.d_model

    num_activations = args.d_model + (2*args.vocab_size + 1)
    num_activations += args.num_layers*(
            8*args.d_model + 2*args.context_length +2*args.d_ff)
    num_activations *= args.batch_size*args.context_length

    print(f"Estimated params: {num_params}, ({4*num_params} bytes)")
    print(f"Estimated gradients: {num_params}, ({4*num_params} bytes)")
    print(f"Estimated opt states: {2*num_params}, ({8*num_params} bytes)")
    print(f"Estimated activations: {num_activations}, ({4*num_activations} bytes)")
    print(f"Estimated total: {num_activations + 4*num_params}, ({4*(num_activations + 4*num_params)} bytes)")

def model_params(model):
    num_params = 0
    num_bytes = 0
    for param in model.parameters():
        num_params += param.nelement()
        num_bytes += param.element_size()*param.nelement()
    print(f"Actual model params: {num_params}, bytes: {num_bytes}")

def debug_memory():
    mem_percent = psutil.virtual_memory().percent
    print(f"Using {mem_percent}% virtual memory")
    num_tensors = 0
    tensor_memory = 0
    obj_stats = {}
    tot_objects = 0
    tot_memory = 0
    objs = list(gc.get_objects())
    random.shuffle(objs)
    for obj in objs:
        tot_objects += 1
        s = obj_stats.get(type(obj), [0, 0, []])
        s[0] += 1
        s[1] += sys.getsizeof(obj)
        if len(s[2]) < 5:
            s[2].append(obj)
        tot_memory += sys.getsizeof(obj)
        obj_stats[type(obj)] = s
        #try:
        #    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #        num_tensors += 1
        #        tensor_memory += obj.element_size()*obj.nelement() 
        #        print(type(obj), obj.dtype, str(list(obj.size()))[1:-1], sys.getsizeof(obj), obj.untyped_storage().data_ptr(), obj.element_size()*obj.nelement(), sep="|")
        #        if tuple(obj.shape) == (36, 34, 10000):
        #            print(obj[:4,:4,:4])
        #except:
        #    pass

    #print(f"Found {num_tensors} tensors, tensor memory: {tensor_memory}")
    print(f"Found {tot_objects} objects, gc memory: {tot_memory}")
    print("Memory stats by object type:")
    for t, (c, m, ex) in sorted(obj_stats.items(), key=lambda x: -x[1][1])[:20]:
        print(f"Type {t}: count {c}, memory {100*m/tot_memory}%, examples: ")
        for e in ex:
            print("- " + str(e)[:100])

def train(args, run_name, logger):
    train_tokens = lm_training.TokenLoader(input_path=args.train_path, batch_size=args.batch_size, 
                                           context_length=args.context_length) 
    eval_tokens = lm_training.TokenLoader(input_path=args.eval_path, batch_size=args.batch_size,
                                          context_length=args.context_length)
    print("Created token loaders")
    model = transformer.TransformerLM(vocab_size=args.vocab_size, num_layers=args.num_layers,
                                      d_model=args.d_model, num_heads=args.num_heads, 
                                      d_ff=args.d_ff, max_seq_len=args.context_length,
                                      theta=args.rope_theta)
    print("Created model")

    model_params(model)

    optimizer = lm_training.AdamW(model.parameters(), lr_schedule_spec=args.lr_schedule, 
                                  weight_decay=args.weight_decay,
                                  betas=(args.beta1, args.beta2))
    print("Created optimizer") 
    start_time = time.time()
    for step in range(1, args.train_steps+1):
        model.train()
        inputs, labels = next(train_tokens)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = lm_training.cross_entropy(outputs, labels)
        #if step % 20 == 0:
        #    debug_memory()
        loss.backward()
        optimizer.step()
        
        logger.log({"step": step, "training_loss": loss.item(), "elapsed_s": time.time() - start_time})

        if step % args.eval_interval == 0:
            print("Evaluation phase")
            model.eval()
            with torch.no_grad():
                validation_loss = 0
                eval_tokens.reset()
                def get_validation_loss():
                    eval_inputs, eval_labels = next(eval_tokens)
                    eval_outputs = model(eval_inputs)
                    return lm_training.cross_entropy(eval_outputs, eval_labels).item()

                for _ in range(args.eval_steps):
                    validation_loss += get_validation_loss()
                validation_loss /= args.eval_steps
                logger.log({"step": step, "eval_loss": validation_loss, 
                            "elapsed_s": time.time() - start_time})

        if step % args.checkpoint_interval == 0:
            out_path = args.checkpoint_dir + "/" + run_name + f"/checkpoint_{step}/data.pt"
            lm_training.save_checkpoint(model, optimizer, step, out_path)
                
        


def parse_explicit_bool(s):
    s = s.strip().lower()
    assert s in ["true", "false"]
    return s == "true"

if __name__ == "__main__":
    print("Running command:\n" + " ".join(sys.argv[:]))

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run_id_prefix", help="A valid python identifier to identify this run", required=True)
    parser.add_argument("--run_desc", help="A description for this run", required=True)
    parser.add_argument("--train_path", help="Path to a numpy array of uint16 with the tokens, serialized as bytes", required=True)
    parser.add_argument("--eval_path", help="Path to a numpy array of uint16 with the tokens, serialized as bytes", required=True)
    parser.add_argument("--train_steps", help="How many training steps to run in total", type=int, default=5000)
    parser.add_argument("--eval_steps", help="How many eval steps to run every time that there is an eval phase", type=int, default=25)
    parser.add_argument("--eval_interval", help="Every eval_interval training steps, run the eval phase", type=int, default=100)
    parser.add_argument("--checkpoint_interval", help="How many steps for saving the checkpoint", type=int, default=100)
    parser.add_argument("--checkpoint_dir", help="Directory where to save checkpoints", required=True)
    parser.add_argument("--logs_path", help="Path to an output file with a json dict per line, if not provided, no logs will be written other than wandb and stdout")
    parser.add_argument("--use_wandb", help="If true, log data to weight and biases", default="True")
    parser.add_argument("--vocab_size", help="Total number of tokens, it will be assumed the largest token is vocab_size -1", type=int, required=True)
    parser.add_argument("--endoftext_token", help="The token that represents the end of text", type=int, default=0)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=32)
    parser.add_argument("--context_length", help="Context length", type=int, default=256)
    parser.add_argument("--num_layers", help="Number of layers", type=int, default=4)
    parser.add_argument("--num_heads", help="Number of heads", type=int, default=16)
    parser.add_argument("--d_model", help="Model embedding size", type=int, default=512)
    parser.add_argument("--d_ff", help="Dimension of feed-forward layer", type=int, default=1344)
    parser.add_argument("--rope_theta", help="Theta param for rotary position embeddings, if zero or negative, no position embeddings", type=float, default=10000)
    parser.add_argument("--lr_schedule", help="Learning rate schedule in the format '0,lr0|fn0|x1,y1|fn1|...|xn,yn', where fni can be LIN or COS", required=True)
    parser.add_argument("--beta1", help="Beta1 param in AdamW", type=float, default=0.9)
    parser.add_argument("--beta2", help="Beta2 param in AdamW", type=float, default=0.999)
    parser.add_argument("--weight_decay", help="Weight decay parameter in AdamW", type=float, default=0.01)
    args = parser.parse_args()
    assert args.run_id_prefix.isidentifier()
    

    resource_estimate(args)
    args_dict = vars(args)

    run_name = args.run_id_prefix + "_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    logger = Logger(args.logs_path, run_name, args.run_desc, args_dict, use_wandb=parse_explicit_bool(args.use_wandb))

    with logger as l:
        train(args, run_name, l)
