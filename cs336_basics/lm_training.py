from collections.abc import Callable, Iterable
import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, Int
from einops import rearrange
from typing import Optional
import math
import os
import copy

def cross_entropy(predictions: Float[torch.Tensor, "... vocab_size"],
                  labels: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    loss = torch.nn.CrossEntropyLoss()
    predictions = predictions.reshape(-1, predictions.shape[-1])
    return loss(predictions, labels.flatten())

    #norm_predictions = predictions - torch.max(predictions, axis=-1, keepdims=True).values
    #chosen_predictions = torch.take_along_dim(norm_predictions, labels[...,None], -1).squeeze(-1)
    #return torch.mean(-chosen_predictions + torch.log(torch.exp(norm_predictions).sum(axis=-1)))



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=None, lr_schedule_spec=None, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        assert (lr is None) != (lr_schedule_spec is None), "Only one of lr or lr_schedule_spec can be provided"
        assert lr is None or lr > 0, f"Invalid negative learning rate: {lr}" 
        if lr_schedule_spec is not None:
            LRScheduler(lr_schedule_spec)  # Check that the specs are valid

        defaults = {"lr": lr, "lr_schedule_spec": lr_schedule_spec, "betas": betas, 
                    "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["m"] = torch.zeros(p.data.shape)
                state["v"] = torch.zeros(p.data.shape)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            fixed_lr = group["lr"]
            lr_scheduler = None
            if fixed_lr is None:
                lr_scheduler = LRScheduler(group["lr_schedule_spec"])

            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0) + 1
                lr = fixed_lr if fixed_lr is not None else lr_scheduler.lr(t)

                grad = p.grad.data

                state["m"] = betas[0] * state["m"] + (1 - betas[0]) * grad
                state["v"] = betas[1] * state["v"] + (1 - betas[1]) * (grad * grad)
                lr_adj = lr * math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t)
                p.data -= lr_adj * state["m"]/(torch.sqrt(state["v"]) + eps)
                if (weight_decay > 0):
                    p.data *= (1 - lr * weight_decay)
                state["t"] = t
        return loss


class LRScheduler(object):
    _LIN_FN = "lin"
    _COS_FN = "cos"
    _ALLOWED_FNS = [_LIN_FN, _COS_FN]
    def __init__(self, specs):
        # Expects a spec like x1,y1|lin|x2,y2|cos|x3,y3
        # where x1 must be present and equal to 0 and the xs must
        # be striclty increasing.
        # The learning rate is interpolated from the ys,
        # if the iteration is after the last xi, the last yi will be used.
        steps = specs.split("|")
        assert steps, "Empty learning rate spec: " + spec
        assert len(steps) % 2 == 1, "Expecting an odd number of steps"

        def parse_pair(step):
            x, y = step.split(",")
            return (float(x), float(y))

        pair = parse_pair(steps[0])
        assert pair[0] == 0, "The first x coordinate must be 0"

        pairs = [pair]
        fns = []
    
        for i in range(1, len(steps), 2):
            fn = steps[i]
            assert fn in self._ALLOWED_FNS, "Expecting a function specification among: " + ",".join(self._ALLOWED_FNS) + ", found " + fn
            next_pair = parse_pair(steps[i+1])
            assert pair[0] < next_pair[0], "x coordinates must be strictly increasing"
            pairs.append(next_pair)
            fns.append(fn)
            pair = next_pair

        self.pairs = pairs
        self.fns = fns

    @staticmethod
    def _interpolate(pair, it, next_pair, fn):
        x_ratio = (it - pair[0])/(next_pair[0] - pair[0])
        y_delta = next_pair[1] - pair[1]
        if fn == LRScheduler._LIN_FN:
            return pair[1] + y_delta*x_ratio
        if fn == LRScheduler._COS_FN:
            return pair[1] + 0.5*(1 - math.cos(x_ratio*math.pi))*y_delta


    def lr(self, it):
       pair = self.pairs[0] 
       for i in range(1, len(self.pairs)):
           next_pair = self.pairs[i]
           if it <= next_pair[0]:
               return self._interpolate(pair, it, next_pair, self.fns[i-1])
           pair = next_pair
       return pair[1]


def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float):
    l2_norm_squared = 0
    for param in params:
        if param.grad is None:
            continue
        l2_norm_squared += torch.sum(param.grad*param.grad)
    
    l2_norm = math.sqrt(l2_norm_squared)

    if l2_norm >= max_l2_norm:
        scale_factor = max_l2_norm/(l2_norm + 1e-6)
        for param in params:
            if param.grad is None:
                continue
            param.grad *= scale_factor


def get_batch(arr: npt.NDArray, generator: np.random.Generator, batch_size: int, context_length: int, device=None):
    starts = generator.integers(0, high=len(arr) - context_length, size=batch_size)
    offsets = np.arange(context_length + 1)
    indices = starts.reshape((batch_size, 1)) + offsets.reshape((1, context_length + 1))
    np_tokens = arr[indices].astype(np.int64)
    tokens = torch.from_numpy(np_tokens)
    if device is not None:
        tokens = tokens.to(device=device)
    return tokens[:,:context_length], tokens[:,1:]


class TokenLoader(object):
    def __init__(self, *, input_path: str, batch_size: int, context_length: int, seed: int=123, device=None):
        self.batch_size = batch_size
        self.context_length = context_length
        self.seed = seed
        self.arr = np.memmap(input_path, dtype=np.uint16)
        self.device = device
        self.reset()

    def reset(self):
        self.generator = np.random.default_rng(self.seed)

    def __next__(self):
        return get_batch(self.arr, self.generator, self.batch_size, self.context_length, self.device)


def save_checkpoint(model, optimizer, iteration, out):
    directory = os.path.dirname(out)
    if not os.path.exists(directory):
        os.makedirs(directory)

    def detach_values(d):
        for k in d.keys():
            if torch.is_tensor(d[k]):
                d[k] = d[k].detach()
    d = {
        "model": detach_values(model.state_dict()),
        "optimizer": detach_values(optimizer.state_dict()),
        "iteration": iteration
    }
    torch.save(d, out)

def load_checkpoint(src, model, optimizer):
    d = torch.load(src)
    model.load_state_dict(d["model"])
    optimizer.load_state_dict(d["optimizer"])
    return d["iteration"]


