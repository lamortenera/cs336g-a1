from collections.abc import Callable, Iterable
import torch
from jaxtyping import Float, Int
from einops import rearrange
from typing import Optional
import math

def cross_entropy(predictions: Float[torch.Tensor, "batch_size vocab_size"],
                  labels: Int[torch.Tensor, "batch_size"]):
    norm_predictions = predictions - torch.max(predictions, axis=-1, keepdims=True).values
    chosen_predictions = torch.take_along_dim(norm_predictions, labels[:,None], -1)
    return torch.mean(-chosen_predictions + torch.log(torch.exp(norm_predictions).sum(axis=-1)))



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, 
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
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0) + 1
                grad = p.grad.data

                state["m"] = betas[0] * state["m"] + (1 - betas[0]) * grad
                state["v"] = betas[1] * state["v"] + (1 - betas[1]) * (grad * grad)
                lr_adj = lr * math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t)
                p.data -= lr_adj * state["m"]/(torch.sqrt(state["v"]) + eps)
                if (weight_decay > 0):
                    p.data *= (1 - lr * weight_decay)
                state["t"] = t
        return loss


def learning_rate_schedule(it: int, alpha_min: float, alpha_max: float, 
                           t_warmup: int, t_cosine: int) -> float:
    assert it >= 0, it
    assert alpha_min > 0, alpha_min
    assert alpha_max >= alpha_min, f"alpha_max: {alpha_max}, alpha_min: {alpha_min}"
    assert t_warmup > 0
    assert t_cosine > t_warmup, f"t_cosine: {t_cosine}, t_warmup: {t_warmup}"

    if it < t_warmup:
        return alpha_max * (it / t_warmup)
    if it < t_cosine:
        ratio = (it - t_warmup)/(t_cosine - t_warmup)
        return alpha_min + 0.5*(1 + math.cos(ratio*math.pi))*(alpha_max - alpha_min)
    return alpha_min

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

