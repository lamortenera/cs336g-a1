from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import argparse
import sys


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad
                state["t"] = t + 1
        return loss


if __name__ == "__main__":
    print("Running command:\n" + " ".join(sys.argv[:]))

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate",
                        help="The learning rate", type=float, required=True)

    args = parser.parse_args()

    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=args.learning_rate)

    for t in range(100):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()

    print("Final weights: ", weights.cpu())
