from torch.nn import Module, Parameter, MSELoss, Linear, LazyLinear
from torch import Tensor
from typing import Any, cast, Self
from collections.abc import Callable
from math import inf


class RAI(Module):
    _forward: Module
    _loss_fn: Module
    _period: int | float

    def __init__(self, func: Module, loss_fn: Module = MSELoss(), period: int | float = inf) -> None:
        super().__init__() # type: ignore
        self._forward = func
        self._loss_fn = loss_fn
        self._period = period
    
    def forward(self, *arg: Any, **kwg: Any) -> Tensor:
        forward = cast(Callable[..., Tensor], self._forward)
        if self.training:
            if not hasattr(self, "count"):
                self.count = 0
            if not hasattr(self, "target") or self.count % self._period == 0:
                self.target = Parameter(forward(*arg, **kwg).detach())
            self.loss = self._loss_fn(self.target, forward(*arg, **kwg))
            self.count += 1
            return self.target
        return forward(*arg, **kwg)
    
    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if not mode:
            if hasattr(self, "target"):
                del self.target
            if hasattr(self, "loss"):
                del self.loss
            if hasattr(self, "count"):
                del self.count
        return self
    
    def pause(self, mode: bool = True) -> Self:
        super().train(not mode)
        for func in self.modules():
            if isinstance(func, RAI):
                func.pause(mode)
        return self


def loss(func: Module) -> float | Tensor:
    gl_loss = 0.
    for sub in func.modules():
        if isinstance(sub, RAI) and hasattr(sub, "loss"):
            gl_loss = gl_loss + sub.loss
    return gl_loss


def wrap(func: Module, loss_fn: Module = MSELoss(), period: int | float = inf, include: tuple[type, ...] | None = (Linear, LazyLinear)) -> Module:
    include = include or tuple()
    assert include, f"It makes no sense to wrap {func} with an empty include list."
    for att, obj in func.named_children():
        setattr(func, att, wrap(obj, loss_fn, period, include))
    if isinstance(func, include):
        return RAI(func, loss_fn, period)
    return func
