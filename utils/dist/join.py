import torch
from torch.distributed.algorithms.join import Join, JoinHook, Joinable

from typing import Callable

class FnJoinable(Joinable):
    """
    :class:`Joinable` that performs `fn(**kwargs)` for across *all* processes,
    even those that have joined. The expected use case assumes that `fn` 
    has some kind of collective communication so that it will synchronise this 
    class's `__call__` method with the `main_hook` method of :class:`_FnJoinHook`, 
    otherwise `fn` will repeatedly execute asyncronously from `__call__`. 
    """
    def __init__(self, fn: Callable, device, process_group, **kwargs):
        super(FnJoinable, self).__init__()
        self.fn = fn
        self.device = device
        self.process_group = process_group
        self.kwargs = kwargs

    def __call__(self):
        Join.notify_join_context(self)
        return self.fn(**self.kwargs)

    def join(
        self,
        enable: bool = True,
        throw_on_early_termination: bool = False,
    ):

        return Join(
            [self],
            enable,
            throw_on_early_termination
        )

    def join_hook(self, **kwargs) -> JoinHook:
        return _FnJoinHook(self)
    
    @property
    def join_device(self) -> torch.device:
        return self.device

    @property
    def join_process_group(self):
        return self.process_group
    
class _FnJoinHook(JoinHook):

    def __init__(self,joinable):
        self.joinable = joinable

    def main_hook(self):
        self.joinable.fn(**self.joinable.kwargs)          

    def post_hook(self, is_last_joiner: bool):
        pass