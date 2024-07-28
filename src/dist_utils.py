import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join, JoinHook, Joinable
from typing import Callable
import os

#========Handle Process Groups

def setup(backend):
    """setup function that needs to be ran on each process."""
    if backend == "gloo": # For training tokenizers
        assert dist.is_gloo_available(), "gloo backend unavailable"
    elif backend == "nccl": # For training models
        assert dist.is_nccl_available(), "nccl backend unavailable"
    else:
        raise ValueError(f"The {backend} backend is not supported.")
    if is_torchrun():
        # `init_method` does not need to be set as the default (`env://`) requires only the
        # env variables that are automatically set by using `torchrun`. 
        dist.init_process_group(backend) 
    else: 
        # If we aren't using distributed training we still need to 
        # init_process_group to keep scripts distribution agnostic.
        dist.init_process_group(backend,init_method='tcp://localhost:12345',rank=0,world_size=1)

def cleanup():
    dist.destroy_process_group()

def is_torchrun():
    return 'RANK' in os.environ #TODO: Better check?

#========Handle Non-Model Collective Comms within Join context
#  e.g. sharing eval metrics 

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
    
#========Distributing Data

#Currently not used, using np.memmap instead
def synced_write_to_file(data_chunk, file_path, rank, world_size):
    for i in range(world_size):
        if rank == i:
            with open(file_path, 'ab') as f:
                f.write(data_chunk)
            print(f"Rank {rank} wrote its part to the file.")
        dist.barrier()

def split_txt(path,rank,world_size):
    """
    Used to split a .txt corpus into `world_size` roughly equal opera, while 
    maintaining UTF-8 decodability. Used in distributed tokenizers.

    Returns only the opus that will be processed on rank `rank` (in byte form).
    """
    # Find start and end idx of opus, making sure that all
    # opera have a pairwise difference in length of at most 1.
    corpus_length = os.stat(path).st_size #size in bytes
    start, end = find_opus_indices(corpus_length,rank,world_size)
    opus_bytes = adjust_split_then_read(start,end,path)

    return opus_bytes

def find_opus_indices(n: int,rank: int,world_size: int):
    """
    Simple mathematical helper function that finds the start and end indices that split 
    `n` objects into `world_size` different "groups" whose sizes differ by at most 1.

    Returns only the indices that will be needed on rank `rank`.

    Note: This is used to split both Hugging Face and .txt corpora. In each case
    the "groups" represent different things:
    - HF: we have groups of examples that are concatenated to form opera.
      Each example is normally a document of some sort.
    - .txt: we have groups of bytes that are concatenated to form opera. 
    """
    quotient = n//world_size
    remainder = n%world_size
    if rank < remainder:
        # There are `remainder` groups that need an extra member
        start = rank * (quotient + 1)
        end = start + quotient + 1
    else:
        start = rank * quotient + remainder
        end = start + quotient
    return start, end

def adjust_split_then_read(start,end,path):
    """
        Helper function to shift corpus split points to just before spaces.
        This prevents words from being split in half.

        Returns the shifted corpus split as bytes.

        In this fucntion and the functions it calls, a lot of "pointer moving"
        occurs behind the scenes, this can make it quite hard to follow.
        When debugging, remember to keep track of `start`, `end` AND where
        the pointer is pointing. A lot of bugs (well only 2 actually) have
        been caused by the pointer ending up in the wrong spot.
    """
    with open(path, 'rb') as f:
        end = shift_while_condition_is_met(is_not_space,end,f)
        start = shift_while_condition_is_met(is_not_space,start,f)
        read_bytes = f.read(end-start)
    return read_bytes

def shift_while_condition_is_met(condition, point_to_shift, binary_being_read):
    binary_being_read.seek(point_to_shift)
    if point_to_shift == 0:
        return point_to_shift
    byte = binary_being_read.read(1)
    while byte and condition(byte):
        byte = binary_being_read.read(1)
    if byte:
        binary_being_read.seek(-1,1)
    return binary_being_read.tell()

def is_not_space(byte):
    return byte[0] != 0x20

#Not currently used but could be useful
def is_continuation_byte(byte):
    return (byte[0]&0b11000000) == 0b10000000

    
