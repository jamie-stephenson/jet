import torch.distributed as dist
import os

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
    - HF: we have groups of shards that are concatenated to form opera.
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

    
