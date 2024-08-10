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