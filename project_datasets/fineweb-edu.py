from datasets import load_dataset
from src.dist_utils import find_opus_indices
import torch.distributed as dist

def get_dataset(path,rank,world_size):
    start, end = find_opus_indices(2101,rank,world_size)
    if rank == 0: # Make sure only one node downloads the dataset
        dataset = load_dataset(
            path='HuggingFaceFW/fineweb-edu',
            name='sample-10BT',
            split=f'train[{start}:{end}]',
            cache_dir=path
        )
    dist.barrier()
    if rank !=0:
        dataset = load_dataset(
            path='HuggingFaceFW/fineweb-edu',
            name='sample-10BT',
            split=f'train[{start}:{end}]',
            cache_dir=path
        )
    return '\n'.join(dataset['text'])