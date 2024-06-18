from datasets import load_dataset
from src.dist_utils import find_opus_indices
import torch.distributed as dist
import argparse

def get_dataset(path,rank,world_size):
    start, end = find_opus_indices(100000,rank,world_size) # We train on 100,000 documents TODO: make this an arg
    if rank == 0: # Make sure at most one node downloads the dataset
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        help="Path to dir to download data to."
    )

    parser.add_argument(
        "--num_proc",
        type=str,
        help="Number of processes to use for downloading."
    )

    args = parser.parse_args()

    load_dataset(
            path='HuggingFaceFW/fineweb-edu',
            name='sample-10BT',
            split='train',
            cache_dir=args.path,
            num_proc=args.num_proc
    )