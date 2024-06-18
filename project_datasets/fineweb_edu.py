from datasets import load_dataset
from src.dist_utils import find_opus_indices
import argparse
import os

def get_dataset(path,rank,world_size):
    assert os.path.exists(path), (
        f"Dataset fineweb-edu does not exist at {path}.\n"
        f"Please download by running `python ./project_datasets/fineweb_edu.py --path {path} --num_proc {os.cpu_count()}`"
    ) #TODO stop this from duplicating across processes while still ending process group gracefully

    start, end = find_opus_indices(100000,rank,world_size) # We train on 100,000 documents TODO: make this an arg

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