from utils.files import PathFetcher

from datasets import load_dataset

import argparse
import os
import yaml

def download_dataset(
        hf_path: str,
        save_path: str,
        name: str | None = None,
        split: str | None = 'train',
        num_proc: int | None = os.cpu_count(),
        **kwargs
    ):
    """Download a Hugging Face dataset, save it to disk and remove the cached files"""
    
    dataset = load_dataset(
        path=hf_path,
        name=name,
        split=split,
        num_proc=num_proc
    )
    
    dataset.save_to_disk(save_path)
    dataset.cleanup_cache_files()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of dataset (as found in the `configs/project_datasets/` directory)."
    )

    args = parser.parse_args()

    paths = PathFetcher(args)

    with open(paths.dataset_config,'r') as file:
        yaml_config = yaml.safe_load(file)

    download_dataset(save_path=paths.dataset,**yaml_config)