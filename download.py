from utils.files import PathFetcher
from project_datasets import download_dataset

import argparse
import os

def main(args):
    
    paths = PathFetcher(args)
    download_dataset(args.corpus, paths.corpus, args.nproc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus",
        type=str,
        help="Name of dataset."
    )

    parser.add_argument(
        "--nproc",
        default=os.cpu_count(),
        type=int,
        help="Number of processes to use for downloading."
    )

    args = parser.parse_args()

    main(args)