from utils.files import PathFetcher
import argparse
import importlib
import os

def main(args):
    
    paths = PathFetcher(args)

    download_dataset(args.corpus, paths.corpus, args.nproc)

def download_dataset(name,path,nproc):
        
    dataset_module = importlib.import_module("project_datasets." + name, package=".")
    dataset_module.download_dataset(path,nproc)

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