import argparse
import importlib
import os

def download_dataset(name,path,nproc):
        
    dataset_module = importlib.import_module("project_datasets." + name, package=".")
    dataset_module.download_dataset(path,nproc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        help="Name of dataset."
    )

    parser.add_argument(
        "--path",
        type=str,
        help="Path to dir to download data to."
    )

    parser.add_argument(
        "--num_proc",
        default=os.cpu_count(),
        type=int,
        help="Number of processes to use for downloading."
    )

    args = parser.parse_args()

    download_dataset(args.name,args.path,args.nproc)