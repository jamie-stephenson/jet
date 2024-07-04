from src.dist_utils import setup, cleanup
from src.tokenizer import Tokenizer
from src.file_utils import PathFetcher, args_from_config_file
import torch.distributed as dist
import importlib
import argparse
import os
import wandb

def main(args):

    rank, world_size = dist.get_rank(), dist.get_world_size()

    paths = PathFetcher(args)

    if not args.no_wandb and rank == 0:
        wandb.init(project='jet',name=f"encode_ws{world_size}_vs{args.vocab_size}",config=args)
    
    corpus = get_dataset(args.corpus,paths.corpus,rank,world_size)

    if args.tokenizer_corpus == args.corpus:
        if os.path.exists(paths.encoded_tokenizer_corpus):
            if rank == 0:
                print(f"Encoded corpus already exists at {paths.encoded_corpus}.")
        else:
            tokenizer = Tokenizer.from_corpus(corpus,args.vocab_size,rank,world_size)
            tokenizer.save_encoded_tokenizer_corpus(paths.encoded_corpus,args.encoded_format)
            tokenizer.save_merges(paths.tokenizer)
    else:
        if os.path.exists(paths.tokenizer):
            tokenizer = Tokenizer.from_pickled_merges(paths.tokenizer,rank,world_size)
        else:
            tokenizer_corpus = get_dataset(args.tokenizer_corpus,paths.tokenizer_corpus,rank,world_size)
            tokenizer = Tokenizer.from_corpus(tokenizer_corpus,args.vocab_size,rank,world_size)
            tokenizer.save_encoded_tokenizer_corpus(paths.encoded_tokenizer_corpus,args.encoded_format)
            tokenizer.save_merges(paths.tokenizer)
        tokenizer.save_encoded_corpus(corpus,paths.encoded_corpus,args.encoded_format)


def get_dataset(name, path, rank, world_size):
    
    dataset_module = importlib.import_module("project_datasets." + name, package=".")
    dataset = dataset_module.get_dataset(path,rank,world_size)

    return dataset

if __name__ == '__main__':
    """Trains and saves new tokenizer based on command line input."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus",
        type=str,
        help="The corpus to be encoded."
    )
    
    parser.add_argument(
        "--encoded_format",
        choices=['shards','mmap'],
        help="The desired format of the encoded corpus. If not already encoded, both corpora will be encoded in this format."
    )

    parser.add_argument(
        "--tokenizer_corpus",
        default=None,
        type=str,
        help='If specified, the corpus will be encoded using a pretrained tokenizer trained on `tokenzier_corpus`.'
    )

    parser.add_argument(
        "--vocab_size",
        type=int
    )

    parser.add_argument(
        "--no-wandb",
        "--no_wandb",
        action="store_true",
        help="If set, wandb logging is disabled."
    )

    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        help="Path to optional config file."
    )

    args = parser.parse_args()

    if args.config_file:
        args = args_from_config_file(args)

    if not args.tokenizer_corpus:
        args.tokenizer_corpus = args.corpus 

    setup('gloo')
    main(args)
    cleanup()

    """ 
    NOTE: To use torchrun on windows you may have to change the path on line 1 of
    {your_env}/Scripts/torchrun-script.py to be the path to the python.exe file
    in your python environment.
    """
