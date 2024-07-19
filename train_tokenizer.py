from src.dist_utils import setup, cleanup
from src.tokenizer import Tokenizer
from src.file_utils import PathFetcher, args_from_config_file
from project_datasets import get_dataset
import torch.distributed as dist
import argparse
import os
import wandb

def main(args):
    rank, world_size = dist.get_rank(), dist.get_world_size()

    paths = PathFetcher(args)

    assert not os.path.exists(paths.tokenizer),(
        "A tokenizer already exists at {}. Have you trained this tokenizer already?"
        .format(paths.tokenizer)
    )

    if not args.no_wandb and rank == 0:
        wandb.init(project='jet',name=f"tokenizer_ws{world_size}_vs{args.vocab_size}",config=args)
    
    corpus = get_dataset(args.tokenizer_corpus,paths.tokenizer_corpus,rank,world_size)

    tokenizer = Tokenizer.from_corpus(corpus,args.vocab_size,rank,world_size)
    tokenizer.save_merges(paths.tokenizer)
    tokenizer.save_encoded_tokenizer_corpus(paths.encoded_tokenizer_corpus)

if __name__ == '__main__':
    """Trains and saves new tokenizer based on command line input."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokenizer_corpus",
        default=None,
        type=str,
        help='The corpus to train the tokenizer on.'
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

    setup('gloo')   
    main(args)
    cleanup()

    """ 
    NOTE: To use torchrun on windows you may have to change the path on line 1 of
    {your_env}/Scripts/torchrun-script.py to be the path to the python.exe file
    in your python environment.
    """
