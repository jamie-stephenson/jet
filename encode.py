from src.dist_utils import setup, cleanup
from src.tokenizer import Tokenizer
from src.file_utils import PathFetcher, args_from_config_file
from project_datasets import get_dataset
import torch.distributed as dist
import argparse
import os

def main(args):

    rank, world_size = dist.get_rank(), dist.get_world_size()

    paths = PathFetcher(args)
    assert os.path.exists(paths.tokenizer),(
        "No tokenizer found at {}. Please train this tokenizer first before attempting to use it."
        .format(paths.tokenizer)
    )

    assert not os.path.exists(paths.encoded_corpus),(
        "A directory named {} already exists. Have you already used {} to encode {}?."
        .format(paths.encoded_corpus,paths.tokenizer,args.corpus)
    )
    
    corpus = get_dataset(args.corpus,paths.corpus,rank,world_size)
    tokenizer = Tokenizer.from_pickled_merges(paths.tokenizer,rank,world_size)
    tokenizer.save_encoded_corpus(corpus,paths.encoded_corpus)

if __name__ == '__main__':
    """Trains and saves new tokenizer based on command line input."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus",
        type=str,
        help="The corpus to be encoded."
    )

    parser.add_argument(
        "--tokenizer_corpus",
        type=str,
        help='If specified, the corpus will be encoded using a pretrained tokenizer trained on `tokenzier_corpus`.'
    )

    parser.add_argument(
        "--vocab_size",
        type=int
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
