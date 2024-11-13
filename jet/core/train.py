from jet.core.model import get_model
from jet.utils import *

from bpekit import Tokenizer
import torch
import torch.distributed as dist
import wandb

import argparse
import os
import yaml
from pathlib import Path


def train_model(cfg: Config):

    paths = cfg.get_paths()

    if not cfg.no_wandb and cfg.rank == 0:
        wandb.init(project='jet',name=paths.wandb,config=cfg)
        wandb.define_metric("Effective Batch Number") 

    # Only used for sample output generation during training
    tokenizer = Tokenizer.from_pickled_merges(paths.tokenizer) 

    model = get_model(cfg)

    train_dataloader = get_dataloader(paths.tokens, 'train', cfg)
    eval_dataloader = get_dataloader(paths.tokens, 'val', cfg)

    optimizer = get_optimizer(
        cfg.optimizer.name, 
        model, 
        **cfg.optimizer.params
    )
    
    lr_scheduler = get_lr_scheduler(
        cfg.lr_schedule.name, 
        optimizer, 
        steps_per_epoch=len(train_dataloader)//cfg.grad_accumulation_steps, 
        **cfg.lr_schedule.params
    )
    
    model = train(
        model,
        tokenizer,
        train_dataloader,
        eval_dataloader,
        optimizer,
        lr_scheduler,
        cfg
    )

    if cfg.rank == 0:
        os.makedirs(os.path.dirname(paths.model))
        torch.save(model.state_dict(),paths.model)
        with open(paths.model_config,'w') as file:
            yaml.dump(vars(cfg), file)


def get_parser():

    # NOTE: Some args cannot be set from command line (e.g. nested args like optimizer).
    # These must be set from a config yaml file.
    # NOTE: No defaults. These are set from the config file.
    # Command line args should just be used as a convenient override of config file
    # for specific args here and there.

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to optional config file."
    )

    parser.add_argument(
        "--name",
        type=str,
        help="Used to name run folder."
    )

    parser.add_argument(
        "--tokens",
        type=Path,
        help="The path to the tokenized dataset."
    )

    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="The path to the tokenzier merges."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="The number of epochs to train for."
    )

    parser.add_argument(
        "--dropout",
        type = float,
        help="Proportion of elements to zero in layer where dropout is used."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size used for training."
    )
    
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        help="The number of batches to accumulate into one effective batch. Used with distributed training."
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        help="Sequence length used for training."
    )

    parser.add_argument(
        "--overlap",
        type=int,
        help="Sequence overlap used for training."
    )

    parser.add_argument(
        "--d_model",
        type=int,
        help="Dimension tokens are embedded into."
    )

    parser.add_argument(
        "--n_heads",
        type=int,
        help="Number of attention heads in each attention layer."
    )

    parser.add_argument(
        "--mask_type",
        type=str,
        help="How the attention is masked."
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        help="If set, attempts to set device to cuda:{$LOCAL_RANK} (if available)."
    )

    parser.add_argument(
        "--autocast",
        action="store_true",
        help="If set, attempts to use bfloat16 autocast (if available on system)."
    )

    parser.add_argument(
        "--n_workers",
        type=int,
        help="Number of subprocesses to use for dataloading each GPU."
    )

    parser.add_argument(
        "--no-wandb",
        "--no_wandb",
        action="store_true",
        help="If set, wandb logging is disabled."
    )

    parser.add_argument(
        "--eff_batch_per_log",
        help="Number of effective batches per log."
    )

    parser.add_argument(
        "--log_per_val",
        type=int,
        help="Number of logs per validation run."
    )

    parser.add_argument(
        "--val_prompt",
        type=str,
        help="Prompt from which to generate sample output during training."
    )

    parser.add_argument(
        "--temp",
        help="Temperature to use for sample output generation during training."
    )

    parser.add_argument(
        "--seed",
        help="Seed for random output during training."
    )

    return parser

if __name__ == '__main__':

    args = get_parser().parse_args()

    cfg = Config.build_from(args.config_file,args)

    torch.manual_seed(cfg.seed)

    setup(cfg.backend) 
    cfg.rank = dist.get_rank()
    cfg.world_size = dist.get_world_size() 
    
    train_model(cfg)

    cleanup()
