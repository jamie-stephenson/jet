from jet import get_model
from utils.model import train, get_dataloader, get_optimizer, get_lr_scheduler
from utils.dist import setup, cleanup
from utils.files import PathFetcher, args_from_config_file 

from bpekit.core import Tokenizer
import torch
import torch.distributed as dist
import wandb

import argparse
import os
import yaml
from pathlib import Path


def main(args):

    paths = PathFetcher(args)

    if not args.no_wandb and args.rank == 0:
        wandb.init(project='jet',name=paths.wandb,config=args)
        wandb.define_metric("Effective Batch Number") 

    tokenizer = Tokenizer.from_pickled_merges(paths.tokenizer) # Only used for sample output generation during training
    model = get_model(args)
    train_dataloader = get_dataloader(paths.tokens, 'train', args)
    eval_dataloader = get_dataloader(paths.tokens, 'val', args)
    optimizer = get_optimizer(args.optimizer, model, args)
    lr_scheduler = get_lr_scheduler(
        args.lr_schedule, 
        optimizer, 
        steps_per_epoch=len(train_dataloader)//args.grad_accumulation_steps, 
        args=args
    )
    
    model = train(model,tokenizer,train_dataloader,eval_dataloader,optimizer,lr_scheduler,args)

    if args.rank == 0:
        os.makedirs(os.path.dirname(paths.model))
        torch.save(model.state_dict(),paths.model)
        with open(paths.model_config,'w') as file:
            yaml.dump(vars(args), file)

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="jet",
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
        default=1,
        help="The number of epochs to train for."
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="The optimizer to use for training."
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0,
        help="The momentum to be used by the optimizer during training."
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="The weight decay to be used by the optimizer during training"#
    )

    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="onecycle",
        help="The learning rate schedule used for training."
    )

    parser.add_argument(
        "--lr_gamma",
        type=str,
        default=0.1,
        help="Multiplicative factor of learning rate decay."
    )

    parser.add_argument(
        "--lr_warmup_end",
        type=float,
        default=0.4,
        help="The number of epochs to warm up the learning rate for"
    )

    parser.add_argument(
        "--lr_max",
        type=float,
        default=0.2,
        help="The maximum learning rate to use for training"
    )

    parser.add_argument(
        "--dropout",
        default=0.2,
        type = float,
        help="Proportion of elements to zero in layer where dropout is used."
    )

    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size used for training."
    )
    
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        default=0,
        help="The number of batches to accumulate into one effective batch. Used with distributed training."
    )

    parser.add_argument(
        "--seq_len",
        default=16,
        type=int,
        help="Sequence length used for training."
    )

    parser.add_argument(
        "--overlap",
        default=2,
        type=int,
        help="Sequence overlap used for training."
    )

    parser.add_argument(
        "--embed_dim",
        default=16,
        type=int,
        help="Dimension tokens are embedded into."
    )

    parser.add_argument(
        "--num_heads",
        default=16,
        type=int,
        help="Number of attention heads in each attention layer."
    )

    parser.add_argument(
        "--mask_type",
        default="causal",
        type=str,
        help="How the attention is masked."
    )

    parser.add_argument(
        "--device",
        default=f"cuda:{os.getenv('LOCAL_RANK','0')}" if torch.cuda.is_available() else "cpu",
        type=str,
        help="Device on which training will occur."
    )

    parser.add_argument(
        "--autocast",
        default=torch.cuda.get_device_properties(0).major >= 8,
        help="Whether to use bfloat16 autocast."
    )

    parser.add_argument(
        "--num_workers",
        default=2 if torch.cuda.is_available() else 0,
        type=int,
        help="Number of subprocesses to use for dataloading each GPU."
    )

    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        help="Path to optional config file."
    )

    parser.add_argument(
        "--no-wandb",
        "--no_wandb",
        action="store_true",
        help="If set, wandb logging is disabled."
    )

    parser.add_argument(
        "--eff_batch_per_log",
        default=50,
        help="Number of effective batches per log."
    )

    parser.add_argument(
        "--log_per_val",
        default=10,
        help="Number of logs per validation run. Set to -1 to disable validation."
    )

    parser.add_argument(
        "--val_prompt",
        default="Hello, my name is Jet. J.E.T. stands for",
        help="Prompt from which to generate sample output during training."
    )

    parser.add_argument(
        "--temp",
        default=1,
        help="Temperature to use for sample output generation during training."
    )

    parser.add_argument(
        "--seed",
        default=90,
        help="Seed for random output during training."
    )

    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()

    if args.config_file:
        args = args_from_config_file(args)

    torch.manual_seed(args.seed)

    if 'cuda' in args.device:
        args.device_type = 'cuda'
        args.device_id = [int(os.getenv('LOCAL_RANK','0'))]
        backend = 'nccl'  
    else:
        args.device_type = 'cpu'
        args.device_id = None
        backend = 'gloo'

    setup(backend) 
    args.rank, args.world_size = dist.get_rank(), dist.get_world_size() 
    print(backend,args.rank)
    
    main(args)

    cleanup()
