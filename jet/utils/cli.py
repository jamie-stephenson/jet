import typer
from jet.utils import Config

from typing import Callable

def common_options(func: Callable):
 
    def wrapper(
        config_file: str = typer.Option(
            None,
            '--config-file',
            '-c',
            help="Path to optional config file."
        ),

        name: str = typer.Option(
            None,
            '--name',
            '-n',
            help="Used to name run folder."
        ),

        dataset: str = typer.Option(
            None,
            '--dataset',
            '-d',
            help="Name of the dataset (must match the name of a valid dataset config yaml file)."
        ),

        epochs: int = typer.Option(
            None,
            '--epochs',
            '-e',
            help="The number of epochs to train for."
        ),

        dropout: float = typer.Option(
            None,
            '--dropout',
            '-do',
            help="Proportion of elements to zero in layer where dropout is used."
        ),

        batch_size: int = typer.Option(
            None,
            '--batch-size',
            '--batch_size',
            '-b',
            help="Batch size used for training."
        ),

        grad_accumulation_steps: int = typer.Option(
            None,
            '--grad-acc-steps',
            '--grad_acc_steps',
            '-gas',
            help="The number of batches to accumulate into one effective batch."
        ),

        n_ctx: int = typer.Option(
            None,
            '--n-ctx',
            '--n_ctx',
            '-ctx',
            help="Max context length used for training."
        ),

        cuda: bool = typer.Option(
            False,
            '--cuda',
            '-cuda',
            is_flag=True,
            help="If set, attempts to set device to cuda:{$LOCAL_RANK} (if available)."
        ),

        autocast: bool = typer.Option(
            False,
            '--autocast',
            '-a',
            is_flag=True,
            help="If set, attempts to use bfloat16 autocast (if available on system)."
        ),

        n_workers: int = typer.Option(
            None,
            '--n-workers',
            '--n_workers',
            '-w',
            help="Number of subprocesses to use for dataloading each GPU."
        ),

        wandb: bool = typer.Option(
            False,
            '--wandb/--no-wandb',
            '-wb/-nwb',
            is_flag=True,
            help="If set, wandb logging is disabled."
        ),

        eff_batch_per_log: int = typer.Option(
            None,
            '--eff-batch-per-log',
            '--eff_batch_per_log',
            '-l',
            help="Number of effective batches per log."
        ),

        log_per_val: int = typer.Option(
            None,
            '--log-per-val',
            '--log_per_val',
            '-v',
            help="Number of logs per validation run."
        ),

        temp: float = typer.Option(
            None,
            '--temp',
            '-t',
            help="Temperature to use for sample output generation during training."
        ),

        seed: int = typer.Option(
            None,
            '--seed',
            '-s',
            help="Seed for random output during training."
        ),

        ctx: typer.Context = typer.Context
    ):
        cfg = Config.build_from(ctx.params.get('config_file'), ctx.params)
        return func(cfg)
    
    return wrapper



def register_command(app: typer.Typer, command_function):
    """
    Registers a command function with a typer app, applying common options and shared logic.
    """
    command_name = command_function.__name__

    @app.command(name=command_name)
    @common_options
    def command(cfg: Config):
        command_function(cfg)
    
    # Set the command's name to the original function's name
    command.__name__ = command_name
