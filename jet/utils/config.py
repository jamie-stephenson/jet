from torch import cuda

from dataclasses import dataclass, field
from pathlib import Path
import os
import yaml
from datetime import datetime
import argparse
from typing import Dict
import warnings

@dataclass
class ParamAttribute:
    """
    Dataclass for attributes with params
    e.g. optimizer
    """
    name: str
    params: dict

def __post_init__(self):
    self.time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


DEFAULT_LR_SCHEDULER = ParamAttribute(
    name='onecycle',
    params={
        'lr_max': 0.01,
        'pct_start': 0.1
    }
)

DEFAULT_OPTIMIZER = ParamAttribute(
    name='adamw',
    params={
        'weight_decay': 0.0001,
        'fused': True
    }
)

@dataclass
class Config:

    """
    Dataclass that holds configuration data.
    """

    # -----MODEL-----
    d_mlp: int = 128
    d_model: int = 32
    mask_type: str = 'causal'
    n_blocks: int = 12
    n_ctx: int = 128
    n_heads: int = 12
    seed: int = 90

    # -----TOKENIZER-----
    vocab_size: int = 16384

    # -----TRAINING-----
    batch_size: int = 128
    epochs: int = 1
    grad_accumulation_steps: int = 1
    dropout: float = 0.2

    # -----DATASET-----
    dataset: str = 'fineweb-edu'
    overlap: int = 8
    n_workers: int = 2

    # -----RESOURCES-----
    cuda: bool = True
    autocast: bool = True

    # -----OPTIMIZER-----
    optimizer: ParamAttribute = field(default_factory=lambda :DEFAULT_OPTIMIZER)

    # -----LR Schedule-----
    lr_schedule: ParamAttribute = field(default_factory=lambda :DEFAULT_LR_SCHEDULER)

    # -----VALIDATION-----
    log_per_val: int | None = None
    temp: float = 1
    val_prompt: str = "Hello my name is JET. JET stands for"

    # -----LOGGING-----
    name: str = 'jet' 
    wandb: bool = False
    eff_batch_per_log: int = 100

    # -----PATH TEMPLATES-----
    templates: str = 'configs/path_templates.yaml'

    @classmethod
    def build_from(
        cls, 
        file: Path | None = None, 
        args: argparse.Namespace | dict | None = None
    ):
        """
        Factory method to create a Config instance from at least one of: 
        - A YAML file 
        - `argparse` args.

        Command-line args take precedence over file settings.
        """

        if not file and not args:
            warnings.warn(
                "No config file or command line arguments given. " 
                "Default config will be used."
            )

        # Start with a default instance
        config = cls()
        
        # Load from file if provided
        if file:
            with open(file, 'r') as yaml_file:
                data = yaml.safe_load(yaml_file)

            for key, value in data.items():

                # Handle `ParamAttribute`s
                if isinstance(value,dict):
                    try:
                        setattr(config, key, ParamAttribute(**value))
                    except (ValueError, TypeError) as e:
                        raise type(e)(
                            f"Unable to create attribute \"{key}\" with the following structure: {value}\n"
                            "NOTE: Attributes with nested strucutre are expected to take the form {\"name\":___, \"params\":{___}}."
                        ) from e
                else:
                    setattr(config, key, value)

        # Override with command-line args if provided
        if args:
            args_dict = args if isinstance(args,dict) else vars(args)
            args_dict.setdefault('time', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            for key, value in args_dict.items():
                if value:
                    setattr(config, key, value)
        
        # Handle attributes that depend on system state (e.g. available resources)

        # cuda
        if config.cuda:
            config.cuda = cuda.is_available()
        
        config.device_id = [int(os.getenv('LOCAL_RANK','0'))] if config.cuda else None
        config.device = f"cuda:{config.device_id[0]}" if config.cuda else "cpu"
        config.device_type = 'cuda' if config.cuda else 'cpu'

        # autocast
        if config.autocast:
            config.autocast = config.cuda and cuda.get_device_properties(0).major >= 8

        # optimizer
        if config.optimizer.params.get('fused'): 
            config.optimizer.params['fused'] = config.cuda

        return config

    class Paths:
        """
        Inner class to represent paths with dynamically set attributes.
        """
        def __init__(self, **paths: Dict[str,Path]):
            for name, path in paths.items():
                setattr(self, name, path)

    def get_paths(self, templates_path: Path | None = None) -> Paths:

        if not self.templates and not templates_path:
            raise ValueError("Path templates must be given to generate paths.")
        
        path = templates_path if templates_path else self.templates

        with open(path,'r') as yaml_file:
            templates = yaml.safe_load(yaml_file)

        paths_dict = {}
        for template_name, template in templates.items():
            try:
                path = template.format(**self.__dict__)
                paths_dict[template_name] = Path(path)
            except KeyError as e:
                print(f"Warning: Missing key {e} for template '{template_name}'")
                pass

        return self.Paths(**paths_dict)
