from torch import cuda

from dataclasses import dataclass
from pathlib import Path
import os
import yaml
from datetime import datetime
import argparse
from typing import Dict

@dataclass
class Config:
    """
    Data class that holds configuration data.
    """

    @dataclass
    class ParamAttribute:
        """
        Inner dataclass for attributes with params
        e.g. optimizer
        """
        name: str
        params: dict

    def __post_init__(self):
        self.time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    @classmethod
    def build_from(cls, file: Path | None = None, args: argparse.Namespace | None = None):
        """
        Factory method to create a Config instance from at least one of: 
        - A YAML file 
        - `argparse` args.

        Command-line args take precedence over file settings.
        """

        assert file or args, "At least one source must be provided from which to build config."

        # Start with an empty instance
        config = cls()
        
        # Load from file if provided
        if file:
            with open(file, 'r') as yaml_file:
                data = yaml.safe_load(yaml_file)

                for key, value in data.items():

                    # Handle `ParamAttribute`s
                    if isinstance(value,dict):
                        try:
                            setattr(config, key, cls.ParamAttribute(**value))
                        except (ValueError, TypeError) as e:
                            raise type(e)(
                                f"Unable to create attribute \"{key}\" with the following structure: {value}\n"
                                "NOTE: Attributes with nested strucutre are expected to take the form {\"name\":___, \"params\":{___}}."
                            ) from e
                    else:
                        setattr(config, key, value)

        # Override with command-line args if provided
        if args:
            args_dict = vars(args)
            args_dict.setdefault('time', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            for key, value in args_dict.items():
                if value:
                    setattr(config, key, value)
        
        # Handle attributes that depend on system state (e.g. available resources)

        # cuda
        if config.cuda:
            config.cuda = cuda.is_available()
            config.device_id = [int(os.getenv('LOCAL_RANK','0'))] if config.cuda else None
            config.device = f"cuda:{config.device_id[0]}" if config.cuda else "cpu",
            config.device_type = 'cuda' if config.cuda else 'cpu'
            config.backend = 'nccl' if config.cuda else 'gloo'

        # autocast
        if config.autocast:
            config.autocast = config.cuda and cuda.get_device_properties(0).major >= 8

        # optmizer
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
