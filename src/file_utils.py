import yaml
from datetime import datetime

class PathFetcher:
    """
    Generates paths used in `train_model.py` and `train_tokenizer.py` based on the templates in the give path config yaml file.
    """

    def __init__(self, args, config_path: str = './configs/path_config.yaml'):
        with open(config_path, 'r') as yaml_file:
            templates = yaml.safe_load(yaml_file)
        self.templates = templates
        self.args = args
        self.args.time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.generate_paths()

    def generate_paths(self):
        for template_name, template in self.templates.items():
            try:
                path = template.format(**self.args.__dict__)
                setattr(self, template_name, path)
            except KeyError:
                pass

def args_from_config_file(args):
    with open(args.config_file, 'r') as yaml_file:
        yaml_args = yaml.safe_load(yaml_file)
    
    for key, value in yaml_args.items():
        setattr(args, key, value)
    
    return args
    


