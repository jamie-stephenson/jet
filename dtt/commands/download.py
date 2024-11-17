from dtt.utils.config import Config

from bpekit import download_dataset

import argparse
import yaml

def download_data(cfg: Config):

    paths = cfg.get_paths()

    with open(paths.dataset_config,'r') as file:
        dataset_cfg = yaml.safe_load(file)

    download_dataset(
        path=paths.dataset,
        **dataset_cfg
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config",
        help="Path to config file."
    )

    args = parser.parse_args()

    cfg = Config.build_from(args.config)

    download_data(cfg)