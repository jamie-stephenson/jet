from jet.utils.config import Config

from bpekit import encode_dataset, train_tokenizer
import yaml
import argparse

def tokenize_data(cfg: Config):

    paths = cfg.get_paths()

    with open(paths.dataset_config,'r') as file:
        dataset_cfg = yaml.safe_load(file)

    train_tokenizer(
        path=paths.dataset,
        vocab_size=cfg.vocab_size,
        merges_path=paths.tokenizer,
        **dataset_cfg
    )

    encode_dataset(
        path=paths.dataset,
        merges_path=paths.tokenizer,
        tokens_path=paths.tokens,
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

    tokenize_data(cfg)
