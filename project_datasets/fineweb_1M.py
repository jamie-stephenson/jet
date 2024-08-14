from utils.dist import find_opus_indices

from datasets import load_dataset

import os

#-----CONFIG------
ndocs = int(1e6) # We use 1,000,000 documents e.g. for training a tokenizer 
shard_size = int(5e6)
#-----------------

def get_dataset(path,rank,world_size):
    path_to_data = path.replace('1M','edu')
    assert os.path.exists(path_to_data), (
        f"Dataset fineweb-edu does not exist at {path_to_data}.\n"
        f"Please download by running `python download.py --corpus fineweb_edu`"
    ) #TODO stop this from duplicating across processes while still ending process group gracefully

    start, end = find_opus_indices(ndocs,rank,world_size)

    dataset = load_dataset(
        path='HuggingFaceFW/fineweb-edu',
        name='sample-10BT',
        split=f'train[{start}:{end}]', 
        cache_dir=path_to_data
    )
    
    dataset.shard_size = shard_size

    return dataset

def download_dataset(path,nproc):
    print("This dataset is a subset of the fineweb_edu dataset, please download that instead.")