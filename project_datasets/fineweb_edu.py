from utils.dist import find_opus_indices

from datasets import load_dataset

import os

#-----CONFIG------
ndocs = 9672101 # All documents 
shard_size = int(1e8)
#-----------------

def get_dataset(path,rank,world_size):
    assert os.path.exists(path), (
        f"Dataset fineweb-edu does not exist at {path}.\n"
        f"Please download by running `python download.py --corpus fineweb_edu`"
    ) #TODO stop this from duplicating across processes while still ending process group gracefully

    start, end = find_opus_indices(ndocs,rank,world_size) # Slightly quicker than using `split_dataset_by_node`

    dataset = load_dataset(
        path='HuggingFaceFW/fineweb-edu',
        name='sample-10BT',
        split=f'train[{start}:{end}]', 
        cache_dir=path
    )

    dataset.shard_size = shard_size

    return dataset

def download_dataset(path,nproc):

    load_dataset(
        path='HuggingFaceFW/fineweb-edu',
        name='sample-10BT',
        split='train',
        cache_dir=path,
        num_proc=nproc
    )