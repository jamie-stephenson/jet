from datasets import load_dataset
from src.dist_utils import find_opus_indices
import os

#-----CONFIG------
train_size = 1e5 # We train on 100,000 documents 
#-----------------

def get_dataset(path,rank,world_size):
    assert os.path.exists(path), (
        f"Dataset fineweb-edu does not exist at {path}.\n"
        f"Please download by running `python download.py --corpus fineweb_edu`"
    ) #TODO stop this from duplicating across processes while still ending process group gracefully

    start, end = find_opus_indices(train_size,rank,world_size)

    dataset = load_dataset(
        path='HuggingFaceFW/fineweb-edu',
        name='sample-10BT',
        split=f'train[{start}:{end}]', 
        cache_dir=path
    )

    return '\n'.join(dataset['text'])

def download_dataset(path,nproc):

    load_dataset(
            path='HuggingFaceFW/fineweb-edu',
            name='sample-10BT',
            split='train',
            cache_dir=path,
            num_proc=nproc
    )