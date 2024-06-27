from datasets import load_dataset
from src.dist_utils import find_opus_indices
import os

#-----CONFIG------
val_size = 1e4   # number of documents in the val split
train_size = 1e5 # We train on 100,000 documents 
#-----------------

def get_dataset(path,split,rank,world_size):
    assert os.path.exists(path), (
        f"Dataset fineweb-edu does not exist at {path}.\n"
        f"Please download by running `python download.py --corpus fineweb_edu`"
    ) #TODO stop this from duplicating across processes while still ending process group gracefully

    if split == 'train':
        start, end = find_opus_indices(100000,rank,world_size) 
        start += val_size
        end += val_size
    else:
        start, end = find_opus_indices(val_size,rank,world_size)

    dataset = load_dataset(
        path='HuggingFaceFW/fineweb-edu',
        name='sample-10BT',
        split=f'train[{start}:{end}]', # I think this dataset only has a train split so we make our own val split
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