from src.dist_utils import split_txt
from datasets import Dataset

#-----CONFIG------
shard_size = int(1e5)
#-----------------

def get_dataset(path,rank,world_size):
    path+='test.txt'
    dataset = Dataset.from_dict({'text':[split_txt(path,rank,world_size).decode(errors='replace')]})
    dataset.shard_size = shard_size
    return dataset