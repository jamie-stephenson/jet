from torch.utils.data import DataLoader, Dataset, Sampler
import torch 
import numpy as np

class MmapDataset(Dataset):
    """
        Dataset from which to sample blocks of tokens of size `block_size`.
    """
    def __init__(self, data, seq_len):
        self.data = data 
        self.seq_len = seq_len
        
    def __len__(self):
        """Number of tokens"""
        return len(self.data)   
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx:idx+self.seq_len],dtype=int)
        targets = torch.tensor(self.data[idx+1:idx+self.seq_len+1],dtype=int)  
        return sample,targets
    
class ShardedDataset(Dataset):
    def __init__(self, path, seq_len):
        self.path = path 
        self.seq_len = seq_len
        
    def __len__(self):
        """Number of shards"""
        return len(self.data)   
    
    def __getitem__(self, idx):
        np.load()
        sample = torch.tensor(self.data[idx:idx+self.seq_len],dtype=int)
        targets = torch.tensor(self.data[idx+1:idx+self.seq_len+1],dtype=int)  
        return sample,targets
    
class RandomBlockBatchSampler(Sampler):
    """
        Sampler to give `batch_size` random indices to a `DataLoader`.
        The loader then uses these to sample random blocks from `BlockDataset`.
        You can only iterate over the loader `max_iter` times.
    """
    def __init__(self, dataset, args):
        self.max_idx = len(dataset) - args.seq_len - 1
        self.batch_size = args.batch_size
        self.max_iter = args.max_iter

    def __iter__(self):
        for _ in range(self.max_iter):
            yield torch.randint(0, self.max_idx,(self.batch_size,))
    
    def __len__(self):
        return self.max_iter #I have no idea if this is right, changing it seems to do nothing.

def get_dataloader(path,args,mode):
    # This test train split is a bit basic, and also might cause problems if np.memmap needs to be recreated every iteration
    if args.encoded_format == 'mmap':
        path+='mmap'
        data = np.memmap(path,dtype=np.uint16,mode='r+')
    elif args.encoded_format == 'shards':

    custom_dataset = BlockDataset(data, args.seq_len)
    custom_sampler = RandomBlockBatchSampler(custom_dataset, args)
    dataloader = DataLoader(dataset=custom_dataset, batch_sampler=custom_sampler)
    return dataloader