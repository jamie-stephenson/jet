from torch.utils.data import DataLoader, Dataset, Sampler
import torch 
import numpy as np
import os

class ArrayDataset(Dataset):
    """
    Dataset to sample blocks of tokens of size `block_size` from an array of tokens.
    "Array" can be anything indexed using slice notation ('arr[start:end]') and that 
    `torch.tensor` accepts as valid 'data', e.g. list, np.array, np.memmap.
    """
    def __init__(self, data, seq_len):
        self.data = data 
        self.seq_len = seq_len
        
    def __len__(self):
        """Number of tokens"""
        return len(self.data)   
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx:idx+self.seq_len],dtype=torch.long)
        targets = torch.tensor(self.data[idx+1:idx+self.seq_len+1],dtype=torch.long)  
        return sample,targets
    
class ShardedDataset(Dataset):
    """
    Class to sample from a dataset that is split into numpy array shards and stored at `path`.
    Each shard is assumed to have equal `shard_size` except the last shard in the dir, which has size <= `shard_size`.
    """
    def __init__(self, path, seq_len):
        self.shards = [os.path.join(path,shard) for shard in sorted(os.listdir(path))]
        self.shard_size = np.load(self.shards[0], mmap_mode='r').shape[0]
        self.shard_remainder = np.load(self.shards[-1], mmap_mode='r').shape[0]
        self.nshards = len(self.shards) 
        self.seq_len = seq_len
        
    def __len__(self):
        return self.shard_size*(self.nshards-1)+self.shard_remainder  
    
    def __getitem__(self, idx):
        shard_idx = idx//self.shard_size
        start_idx = idx%self.shard_size
        # Check if sample will fit within shard. If not go to next shard. TODO find better alternative to this
        if start_idx+self.seq_len+1>self.shard_size:
            shard_idx += 1
            start_idx = 0
        np_tokens = np.load(self.shards[shard_idx], mmap_mode='r').astype(np.int32)[start_idx:start_idx+self.seq_len+1]
        sample = torch.tensor(np_tokens[:-1],dtype=torch.long)
        targets = torch.tensor(np_tokens[1:],dtype=torch.long)  
        return sample,targets
    
class CustomBatchSampler(Sampler):
    """
    Samples `batch_size` valid indices randomly without replacement.
    The valid indices are multiples of `seq_len-overlap`. 
    """
    def __init__(self, length, rank, world_size, args):
        max_idx = length - args.seq_len - 1
        self.indices = torch.arange(0,max_idx,args.seq_len-args.overlap)
        self.nsamples = len(self.indices) # Total number of samples in dataset
        self.length = self.nsamples//(world_size*args.batch_size) # Length of sampler 
        self.max_idx = self.length*world_size*args.batch_size # Trim nsamples so that every rank gets sampler with same length
        self.rank = rank
        self.world_size = world_size  
        self.batch_size = args.batch_size      

    def __iter__(self):
        shuffle = torch.randperm(self.nsamples)[self.rank:self.max_idx:self.world_size]
        shuffled_indices = self.indices[shuffle].view(-1,self.batch_size)
        for i in range(self.length):
            yield shuffled_indices[i]
    
    def __len__(self):
        return self.length #I have no idea if this is right, changing it seems to do nothing.

def get_dataloader(path,args,mode):
    if args.encoded_format == 'mmap':
        path+='.mmap'
        data = np.memmap(path,dtype=np.uint16,mode='r+')
        dataset = ArrayDataset(data,args.seq_len)
    elif args.encoded_format == 'shards':
        dataset = ShardedDataset(path,args.seq_len)

    sampler = CustomBatchSampler(len(dataset), args.rank, args.world_size, args)
    dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=args.num_workers)
    return dataloader