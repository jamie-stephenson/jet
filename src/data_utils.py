from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler
import torch 
import numpy as np
import os

class ArrayDataset(Dataset):
    """
    Dataset to sample blocks of tokens of size `seq_len` from an array of tokens.
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
    
class CustomBatchSampler(Sampler):
    """
    Samples `batch_size` valid indices randomly without replacement from an ArrayDataset.
    The valid indices are multiples of `seq_len-overlap`. 
    """
    def __init__(self, split_length, rank, world_size, args):
        max_idx = split_length - args.seq_len - 1
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
    
class ShardedDataset(IterableDataset):
    """
    Dataset to sample blocks of tokens of size `seq_len` from a directory containing numpy shards.
    Each epoch all shards will be shuffled and shared across all available ranks.
    Each rank will then shuffle and then iterate over each of its shards.
    """
    def __init__(self, shard_paths, seq_len, overlap, rank, world_size):
        self.seq_len = seq_len
        self.idx_step = seq_len - overlap
        self.rank = rank
        self.world_size = world_size 
        self.shard_paths = shard_paths
        self.global_seed = torch.initial_seed() 
        self.worker_seed = torch.initial_seed()
        
    def _load_shard(self, shard_path):
        """Generator to yield sequences of tokens from a given shard"""
        data = np.load(shard_path, allow_pickle=True).astype(np.int32)
        torch_data = torch.from_numpy(data)
        max_idx = len(torch_data) - self.seq_len - 1
        ids = torch.arange(0,max_idx,self.idx_step)
        print(torch.randperm(10))
        shuffled_ids = ids[torch.randperm(len(ids))] # This shuffle is different on all ranks 
        for idx in shuffled_ids:
            sample = torch_data[idx:idx+self.seq_len]
            targets = torch_data[idx+1:idx+self.seq_len+1]
            yield sample, targets

    def __iter__(self):

        torch.manual_seed(self.global_seed)
        self.shard_paths = [self.shard_paths[i] for i in torch.randperm(len(self.shard_paths))]  # This shuffle is the same on all ranks (as desired)
        torch.manual_seed(self.worker_seed)

        shard_paths = self.shard_paths[self.rank::self.world_size]
        for shard_path in shard_paths:
            yield from self._load_shard(shard_path)

def sharded_init_fn(worker_id): 
    dataset = torch.utils.data.get_worker_info().dataset
    dataset.worker_seed += worker_id
    pass

def get_dataloader(path,split,args):     
    if args.encoded_format == 'mmap':
        dtype=np.uint16 
        path = os.path.join(path,f"{split}.mmap")
        data = np.memmap(path,dtype,mode='r+')
        dataset = ArrayDataset(data,args.seq_len)
        sampler = CustomBatchSampler(len(dataset), args.rank, args.world_size, args)
        dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=args.num_workers)
    elif args.encoded_format == 'shards':
        paths = [os.path.join(path,shard) for shard in sorted(os.listdir(path)) if split in shard]
        dataset = ShardedDataset(paths,args.seq_len,args.overlap,args.rank,args.world_size)
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=sharded_init_fn)
    
    return dataloader

