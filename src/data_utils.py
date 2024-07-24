from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler
import torch 
import numpy as np
import os

class ArrayDataset(Dataset):
    """
    Dataset to sample sequences of tokens of size `seq_len` from an array of tokens.
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
    def __init__(self, dataset_length, rank, world_size, args):
        idx_upper_bound = dataset_length - args.seq_len
        self.indices = torch.arange(0,idx_upper_bound,args.seq_len-args.overlap)
        self.nsamples = len(self.indices) # Total number of samples in dataset
        self.length = self.nsamples//(world_size*args.batch_size) # Length of sampler 
        self.nsamples_per_epoch = self.length*world_size*args.batch_size # Trim nsamples so that each epoch every rank gets sampler with same length
        self.rank = rank
        self.world_size = world_size  
        self.batch_size = args.batch_size      

    def __iter__(self):
        shuffle = torch.randperm(self.nsamples)[self.rank:self.nsamples_per_epoch:self.world_size]
        shuffled_indices = self.indices[shuffle].view(-1,self.batch_size)
        for i in range(self.length):
            yield shuffled_indices[i]
    
    def __len__(self):
        return self.length #I have no idea if this is right, changing it seems to do nothing.
    
class ShardedDataset(IterableDataset):
    """
    Dataset to sample sequences of tokens of size `seq_len` from a directory containing numpy shards.
    Each epoch all* shards will be shuffled and shared across all available ranks.
    Each rank will then shuffle and then iterate over each of its shards.

    * (not quite all: see `nshards_per_epoch` below)
    """
    def __init__(self, shard_paths, seq_len, overlap, rank, world_size):
        self.seq_len = seq_len
        self.idx_step = seq_len - overlap

        self.rank = rank
        self.world_size = world_size 

        self.shard_paths = shard_paths
        self.nshards = len(shard_paths)

        # To minimise differnce in dataloader length on each rank, each
        # epoch all ranks will form dataset from same number of shards.
        # This means the total number of shards we consider each epoch
        # needs to be a multiple of world_size: 
        self.nshards_per_epoch = world_size * (self.nshards // world_size) 

        # The following attributes will be set by the worker_init_fn (iff num_workers > 0):
        self.worker_id = 0
        self.num_workers = 1
        self.rank_rng = None

    def __iter__(self):
        # This shuffle is the same on all ranks:
        shuffled_path_ids = torch.randperm(self.nshards)[:self.nshards_per_epoch] 
        shard_paths = [self.shard_paths[i] for i in shuffled_path_ids]  

        shard_paths = shard_paths[self.rank::self.world_size]
        for shard_path in shard_paths:
            yield from self._load_shard(shard_path)

    def _load_shard(self, shard_path):
        """Generator to yield sequences of tokens from a given shard"""
        data = np.load(shard_path, allow_pickle=True).astype(np.int64)
        torch_data = torch.from_numpy(data)
        max_idx = len(torch_data) - self.seq_len - 1
        ids = torch.arange(0,max_idx,self.idx_step)
        # This shuffle is different on all ranks but the same for all workers on a given rank:
        shuffled_ids = ids[torch.randperm(len(ids),generator=self.rank_rng)[self.worker_id::self.num_workers]] 
        for idx in shuffled_ids:
            sample = torch_data[idx:idx+self.seq_len]
            targets = torch_data[idx+1:idx+self.seq_len+1]
            yield sample, targets

    def __len__(self):
        """Rough estimate of dataset length (currently only used for input to OneCycleLR)"""
        shard_size = np.load(self.shard_paths[0],mmap_mode='r').size
        length = (self.nshards_per_epoch//self.world_size)*(shard_size/self.idx_step)
        return int(length)

def seed_worker(worker_id):
    """
    This is a worker_init_fn that establishes two sources of rng for use in dataloader workers:
    1) The global torch source of rng. This means that all workers on all ranks can perform the same shuffles when needed.
    2) A local source of rng that is the same for all workers on a given rank but different for different ranks.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset =  worker_info.dataset
    dataset.worker_id = worker_id
    dataset.num_workers = worker_info.num_workers

    global_seed = torch.initial_seed()-worker_id # Undo torch's automatic seeding
    torch.manual_seed(global_seed)

    dataset.rank_rng = torch.Generator() # Create second source of rng that is rankwise unique
    dataset.rank_rng.manual_seed(global_seed+dataset.rank)

def get_dataloader(path,split,args):     
    if args.encoded_format == 'mmap':
        dtype=np.uint16 
        path = os.path.join(path,f"{split}.mmap")
        data = np.memmap(path,dtype,mode='r+')
        dataset = ArrayDataset(data,args.seq_len)
        sampler = CustomBatchSampler(len(dataset), args.rank, args.world_size, args)
        dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=args.num_workers)
    elif args.encoded_format == 'shards':
        paths = [os.path.join(path,shard) for shard in sorted(os.listdir(path)) if split in shard and 'mmap' not in shard]
        dataset = ShardedDataset(paths,args.seq_len,args.overlap,args.rank,args.world_size)
        dataloader = DataLoader(
                        dataset=dataset, 
                        batch_size=args.batch_size, 
                        num_workers=args.num_workers,
                        worker_init_fn=seed_worker
                    )
    
    return dataloader

