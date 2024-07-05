import os
import torch
import torch.distributed as dist
import numpy as np
import re
from collections import Counter
import pickle
from typing import Tuple, List
from time import time
import wandb
    
class Tokenizer:
    """
    Class for training and using tokenizers that is (almost) distribution agnositic.
    """
    def __init__(self, merges, rank, world_size) -> None:
        self.rank = rank
        self.world_size = world_size
        self.merges = merges

    @classmethod
    def from_pickled_merges(cls, path, rank=0, world_size=1):
        return cls(cls.load_merges(path), rank, world_size)

    @classmethod
    def from_corpus(cls, corpus, vocab_size, rank, world_size):
        """
        Trains new tokenizer from a dataset. When using distributed training, `corpus` 
        should be the chunk of the dataset ("opus") that the current rank will handle.
        """
        tokenizer = cls({}, rank, world_size)
        tokenizer.corpus = corpus
        tokenizer.__train(vocab_size)

        return tokenizer
    
    #--------------------TRAINING-METHODS---------------------

    def __train(self, vocab_size):
        """This method is only supposed to be accessed by the `from_corpus` factory method."""
        self.current_vocab_size = 256
        self.max_vocab_size = vocab_size

        blocks = self._regex_split(self.corpus)
        self.blocks = [list(block.encode('utf-8')) for block in blocks]

        print(f"Rank {self.rank} ready to train.")
        dist.barrier()
        if self.rank == 0:
            t0 = time() 
            t_log = t0  
            print("\nTraining tokenizer...")

        while self.current_vocab_size < self.max_vocab_size:
            pair_to_merge = self._sync_bp_max()
            if self.rank == 0:
                self.merges[pair_to_merge] = self.current_vocab_size                
                if wandb.run is not None:                    
                    wandb.log({
                        "Total Time": time()-t0,
                        "Iter Time": time()-t_log,
                    })
                    t_log = time()
            self._merge_and_update_bp_counts(pair_to_merge)
            self.current_vocab_size += 1

        if self.rank == 0:
            print(f"\nTraining completed in {time()-t0:.2f} seconds.")

    def _sync_bp_max(self) -> Tuple:

        if self.current_vocab_size%256==0:
            self.bp_counts = {}
            for block in self.blocks:
                self._count_bps(block,self.bp_counts)

        all_bp_counts = [None]*self.world_size
        dist.all_gather_object(object_list=all_bp_counts,obj=self.bp_counts)
        
        unique_bps = set(bp for bp_counts in all_bp_counts for bp in bp_counts.keys())
        total_bp_counts = {bp:sum(bp_counts[bp] for bp_counts in all_bp_counts if bp in bp_counts) 
                            for bp in unique_bps}
        
        if self.current_vocab_size%256==0:
            # Claim: bp_counts[pair_to_merge] > a/(pair_to_merge's overall position in merge ordering) for some a.
            # e.g. if pair_to_merge is the 3rd bp we chose to merge, then bp_counts[pair_to_merge] > a/3 = a/(current_vocab_size - 255)
            # This claim is more of a hope that the initial bp_counts and, by extension, 
            # the counts of the bps that we merge, Follow Zipf's law.
            #
            # We estimate a and then set min_freq to a/"vocab_size n merges from now"
            # This way, if our claim holds, we will track fewer bp_counts but still calculate the correct merges
            # The only downside is that it requires recounting all bp_counts and resetting min_freq every n merges.
            # Setting n = 256 means the recounting overhead is small compared to the time saved from tracking fewer bp_counts.
            # a is estimated as follows:
            # Suppose a bp with count k is the pth bp to be merged. Then p*k is an approximation for a.
            # Repeat for many bps and average to get an estimate for a.
            #
            # So far in practice this results in very safe values for min_freq, while still giving significant speedup for highly distributed workloads.
            x = self.current_vocab_size - 255
            a_estimate = np.mean([(x+i)*k[1] for i,k in enumerate(Counter(total_bp_counts).most_common())])
            self.min_freq = int(a_estimate/(self.current_vocab_size+256)) 
            if self.rank==0:
                print(f"Minimum frequency set to {self.min_freq}.")

        self.bp_counts = {k:v for k,v in self.bp_counts.items() if total_bp_counts[k]>=self.min_freq}
        pair_to_merge = max(total_bp_counts, key=total_bp_counts.get)
        
        if self.rank == 0:
            print(f"New bytepair merge {pair_to_merge} -> {self.current_vocab_size}"+ 
                  f" with count {total_bp_counts[pair_to_merge]}.")
            if wandb.run is not None:
                wandb.log({
                    "Merged bytepair count": total_bp_counts[pair_to_merge],
                    "Length total_bp_counts": len(unique_bps),
                    "Length rank 0 bp_counts": len(self.bp_counts),
                    "Number of tokens": sum(len(block) for block in self.blocks)
                })

        return pair_to_merge
    
    def _merge_and_update_bp_counts(self, bytepair):
        for block_idx in range(len(self.blocks)):
            i = 1
            while i < len(self.blocks[block_idx]):
                if self.blocks[block_idx][i-1:i+1]==list(bytepair):
                    # If X=bc is our bytepair to merge and our string is abcd, then we need 
                    # to decrease the ab and cd counts and increase the aX and Xd counts.
                    # We say ab and cd have `location` = "before" and "after" respectively
                    if i > 1:
                        self._update_bp_counts((self.blocks[block_idx][i-2],self.blocks[block_idx][i-1]),"before")
                    if i < len(self.blocks[block_idx]) - 1: 
                        self._update_bp_counts((self.blocks[block_idx][i],self.blocks[block_idx][i+1]),"after")
                    
                    del self.blocks[block_idx][i]
                    self.blocks[block_idx][i-1] = self.current_vocab_size
                i+=1
        self.bp_counts.pop(bytepair,None)

    def _update_bp_counts(self, bp, location):
        if bp in self.bp_counts:
            self.bp_counts[bp] -= 1
        new_bp = (bp[0],self.current_vocab_size) if location == "before" else (self.current_vocab_size,bp[1])
        self.bp_counts[new_bp] = self.bp_counts.get(new_bp,0) + 1
    
    @staticmethod
    def _count_bps(block, bp_counts=None):
        """bp_counts for a single block"""
        bp_counts = {} if bp_counts is None else bp_counts
        for pair in zip(block, block[1:]):
            bp_counts[pair] = bp_counts.get(pair, 0) + 1
        return bp_counts
    
    @staticmethod
    def _regex_split(string):
        return re.findall(re.compile(r'\s*\w+|\d+|\s*[!"#$%&\'‘’()*+,-./:;<=>?@\[\]^_`{}~]+'), string)
    
    #------------------END-OF-TRAINING-METHODS---------------------

    #----------------------ENCODING-METHODS------------------------

    def encode(self, corpus: str):
        blocks = [list(block.encode('utf-8')) for block in self._regex_split(corpus)]
        for block in blocks:
            while len(block) >= 2:
                bp_counts = self._count_bps(block)
                bp = min(bp_counts, key=lambda p: self.merges.get(p, float("inf")))
                if bp not in self.merges:
                    break
                merged_byte = self.merges[bp]
                block = self._merge_bytepair(block,bp,merged_byte)
        return self.flatten_blocks(blocks)
    
    @staticmethod
    def _merge_bytepair(block: List[int], bytepair: Tuple[int,int],merged_byte: int) -> List[int]:  
        i = 1
        while i < len(block):
            if block[i-1:i+1]==list(bytepair):                
                del block[i]
                block[i-1] = merged_byte
            i+=1
        return block
    
    def save_encoded_corpus(self,corpus,path,encoded_format):
        """
        Janky way of encoding and saving a corpus (that differs from the tokenizer corpus) 
        that supports two different output formats.
        """
        save_mthd = f"_save_to_{encoded_format}"
        merges_list = [self.merges]
        dist.broadcast_object_list(merges_list) # Ensure all ranks know correct merges
        self.merges = merges_list[0]
        tokens = self.encode(corpus)
        getattr(self,save_mthd)(tokens,path) # same as `self.save_mthd(path)` 

    def save_encoded_tokenizer_corpus(self, path, encoded_format):
        """
        Save encoded tokenizer corpus as np.memmap or shards.
        Must be called after `__train`.
        """
        save_mthd = f"_save_to_{encoded_format}"
        tokens = self.flatten_blocks(self.blocks)
        getattr(self,save_mthd)(tokens,path) # same as `self.save_mthd(path)`

    @staticmethod
    def flatten_blocks(blocks: List[List[int]]) -> List[int]:
        tokens = []
        for block in blocks:
            tokens += block
        return tokens
    
    #------------------END-OF-ENCODING-METHODS--------------------

    #----------------------DECODING-METHODS-----------------------
        
    def decode(self, tokens: list) -> str:
        print("WARNING: Execution of `Tokenizer.decode` currently only supported on a single process.")
        for bytepair,merged_byte in reversed(list(self.merges.items())):
            tokens = self._unmerge_byte(tokens,merged_byte,bytepair)
        return bytes(tokens).decode('utf-8',errors='replace')
    
    @staticmethod
    def _unmerge_byte(lst: List[int], merged_byte: int, bytepair: Tuple[int,int]) -> List[int]:
        new_lst = []
        for i in range(len(lst)):
            if lst[i] == merged_byte:
                new_lst += list(bytepair)
            else:
                new_lst.append(lst[i])
        return new_lst
    
    def decoded_tokens(self):
        """lists all merged token chunks as plain text"""
        for token in self.merges.values():
            print(f"Token {token} decodes to {self.decode([token])}")

    #------------------END-OF-DECODING-METHODS--------------------

    #----------------------SAVING-METHODS-------------------------

    def _save_to_shards(self,tokens,path):
        """
        Saves the encoded opera on all processes to shards of `shard_size` tokens.
        The last shard will have <= `shard_size` tokens.

        Given that all shards have a fixed number of tokens, and each rank will end up
        with a different number of tokens, each rank distributes their tokens as follows:
        - Fill up as many shards as possible with the tokens on that rank
        - Partially fill a shard with the remaining tokens and leave a pointer where these tokens get to
        - Pass the partially full shard, the pointer and the shard_idx to the next rank
        This process continues on all ranks and then rank 0 will put all remaining tokens into a final partially full shard.           
        """

        shard_size = int(1e8) #TODO make this flexible

        shard =  torch.zeros((shard_size,), dtype=torch.int16)
        shard_idx = torch.tensor(0)
        pointer = torch.tensor(0) # pointer within a shard

        for rank in range(self.world_size):
            if self.rank == rank:
                if rank == 0:
                    os.makedirs(path, exist_ok=True)
                else:
                    dist.recv(shard,rank-1)
                    dist.recv(pointer,rank-1)
                    dist.recv(shard_idx,rank-1)
                ntokens_to_write = len(tokens)

                while ntokens_to_write > shard_size-pointer:
                    shard[pointer:] = torch.tensor(tokens[-ntokens_to_write:-ntokens_to_write+shard_size-pointer])
                    np.save(os.path.join(path, f"{shard_idx:06d}"),shard.numpy().astype(np.uint16)) # torch.uint16 doesn't exist yet :(
                    ntokens_to_write -= shard_size-pointer
                    shard_idx += 1
                    pointer = torch.tensor(0)

                if ntokens_to_write > 0:    
                    shard[pointer:pointer+ntokens_to_write] = torch.tensor(tokens[-ntokens_to_write:])
                pointer += ntokens_to_write
                dist.send(shard,(rank+1)%self.world_size)
                dist.send(pointer,(rank+1)%self.world_size)
                dist.send(shard_idx,(rank+1)%self.world_size)

        if self.rank == 0:
            dist.recv(shard,self.world_size-1)
            dist.recv(pointer,self.world_size-1)
            dist.recv(shard_idx,self.world_size-1)
            if pointer !=0:
                np.save(os.path.join(path, f"{shard_idx:06d}"),shard[:pointer].numpy().astype(np.uint16))

    def _save_to_mmap(self,tokens,path):
        """
        Saves the encoded opera on all processes to one `np.memmap` in a coordinated fashion.
        """
        path+=".mmap"
        all_opera_lengths = [None]*self.world_size

        dist.all_gather_object(object_list=all_opera_lengths,obj=len(tokens))

        if self.rank == 0:
            print("\nCreating memory mapped array for encoded corpus storage...")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            arr = np.memmap(path,dtype=np.uint16,mode='w+',shape=(sum(all_opera_lengths)))
            arr.flush()
            del arr
            print("Memory mapped array successfully created.")

        dist.barrier()

        # Then write to that array
        # TODO Make this faster, maybe dont use memmap?
        # Here we write one at a time, can we write simultaneously? 
        for rank in range(self.world_size):
            if self.rank == rank:
                start = sum(all_opera_lengths[:rank])
                end = start + all_opera_lengths[rank]
                arr = np.memmap(path,dtype=np.uint16,mode='r+')
                arr[start:end] = np.array(tokens)
                arr.flush()
                del arr
            dist.barrier()

    def save_merges(self, path):
        if self.rank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as file:
                pickle.dump(self.merges, file)

    #-----------------END-OF-SAVING-METHODS-----------------------

    #--------------------LOADING-METHODS--------------------------

    @staticmethod
    def load_merges(path):
        with open(path, 'rb') as file:
            merges = pickle.load(file)
        return merges

    @staticmethod
    def load_corpus(path):
        with open(path, 'r') as file:
            file_contents = file.read()
        return file_contents
    
    #-------------------END-OF-LOADING-METHODS---------------------