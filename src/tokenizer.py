import os
import torch
import torch.distributed as dist
import numpy as np
import re
from collections import Counter
import pickle
from typing import Tuple, List
from time import time
    
class Tokenizer:
    """
    Class for training and using tokenizers in an (almost) distribution agnositic manner.
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
        should be the chunk of the dataset that the current rank will handle.
        """
        tokenizer = cls({}, rank, world_size)
        tokenizer.corpus = corpus
        tokenizer.__train(vocab_size)

        return tokenizer

    def save_encoded_corpus(self,corpus,path,encoded_format):
        """
        Janky way of encoding and saving a corpus (that differs from the tokenizer corpus) 
        that supports two different output formats.
        """
        save_mthd = f"_save_to_{encoded_format}"
        tokens = self.encode(corpus)
        getattr(self,save_mthd)(tokens,path) # same as `self.save_mthd(path)` 
    
    def encode(self, corpus):
        dist.broadcast_object_list([self.merges]) # Ensure all ranks know correct merges
        print(f"Rank {self.rank} ready to encode.")
        dist.barrier()
        for bytepair,merged_byte in self.merges.items():
            corpus = self._merge_bytepair(corpus,bytepair,merged_byte)
        return corpus
        
    def decode(self, tokens: list) -> str:
        print("WARNING: Execution of `Tokenizer.decode` currently only supported on a single process.")
        for bytepair,merged_byte in reversed(list(self.merges.items())):
            tokens = self._unmerge_byte(tokens,merged_byte,bytepair)
        return bytes(tokens).decode('utf-8',errors='replace')

    def __train(self, vocab_size):
        """This method is only supposed to be accessed by the `from_corpus` factory method."""
        self.current_vocab_size = 256
        self.max_vocab_size = vocab_size

        blocks = self._regex_split(self.corpus)
        self.blocks = [list(block.encode('utf-8')) for block in blocks]
        self.bp_counts = self._count_bytepairs(self.blocks)

        print(f"Rank {self.rank} ready to train.")
        dist.barrier()
        if self.rank == 0:
            t0 = time()   
            print("\nTraining tokenizer...")
        while self.current_vocab_size < self.max_vocab_size:
            pair_to_merge = self._sync_bp_counts()
            if self.rank == 0:
                self.merges[pair_to_merge] = self.current_vocab_size
            self._merge_and_update_bp_counts(pair_to_merge)
            self.current_vocab_size += 1

        if self.rank == 0:
            print(f"\nTraining completed in {time()-t0:.2f} seconds.")

    def _sync_bp_counts(self) -> Tuple:
        all_bp_counts = [None]*self.world_size
        dist.all_gather_object(object_list=all_bp_counts,obj=self.bp_counts)
        
        unique_bps = set(bp for bp_counts in all_bp_counts for bp in bp_counts.keys())
        total_bp_counts = {bp:sum(bp_counts[bp] for bp_counts in all_bp_counts if bp in bp_counts) 
                            for bp in unique_bps}
        pair_to_merge = max(total_bp_counts, key=total_bp_counts.get)
        
        if self.rank == 0:
            print(f"New bytepair merge {pair_to_merge} -> {self.current_vocab_size}"+ 
                  f" with count {total_bp_counts[pair_to_merge]}.")

        return pair_to_merge
    
    def _merge_and_update_bp_counts(self, bytepair):
        for block_idx in range(len(self.blocks)):
            i = 1
            while i < len(self.blocks[block_idx]):
                if self.blocks[block_idx][i-1:i+1]==list(bytepair):
                    #if X=bc is our bytepair to merge and our string is abcd, then we need 
                    #to decrease the ab and cd counts and increase the aX and Xd counts.
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
        if self.bp_counts[bp] == 1:
            self.bp_counts.pop(bp,None)
        else:
            self.bp_counts[bp] -= 1
        new_bp = (bp[0],self.current_vocab_size) if location == "before" else (self.current_vocab_size,bp[1])
        self.bp_counts[new_bp] = self.bp_counts.get(new_bp,0) + 1

    @staticmethod
    def _count_bytepairs(blocks: List[list]) -> dict:
        """
            At the moment this is only used once, at the start of training.
            However, I have made this a staticmethod incase I end up using 
            it more frequently e.g. instead of updating bp_counts in place
            I could just recount bps every loop, maybe this is quicker?
        """
        bytepairs = []
        for block in blocks:
            bytepairs += [(block[i],block[i+1]) for i in range(len(block)-1)]
        return dict(Counter(bytepairs))
    
    @staticmethod
    def _merge_bytepair(lst: List[int],bytepair: Tuple[int,int],merged_byte: int) -> List[int]:
        new_lst = []
        for i in range(len(lst)):
            if i > 0 and lst[i-1:i+1]==list(bytepair) and new_lst[-1]!=merged_byte:
                new_lst[-1] = merged_byte
            else:
                new_lst.append(lst[i]) 
        return new_lst
    
    @staticmethod
    def _unmerge_byte(lst: List[int], merged_byte: int, bytepair: Tuple[int,int]) -> List[int]:
        new_lst = []
        for i in range(len(lst)):
            if lst[i] == merged_byte:
                new_lst += list(bytepair)
            else:
                new_lst.append(lst[i])
        return new_lst
    
    @staticmethod
    def load_corpus(path):
        with open(path, 'r') as file:
            file_contents = file.read()
        return file_contents

    def save_encoded_tokenizer_corpus(self, path, encoded_format):
        """Save encode tokenizer corpus as np.memmap or shards"""
        save_mthd = f"_save_to_{encoded_format}"
        tokens = self.get_encoded_tokenizer_corpus()
        getattr(self,save_mthd)(tokens,path) # same as `self.save_mthd(path)`

    def get_encoded_tokenizer_corpus(self):
        #must be called **after** `self.train()` to return fully encoded tokenizer corpus.
        encoded_corpus = []
        for block in self.blocks:
            encoded_corpus += block
        return encoded_corpus

    def _save_to_shards(self,tokens,path):
        """
        Saves the encoded opera on all processes to shards of `shard_size` tokens.
        The last shard will have <= `shard_size` tokens.

        Given that all shards have a fixed number of tokens, and each rank will end up
        with a different number of tokens, each rank distributed their tokens as follows:
        - Fill up as many shards as possible with the tokens on that rank
        - Partially fill a shard with the remaining tokens and leave a pointer where these tokens get to
        - Pass the partially full shard, the pointer and the shard_idx to the next rank
          Here we have encoded that the pointer is at position 5 in shard 8
        This process continues on all ranks and then rank 0 will put all remaining tokens into a final partially full shard.           
        """

        shard_size = int(1e8) #TODO make this an argument 

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

    @staticmethod
    def load_merges(path):
        with open(path, 'rb') as file:
            merges = pickle.load(file)
        return merges
    
    def save_merges(self, path):
        if self.rank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as file:
                pickle.dump(self.merges, file)

    @staticmethod
    def _regex_split(string):
        return re.findall(re.compile(r'\s*\w+|\d+|\s*[!"#$%&\'‘’()*+,-./:;<=>?@\[\]^_`{}~]+'), string)
    
    def decoded_tokens(self):
        """lists all merged token chunks as plain text"""
        for token in self.merges.values():
            print(f"Token {token} decodes to {self.decode([token])}")