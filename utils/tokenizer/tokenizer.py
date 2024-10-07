from .datastructures import IndexedList
from .bpe import bpe

import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import wandb
from tqdm.auto import tqdm

import os
import re
import pickle
from typing import Tuple, List
from time import time
from itertools import pairwise, chain


class Tokenizer:
    """
    Class for training and using tokenizers that is (almost) distribution agnositic.
    """
    def __init__(self, merges, rank, world_size, pattern=r'\s?\w+|\s?[^a-zA-Z0-9\s]+|\s+(?=\s)') -> None:
        self.rank = rank
        self.world_size = world_size
        self.merges = merges
        self.pattern = re.compile(pattern)

    @classmethod
    def from_pickled_merges(cls, path, rank=0, world_size=1):
        return cls(cls.load_merges(path), rank, world_size)

    @classmethod
    def from_corpus(cls, dataset, vocab_size, rank, world_size):
        """
        Trains new tokenizer from a dataset. When using distributed training, `corpus` 
        should be the chunk of the dataset that the current rank will handle.
        """
        tokenizer = cls([], rank, world_size)
        tokenizer.corpus = dataset
        tokenizer.__train(vocab_size)

        return tokenizer
    
    def __train(self, vocab_size: int):
        """This method is only supposed to be accessed by the `from_corpus` factory method."""

        print(f"Rank {self.rank} ready to train.")
        dist.barrier()

        blocks_str = self.regex_split('\n'.join(self.corpus['text']))
        blocks_utf8 = [block_str.encode('utf-8') for block_str in blocks_str]
        
        self.merges, self.blocks = bpe(blocks_utf8,vocab_size,self.rank,self.world_size) 


    #----------------------ENCODING-METHODS------------------------

    def encode(self, text: str) -> List[int]:
        indexed_list = IndexedList(text.encode('utf-8'))
        for bp, token in self.merges:
            for node in indexed_list.index.get(bp,[]):
                if node.val != bp[0] or node.next is None or node.next.val != bp[1]:
                    continue  # The index was stale - continue.
                node.next.delete() 
                node.val = token 
                indexed_list.update_index(node)
        return [node.val for node in indexed_list]
    
    def save_encoded_corpus(self,dataset,path):
        """
        Encode and save a corpus (that differs from the tokenizer corpus) 
        to shards.
        """
        dist.broadcast_object_list(self.merges) # Ensure all ranks know correct merges

        if self.rank==0:
            t0 = time()

        with mp.Pool(os.cpu_count()) as pool:
            tokens_iter = pool.imap(self.encode, dataset['text'], chunksize=16)
            self._save_tokens(tokens_iter,path,dataset.shard_size) 

        if self.rank==0:
            print(f"Encoding and saving took {time()-t0:.2f} seconds.")

    def save_encoded_tokenizer_corpus(self, path):
        """
        Save encoded tokenizer corpus as shards.
        Must be called after `__train`.
        """
        self._save_tokens(self.blocks,path,self.corpus.shard_size)
    
    #------------------END-OF-ENCODING-METHODS--------------------

    #----------------------DECODING-METHODS-----------------------
        
    def decode(self, tokens: list) -> str:
        for bytepair,merged_byte in reversed(self.merges):
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
        for _,token in self.merges:
            print(f"Token {token} decodes to {self.decode([token])}")

    #------------------END-OF-DECODING-METHODS--------------------

    #-------------------SAVING/LOADING-METHODS--------------------

    def _save_tokens(self, tokens_iter, path, shard_size):
        """
        Save tokens from an iterable to shards and mmap. 
        `tokens_iter` must be an iterable that yields lists (or numpy arrays) of tokens
        """

        os.makedirs(path, exist_ok=True)
        
        dist.barrier()
        
        dtype = np.uint16
        split = "train"
        shard_index = 0
        # Preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=dtype)
        token_count = 0
        if self.rank == 0:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

        for tokens in tokens_iter:
            while token_count + len(tokens) >= shard_size:
                # Write the current shard and start a new one
                filename = os.path.join(path, f"{self.rank}_{split}_{shard_index:06d}")
                
                # Split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]

                if self.rank == 0:
                    progress_bar.update(remainder)
                
                np.save(filename, all_tokens_np)
                shard_index += 1

                token_count = 0
                tokens = tokens[remainder:]

                if self.rank == 0:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            
            if self.rank == 0:
                progress_bar.update(len(tokens))

        if token_count != 0:
            split = "train" if shard_index == 0 else "val"
            filename = os.path.join(path, f"{self.rank}_{split}_{shard_index:06d}")
            np.save(filename, all_tokens_np[:token_count])

        dist.barrier()

    def save_merges(self, path):
        if self.rank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as file:
                pickle.dump(self.merges, file)

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
    
    #-----------------END-OF-SAVING/LOADING-METHODS------------------

    def regex_split(self, string):
        return re.findall(self.pattern, string)