from .datastructures import DistributedMultiset, IndexedBlocks, IndexedList

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
    
    #--------------------TRAINING-METHODS---------------------

    def __train(self, vocab_size):
        """This method is only supposed to be accessed by the `from_corpus` factory method."""
        self.current_vocab_size = 256
        self.max_vocab_size = vocab_size

        print(f"Rank {self.rank} ready to train.")
        dist.barrier()
        if self.rank == 0:
            t0 = time() 
            t_log = t0  
            print("\nTraining tokenizer...")

        blocks_str = self._regex_split('\n'.join(self.corpus['text']))
        blocks_utf8 = [block_str.encode('utf-8') for block_str in blocks_str]
        self.blocks = IndexedBlocks(blocks_utf8)
        self.bp_counts = DistributedMultiset(
            chain(*[pairwise(block) for block in blocks_utf8]),
            world_size=self.world_size
        )

        if self.rank == 0: 
            print(f"Initialising data structures took {time()-t0}.")

        del blocks_str, blocks_utf8

        while self.current_vocab_size < self.max_vocab_size:
            
            if self.rank == 0:
                len_comms = len(self.bp_counts.to_add)+len(self.bp_counts.to_remove)
            
            pair_to_merge = self.bp_counts.most_common

            if self.rank == 0:
                self.merges.append((pair_to_merge,self.current_vocab_size))

                count = self.bp_counts.l[0].count
                print(f"New bytepair merge {pair_to_merge} -> {self.current_vocab_size}"+ 
                  f" with count {count}.")
                
                if wandb.run is not None:                    
                    wandb.log({
                        "Total Time": time()-t0,
                        "Iter Time": time()-t_log,
                        "Merged bytepair count": count,
                        "Length bp_counts": len(self.bp_counts.l),
                        "Length rank 0 comms (to_add + to_remove)": len_comms                        
                    })
                    t_log = time()

            self._merge_and_update_bp_counts(pair_to_merge)
            self.current_vocab_size += 1

        if self.rank == 0:
            print(f"\nTraining completed in {time()-t0:.2f} seconds.")
    
    def _merge_and_update_bp_counts(self, bytepair):
        for node in self.blocks.index.get(bytepair,[]):
            if node.val != bytepair[0] or node.next is None or node.next.val != bytepair[1]:
                continue  # The index was stale - continue.
            # Say we're merging "bc" to "X" in "abcd", and the node we're visiting now is "b".
            self.bp_counts.remove(bytepair) # Remove "bc".
            if node.next.next is not None:
                self.bp_counts.remove((node.next.val, node.next.next.val))  # Remove "cd".
                self.bp_counts.add((self.current_vocab_size, node.next.next.val))  # Add "Xd".
            if node.prev is not None:
                self.bp_counts.remove((node.prev.val, bytepair[0]))  # Remove "ab".
                self.bp_counts.add((node.prev.val, self.current_vocab_size))  # Add "aX".
            node.next.delete()  # Delete "c", we now have "abd".
            node.val = self.current_vocab_size  # Update "b" to "X", we now have "aXd".
            self.blocks.update_index(node)  # Add "aX" and "Xd" to the index.    
    
    def _regex_split(self, string):
        return re.findall(self.pattern, string)
    
    #------------------END-OF-TRAINING-METHODS---------------------

    #----------------------ENCODING-METHODS------------------------

    def encode(self, text: str) -> List[int]:
        indexed_list = IndexedList(text.encode('utf-8'))
        for bp, token in self.merges:
            for node in indexed_list.index[bp]:
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