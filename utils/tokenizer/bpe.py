from .datastructures import DistributedMultiset, IndexedBlocks

import wandb

from typing import List, Tuple
from itertools import pairwise, chain
from time import time


def bpe(
    blocks: List[List[int]], 
    vocab_size:int,
    rank: int = 0, 
    world_size: int = 1
) -> Tuple[
        List[Tuple[Tuple[int,int],int]],
        IndexedBlocks
    ]: 

    if rank == 0:
        t0 = t_log = time()   
        print("\nRunning BPE algorithm...")

    indexed_blocks = IndexedBlocks(blocks)
    bp_counts = DistributedMultiset(
        chain(*[pairwise(block) for block in blocks]),
        world_size=world_size
    )

    if rank == 0: 
        print(f"Initialising data structures took {time()-t0:.2f} seconds.")

    current_vocab_size = 256
    merges = []

    while current_vocab_size < vocab_size:
        
        if rank == 0:
            len_comms = len(bp_counts.to_add)+len(bp_counts.to_remove)
        
        pair_to_merge = bp_counts.most_common

        if rank == 0:
            merges.append((pair_to_merge,current_vocab_size))

            count = bp_counts.l[0].count
            print(f"New bytepair merge {pair_to_merge} -> {current_vocab_size}"+ 
                f" with count {count}.")
            
            if wandb.run is not None:                    
                wandb.log({
                    "Total Time": time()-t0,
                    "Iter Time": time()-t_log,
                    "Merged bytepair count": count,
                    "Length bp_counts": len(bp_counts.l),
                    "Length rank 0 comms (to_add + to_remove)": len_comms                        
                })
                t_log = time()

        for node in indexed_blocks.index.get(pair_to_merge,[]):

            if node.val != pair_to_merge[0] or node.next is None or node.next.val != pair_to_merge[1]:
                continue  # The index was stale - continue.

            # Say we're merging "bc" to "X" in "abcd", and the node we're visiting now is "b".
            bp_counts.remove(pair_to_merge) # Remove "bc".

            if node.next.next is not None:
                bp_counts.remove((node.next.val, node.next.next.val))  # Remove "cd".
                bp_counts.add((current_vocab_size, node.next.next.val))  # Add "Xd".

            if node.prev is not None:
                bp_counts.remove((node.prev.val, pair_to_merge[0]))  # Remove "ab".
                bp_counts.add((node.prev.val, current_vocab_size))  # Add "aX".

            node.next.delete()  # Delete "c", we now have "abd".
            node.val = current_vocab_size  # Update "b" to "X", we now have "aXd".
            indexed_blocks.update_index(node)  # Add "aX" and "Xd" to the index.    

        current_vocab_size += 1

    if rank == 0:
        print(f"\nBPE algorithm complete in {time()-t0:.2f} seconds.")

    return merges, indexed_blocks