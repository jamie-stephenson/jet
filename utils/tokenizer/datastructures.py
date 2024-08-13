import torch.distributed as dist

from collections import Counter, defaultdict
from typing import Any
import heapq

#=========EFFICIENT DATA STRUCTURES========
# Adapted from https://github.com/yanivle/fast_minbpe

class IndexedList:
    __slots__ = 'index', 'start'
    class Node:
        __slots__ = 'val', 'prev', 'next'
        def __init__(self, val, prev, next):
            self.val, self.prev, self.next = val, prev, next

        def delete(self):
            if self.prev is not None:
                self.prev.next = self.next
            if self.next is not None:
                self.next.prev = self.prev
            self.next = self.prev = None

    def __init__(self, l):
        self.index = {}
        l = iter(l)
        a = next(l)
        self.start = prev_node = IndexedList.Node(a, None, None)
        for b in l:
            prev_node.next = node = IndexedList.Node(b, prev_node, None)
            self.add_to_index((a, b), prev_node)
            a, prev_node = b, node

    def __iter__(self):
        node = self.start
        while node is not None:
            yield node
            node = node.next

    def update_index(self, node):  # Update index before/after node.
        if node.prev is not None:
            self.add_to_index((node.prev.val, node.val), node.prev)
        if node.next is not None:
            self.add_to_index((node.val, node.next.val), node)

    def add_to_index(self, pair, node):
        self.index.setdefault(pair, []).append(node)

class IndexedBlocks(IndexedList):
    """
    Like `IndexedList`, but maintains index over list of disjointed `IndexedList`s
    """
    def __init__(self, blocks: list):
        self._blocks = []
        self.index = {}
        for block in blocks:
            next_block = IndexedList(block)
            for pair, nodes in next_block.index.items():
                self.index.setdefault(pair, []).extend(nodes)
            del next_block.index
            self._blocks.append(next_block)

    def __iter__(self):
        for block in self._blocks:
            node_values = []
            for node in block:
                node_values.append(node.val)
            yield node_values

class Node:
    __slots__ = 'count', 'val', 'pos'

    def __init__(self, count: int, val: Any, pos: int):
        self.count = count
        self.val = val
        self.pos = pos

    @property
    def key(self):  # key for comparisons
        return self.count
        # Breaking ties explicitly, forcing more heap update, results in a significant slowdown:
        # return (self.count, self.val, self.pos)

    def __lt__(self, other):
        return self.key < other.key

class Multiset:
    def __init__(self, init=None, node_type=Node):
        self.l = []  # A heap of nodes.
        self.d = {}  # A map from value to its node.
        self.node_type = node_type
        self.to_add = defaultdict(int)
        self.to_remove = defaultdict(int)
        self.to_add.update(Counter(init))

    def add(self, item, count=1):
        self.to_add[item] += count

    def remove(self, item, count=1):
        self.to_remove[item] += count

    def _add(self, item, count=1):
        node = self.d.get(item)
        if node is None:
            node = self.d[item] = self.node_type(0, item, len(self.l))
            self.l.append(node)
        node.count += count
        self._item_increased(node.pos)

    def _remove(self, item, count=1):
        node = self.d[item]
        node.count -= count
        self._item_decreased(node.pos)
        # We could actually remove items with 0-count from the list, but
        # since for some scores its helpful to have items with arbitrary
        # counts, including negative, we're never actually removing items.

    def _commit(self):
        for pair, count in self.to_add.items():
            self._add(pair, count)
        for pair, count in self.to_remove.items():
            self._remove(pair, count)
        self.to_add.clear()
        self.to_remove.clear()

    def count(self, item):
        self._commit()
        if item not in self.d: return 0
        return self.d[item].count

    @property
    def most_common(self):
        self._commit()
        return self.l[0].val

    def top_k(self, k: int) -> list[tuple[Any, int]]:
        self._commit()
        totup = lambda n: (-n.key, n.val, n.count, n.pos)
        res, heap = [], [totup(self.l[0])]
        for _ in range(k):
            if not heap: break
            _key, val, count, pos = heapq.heappop(heap)
            res.append((val, count))
            for child_pos in [pos * 2 + 1, pos * 2 + 2]:
                if child_pos < len(self.l):
                    heapq.heappush(heap, totup(self.l[child_pos]))
        return res

    def __bool__(self):
        self._commit()
        return bool(self.l)

    # The below functions maintain the heap property:

    def _item_increased(self, pos):
        # Adapted from heapq._siftdown_max.
        node = self.l[pos]
        while pos > 0:
            parentpos = (pos - 1) >> 1
            parent = self.l[parentpos]
            if parent < node:
                self.l[pos] = parent
                parent.pos = pos
                pos = parentpos
                continue
            break
        self.l[pos] = node
        node.pos = pos

    def _item_decreased(self, pos):
        # Adapted from heapq._siftup_max.
        endpos = len(self.l)
        node = self.l[pos]
        childpos = 2 * pos + 1  # leftmost child position
        while childpos < endpos:
            # Set childpos to index of larger child.
            rightpos = childpos + 1
            if rightpos < endpos and not self.l[rightpos] < self.l[childpos]:
                childpos = rightpos
            childnode = self.l[childpos]
            if node < childnode:  # Move the larger child up.
                self.l[pos] = childnode
                childnode.pos = pos
                pos = childpos
                childpos = 2 * pos + 1
            else:
                break
        self.l[pos] = node
        node.pos = pos

class DistributedMultiset(Multiset):
    def __init__(self, init=None, node_type=Node, world_size=1):
        self.world_size = world_size
        self.l = []  # A heap of nodes.
        self.d = {}  # A map from value to its node.
        self.node_type = node_type
        self.to_add = Counter(init)
        self.to_remove = Counter()

    def _commit(self): 
        self.to_add = counter_reduce(self.to_add,self.world_size)
        self.to_remove = counter_reduce(self.to_remove,self.world_size)
        super()._commit()

def counter_reduce(c, world_size):
    counter_list = [None]*world_size
    dist.all_gather_object(object_list=counter_list,obj=c)
    counter_reduced = sum(counter_list,Counter())
    return counter_reduced
