import random


import numpy as np
from .sum_tree import SumTree


class ReplaySamplerPriority:
    '''
    Samples memories with priority given by loss
    '''

    MAX_DUPLICATE_RETRIES = 100


    def __init__(self, replay_memory):
        self.replay_memory = replay_memory
        self.sum_tree = SumTree(self.replay_memory.max_size, dtype='uint32')
        self.add_losses()


    def append(self, state, action, reward, next_state, cont, loss):
        self.sum_tree.add(loss, self.replay_memory.cur_idx)
        self.replay_memory.append(state, action, reward, next_state, cont, loss)


    def sample_memories(self, target, batch_size=32, priorities=None, tree_idxes=None, skip_duplicates=True):
        dup_indexes = {}
        dup_count = 0
        size = self.sum_tree.total / batch_size

        for i in range(batch_size):
            s = random.random() * size + i * size

            t_idx, d_idx, score, memory_idx = self.sum_tree.get_with_info(s)

            self.replay_memory.copy(memory_idx, target, i)

            if priorities is not None:
                priorities[i] = score

            if tree_idxes is not None:
                tree_idxes[i] = t_idx


    def update_sum_tree(self, tree_idxes, losses):
        for t_idx, loss in zip(tree_idxes, losses):
            self.sum_tree.update_score(t_idx, loss)
            memory_idx = self.sum_tree.get_data(t_idx)
            self.replay_memory.set(memory_idx, loss=loss)


    def close(self):
        self.replay_memory.close()


    def add_losses(self):
        for i in range(len(self.replay_memory)):
            self.sum_tree.add(self.replay_memory.losses[i], i)


    def get_min(self):
        return self.sum_tree.get_min()


    def get_max(self):
        return self.sum_tree.get_max()


    def get_average(self):
        return self.sum_tree.get_average()



    def __getitem__(self, idx):
        return self.replay_memory[idx]


    def __len__(self):
        return len(self.replay_memory)


    @property
    def cache_size(self):
        return len(self.replay_memory.cache)


    @property
    def total(self):
        return self.sum_tree.total
    