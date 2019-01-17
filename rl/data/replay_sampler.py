import random


from .replay_memory import ReplayMemory
from .sum_tree import SumTree


class ReplaySampler:
    MAX_SIZE = ReplayMemory.MAX_SIZE
    CACHE_SIZE = ReplayMemory.CACHE_SIZE


    def __init__(self,
                 fn,
                 input_height,
                 input_width,
                 input_channels,
                 state_type='uint8',
                 max_size=MAX_SIZE,
                 cache_size=CACHE_SIZE):
        self.replay_memory = ReplayMemory(fn,
                                          input_height,
                                          input_width,
                                          input_channels,
                                          state_type=state_type,
                                          max_size=max_size,
                                          cache_size=cache_size)
        self.sum_tree = SumTree(max_size, dtype='uint32')

        # self.last_idx = 0
        # self.max_size = max_size
        # self.is_max_capacity = False

        self.add_losses()


    def append(self, state, action, reward, next_state, cont, loss):
        # self.replay_memory.set_abs(self.last_idx, state, action, reward, next_state, cont, loss)
        # self.sum_tree.add(loss, self.last_idx)
        #
        # self.last_idx = (self.last_idx + 1) % (self.replay_memory.max_size_plus_one - 1)
        #
        # if self.last_idx == 0:
        #     self.is_max_capacity = True

        self.replay_memory.append(state, action, reward, next_state, cont, loss)
        self.sum_tree.add(loss, self.replay_memory.last_idx_abs)


    def sample_memories(self, states, actions, rewards, next_states, continues, losses, batch_size=32, tree_idxes=None):
        # num_tries = 0

        indexes = {}
        for i in range(batch_size):
            memory_idx = None
            count = 0
            while memory_idx is None or memory_idx in indexes:
                s = random.random() * self.sum_tree.total()
                t_idx, d_idx, score, memory_idx = self.sum_tree.get_with_info(s)
                indexes[memory_idx] = True
                count += 1

                if count > 1000:
                    break

            row = self.replay_memory.get_abs(memory_idx)

            # print('adding', t_idx)
            if tree_idxes is not None:
                tree_idxes.append(t_idx)

            for j, col in enumerate([states, actions, rewards, next_states, continues, losses]):
                col[i] = row[j]


    def update_sum_tree(self, tree_idxes, losses):
        for t_idx, loss in zip(tree_idxes, losses):
            self.sum_tree.update_score(t_idx, loss)
            # memory_idx = self.sum_tree[self.sum_tree.get_data_idx(t_idx)]
            memory_idx = self.sum_tree.get_data(t_idx)
            self.replay_memory.set_loss_abs(memory_idx, loss)


    def close(self):
        self.replay_memory.close()


    def add_losses(self):
        for i in range(len(self.replay_memory)):
            row = self.replay_memory.get_abs(i)
            self.sum_tree.add(row[-1], i)

    # def __getitem__(self, idx):
    #     return self.replay_memory.get_abs(idx)


    # def __len__(self):
    #     if self.is_max_capacity:
    #         return self.max_size
    #     else:
    #         return self.last_idx

    def __getitem__(self, idx):
        return self.replay_memory[idx]


    def __len__(self):
        return len(self.replay_memory)


    @property
    def cache_size(self):
        return len(self.replay_memory.cache)
