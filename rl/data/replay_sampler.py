import random


from .sum_tree import SumTree


class ReplaySampler:
    MAX_DUPLICATE_RETRIES = 100


    def __init__(self, replay_memory):
        self.replay_memory = replay_memory
        self.sum_tree = SumTree(self.replay_memory.max_size, dtype='uint32')
        self.add_losses()


    def append(self, state, action, reward, next_state, cont, loss):
        self.sum_tree.add(loss, self.replay_memory.cur_idx)
        self.replay_memory.append(state, action, reward, next_state, cont, loss)


    def sample_memories(self, target, batch_size=32, tree_idxes=None):
        dup_count = 0
        indexes = {}
        for i in range(batch_size):
            s = random.random() * self.sum_tree.total()
            t_idx, d_idx, score, memory_idx = self.sum_tree.get_with_info(s)
            while memory_idx in indexes:
                s = random.random() * self.sum_tree.total()
                t_idx, d_idx, score, memory_idx = self.sum_tree.get_with_info(s)
                indexes[memory_idx] = True
                dup_count += 1

                if dup_count > self.MAX_DUPLICATE_RETRIES:
                    break

            self.replay_memory.copy(memory_idx, target, i)

            # print('adding', t_idx)
            if tree_idxes is not None:
                tree_idxes.append(t_idx)


    def update_sum_tree(self, tree_idxes, losses):
        for t_idx, loss in zip(tree_idxes, losses):
            self.sum_tree.update_score(t_idx, loss)
            # memory_idx = self.sum_tree[self.sum_tree.get_data_idx(t_idx)]
            memory_idx = self.sum_tree.get_data(t_idx)
            self.replay_memory.set(memory_idx, loss=loss)


    def close(self):
        self.replay_memory.close()


    def add_losses(self):
        for i in range(len(self.replay_memory)):
            self.sum_tree.add(self.replay_memory.losses[i], i)


    def __getitem__(self, idx):
        return self.replay_memory[idx]


    def __len__(self):
        return len(self.replay_memory)


    @property
    def cache_size(self):
        return len(self.replay_memory.cache)
