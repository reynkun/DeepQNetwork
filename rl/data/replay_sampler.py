import random


class ReplaySampler:
    def __init__(self, replay_memory):
        self.replay_memory = replay_memory


    def append(self, state, action, reward, next_state, cont, loss):
        self.replay_memory.append(state, action, reward, next_state, cont, loss)


    def sample_memories(self, target, batch_size=32):
        size = len(self) / batch_size

        for i in range(batch_size):
            idx = random.randint(int(i * size), int((i + 1) * size - 1))

            self.replay_memory.copy(idx, target, i)

    def close(self):
        self.replay_memory.close()


    def add_losses(self):
        for i in range(len(self.replay_memory)):
            self.sum_tree.add(self.replay_memory.losses[i], i)


    def __getitem__(self, idx):
        return self.replay_memory[idx]


    def __len__(self):
        return len(self.replay_memory)