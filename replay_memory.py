import numpy as np
import itertools


class ReplayMemory:
    def __init__(self, size, input_height, input_width, input_channels, state_type='uint8'):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        # self.data = [None] * (size + 1)

        self.size_plus_one = size + 1

        self.memory_states = np.ndarray((self.size_plus_one, input_height, input_width, input_channels), dtype=state_type)
        self.memory_actions = np.ndarray((self.size_plus_one), dtype='uint8')
        self.memory_rewards = np.ndarray((self.size_plus_one, 1), dtype='uint8')
        self.memory_next_states = np.ndarray((self.size_plus_one, input_height, input_width, input_channels), dtype=state_type)
        self.memory_continues = np.ndarray((self.size_plus_one, 1), dtype='bool')

        self.start = 0
        self.end = 0
      

    def clear(self):
        self.start = 0
        self.end = 0


    def append(self, state, action, reward, next_state, cont):
        # self.data[self.end] = element

        self.memory_states[self.end] = state
        self.memory_actions[self.end] = action
        self.memory_rewards[self.end] = reward
        self.memory_next_states[self.end] = next_state
        self.memory_continues[self.end] = cont


        self.end = (self.end + 1) % self.size_plus_one
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % self.size_plus_one
        

    def __getitem__(self, idx):
        if isinstance(idx, int):
            new_idx = (self.start + idx) % self.size_plus_one

            return (self.memory_states[new_idx],
                    self.memory_actions[new_idx],
                    self.memory_rewards[new_idx],
                    self.memory_next_states[new_idx],
                    self.memory_continues[new_idx])        
        else:
            if idx.start is None:
                start = 0
            else:
                start = idx.start

            if idx.stop is None:
                stop = self.size_plus_one - 1
            else:
                stop = idx.stop

            if idx.step is None:
                step = 1
            else:
                step = idx.step

            return [self[i] for i in range(start, stop, step)]
    

    def __len__(self):
        if self.end < self.start:
            return self.end + self.size_plus_one - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


if __name__ == '__main__':
    rp = ReplayMemory(10, 1, 1, 1)

    # for i in range(5):
    #     rp.memory_actions[i] = i
    #     print(len)

    # for i in range(10):
    #     print(i, rp[i])


    # for i in range(15):
    #     rp.append(np.zeros((1, 1, 1)), i, i, i, np.zeros((1, 1, 1)))

    # rp.clear()

    # for i in range(10):
    #     rp.append(np.zeros((1, 1, 1)), i*2, i*2, i*2, np.zeros((1, 1, 1)))
    #     print(i, len(rp))


    # print(len(rp))
    # for i in range(10):
    #     print(i, rp[i])

    # for row in rp[2:10]:
    #     print(row)
    #     
    #     
    #     
    import random
    rp = ReplayMemory(100, 1, 1, 1)
    for i in range(100):
        rp.append(np.zeros((1, 1, 1)), i, i, i, np.zeros((1, 1, 1)))
    batch_size = 10

    period = len(rp) / batch_size
    idx = random.randint(0, int(period)-1)
    print(period, idx)

    for i in range(batch_size):
        print(int(period * i + idx))
