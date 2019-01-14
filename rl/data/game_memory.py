import numpy as np


class GameMemory:
    def __init__(self, input_height, input_width, input_channels, max_size=10000, state_type='uint8'):
        self.size_plus_one = max_size + 1

        self.states = np.ndarray((self.size_plus_one,
                                  input_height,
                                  input_width,
                                  input_channels),
                                 dtype=state_type)
        self.actions = np.ndarray((self.size_plus_one,),
                                  dtype='uint8')
        self.rewards = np.ndarray((self.size_plus_one,),
                                  dtype='uint8')
        self.next_states = np.ndarray((self.size_plus_one,
                                       input_height,
                                       input_width,
                                       input_channels),
                                      dtype=state_type)
        self.continues = np.ndarray((self.size_plus_one,),
                                    dtype='bool')

        self.start = 0
        self.end = 0


    def clear(self):
        self.start = 0
        self.end = 0


    def append(self, state, action, reward, next_state, cont):
        self.states[self.end] = state
        self.actions[self.end] = action
        self.rewards[self.end] = reward
        self.next_states[self.end] = next_state
        self.continues[self.end] = cont

        last_idx = self.end

        self.end = (self.end + 1) % self.size_plus_one
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % self.size_plus_one

        return last_idx


    def __setitem__(self, idx, row):
        state, action, reward, next_state, cont = row

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.continues[idx] = cont


    def __getitem__(self, idx):
        if isinstance(idx, int):
            new_idx = (self.start + idx) % self.size_plus_one

            return (self.states[new_idx],
                    self.actions[new_idx],
                    self.rewards[new_idx],
                    self.next_states[new_idx],
                    self.continues[new_idx])
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
