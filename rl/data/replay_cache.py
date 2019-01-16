import numpy as np


class ReplayCache:
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
        self.losses = np.ndarray((self.size_plus_one,),
                                    dtype='float16')

        self.start = 0
        self.end = 0
        self.last_idx_abs = None


    def clear(self):
        self.start = 0
        self.end = 0


    def append(self, state, action, reward, next_state, cont, loss):
        self.set_abs(self.end, state, action, reward, next_state, cont, loss)
        self.last_idx_abs = self.end

        self.end = (self.end + 1) % self.size_plus_one
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % self.size_plus_one


    def set_loss_abs(self, idx_abs, loss):
        self.losses[idx_abs] = loss


    def set_abs(self, idx_abs, state, action, reward, next_state, cont, loss):
        self.states[idx_abs] = state
        self.actions[idx_abs] = action
        self.rewards[idx_abs] = reward
        self.next_states[idx_abs] = next_state
        self.continues[idx_abs] = cont
        self.losses[idx_abs] = loss


    def get_abs(self, idx_abs):
        return (self.states[idx_abs],
                self.actions[idx_abs],
                self.rewards[idx_abs],
                self.next_states[idx_abs],
                self.continues[idx_abs],
                self.losses[idx_abs])


    def __setitem__(self, idx, row):
        new_idx = (self.start + idx) % self.size_plus_one

        self.set_abs(new_idx, row)


    def __getitem__(self, idx):
        if isinstance(idx, int):
            new_idx = (self.start + idx) % self.size_plus_one

            return self.get_abs(new_idx)
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


