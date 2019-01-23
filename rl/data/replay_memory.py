import numpy as np


class ReplayMemory:
    def __init__(self, input_height, input_width, input_channels, max_size=10000, state_type='uint8'):
        self.cur_idx = 0
        self.is_full = False
        self.max_size = max_size

        self.states = np.zeros((self.max_size,
                                input_height,
                                input_width,
                                input_channels),
                                dtype=state_type)
        self.actions = np.zeros((self.max_size,),
                                dtype='uint8')
        self.rewards = np.zeros((self.max_size,),
                                dtype='uint8')
        self.next_states = np.zeros((self.max_size,
                                     input_height,
                                     input_width,
                                     input_channels),
                                    dtype=state_type)
        self.continues = np.zeros((self.max_size,),
                                   dtype='bool')
        self.losses = np.zeros((self.max_size,),
                                dtype='float16')


    def clear(self):
        self.cur_idx = 0
        self.is_full = False


    def append(self, state=None, action=None, reward=None, next_state=None, cont=None, loss=None):
        self.set(self.cur_idx,
                 state=state,
                 action=action,
                 reward=reward,
                 next_state=next_state,
                 cont=cont,
                 loss=loss)

        self.increment_idx()
        # self.cur_idx = (self.cur_idx + 1) % self.max_size


    def extend(self, target):
        for i in range(len(target)):
            target.copy(i, self, self.cur_idx)
            self.increment_idx()


    def get(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'next_state': self.next_states[idx],
            'cont': self.continues[idx],
            'loss': self.losses[idx]
        }


    def set(self, idx, state=None, action=None, reward=None, next_state=None, cont=None, loss=None):
        if state is not None:
            self.states[idx] = state
        if action is not None:
            self.actions[idx] = action
        if reward is not None:
            self.rewards[idx] = reward
        if next_state is not None:
            self.next_states[idx] = next_state
        if cont is not None:
            self.continues[idx] = cont
        if loss is not None:
            self.losses[idx] = loss


    def copy(self, idx, target, target_idx):
        target.states[target_idx] = self.states[idx]
        target.actions[target_idx] = self.actions[idx]
        target.rewards[target_idx] = self.rewards[idx]
        target.next_states[target_idx] = self.next_states[idx]
        target.continues[target_idx] = self.continues[idx]
        target.losses[target_idx] = self.losses[idx]


    def increment_idx(self):
        if self.cur_idx + 1 >= self.max_size:
            self.is_full = True
        self.cur_idx = (self.cur_idx + 1) % self.max_size


    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.get(idx)
        else:
            if idx.start is None:
                start = 0
            else:
                start = idx.start

            if idx.stop is None:
                stop = self.max_size
            else:
                stop = idx.stop

            if idx.step is None:
                step = 1
            else:
                step = idx.step

            return [self[i] for i in range(start, stop, step)]


    def __len__(self):
        if self.is_full:
            return self.max_size

        return self.cur_idx


    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


