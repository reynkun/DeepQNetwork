import numpy as np
import h5py
import os
import pickle
import random


from .replay_memory import ReplayMemory


class ReplayMemoryDisk:
    MAX_SIZE = 2000000
    CACHE_SIZE = 350000


    def __init__(self,
                 fn,
                 input_height=1,
                 input_width=1,
                 input_channels=1,
                 state_type='uint8',
                 max_size=MAX_SIZE,
                 cache_size=CACHE_SIZE):
        if not os.path.exists(fn):
            init = True
        else:
            init = False

        self.filename = fn
        self.data_file = h5py.File(fn, 'a')

        if init:
            self.create_dataset(input_height, input_width, input_channels, state_type, max_size)

            self._cur_idx = 0
            self._is_full = False
            self.max_size = max_size
        else:
            self._cur_idx = self.data_file.attrs['cur_idx']
            self._is_full = self.data_file.attrs['is_full']
            self.max_size = self.data_file.attrs['max_size']

            input_height = self.data_file.attrs['input_height']
            input_width = self.data_file.attrs['input_width']
            input_channels = self.data_file.attrs['input_channels']
            state_type = self.data_file.attrs['state_type']

        # init cache
        if cache_size:
            self.cache = ReplayMemory(input_height,
                                      input_width,
                                      input_channels,
                                      max_size=cache_size,
                                      state_type=state_type)

            self.cache_map = {}
            self.cache_map_rev = {}
        else:
            self.cache = None


    def create_dataset(self, input_height, input_width, input_channels, state_type, max_size):
        self.data_file.attrs['cur_idx'] = 0
        self.data_file.attrs['is_full'] = False
        self.data_file.attrs['max_size'] = max_size
        self.data_file.attrs['input_height'] = input_height
        self.data_file.attrs['input_width'] = input_width
        self.data_file.attrs['input_channels'] = input_channels
        self.data_file.attrs['state_type'] = state_type


        self.data_file.create_dataset('states',
                                      shape=(max_size, input_height, input_width, input_channels),
                                      dtype=state_type)
        self.data_file.create_dataset('actions',
                                      shape=(max_size,),
                                      dtype='uint8')
        self.data_file.create_dataset('rewards',
                                      shape=(max_size,),
                                      dtype='uint8')
        self.data_file.create_dataset('next_states',
                                      shape=(max_size, input_height, input_width, input_channels),
                                      dtype=state_type)
        self.data_file.create_dataset('continues',
                                      shape=(max_size,),
                                      dtype='bool')
        self.data_file.create_dataset('losses',
                                      shape=(max_size,),
                                      dtype='float16')

    def clear(self):
        self.cur_idx = 0
        self.is_full = False

        if self.cache:
            self.cache.clear()


    def load_memory(self, memory_fn, delete=True):
        with open(memory_fn, 'rb') as fin:
            size = pickle.load(fin)

            for i in range(size):
                self.append(*pickle.load(fin))

        if delete:
            os.unlink(memory_fn)


    def sample_memories(self, target, batch_size=32):
        idxs = {}
        dup_count = 0

        for i in range(batch_size):
            idx = random.randint(0, len(self)-1)
            while idx in idxs and dup_count < batch_size*3:
                idx = random.randint(0, len(self)-1)
                dup_count += 1
                continue

            idxs[idx] = True
            self.copy(idx, target, i)


    def append(self, state=None, action=None, reward=None, next_state=None, cont=None, loss=None):
        self.set(self.cur_idx,
                 state=state,
                 action=action,
                 reward=reward,
                 next_state=next_state,
                 cont=cont,
                 loss=loss)

        if self.cur_idx == self.max_size - 1:
            self.is_full = True

        self.cur_idx = (self.cur_idx + 1) % self.max_size


    def set(self, idx, state=None, action=None, reward=None, next_state=None, cont=None, loss=None):
        if state is not None:
            self.data_file['states'][idx] = state
        if action is not None:
            self.data_file['actions'][idx] = action
        if reward is not None:
            self.data_file['rewards'][idx] = reward
        if next_state is not None:
            self.data_file['next_states'][idx] = next_state
        if cont is not None:
            self.data_file['continues'][idx] = cont
        if loss is not None:
            self.data_file['losses'][idx] = loss


        if self.cache:
            cache_idx = self.cache_map.get(idx, None)
            if cache_idx is not None:
                del self.cache_map[idx]
                del self.cache_map_rev[cache_idx]


    def get(self, idx):
        if self.cache:
            cache_idx = self.cache_map.get(idx, None)

            if cache_idx is not None:
                return self.cache.get(cache_idx)

        row = self.get_row(idx)

        if self.cache:
            # cache_cur_idx = self.cache.cur_idx
            # self.cache_map[idx] = cache_cur_idx
            # self.cache.append(**row)
            #
            # old_idx = self.cache_map_rev.get(cache_cur_idx, None)
            # if old_idx is not None:
            #     del self.cache_map[old_idx]
            #
            # self.cache_map_rev[cache_cur_idx] = idx

            self.cache_row(idx, row)

        return row


    def copy(self, idx, target, target_idx):
        if self.cache:

            cache_idx = self.cache_map.get(idx, None)

            if cache_idx is not None:

                # print('using cache', cache_idx)
                target.states[target_idx] = self.cache.states[cache_idx]
                target.actions[target_idx] = self.cache.actions[cache_idx]
                target.rewards[target_idx] = self.cache.rewards[cache_idx]
                target.next_states[target_idx] = self.cache.next_states[cache_idx]
                target.continues[target_idx] = self.cache.continues[cache_idx]
                target.losses[target_idx] = self.cache.losses[cache_idx]

                return


        target.states[target_idx] = self.states[idx]
        target.actions[target_idx] = self.actions[idx]
        target.rewards[target_idx] = self.rewards[idx]
        target.next_states[target_idx] = self.next_states[idx]
        target.continues[target_idx] = self.continues[idx]
        target.losses[target_idx] = self.losses[idx]

        # print(idx, target_idx, target.states[target_idx])
        #
        if self.cache:
            row = self.get_row(idx)
            self.cache_row(idx, row)


    def close(self):
        self.data_file.close()


    def get_row(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'next_state': self.next_states[idx],
            'cont': self.continues[idx],
            'loss': self.losses[idx]
        }

    def cache_row(self, idx, row):
        cache_cur_idx = self.cache.cur_idx
        self.cache_map[idx] = cache_cur_idx
        self.cache.append(**row)

        old_idx = self.cache_map_rev.get(cache_cur_idx, None)
        if old_idx is not None:
            del self.cache_map[old_idx]

        self.cache_map_rev[cache_cur_idx] = idx


    def __getitem__(self, idx):
        try:
            return self.get(idx)
        except TypeError:
            if idx.start is None:
                start = 0
            else:
                start = idx.start

            if idx.stop is None:
                stop = self.max_size - 1
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


    @property
    def states(self):
        return self.data_file['states']


    @property
    def actions(self):
        return self.data_file['actions']


    @property
    def rewards(self):
        return self.data_file['rewards']


    @property
    def next_states(self):
        return self.data_file['next_states']


    @property
    def continues(self):
        return self.data_file['continues']


    @property
    def losses(self):
        return self.data_file['losses']


    @property
    def cur_idx(self):
        return self._cur_idx


    @cur_idx.setter
    def cur_idx(self, idx):
        self._cur_idx = idx
        self.data_file.attrs['cur_idx'] = self._cur_idx


    @property
    def is_full(self):
        return self._is_full


    @is_full.setter
    def is_full(self, val):
        self._is_full = val
        self.data_file.attrs['is_full'] = self._is_full


    @property
    def cache_size(self):
        return len(self.cache)

