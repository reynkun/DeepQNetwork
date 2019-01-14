import numpy as np
import h5py
import os


from .game_memory import GameMemory


class ReplayMemory:
    MAX_SIZE = 2000000


    def __init__(self,
                 fn,
                 input_height,
                 input_width,
                 input_channels,
                 state_type='uint8',
                 max_size=MAX_SIZE,
                 cache_size=300000):
        if not os.path.exists(fn):
            init = True
        else:
            init = False

        self.data_file = h5py.File(fn, 'a')
        self.state_type = state_type

        if init:
            self.create_dataset(input_height, input_width, input_channels, max_size)
            self.clear()

        self._start = self.data_file.attrs['start']
        self._end = self.data_file.attrs['end']
        self._max_size_plus_one = self.data_file.attrs['max_size_plus_one']

        self.init_cache(input_height,
                        input_width,
                        input_channels,
                        cache_size,
                        state_type)


    def create_dataset(self, input_height, input_width, input_channels, max_size):
        self.data_file.attrs['start'] = 0
        self.data_file.attrs['end'] = 0
        self.data_file.attrs['max_size_plus_one'] = max_size + 1


        self._max_size_plus_one = max_size + 1

        self.data_file.create_dataset('state',
                                      shape=(self.max_size_plus_one, input_height, input_width, input_channels),
                                      dtype=self.state_type)
        self.data_file.create_dataset('action',
                                      shape=(self.max_size_plus_one,),
                                      dtype='uint8')
        self.data_file.create_dataset('reward',
                                      shape=(self.max_size_plus_one,),
                                      dtype='uint8')
        self.data_file.create_dataset('next_state',
                                      shape=(self.max_size_plus_one, input_height, input_width, input_channels),
                                      dtype=self.state_type)
        self.data_file.create_dataset('continue',
                                      shape=(self.max_size_plus_one,),
                                      dtype='bool')


    def init_cache(self, input_height, input_width, input_channels, cache_size, state_type='uint8'):
        self.cache = GameMemory(input_height,
                                input_width,
                                input_channels,
                                max_size=cache_size,
                                state_type=state_type)

        self.cache_map = {}
        self.cache_map_rev = {}


    def clear(self):
        self.start = 0
        self.end = 0


    def append(self, state, action, reward, next_state, cont):
        # add memory to last position
        self.data_file['state'][self.end] = state
        self.data_file['action'][self.end] = action
        self.data_file['reward'][self.end] = reward
        self.data_file['next_state'][self.end] = next_state
        self.data_file['continue'][self.end] = cont

        last_index = self.end

        # move end pointer
        self.end = (self.end + 1) % self.max_size_plus_one

        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % self.max_size_plus_one

        return last_index


    def close(self):
        self.data_file.close()


    def __getitem__(self, idx):
        cache_idx = self.cache_map.get(idx, None)

        if cache_idx is not None:
            return self.cache[cache_idx]

        if isinstance(idx, int):
            new_idx = (self.start + idx) % self.max_size_plus_one

            row = (self.data_file['state'][new_idx],
                   self.data_file['action'][new_idx],
                   self.data_file['reward'][new_idx],
                   self.data_file['next_state'][new_idx],
                   self.data_file['continue'][new_idx])

            last_idx = self.cache.append(*row)

            self.cache_map[idx] = last_idx

            old_idx = self.cache_map_rev.get(last_idx, None)

            if old_idx is not None:
                del self.cache_map[old_idx]

            self.cache_map_rev[last_idx] = idx

            return row
        else:
            if idx.start is None:
                start = 0
            else:
                start = idx.start

            if idx.stop is None:
                stop = self.max_size_plus_one - 1
            else:
                stop = idx.stop

            if idx.step is None:
                step = 1
            else:
                step = idx.step

            return [self[i] for i in range(start, stop, step)]


    def __len__(self):
        if self.end < self.start:
            return self.end + self.max_size_plus_one - self.start
        else:
            return self.end - self.start


    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


    @property
    def max_size_plus_one(self):
        return self._max_size_plus_one


    @property
    def start(self):
        return self._start


    @start.setter
    def start(self, s):
        self._start = s
        self.data_file.attrs['start'] = self._start


    @property
    def end(self):
        return self._end


    @end.setter
    def end(self, s):
        self._end = s
        self.data_file.attrs['end'] = self._end




if __name__ == '__main__':
    # rp = ReplayMemory(10, 1, 1, 1)

    # for i in range(5):
    #     rp.actions[i] = i
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

    # def __init__(self,
    #              fn,
    #              input_height,
    #              input_width,
    #              input_channels,
    #              state_type='uint8',
    #              max_size=MAX_SIZE):

    # def append(self, state, action, reward, next_state, cont):


    # add memory to last position

    import random
    rp = ReplayMemory('test.hdf5', 1, 1, 1, max_size=10)

    print(rp.start, rp.end, rp.max_size_plus_one, len(rp))

    for i in range(100):
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i%2)

    for i in range(10):
        row = rp[i]

        print(row)

    # batch_size = 10
    #
    # period = len(rp) / batch_size
    # idx = random.randint(0, int(period)-1)
    # print(period, idx)
    #
    # for i in range(batch_size):
    #     print(int(period * i + idx))
