import numpy as np
import itertools
import h5py


class ReplayMemory:
    MAX_SIZE = 2000000


    def __init__(self,
                 fn,
                 input_height,
                 input_width,
                 input_channels,
                 state_type='uint8',
                 max_size=MAX_SIZE):
        if not os.exists(fn):
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


    def create_dataset(self, input_height, input_width, input_channels, max_size):
        self.data_file.attrs['start'] = 0
        self.data_file.attrs['end'] = 0
        self.data_file.attrs['max_size_plus_one'] = max_size + 1


        self._max_size_plus_one = max_size + 1


        self.data_file.create_dataset('states',
                                      shape=(self.max_size_plus_one, input_height, input_width, input_channels),
                                      dtype=self.state_type)
        self.data_file.create_dataset('actions',
                                      shape=(self.max_size_plus_one),
                                      dtype='uint8')
        self.data_file.create_dataset('rewards',
                                      shape=(self.max_size_plus_one, 1),
                                      dtype='uint8')
        self.data_file.create_dataset('next_states',
                                      shape=(self.max_size_plus_one, input_height, input_width, input_channels),
                                      dtype=self.state_type)
        self.data_file.create_dataset('continues',
                                      shape=(self.max_size_plus_one),
                                      dtype='bool')


    def clear(self):
        self.start = 0
        self.end = 0


    def append(self, state, action, reward, next_state, cont):
        # add memory to last position
        self.data_file['state'][self.end] = state
        self.data_file['actions'][self.end] = action
        self.data_file['rewards'][self.end] = reward
        self.data_file['next_states'][self.end] = next_state
        self.data_file['continues'][self.end] = cont

        # move end pointer
        self.end = (self.end + 1) % self.max_size_plus_one

        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % self.max_size_plus_one


    def __getitem__(self, idx):
        if isinstance(idx, int):
            new_idx = (self.start + idx) % self.max_size_plus_one

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
        return self._max_size_plus_one_plus_one


    @property
    def start(self):
        return self._start


    @start.setter
    def start(self, s):
        self._start = s
        self.data_file.attrs['start'] = self._start


    @end.setter
    def end(self, s):
        self._end = s
        self.data_file.attrs['end'] = self._end


    @property
    def end(self):
        return self._end




if __name__ == '__main__':
    # rp = ReplayMemory(10, 1, 1, 1)

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
