from .replay_memory import ReplayMemory
from .sum_tree import SumTree


class ReplaySampler:
    def __init__(self,
                 fn,
                 input_height,
                 input_width,
                 input_channels,
                 state_type='uint8',
                 max_size=MAX_SIZE,
                 cache_size=CACHE_SIZE):
        self.memory = ReplayMemory(fn,
                                   input_height,
                                   input_width,
                                   input_channels,
                                   state_type=state_type,
                                   max_size=max_size,
                                   cache_size=cache_size)
        self.sum_tree = SumTree(max_size, dtype='uint32')


