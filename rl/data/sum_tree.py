import numpy as np


class SumTree:
    '''
    Implements sum tree
    '''

    write = 0

    def __init__(self, capacity, dtype='uint32'):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1, dtype=np.float16)
        self.data = np.zeros( capacity, dtype=dtype)
        self.size = 0
        self.max_reached = False


    def add(self, score, data):
        idx = self.write + self.capacity - 1

        last_idx = self.write

        self.data[self.write] = data
        self.update_score(idx, score)

        if not self.max_reached:
            self.size += 1

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            self.max_reached = True

        return last_idx


    def update_score(self, tree_idx, score):
        change = score - self.tree[tree_idx]

        self.tree[tree_idx] = score
        self._propagate(tree_idx, change)


    def update_value(self, tree_idx, val):
        data_idx = self.get_data_idx(tree_idx)

        self.data[data_idx] = val


    def get(self, score):
        tree_idx = self._retrieve(0, score)
        data_idx = self.get_data_idx(tree_idx)

        return self.data[data_idx]


    def get_with_info(self, s):
        tree_idx = self._retrieve(0, s)
        data_idx = self.get_data_idx(tree_idx)

        return (tree_idx, data_idx, self.tree[tree_idx], self.data[data_idx])


    def get_data_idx(self, tree_idx):
        return tree_idx - self.capacity + 1


    def get_data(self, tree_idx):
        return self.data[self.get_data_idx(tree_idx)]


    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)


    def _retrieve(self, idx, score):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if score <= self.tree[left]:
            return self._retrieve(left, score)
        else:
            return self._retrieve(right, score - self.tree[left])


    @property
    def total(self):
        return self.tree[0]


    def __len__(self):
        return self.size


    def __getitem__(self, item):
        return self.data[item]






