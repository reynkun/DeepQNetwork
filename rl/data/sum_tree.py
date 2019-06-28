import numpy as np


class SumTree:
    '''
    Implements sum tree.  Modified from:

    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb

    '''

    def __init__(self, capacity, dtype='uint32'):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1, dtype=np.float32)
        self.data = np.zeros(capacity, dtype=dtype)
        self.size = 0
        self.max_reached = False
        self.write = 0


    def add(self, score, data):
        '''
        Add new data with score
        '''

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
        '''
        Updates score for tree_idx
        '''
        change = score - self.tree[tree_idx]

        self.tree[tree_idx] = score
        self._propagate(tree_idx, change)


    def update_value(self, tree_idx, val):
        '''
        Update the value for tree_idx
        '''

        data_idx = self.get_data_idx(tree_idx)

        self.data[data_idx] = val


    def get(self, score):
        '''
        Data with value at score
        '''

        tree_idx = self._retrieve(0, score)
        data_idx = self.get_data_idx(tree_idx)

        return self.data[data_idx]


    def get_with_info(self, s):
        '''
        Get data and index info for score s
        '''
        tree_idx = self._retrieve(0, s)
        data_idx = self.get_data_idx(tree_idx)

        return (tree_idx, data_idx, self.tree[tree_idx], self.data[data_idx])


    def get_data_idx(self, tree_idx):
        '''
        Get index of data for tree_idx
        '''

        return tree_idx - self.capacity + 1


    def get_data(self, tree_idx):
        '''
        Get data stored at tree_idx
        '''
        return self.data[self.get_data_idx(tree_idx)]


    def _propagate(self, idx, change):
        '''
        Recursively propagate changes through the sum tree
        '''
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)


    def _retrieve(self, idx, score):
        '''

        '''
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            # check for float rounding bug
            if self.get_data_idx(idx) - self.size + 1 > 0:
                idx -= self.get_data_idx(idx) - self.size + 1
            return idx

        if score <= self.tree[left]:
            return self._retrieve(left, score)
        else:
            return self._retrieve(right, score - self.tree[left])


    def _start_data_index(self):
        '''
        Get start index for leafs
        '''
        return self.capacity - 1


    @property
    def total(self):
        '''
        Total score sum of all leafs 
        '''        
        return self.tree[0]


    def get_min(self):
        '''
        Get min score
        '''
        return np.min(self.tree[self._start_data_index():self._start_data_index()+len(self)])


    def get_max(self):
        '''
        Get max score
        '''
        return np.max(self.tree[self._start_data_index():self._start_data_index()+len(self)])


    def get_average(self):
        '''
        Get average score
        '''
        return np.average(self.tree[self._start_data_index():self._start_data_index()+len(self)])


    def __len__(self):
        return self.size


    def __getitem__(self, item):
        return self.data[item]


