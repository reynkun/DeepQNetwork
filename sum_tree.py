import numpy as np


class SumTree:
    write = 0

    def __init__(self, capacity, dtype='bool'):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1, dtype=np.float16)
        self.data = np.zeros( capacity, dtype='bool')
        self.size = 0
        self.max_reached = False


    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)


    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])


    def total(self):
        return self.tree[0]


    def add(self, p, data):
        idx = self.write + self.capacity - 1

        last_idx = self.write

        self.data[self.write] = data
        self.update(idx, p)

        if not self.max_reached:
            self.size += 1

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            self.max_reached = True

        return last_idx


    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)


    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (dataIdx, self.tree[idx], self.data[dataIdx])

    def __len__(self):
        return self.size


if __name__ == '__main__':
    import random

    st = SumTree(100)
    st.add(0.01, 'A')
    st.add(0.09, 'B')

    print(st.total())



    # for i in range(100):
    #     sc = random.random() * st.total()
    #     index, node, value = st.get(sc)
    #     print(st.total(), sc, value)

    sc = 1000
    index, node, value = st.get(sc)
    print(st.total(), sc, value)

