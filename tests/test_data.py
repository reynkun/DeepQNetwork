import os
import test_base
import numpy as np


test_base.TestBase.add_lib_dir_sys_path()


from rl.data.replay_memory_disk import ReplayMemoryDisk
from rl.data.replay_memory import ReplayMemory
from rl.data.sum_tree import SumTree
from rl.data.replay_sampler import ReplaySampler


class TestData(test_base.TestBase):
    def check_row(self, rp, i, val):
        self.assertEqual(rp[i]['state'][0], val)
        self.assertEqual(rp[i]['action'], val)
        self.assertEqual(rp[i]['reward'], val)
        self.assertEqual(rp[i]['next_state'][0], val)
        self.assertEqual(rp[i]['cont'], val > 0)
        self.assertAlmostEqual(rp[i]['loss'], val * 0.01, places=3)


    def test_replay_memory_disk_append(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemoryDisk(data_fn,
                              1,
                              1,
                              1,
                              max_size=2)

        for i in range(2):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

            row = rp[i]

            self.assertEqual(len(rp), i+1)
            self.assertEqual(row['state'][0], i)
            self.assertEqual(row['action'], i)
            self.assertEqual(row['reward'], i)
            self.assertEqual(row['next_state'][0], i)
            self.assertEqual(row['cont'], i)
            self.assertAlmostEqual(row['loss'], i * 0.01, places=3)


    def test_replay_memory_disk_copy(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemoryDisk(data_fn,
                              1,
                              1,
                              1,
                              max_size=2)

        for i in range(2):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

            row = rp[i]

        target = ReplayMemory(1, 1, 1, 2)
        for i in range(2):
            rp.copy(i, target, i)

            self.assertEqual(target[i]['state'][0][0][0], i)
            self.assertEqual(target[i]['action'], i)
            self.assertEqual(target[i]['reward'], i)
            self.assertEqual(target[i]['next_state'][0][0][0], i)
            self.assertEqual(target[i]['cont'], i > 0)
            self.assertAlmostEqual(target[i]['loss'], i * 0.01, places=3)

            self.assertEqual(target.states[i][0][0][0], i)
            self.assertEqual(target.actions[i], i)
            self.assertEqual(target.rewards[i], i)
            self.assertEqual(target.next_states[i][0][0][0], i)
            self.assertEqual(target.continues[i], i > 0)
            self.assertAlmostEqual(target.losses[i], i * 0.01, places=3)


    def test_replay_memory_disk_overwrite_1(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemoryDisk(data_fn,
                              1,
                              1,
                              1,
                              max_size=2,
                              cache_size=0)

        for i in range(3):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)


        self.assertEqual(len(rp), 2)
        self.assertEqual(rp[0]['action'], 2)
        self.assertAlmostEqual (float(rp[0]['loss']), 0.02, places=3)
        self.assertEqual(rp[1]['action'], 1)
        self.assertAlmostEqual(float(rp[1]['loss']), 0.01, places=3)


    def test_replay_memory_disk_overwrite_2(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemoryDisk(data_fn,
                              1,
                              1,
                              1,
                              max_size=2,
                              cache_size=5)

        i = 0
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        i = 1
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        self.assertEqual(rp[1]['action'], 1)
        self.assertEqual(rp[0]['action'], 0)
        i = 2
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)


        self.assertEqual(rp[0]['action'], 2)
        self.assertEqual(rp[1]['action'], 1)


    def test_replay_memory_disk_overwrite_3(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemoryDisk(data_fn,
                              1,
                              1,
                              1,
                              max_size=2,
                              cache_size=5)

        for i in range(5):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        self.assertEqual(rp[0]['action'], 4)
        self.assertEqual(rp[1]['action'], 3)


    def test_replay_memory_disk_cache_1(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemoryDisk(data_fn,
                              1,
                              1,
                              1,
                              max_size=10,
                              cache_size=1)

        for i in range(10):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        self.assertEqual(rp[0]['action'], 0)
        self.assertEqual(rp[1]['action'], 1)
        self.assertEqual(rp[2]['action'], 2)
        self.assertEqual(rp[0]['action'], 0)


    def test_replay_memory_disk_cache_2(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemoryDisk(data_fn,
                              1,
                              1,
                              1,
                              max_size=2,
                              cache_size=2)

        i = 0
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        self.assertEqual(rp[0]['action'], 0)

        i = 1
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        self.assertEqual(rp[1]['action'], 1)
        self.assertEqual(rp[0]['action'], 0)

        i = 2
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        self.assertEqual(rp[1]['action'], 1)

        i = 3
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        self.check_row(rp, 0, 2)
        self.check_row(rp, 1, 3)
        # self.assertEqual(rp[0]['action'], 2)
        # self.assertEqual(rp[1]['action'], 3)


    def test_replay_memory_disk_sample_1(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemoryDisk(data_fn,
                              1,
                              1,
                              1,
                              max_size=10,
                              cache_size=1)

        num_values = 10
        for i in range(num_values):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)


        batch_size = 6
        # target = (np.zeros((batch_size, 1, 1, 1), dtype='uint8'),
        #           np.zeros((batch_size,), dtype='uint8'),
        #           np.zeros((batch_size,), dtype='uint8'),
        #           np.zeros((batch_size, 1, 1, 1), dtype='uint8'),
        #           np.zeros((batch_size,), dtype='bool'),
        #           np.zeros((batch_size,), dtype='float16'))

        target = ReplayMemory(1, 1, 1, max_size=batch_size)


        values = {}
        count = 0
        num_batches = 0
        while len(values) < num_values:
            if num_batches > 10:
                break

            rp.sample_memories(target, batch_size=batch_size)

            for i in range(batch_size):
                val = target[i]['action']

                if val not in values:
                    values[val] = True
                    count += 1

                self.assertGreater(target[5]['loss'], -0.001)
                self.assertLess(target[5]['loss'], 0.10)

            num_batches += 1

        self.assertEqual(len(values), num_values)


    def test_replay_memory_disk_sample_2(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemoryDisk(data_fn,
                              1,
                              1,
                              1,
                              max_size=10,
                              cache_size=1,
                              state_type='uint8')

        num_values = 10
        for i in range(num_values):
            rp.append(np.array([[[i]]]), i+1, i+2, np.array([[[i+3]]]), i+4, (i+5) * 0.01)

        batch_size = 6

        target = ReplayMemory(1, 1, 1, max_size=batch_size)

        values = {}
        count = 0
        num_batches = 0

        while len(values) < num_values:
            if num_batches > 10:
                break

            rp.sample_memories(target,
                               batch_size=batch_size)

            for i in range(batch_size):
                # print(i, target[i])

                val = target.states[i][0][0][0]

                if val not in values:
                    values[val] = True
                    count += 1

                self.assertEqual(target.states[i][0][0][0], val)
                self.assertEqual(target.actions[i], val+1)
                self.assertEqual(target.rewards[i], val+2)
                self.assertEqual(target.next_states[i][0][0][0], val+3)
                self.assertEqual(target.continues[i], val+4 > 0)
                self.assertAlmostEqual(target.losses[i], (val+5) * 0.01, places=2)

            num_batches += 1

        self.assertEqual(len(values), num_values)

    def test_replay_memory_disk_extend_1(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemoryDisk(data_fn,
                              1,
                              1,
                              1,
                              max_size=2,
                              cache_size=0)

        rp2 = ReplayMemoryDisk(data_fn,
                              1,
                              1,
                              1,
                              max_size=3,
                              cache_size=0)

        for i in range(3):
            rp2.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        rp.extend(rp2)

        self.assertEqual(len(rp), 2)
        self.assertEqual(rp[0]['action'], 2)
        self.assertAlmostEqual(float(rp[0]['loss']), 0.02, places=3)
        self.assertEqual(rp[1]['action'], 1)
        self.assertAlmostEqual(float(rp[1]['loss']), 0.01, places=3)
        self.assertEqual(len(rp), 2)


    def test_replay_memory_1(self):
        rp = ReplayMemory(1,
                          1,
                          1,
                          max_size=2)

        i = 0
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        self.assertEqual(rp[0]['action'], 0)

        i = 1
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        self.assertEqual(rp[1]['action'], 1)

        i = 2
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        i = 3
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        self.assertEqual(len(rp), 2)

        for i, val in enumerate([2, 3]):
            self.assertEqual(rp[i]['state'], val)
            self.assertEqual(rp[i]['action'], val)
            self.assertEqual(rp[i]['reward'], val)
            self.assertEqual(rp[i]['next_state'], val)
            self.assertEqual(rp[i]['cont'], val>0)
            self.assertAlmostEqual(rp[i]['loss'], val * 0.01, places=3)

        self.assertEqual(len(rp), 2)
        self.assertEqual(rp.is_full, True)


    def test_replay_memory_2(self):
        rp = ReplayMemory(1,
                          1,
                          1,
                          max_size=2)

        i = 0

        for i in range(2):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)


    def test_replay_memory_extend_1(self):
        rp = ReplayMemory(1,
                          1,
                          1,
                          max_size=2)


        for i in range(2):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        rp2 = ReplayMemory(1,
                          1,
                          1,
                          max_size=3)

        for i in range(3):
            rp.append(np.array([[[(i+5)]]]), (i+5), (i+5), np.array([[[(i+5)]]]), (i+5), (i+5) * 0.01)

        rp.extend(rp2)

        self.assertEqual(rp[0]['state'][0][0][0], 7)
        self.assertEqual(rp[0]['action'], 7)
        self.assertEqual(rp[0]['reward'], 7)
        self.assertEqual(rp[0]['next_state'][0][0][0], 7)
        self.assertAlmostEqual(rp[0]['loss'], 0.07, places=2)
        self.assertEqual(rp[0]['action'], 7)

        self.assertEqual(rp[1]['action'], 6)
        self.assertEqual(len(rp), 2)


    def test_sum_tree_empty(self):
        st = SumTree(2, dtype='uint8')

        st.get(1.0)


    def test_sum_tree_set_1(self):
        st = SumTree(2, dtype='uint8')

        st.add(1, 1)
        self.assertEqual(st.total(), 1)
        self.assertEqual(st.get(1.0), 1)


    def test_sum_tree_set_2(self):
        st = SumTree(2, dtype='uint8')

        st.add(1, 1)
        st.add(1, 2)

        self.assertEqual(st.total(), 2)
        self.assertEqual(st.get(0.5), 1)
        self.assertEqual(st.get(1.5), 2)


    def test_sum_tree_overwrite_1(self):
        st = SumTree(2, dtype='uint8')

        st.add(1, 1)
        st.add(1, 2)
        st.add(0.5, 3)

        self.assertEqual(st.total(), 1.5)
        self.assertEqual(st.get(0.25), 3)
        self.assertEqual(st.get(1), 2)


    def test_sum_tree_overwrite_2(self):
        st = SumTree(2, dtype='uint8')

        st.add(1, 1)
        st.add(1, 2)
        st.add(0.5, 3)
        st.add(0.5, 4)

        self.assertEqual(st.total(), 1)
        self.assertEqual(st.get(0.5), 3)
        self.assertEqual(st.get(1), 4)


    def test_sum_tree_get_with_info(self):
        st = SumTree(2, dtype='uint8')

        st.add(1, 1)
        st.add(1, 2)
        self.assertEqual(st.total(), 2)

        data = st.get(1.1)
        self.assertEqual(data, 2)

        tree_idx, data_idx, node, data = st.get_with_info(1.0)

        self.assertEqual(data_idx, 0)
        self.assertEqual(data, 1)

        st.update_score(tree_idx, 2)

        self.assertEqual(st.total(), 3)

        data = st.get(1.1)
        self.assertEqual(data, 1)


    def test_replay_sampler_overwrite_1(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplaySampler(ReplayMemoryDisk(data_fn,
                                            1,
                                            1,
                                            1,
                                            max_size=2,
                                            cache_size=0))

        for i in range(3):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        self.assertEqual(len(rp), 2)
        # self.assertEqual(rp[0][1], 2)
        # self.assertAlmostEqual(float(rp[0][5]), 0.02, places=3)
        # self.assertEqual(rp[1][1], 1)
        # self.assertAlmostEqual(float(rp[1][5]), 0.01, places=3)

        self.assertEqual(rp[0]['action'], 2)
        self.assertAlmostEqual(float(rp[0]['loss']), 0.02, places=3)
        self.assertEqual(rp[1]['action'], 1)
        self.assertAlmostEqual(float(rp[1]['loss']), 0.01, places=3)


    def test_replay_sampler_overwrite_2(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplaySampler(ReplayMemoryDisk(data_fn,
                                            1,
                                            1,
                                            1,
                                            max_size=2,
                                            cache_size=2))

        i = 0
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        self.assertEqual(rp[0]['action'], 0)
        self.assertEqual(len(rp), 1)

        i = 1
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        self.assertEqual(rp[1]['action'], 1)
        self.assertEqual(len(rp), 2)

        i = 2
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        i = 3
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        self.assertEqual(len(rp), 2)
        self.assertEqual(rp[0]['action'], 2)
        self.assertEqual(rp[1]['action'], 3)


    def test_replay_sampler_overwrite_3(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplaySampler(ReplayMemoryDisk(data_fn,
                                            1,
                                            1,
                                            1,
                                            max_size=2,
                                            cache_size=0))

        for i in range(5):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        self.assertEqual(len(rp), 2)


        for i, val in enumerate([4, 3]):
            self.assertEqual(rp[i]['state'], val)
            self.assertEqual(rp[i]['action'], val)
            self.assertEqual(rp[i]['reward'], val)
            self.assertEqual(rp[i]['next_state'], val)
            self.assertEqual(rp[i]['cont'], val > 0)
            self.assertAlmostEqual(rp[i]['loss'], val * 0.01, places=3)


    def test_replay_sampler_sample_1(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)
        rp = ReplaySampler(ReplayMemoryDisk(data_fn,
                                            1,
                                            1,
                                            1,
                                            max_size=2,
                                            cache_size=2))

        i = 5
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        for i in range(3):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        # print('total:', rp.sum_tree.total())
        self.assertAlmostEqual(rp.sum_tree.total(), 0.03, places=2)

        batch_size = 6
        # target = (np.zeros((batch_size, 1, 1, 1), dtype='uint8'),
        #           np.zeros((batch_size,), dtype='uint8'),
        #           np.zeros((batch_size,), dtype='uint8'),
        #           np.zeros((batch_size, 1, 1, 1), dtype='uint8'),
        #           np.zeros((batch_size,), dtype='bool'),
        #           np.zeros((batch_size,), dtype='float16'))
        target = ReplayMemory(1, 1, 1, max_size=batch_size)


        values = {}
        count = 0
        num_batches = 0
        for i in range(100):
            rp.sample_memories(target, batch_size=batch_size)

            for j in range(batch_size):
                val = target[j]['action']

                if val not in values:
                    values[val] = 0
                    count += 1
                else:
                    values[val] += 1

            num_batches += 1

        total = sum([val for val in values.values()])

        self.assertAlmostEqual(values[2] / total, 0.66, places=1)
        self.assertAlmostEqual(values[1] / total, 0.33, places=1)


    def test_replay_sampler_sample_2(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplaySampler(ReplayMemoryDisk(data_fn,
                                            1,
                                            1,
                                            1,
                                            max_size=10,
                                            cache_size=1))

        num_values = 10
        for i in range(num_values):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        batch_size = 6

        target = ReplayMemory(1, 1, 1, batch_size)

        values = {}
        count = 0
        num_batches = 0

        while len(values) < num_values:
            if num_batches > 10:
                break

            rp.sample_memories(target, batch_size=batch_size)

            for i in range(batch_size):
                val = target.actions[i]

                if val not in values:
                    values[val] = 0
                    count += 1

                values[val] += 1

                self.assertEqual(target.states[i][0][0][0], val)
                self.assertEqual(target.actions[i], val)
                self.assertEqual(target.rewards[i], val)
                self.assertEqual(target.next_states[i][0][0], val)
                self.assertEqual(target.continues[i], val > 0)
                self.assertAlmostEqual(target.losses[i], val * 0.01, places=2)

            num_batches += 1

        # for key, val in values.items():
        #     print(key, val)
        self.assertGreater(len(values), num_values - 3)


    def test_replay_sampler_update_1(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplaySampler(ReplayMemoryDisk(data_fn,
                                            1,
                                            1,
                                            1,
                                            max_size=2,
                                            cache_size=2))

        i = 5
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        self.assertEqual(rp.sum_tree.data[rp.sum_tree.get_data_idx(1)], 0)
        self.assertEqual(rp.sum_tree.data[rp.sum_tree.get_data_idx(2)], 1)

        for i in range(3):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        self.assertAlmostEqual(rp.sum_tree.total(), 0.03, places=2)
        self.assertAlmostEqual(rp.replay_memory.data_file['losses'][0], 0.02, places=2)
        self.assertAlmostEqual(rp.replay_memory.data_file['losses'][1], 0.01, places=2)


        batch_size = 6
        target = ReplayMemory(1, 1, 1, batch_size)

        tree_idxes = []
        rp.sample_memories(target,
                           batch_size=batch_size,
                           tree_idxes=tree_idxes)

        self.assertTrue(len(tree_idxes) > 0)

        # print(tree_idxes)
        # indexes = {}
        # for i in range(6):
        #     print(target[i]['action'], target[i]['loss'], tree_idxes[i])
        #     indexes[target[i]['action']] = tree_idxes[i]


        rp.update_sum_tree([1, 2], [0.4, 0.6])

        self.assertEqual(rp.sum_tree.total(), 1.0)

        values = {}
        count = 0
        num_batches = 0
        for i in range(100):
            rp.sample_memories(target, batch_size=batch_size, tree_idxes=tree_idxes)
            for i in range(batch_size):
                val = target[1]['action']

                if val not in values:
                    values[val] = 0
                    count += 1
                else:
                    values[val] += 1

            num_batches += 1

        total = sum([val for val in values.values()])

        self.assertAlmostEqual(values[2] / total, 0.4, places=1)
        self.assertAlmostEqual(values[1] / total, 0.6, places=1)

        self.assertAlmostEqual(rp.replay_memory.data_file['losses'][1], 0.6, places=1)
        self.assertAlmostEqual(rp.replay_memory.data_file['actions'][1], 1, places=1)
        self.assertAlmostEqual(rp.replay_memory.data_file['losses'][0], 0.4, places=1)
        self.assertAlmostEqual(rp.replay_memory.data_file['actions'][0], 2, places=1)


    def test_replay_sampler_reload_2(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplaySampler(ReplayMemoryDisk(data_fn,
                                            1,
                                            1,
                                            1,
                                            max_size=2,
                                            cache_size=2))

        i = 5
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        # print(rp.sum_tree.data[rp.sum_tree.get_data_idx(0)])
        # print('-->', rp.sum_tree.data[rp.sum_tree.get_data_idx(1)])
        # print('-->', rp.sum_tree.data[rp.sum_tree.get_data_idx(2)])

        self.assertEqual(rp.sum_tree.data[rp.sum_tree.get_data_idx(1)], 0)
        self.assertEqual(rp.sum_tree.data[rp.sum_tree.get_data_idx(2)], 1)

        for i in range(3):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

            # print(i, '-->', rp.sum_tree.data[rp.sum_tree.get_data_idx(1)])
            # print(i, '-->', rp.sum_tree.data[rp.sum_tree.get_data_idx(2)])

        # print('total:', rp.sum_tree.total())
        self.assertAlmostEqual(rp.sum_tree.total(), 0.03, places=2)

        self.assertAlmostEqual(rp.replay_memory.data_file['losses'][0], 0.02, places=2)
        self.assertAlmostEqual(rp.replay_memory.data_file['losses'][1], 0.01, places=2)

        batch_size = 6
        target = ReplayMemory(1, 1, 1, batch_size)
        # target = (np.zeros((batch_size, 1, 1, 1), dtype='uint8'),
        #           np.zeros((batch_size,), dtype='uint8'),
        #           np.zeros((batch_size,), dtype='uint8'),
        #           np.zeros((batch_size, 1, 1, 1), dtype='uint8'),
        #           np.zeros((batch_size,), dtype='bool'),
        #           np.zeros((batch_size,), dtype='float16'))

        tree_idxes = []
        rp.sample_memories(target,
                           batch_size=batch_size,
                           tree_idxes=tree_idxes)

        self.assertTrue(len(tree_idxes) > 0)

        # print(tree_idxes)
        # for i in range(6):
        #     print(target[1][i], target[5][i], tree_idxes[i])

        # self.assertEqual(rp.sum_tree.data[rp.sum_tree.get_data_idx(2)], 2)
        # self.assertEqual(rp.sum_tree.data[rp.sum_tree.get_data_idx(1)], 1)

        rp.update_sum_tree([1, 2], [0.4, 0.6])



        rp.close()

        rp2 = ReplaySampler(ReplayMemoryDisk(data_fn,
                                             1,
                                             1,
                                             1,
                                             max_size=2,
                                            cache_size=2))

        self.assertEqual(rp2.sum_tree.total(), 1.0)

        values = {}
        count = 0
        num_batches = 0
        for i in range(100):
            rp2.sample_memories(target, batch_size=batch_size, tree_idxes=tree_idxes)
            for i in range(batch_size):
                val = target[1]['action']

                if val not in values:
                    values[val] = 0
                    count += 1
                else:
                    values[val] += 1

            num_batches += 1

        total = sum([val for val in values.values()])

        self.assertAlmostEqual(values[2] / total, 0.4, places=1)
        self.assertAlmostEqual(values[1] / total, 0.6, places=1)

        self.assertAlmostEqual(rp2.replay_memory.data_file['losses'][1], 0.6, places=1)
        self.assertAlmostEqual(rp2.replay_memory.data_file['actions'][1], 1, places=1)
        self.assertAlmostEqual(rp2.replay_memory.data_file['losses'][0], 0.4, places=1)
        self.assertAlmostEqual(rp2.replay_memory.data_file['actions'][0], 2, places=1)


    def test_replay_sampler_reload_1(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplaySampler(ReplayMemoryDisk(data_fn,
                                            1,
                                            1,
                                            1,
                                            max_size=2,
                                            cache_size=2))

        i = 5
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        for i in range(3):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        rp.close()
        rp = None

        rp2 = ReplaySampler(ReplayMemoryDisk(data_fn,
                                            1,
                                            1,
                                            1,
                                            max_size=2,
                                            cache_size=2))

        self.assertAlmostEqual(rp2.sum_tree.total(), 0.03, places=2)

        batch_size = 6
        target = ReplayMemory(1, 1, 1, batch_size)

        values = {}
        count = 0
        num_batches = 0
        for i in range(100):
            rp2.sample_memories(target, batch_size=batch_size)

            for i in range(batch_size):
                val = target[i]['action']

                if val not in values:
                    values[val] = 0
                    count += 1
                else:
                    values[val] += 1

            num_batches += 1

        total = sum([val for val in values.values()])

        self.assertAlmostEqual(values[2] / total, 0.66, places=1)
        self.assertAlmostEqual(values[1] / total, 0.33, places=1)


if __name__ == '__main__':
    TestData.do()

