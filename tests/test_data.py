import os
import test_base
import numpy as np


test_base.TestBase.add_lib_dir_sys_path()


from rl.data.replay_memory import ReplayMemory
from rl.data.replay_cache import ReplayCache
from rl.data.sum_tree import SumTree
from rl.data.replay_sampler import ReplaySampler


class TestData(test_base.TestBase):
    def test_replay_memory_append(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemory(data_fn,
                          1,
                          1,
                          1,
                          max_size=2)

        for i in range(2):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

            self.assertEqual(len(rp), i+1)
            self.assertEqual(rp[rp.last_index][1], i)
            self.assertEqual(rp[rp.last_index][2], i)
            self.assertEqual(rp[rp.last_index][3], i)
            self.assertEqual(rp[rp.last_index][4], i)
            self.assertAlmostEqual(rp[rp.last_index][5], i * 0.01, places=3)


    def test_replay_memory_overwrite_1(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemory(data_fn,
                          1,
                          1,
                          1,
                          max_size=2,
                          cache_size=0)

        for i in range(3):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)


        self.assertEqual(len(rp), 2)
        self.assertEqual(rp[0][1], 1)
        self.assertAlmostEqual (float(rp[0][5]), 0.01, places=3)
        self.assertEqual(rp[1][1], 2)
        self.assertAlmostEqual(float(rp[1][5]), 0.02, places=3)


    def test_replay_memory_overwrite_2(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemory(data_fn,
                          1,
                          1,
                          1,
                          max_size=2,
                          cache_size=5)

        i = 0
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        i = 1
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        self.assertEqual(rp[1][1], 1)
        self.assertEqual(rp[0][1], 0)
        i = 2
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)


        self.assertEqual(rp[0][1], 1)
        self.assertEqual(rp[1][1], 2)


    def test_replay_memory_overwrite_2(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemory(data_fn,
                          1,
                          1,
                          1,
                          max_size=2,
                          cache_size=5)

        for i in range(5):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        self.assertEqual(rp[0][1], 3)
        self.assertEqual(rp[1][1], 4)


    def test_replay_memory_cache_1(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemory(data_fn,
                          1,
                          1,
                          1,
                          max_size=10,
                          cache_size=1)

        for i in range(10):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        self.assertEqual(rp[0][1], 0)
        self.assertEqual(rp[1][1], 1)
        self.assertEqual(rp[2][1], 2)
        self.assertEqual(rp[0][1], 0)


    def test_replay_memory_cache_2(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemory(data_fn,
                          1,
                          1,
                          1,
                          max_size=2,
                          cache_size=2)

        i = 0
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        self.assertEqual(rp[0][1], 0)

        i = 1
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        self.assertEqual(rp[1][1], 1)

        i = 2
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        i = 3
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        self.assertEqual(rp[0][1], 2)
        self.assertEqual(rp[1][1], 3)


    def test_replay_memory_sample_1(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemory(data_fn,
                          1,
                          1,
                          1,
                          max_size=10,
                          cache_size=1)

        num_values = 10
        for i in range(num_values):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)


        batch_size = 6
        target = (np.zeros((batch_size, 1, 1, 1), dtype='uint8'),
                  np.zeros((batch_size,), dtype='uint8'),
                  np.zeros((batch_size,), dtype='uint8'),
                  np.zeros((batch_size, 1, 1, 1), dtype='uint8'),
                  np.zeros((batch_size,), dtype='bool'),
                  np.zeros((batch_size,), dtype='float16'))

        values = {}
        count = 0
        num_batches = 0
        while len(values) < num_values:
            if num_batches > 10:
                break

            rp.sample_memories(*target, batch_size=batch_size)

            for i in range(batch_size):
                val = target[1][i]

                if val not in values:
                    values[val] = True
                    count += 1

                self.assertGreater(target[5][i], -0.001)
                self.assertLess(target[5][i], 0.10)

            num_batches += 1

        self.assertEqual(len(values), num_values)


    def test_replay_memory_sample_2(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplayMemory(data_fn,
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

        states = np.zeros((batch_size,
                           1,
                           1,
                           1), dtype='uint8')
        actions = np.zeros((batch_size,), dtype='uint8')
        rewards = np.zeros((batch_size,), dtype='uint8')
        next_states = np.zeros((batch_size,
                                1,
                                1,
                                1), dtype='uint8')
        continues = np.zeros((batch_size,), dtype='bool')
        losses = np.zeros((batch_size,), dtype='float16')

        values = {}
        count = 0
        num_batches = 0

        while len(values) < num_values:
            if num_batches > 10:
                break

            rp.sample_memories(states,
                               actions,
                               rewards,
                               next_states,
                               continues,
                               losses,
                               batch_size=batch_size)

            for i in range(batch_size):
                val = states[i][0][0][0]

                if val not in values:
                    values[val] = True
                    count += 1

                self.assertEqual(states[i][0][0][0], val)
                self.assertEqual(actions[i], val+1)
                self.assertEqual(rewards[i], val+2)
                self.assertEqual(next_states[i][0][0][0], val+3)
                self.assertEqual(continues[i], val+4 > 0)
                self.assertAlmostEqual(losses[i], (val+5) * 0.01, places=2)

            num_batches += 1

        self.assertEqual(len(values), num_values)


    def test_game_memory_cache_1(self):
        rp = ReplayCache(  1,
                          1,
                          1,
                          max_size=2)

        i = 0
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        self.assertEqual(rp[0][1], 0)

        i = 1
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        self.assertEqual(rp[1][1], 1)

        i = 2
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        i = 3
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        self.assertEqual(len(rp), 2)
        self.assertEqual(rp[0][1], 2)
        self.assertEqual(rp[1][1], 3)


    def test_game_memory_cache_2(self):
        rp = ReplayCache(1,
                         1,
                         1,
                         max_size=2)

        i = 0

        for i in range(2):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)


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

        rp = ReplaySampler(data_fn,
                           1,
                           1,
                           1,
                           max_size=2,
                           cache_size=0)

        for i in range(3):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        self.assertEqual(len(rp), 2)
        # self.assertEqual(rp[0][1], 2)
        # self.assertAlmostEqual(float(rp[0][5]), 0.02, places=3)
        # self.assertEqual(rp[1][1], 1)
        # self.assertAlmostEqual(float(rp[1][5]), 0.01, places=3)

        self.assertEqual(rp[0][1], 1)
        self.assertAlmostEqual(float(rp[0][5]), 0.01, places=3)
        self.assertEqual(rp[1][1], 2)
        self.assertAlmostEqual(float(rp[1][5]), 0.02, places=3)


    def test_replay_sampler_overwrite_2(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplaySampler(data_fn,
                           1,
                           1,
                           1,
                           max_size=2,
                           cache_size=2)

        i = 0
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        self.assertEqual(rp[0][1], 0)
        self.assertEqual(len(rp), 1)

        i = 1
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        self.assertEqual(rp[1][1], 1)
        self.assertEqual(len(rp), 2)

        i = 2
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        i = 3
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        self.assertEqual(len(rp), 2)
        self.assertEqual(rp[0][1], 2)
        self.assertEqual(rp[1][1], 3)


    def test_replay_sampler_overwrite_3(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplaySampler(data_fn,
                           1,
                           1,
                           1,
                           max_size=2,
                           cache_size=0)

        for i in range(5):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        self.assertEqual(len(rp), 2)
        # self.assertEqual(rp[0][1], 4)
        # self.assertAlmostEqual(float(rp[0][5]), 0.04, places=3)
        # self.assertEqual(rp[1][1], 3)
        # self.assertAlmostEqual(float(rp[1][5]), 0.03, places=3)

        self.assertEqual(rp[0][1], 3)
        self.assertAlmostEqual(float(rp[0][5]), 0.03, places=3)
        self.assertEqual(rp[1][1], 4)
        self.assertAlmostEqual(float(rp[1][5]), 0.04, places=3)


    def test_replay_sampler_sample_1(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplaySampler(data_fn,
                           1,
                           1,
                           1,
                           max_size=2,
                           cache_size=2)

        i = 5
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        for i in range(3):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        # print('total:', rp.sum_tree.total())
        self.assertAlmostEqual(rp.sum_tree.total(), 0.03, places=2)

        batch_size = 6
        target = (np.zeros((batch_size, 1, 1, 1), dtype='uint8'),
                  np.zeros((batch_size,), dtype='uint8'),
                  np.zeros((batch_size,), dtype='uint8'),
                  np.zeros((batch_size, 1, 1, 1), dtype='uint8'),
                  np.zeros((batch_size,), dtype='bool'),
                  np.zeros((batch_size,), dtype='float16'))


        values = {}
        count = 0
        num_batches = 0
        for i in range(100):
            rp.sample_memories(*target, batch_size=batch_size)

            for i in range(batch_size):
                val = target[1][i]

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

        rp = ReplaySampler(data_fn,
                          1,
                          1,
                          1,
                          max_size=10,
                          cache_size=1,
                          state_type='uint8')

        num_values = 10
        for i in range(num_values):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        batch_size = 6

        states = np.zeros((batch_size,
                           1,
                           1,
                           1), dtype='uint8')
        actions = np.zeros((batch_size,), dtype='uint8')
        rewards = np.zeros((batch_size,), dtype='uint8')
        next_states = np.zeros((batch_size,
                                1,
                                1,
                                1), dtype='uint8')
        continues = np.zeros((batch_size,), dtype='bool')
        losses = np.zeros((batch_size,), dtype='float16')

        values = {}
        count = 0
        num_batches = 0

        while len(values) < num_values:
            if num_batches > 10:
                break

            rp.sample_memories(states,
                               actions,
                               rewards,
                               next_states,
                               continues,
                               losses,
                               batch_size=batch_size)

            for i in range(batch_size):
                val = actions[i]

                if val not in values:
                    values[val] = 0
                    count += 1

                values[val] += 1

                self.assertEqual(states[i][0][0][0], val)
                self.assertEqual(actions[i], val)
                self.assertEqual(rewards[i], val)
                self.assertEqual(next_states[i][0][0], val)
                self.assertEqual(continues[i], val > 0)
                self.assertAlmostEqual(losses[i], val * 0.01, places=2)

            num_batches += 1

        # for key, val in values.items():
        #     print(key, val)
        self.assertGreater(len(values), num_values - 3)


    def test_replay_sampler_update_1(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplaySampler(data_fn,
                           1,
                           1,
                           1,
                           max_size=2,
                           cache_size=2)

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

        self.assertAlmostEqual(rp.replay_memory.data_file['loss'][0], 0.01, places=2)
        self.assertAlmostEqual(rp.replay_memory.data_file['loss'][1], 0.02, places=2)
        self.assertAlmostEqual(rp.replay_memory.data_file['loss'][2], 0.0, places=2)


        batch_size = 6
        target = (np.zeros((batch_size, 1, 1, 1), dtype='uint8'),
                  np.zeros((batch_size,), dtype='uint8'),
                  np.zeros((batch_size,), dtype='uint8'),
                  np.zeros((batch_size, 1, 1, 1), dtype='uint8'),
                  np.zeros((batch_size,), dtype='bool'),
                  np.zeros((batch_size,), dtype='float16'))

        tree_idxes = []
        rp.sample_memories(*target,
                           batch_size=batch_size,
                           tree_idxes=tree_idxes)

        self.assertTrue(len(tree_idxes) > 0)

        # print(tree_idxes)
        # for i in range(6):
        #     print(target[1][i], target[5][i], tree_idxes[i])

        # self.assertEqual(rp.sum_tree.data[rp.sum_tree.get_data_idx(2)], 2)
        # self.assertEqual(rp.sum_tree.data[rp.sum_tree.get_data_idx(1)], 1)

        rp.update_sum_tree([1, 2], [0.4, 0.6])

        self.assertEqual(rp.sum_tree.total(), 1.0)

        values = {}
        count = 0
        num_batches = 0
        for i in range(100):
            rp.sample_memories(*target, batch_size=batch_size, tree_idxes=tree_idxes)
            for i in range(batch_size):
                val = target[1][i]

                if val not in values:
                    values[val] = 0
                    count += 1
                else:
                    values[val] += 1

            num_batches += 1

        total = sum([val for val in values.values()])

        self.assertAlmostEqual(values[2] / total, 0.4, places=1)
        self.assertAlmostEqual(values[1] / total, 0.6, places=1)

        self.assertAlmostEqual(rp.replay_memory.data_file['loss'][0], 0.6, places=1)
        self.assertAlmostEqual(rp.replay_memory.data_file['action'][0], 1, places=1)
        self.assertAlmostEqual(rp.replay_memory.data_file['loss'][1], 0.4, places=1)
        self.assertAlmostEqual(rp.replay_memory.data_file['action'][1], 2, places=1)


    def test_replay_sampler_reload_2(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplaySampler(data_fn,
                           1,
                           1,
                           1,
                           max_size=2,
                           cache_size=2)

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

        self.assertAlmostEqual(rp.replay_memory.data_file['loss'][0], 0.01, places=2)
        self.assertAlmostEqual(rp.replay_memory.data_file['loss'][1], 0.02, places=2)
        self.assertAlmostEqual(rp.replay_memory.data_file['loss'][2], 0.0, places=2)

        batch_size = 6
        target = (np.zeros((batch_size, 1, 1, 1), dtype='uint8'),
                  np.zeros((batch_size,), dtype='uint8'),
                  np.zeros((batch_size,), dtype='uint8'),
                  np.zeros((batch_size, 1, 1, 1), dtype='uint8'),
                  np.zeros((batch_size,), dtype='bool'),
                  np.zeros((batch_size,), dtype='float16'))

        tree_idxes = []
        rp.sample_memories(*target,
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


        rp2 = ReplaySampler(data_fn,
                                 1,
                                 1,
                                 1,
                                 max_size=2,
                                 cache_size=2)

        self.assertEqual(rp2.sum_tree.total(), 1.0)

        values = {}
        count = 0
        num_batches = 0
        for i in range(100):
            rp2.sample_memories(*target, batch_size=batch_size, tree_idxes=tree_idxes)
            for i in range(batch_size):
                val = target[1][i]

                if val not in values:
                    values[val] = 0
                    count += 1
                else:
                    values[val] += 1

            num_batches += 1

        total = sum([val for val in values.values()])

        self.assertAlmostEqual(values[2] / total, 0.4, places=1)
        self.assertAlmostEqual(values[1] / total, 0.6, places=1)

        self.assertAlmostEqual(rp2.replay_memory.data_file['loss'][0], 0.6, places=1)
        self.assertAlmostEqual(rp2.replay_memory.data_file['action'][0], 1, places=1)
        self.assertAlmostEqual(rp2.replay_memory.data_file['loss'][1], 0.4, places=1)
        self.assertAlmostEqual(rp2.replay_memory.data_file['action'][1], 2, places=1)


    def test_replay_sampler_reload_1(self):
        data_fn = os.path.join(self.get_data_dir(), 'replay_test.hdf5')

        if os.path.exists(data_fn):
            os.unlink(data_fn)

        rp = ReplaySampler(data_fn,
                           1,
                           1,
                           1,
                           max_size=2,
                           cache_size=2)

        i = 5
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)
        rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        for i in range(3):
            rp.append(np.array([[[i]]]), i, i, np.array([[[i]]]), i, i * 0.01)

        rp.close()
        rp = None

        rp2 = ReplaySampler(data_fn,
                            1,
                            1,
                            1,
                            max_size=2,
                            cache_size=2)

        self.assertAlmostEqual(rp2.sum_tree.total(), 0.03, places=2)

        batch_size = 6
        target = (np.zeros((batch_size, 1, 1, 1), dtype='uint8'),
                  np.zeros((batch_size,), dtype='uint8'),
                  np.zeros((batch_size,), dtype='uint8'),
                  np.zeros((batch_size, 1, 1, 1), dtype='uint8'),
                  np.zeros((batch_size,), dtype='bool'),
                  np.zeros((batch_size,), dtype='float16'))

        values = {}
        count = 0
        num_batches = 0
        for i in range(100):
            rp2.sample_memories(*target, batch_size=batch_size)

            for i in range(batch_size):
                val = target[1][i]

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

