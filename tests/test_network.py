import os
import shutil
import numpy as np
import test_base


test_base.TestBase.add_lib_dir_sys_path()


from rl.deep_q_network import DeepQNetwork


class TestNetwork(test_base.TestBase):
    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(os.path.abspath(__file__))
        cls.data_dir = os.path.join(cls.cur_dir, 'data')


    def setUp(self):
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)

        os.makedirs(self.data_dir)

        self.conf = self.get_conf()
        self.network = DeepQNetwork(conf=self.conf,
                                    initialize=True)


    def get_conf(self):
        return {
            'game_id': 'Breakout-v0',
            'agent': 'rl.game_agent.BreakoutAgent',
            'num_game_frames_before_training': 100,
            'save_dir': self.data_dir,
            'use_memory': True,
            'save_model_steps': 8,
            'replay_max_memory_length': 1000,
            'batch_size': 1
        }


    def test_train_init(self):
        self.network.sess = self.network.get_session(init_model=True)

        with self.network.sess:
            self.network.train_init()

        self.assertGreater(len(self.network.memories), 100)


    def test_train_step(self):
        self.network.sess = self.network.get_session(init_model=True)

        with self.network.sess:
            self.network.train_init()

            old_count = len(self.network.memories)

            num_train_steps = int(self.network.batch_size / self.network.num_game_steps_per_train)
            for i in range(num_train_steps):
                self.network.train_step()

            self.assertEqual(self.network.step, num_train_steps)
            self.assertEqual(len(self.network.total_losses), num_train_steps)
            self.assertEqual(len(self.network.memories), old_count + self.network.batch_size)
            self.assertTrue(os.path.exists(os.path.join(self.data_dir, '{}.log'.format(self.network.game_id))))
            self.assertTrue(os.path.exists(os.path.join(self.data_dir, '{}.meta'.format(self.network.game_id))))


    def test_play_game(self):
        self.network.sess = self.network.get_session(init_model=True)

        with self.network.sess:
            self.network.train_init(init_play_step=False, fill_memories=False)
            self.network.play_init()

            num_steps = 0

            game_state = self.network.make_game_state()
            for _ in self.network.play_game_generator(game_state=game_state):
                num_steps += 1

            self.assertGreater(num_steps, 50)
            self.assertEqual(len(self.network.play_game_scores), 1)
            self.assertEqual(len(self.network.play_max_qs), 1)
            self.assertGreater(len(self.network.memories), 0)
            self.assertEqual(len(self.network.memories) + 1, num_steps)
            self.assertEqual(len(self.network.play_batch), 0)
            self.assertEqual(game_state['episode_done'], True)
            self.assertEqual(game_state['game_done'], True)
            self.assertEqual(game_state['num_lives'], 0)

            false_count = 0
            for i in range(len(self.network.memories)):
                if not self.network.memories.continues[i]:
                    false_count += 1

            self.assertEqual(false_count, 5)


    def test_play_episode(self):
        self.network.sess = self.network.get_session(init_model=True)

        with self.network.sess:
            self.network.train_init(init_play_step=False, fill_memories=False)
            self.network.play_init()

            num_steps = 0

            game_state = self.network.make_game_state()
            for _ in self.network.play_episode_generator(game_state=game_state):
                num_steps += 1

            self.assertGreater(num_steps, 10)
            self.assertLess(num_steps, 10000)
            self.assertEqual(len(self.network.play_game_scores), 0)
            self.assertEqual(len(self.network.play_max_qs), 0)
            self.assertGreater(len(self.network.memories), 0)
            self.assertEqual(len(self.network.memories) - 1, num_steps)
            self.assertEqual(len(self.network.play_batch), 0)
            self.assertEqual(game_state['episode_done'], True)
            self.assertEqual(game_state['game_done'], False)
            self.assertEqual(game_state['num_lives'], 4)

            for i in range(len(self.network.memories)):
                self.assertEqual(self.network.memories.continues[i], 1)


if __name__ == '__main__':
    TestNetwork.do()

