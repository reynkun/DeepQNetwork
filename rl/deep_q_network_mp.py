import math
import re
import os
import time
import random
import datetime
import multiprocessing
import pickle
import signal
import json

from collections import deque

import gym
import tensorflow as tf
import numpy as np

from .game.render import render_game
from .data.replay_memory import ReplayMemoryDisk
from .data.replay_cache import ReplayMemory
from .data.replay_sampler import ReplaySampler
from .data.sum_tree import SumTree


def time_string():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


DEFAULT_OPTIONS = {
    'save_dir': './models',
    'eps_min': 0.1,
    'eps_max': 1.0,
    'eps_decay_steps': 2000000,
    'discount_rate': 0.99,
    'mem_save_size': 10000,
    # 'mem_save_size': 1000,
    'batch_size': 32,
    'testing_predict': False,
    'testing_train': False,
    'testing_weights': False,
    'model_save_prefix': None,
    'replay_max_memory_length': 2000000,
    'replay_cache_size': 300000,
    'max_num_training_steps': 20000000,
    'num_game_frames_before_training': 10000,
    'game_report_interval': 10,
    'train_report_interval': 100,
    'game_render_interval': 20000,
    'use_episodes': True,
    'use_double': True,
    'use_priority': False,
    'use_momentum': False,
    'frame_skip': 1,
    'tf_log_level': 3
}


class DeepQNetwork:
    def __init__(self, 
                 game_id,
                 model_class, 
                 options=None):
        self.header = 'init'
        self.save_dir = options['save_dir']
        if options['model_save_prefix'] is not None:
            self.file_prefix = '{}'.format(options['model_save_prefix'])
        else:
            self.file_prefix = game_id

        # create save dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # load conf file if exists
        path = os.path.join(self.save_dir, '{}.conf'.format(self.file_prefix))
        if os.path.exists(path):
            with open(path) as fin:
                self.options = json.load(fin)
                self.log(json.dumps(self.options, sort_keys=True, indent=4))
                self.log('options loaded from:', path)
        else:
            self.options = DEFAULT_OPTIONS

        # override saved options with parameters
        for key, value in options.items():
            if value is not None:
                self.options[key] = value

        # add missing keys if not there
        for key, value in DEFAULT_OPTIONS.items():
            if key not in self.options:
                self.options[key] = value

        # now write final options
        with open(path, 'w+') as fo:
            self.log(json.dumps(self.options, sort_keys=True, indent=4))
            self.log('saving options to:', path)

            json.dump(self.options, fo, sort_keys=True, indent=4)


        self.model_class = model_class
        self.game_id = game_id

        if self.options['model_save_prefix'] is None:
            self.options['model_save_prefix'] = self.game_id

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(self.options['tf_log_level'])

        self.save_path_prefix = os.path.join(self.save_dir, self.file_prefix)
        self.memory_file_prefix = '{}.memory'.format(self.file_prefix)

        env = gym.make(self.game_id)
        self.num_outputs = env.action_space.n

        # run game settings
        self.eps_min = self.options['eps_min']
        self.eps_max = self.options['eps_max']
        self.eps_decay_steps = self.options['eps_decay_steps']
        self.discount_rate = self.options['discount_rate']
        self.mem_save_size = self.options['mem_save_size']
        self.use_episodes = self.options['use_episodes']
        self.frame_skip = self.options['frame_skip']

        # train settings
        self.max_num_training_steps = self.options['max_num_training_steps']
        self.replay_max_memory_length = self.options['replay_max_memory_length']
        self.num_game_frames_before_training = self.options['num_game_frames_before_training']
        self.batch_size = self.options['batch_size']
        self.game_render_interval = self.options['game_render_interval']

        # interprocess communication
        self.memory_queue = multiprocessing.Queue()
        self.save_count = multiprocessing.Value('i', 0)
        self.game_count = multiprocessing.Value('i', 0)

        # testing flags
        self.testing_predict = self.options['testing_predict']
        self.testing_train = self.options['testing_train']
        self.testing_weights = self.options['testing_weights']


    def train(self):
        p_t = multiprocessing.Process(target=self.train_func)
        p_t.start()

        while True:
            g_t = multiprocessing.Process(target=self.run_game_func, 
                                          kwargs={'is_training': True})
            g_t.start()

            while g_t.is_alive():
                if not p_t.is_alive():
                    self.log('train process not not alive!')
                    os.kill(g_t.pid, signal.SIGTERM)
                    raise Exception('train process not alive')
                g_t.join(5)

            if g_t.exitcode != 0:
                self.log('game process nonzero exit!')
                os.kill(p_t.pid, signal.SIGINT)
                break

        p_t.join()


    def train_func(self):
        self.header = 'train'
        env = gym.make(self.game_id)
        env.seed(int(time.time()))

        self.delete_old_memories()

        start_time = time.time()

        with self.get_session(load_model=True, save_model=True, env=env) as sess:
            save_steps = 10000
            copy_steps = 10000
            train_report_interval = self.options['train_report_interval']

            report_start_time = time.time()
            step = sess.model.step.eval()
            report_last_step = step
            total_losses = []

            states = np.zeros((self.batch_size,
                               sess.model.input_height,
                               sess.model.input_width,
                               sess.model.input_channels), dtype=sess.model.state_type)
            actions = np.zeros((self.batch_size,), dtype='uint8')
            rewards = np.zeros((self.batch_size,), dtype='uint8')
            next_states = np.zeros((self.batch_size,
                                    sess.model.input_height,
                                    sess.model.input_width,
                                    sess.model.input_channels), dtype=sess.model.state_type)
            continues = np.zeros((self.batch_size,), dtype='bool')
            losses = np.zeros((self.batch_size,), dtype='float16')

            # replay_batch = (np.zeros((self.batch_size,
            #                           sess.model.input_height,
            #                           sess.model.input_width,
            #                           sess.model.input_channels), dtype='uint8'),
            #                 np.zeros((self.batch_size,), dtype='uint8'),
            #                 np.zeros((self.batch_size,), dtype='uint8'),
            #                 np.zeros((self.batch_size,
            #                           sess.model.input_height,
            #                           sess.model.input_width,
            #                           sess.model.input_channels), dtype='uint8'),
            #                 np.zeros((self.batch_size,), dtype='bool'),
            #                 np.zeros((self.batch_size,), dtype='float16'))


            # allocate memory
            if self.options['use_priority']:
                ReplayClass = ReplaySampler
            else:
                ReplayClass = ReplayMemoryDisk

            self.replay_memory = ReplayClass(os.path.join(self.options['save_dir'],
                                                          '{}_replay_memory.hdf5'.format(
                                                          self.options['model_save_prefix'])),
                                             sess.model.input_height,
                                             sess.model.input_width,
                                             sess.model.input_channels,
                                             state_type=sess.model.state_type,
                                             max_size=self.replay_max_memory_length,
                                             cache_size=self.options['replay_cache_size'])

            replay_memory_size = len(self.replay_memory)


            while replay_memory_size < self.num_game_frames_before_training:
                self.log('waiting for memory',
                      self.num_game_frames_before_training,
                      'cur size:',
                      replay_memory_size)

                fn = self.memory_queue.get()
                self.load_memory(fn)

                replay_memory_size = len(self.replay_memory)
                # if self.options['use_priority']:
                #     replay_memory_size = len(self.replay_sum_tree)
                # else:


            try:
                self.log('start training')

                while step < self.max_num_training_steps:
                    step = sess.model.step.eval()

                    # check for new game replay memories
                    while self.memory_queue.qsize():
                        fn = self.memory_queue.get()
                        self.load_memory(fn)

                        replay_memory_size = len(self.replay_memory)

                    # sample memories and use the target DQN to produce the target Q-Value
                    if self.options['use_priority']:
                        tree_idxes = []
                        self.replay_memory.sample_memories(states, actions, rewards, next_states, continues, losses,
                                                           batch_size=self.batch_size,
                                                           tree_idxes=tree_idxes)
                        # states, actions, rewards, next_states, continues = self.sample_memories(self.batch_size, tree_idxes)
                    else:
                        self.replay_memory.sample_memories(states, actions, rewards, next_states, continues, losses,
                                                           batch_size=self.batch_size)
                        # states, actions, rewards, next_states, continues = self.sample_memories(self.batch_size)

                    target_max_q_values = self.get_target_max_q_values(sess,
                                                                       rewards,
                                                                       continues,
                                                                       next_states)
                    # target_max_q_values = self.get_target_max_q_values(sess, replay_batch[2], replay_batch[4], next_states)

                    # Train the online DQN
                    tr_res, new_losses, loss_val = sess.run([sess.model.training_op,
                                                             sess.model.losses,
                                                             sess.model.loss],
                                                             feed_dict={
                                                                 sess.model.X_state: states,
                                                                 sess.model.X_action: actions,
                                                                 sess.model.y: target_max_q_values
                                                             })

                    if self.options['use_priority']:
                        self.replay_memory.update_sum_tree(tree_idxes, new_losses)

                    total_losses.append(loss_val)

                    # Regularly copy the online DQN to the target DQN
                    if step % copy_steps == 0:
                        self.log('copying online to target dqn')
                        sess.model.copy_online_to_target.run()


                    # And save regularly
                    if step % save_steps == 0:
                        self.log('saving model')
                        sess.model.game_count.load(self.game_count.value)
                        sess.save(self.save_path_prefix)


                        self.save_count.value += 1

                    # report 
                    if step % train_report_interval == 0:
                        elapsed = time.time() - report_start_time
                        if elapsed > 0:
                            frame_rate = (step - report_last_step) / elapsed
                        else:
                            frame_rate = 0.0

                        report_last_step = step
                        report_start_time = time.time()

                        if len(total_losses) > 0:
                            avg_loss = sum(total_losses) / len(total_losses)
                        else:
                            avg_loss = 0

                        total_losses = []

                        self.log('step {} avg loss: {:0.5f} mem: {:d} fr: {:0.1f} cache: {:d}'.format(step,
                                                                                                      avg_loss,
                                                                                                      replay_memory_size,
                                                                                                      frame_rate,
                                                                                                      self.replay_memory.cache_size))
                            
            except KeyboardInterrupt:
                self.log('interrupted')

            sess.model.game_count.load(self.game_count.value)

        env.close()

        elapsed = time.time() - start_time 

        self.log('closing replay memory')
        self.replay_memory.close()
        self.log('train finished in {:0.1f} seconds / {:0.1f} mins'.format(elapsed, elapsed / 60))


    def run_game_func(self, 
                      is_training=False,
                      num_games=None,
                      use_epsilon=False,
                      interval=60,
                      no_display=True):
        self.header = 'game'

        env = gym.make(self.game_id)
        env.seed(int(time.time()))

        self.log('run_game_func')

        with self.get_session(load_model=True, save_model=False, env=env) as sess:
            game_scores = deque(maxlen=1000)
            max_qs = deque(maxlen=1000)
            report_interval = 60
            max_game_length = 50000

            if is_training:
                game_memory = ReplayMemoryDisk(self.get_memory_save_path(),
                                               sess.model.input_height,
                                               sess.model.input_width,
                                               sess.model.input_channels,
                                               state_type=sess.model.state_type,
                                               max_size=self.mem_save_size,
                                               cache_size=0)

            iteration = 0
            report_start_time = time.time()
            report_last_iteration = 0
            report_rate = 0
            step = sess.run([sess.model.step])[0]
            num_episodes = 0

            if is_training and self.game_count.value <= 0:
                self.game_count.value = sess.run([sess.model.game_count])[0]
                self.log('game_count:', self.game_count.value)

            last_save_count = self.save_count.value

            try:
                while True:
                    epoch_start_time = time.time()

                    total_max_q = 0.0
                    game_length = 0
                    game_score = 0
                    state_frames = deque(maxlen=sess.model.input_channels)
                    action = 0
                    reward = None
                    info = None
                    game_done = False
                    state = None
                    num_lives = 0

                    # for not training only
                    if not is_training:
                        actions = []
                        game_frames = []

                    if is_training and self.save_count.value > last_save_count:
                        self.log('save count changed reloading process')
                        break

                    if not is_training:
                        if self.game_count.value >= num_games:
                            break

                    self.game_count.value += 1

                    obs = env.reset()

                    obs = sess.model.preprocess_observation(obs)
                    state_frames.append(obs)

                    while not game_done:
                        episode_done = False
                        episode_length = 0

                        while not episode_done and not game_done:
                            iteration += 1
                            game_length += 1
                            episode_length += 1

                            if game_length > max_game_length:
                                self.log('game too long, breaking')
                                break

                            step = sess.run([sess.model.step])[0]

                            if len(state_frames) >= sess.model.input_channels:
                                next_state = self.make_state(state_frames)

                                if is_training and state is not None:
                                    game_memory.append(state, action, reward, next_state, 1, 0)

                                    # self.append_replay(game_memory, state, action, reward, next_state, 1)

                                    if len(game_memory) >= self.mem_save_size:
                                        self.save_memory(game_memory, sess=sess)
                                        game_memory = ReplayMemoryDisk(self.get_memory_save_path(),
                                                                       sess.model.input_height,
                                                                       sess.model.input_width,
                                                                       sess.model.input_channels,
                                                                       state_type=sess.model.state_type,
                                                                       max_size=self.mem_save_size,
                                                                       cache_size=0)

                                state = next_state

                                # Online DQN evaluates what to do
                                q_values = sess.model.online_q_values.eval(feed_dict={sess.model.X_state: [next_state]})
                                total_max_q += q_values.max()

                                if is_training or use_epsilon:
                                    action = self.epsilon_greedy(q_values,
                                                                 step)
                                else:
                                    action = np.argmax(q_values)

                            action = sess.model.before_action(action, obs, reward, game_done, info)

                            # run action for frame_skip steps 
                            reward = 0
                            for i in range(self.frame_skip):
                                # Online DQN plays
                                obs, step_reward, game_done, info = env.step(action)

                                if not is_training:
                                    actions.append(action)
                                    game_frames.append(sess.model.render_obs(obs))
                                
                                reward += step_reward

                                # check for episode change
                                if self.use_episodes and 'ale.lives' in info and info['ale.lives'] != num_lives:
                                    if num_lives > 0:
                                        episode_done = True
                                    num_lives = info['ale.lives']

                                if game_done:
                                    break

                                if episode_done:
                                    break

                            game_score += reward

                            obs = sess.model.preprocess_observation(obs)
                            state_frames.append(obs)

                        num_episodes += 1

                        # game / episode done, save last step
                        next_state = self.make_state(state_frames)

                        if is_training:
                            game_memory.append(state, action, reward, next_state, 0, 0)


                            # self.append_replay(game_memory, state, action, reward, next_state, 0)

                            if len(game_memory) >= self.mem_save_size:
                                self.save_memory(game_memory, sess=sess)
                                game_memory = ReplayMemoryDisk(self.get_memory_save_path(),
                                                               sess.model.input_height,
                                                               sess.model.input_width,
                                                               sess.model.input_channels,
                                                               state_type=sess.model.state_type,
                                                               max_size=self.mem_save_size,
                                                               cache_size=0)

                        state = next_state


                    if game_length > 0:
                        mean_max_q = total_max_q / game_length
                    else:
                        mean_max_q = 0

                    elapsed = time.time() - epoch_start_time
                    if elapsed > 0:
                        frame_rate = game_length / (time.time() - epoch_start_time)
                    else:
                        frame_rate = 0.0

                    report_elapsed = time.time() - report_start_time
                    if report_elapsed > report_interval:
                        report_rate = (iteration - report_last_iteration) / (report_elapsed)
                        report_last_iteration = iteration
                        report_start_time = time.time()

                    game_scores.append(game_score)

                    if len(game_scores) > 0:
                        avg_score = sum(game_scores) / len(game_scores)
                    else: 
                        avg_score = 0

                    max_qs.append(mean_max_q)

                    avg_max_q = sum(max_qs) / len(max_qs)

                    epsilon = self.epsilon(step)

                    if is_training:
                        mem_len = len(game_memory)
                    else:
                        mem_len = 0

                    if self.game_count.value % sess.model.game_report_interval == 0 or not is_training:
                        self.log('step {} game {} epi {} len: {:d} max_q: {:0.3f}/{:0.3f} score: {:0.1f} avg: {:0.1f} mem: {:d} eps: {:0.3f} fr: {:0.1f}/{:0.1f}'.format(
                                   step,
                                   self.game_count.value, 
                                   num_episodes,
                                   game_length, 
                                   mean_max_q,
                                   avg_max_q,
                                   game_score, 
                                   avg_score,
                                   mem_len,
                                   epsilon,
                                   frame_rate,
                                   report_rate))

                    if not is_training and not no_display:
                        render_game(game_frames, 
                                    actions, 
                                    repeat=False, 
                                    interval=interval,
                                    save_path=os.path.join(self.save_dir, 'video-{}-{}.mp4'.format(step, self.game_count.value)))

                    # if is_training and self.game_count.value % self.game_render_interval == 0:
                    #     render_game(game_frames, 
                    #                 actions, 
                    #                 repeat=False, 
                    #                 interval=interval,
                    #                 save_path=os.path.join(self.save_dir, 'video-{}-{}.mp4'.format(step, self.game_count.value)),
                    #                 no_display=True)



            except KeyboardInterrupt:
                self.log('run game interrupted')

            if is_training and len(game_memory) > 0:
                self.save_memory(game_memory, sess=sess)


        env.close()

        return 0


    def play(self, num_games=1, use_epsilon=False, interval=60, no_display=False):
        self.run_game_func(is_training=False,
                           num_games=num_games,
                           use_epsilon=use_epsilon,
                           interval=interval,
                           no_display=no_display)


    # def sample_memories(self, batch_size, with_replacement=False):
    #     memories = []
    #     if with_replacement:
    #         period = len(self.replay_memory) / batch_size
    #         idx = random.randint(0, int(period)-1)
    #
    #         for i in range(batch_size):
    #             memories.append(self.replay_memory[period * i + idx])
    #     else:
    #         for i in range(batch_size):
    #             idx = random.randint(0, len(self.replay_memory)-1)
    #             memories.append(self.replay_memory[idx])
    #
    #     return self.make_batch(memories)
    #
    #
    # def sample_memories_sum_tree(self, batch_size, tree_idxes=None):
    #     memories = []
    #     num_tries = 0
    #
    #     while len(memories) < batch_size:
    #         s = random.random() * self.replay_sum_tree.total()
    #         t_idx, idx, score, memory = self.replay_sum_tree.get(s)
    #
    #         if idx >= self.replay_sum_tree.capacity:
    #             if num_tries > 10:
    #                 self.log('sample_memories exceeded max tries, breaking')
    #                 break
    #             self.log('warning: invalid index:', idx)
    #             continue
    #
    #         if tree_idxes:
    #             tree_idxes.append(t_idx)
    #
    #         memories.append(self.replay_memory[idx])
    #
    #         # memories.append((self.replay_memory.memory_states[idx],
    #         #                  self.replay_memory.memory_actions[idx],
    #         #                  self.replay_memory.memory_rewards[idx],
    #         #                  self.replay_memory.memory_next_states[idx],
    #         #                  self.replay_memory.memory_continues[idx]))
    #
    #
    #     return self.make_batch(memories)
    #
    #
    # def update_sum_tree(self, tree_idxes, losses):
    #     for t_idx, loss in zip(tree_idxes, losses):
    #         self.sum_tree.update(t_idx, loss)


    def epsilon(self, step):
        return max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)


    def epsilon_greedy(self, q_values, step):
        epsilon = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_outputs) # random action
        else:
            return np.argmax(q_values) # optimal action


    def make_state(self, frames):
        return np.concatenate(frames, axis=2)


    def make_batch(self, memories):
        cols = [[], [], [], [], [], []] # state, action, reward, next_state, continue

        for memory in memories:
            try:
                for col, value in zip(cols, memory):
                    col.append(value)
            except TypeError as e:
                self.log('TypeError: ', str(e), type(memory))
                pass

        cols = [np.array(col) for col in cols]

        return cols


    def get_target_max_q_values(self, sess, rewards, continues, next_states):
        if self.options['use_double']:
            max_next_q_values = sess.model.double_max_q_values.eval(
                feed_dict={sess.model.X_state: next_states})
        else:
            next_q_values = sess.model.target_q_values.eval(
                feed_dict={sess.model.X_state: next_states})
            max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)

        return rewards + continues * self.discount_rate * max_next_q_values


    def get_memory_file_list(self):
        fns = []
        for fn in os.listdir(self.save_dir):
            match = re.match(self.memory_file_prefix + '-(\\d+)-(\\d+)', fn)
            if not match:
                continue
            path = os.path.join(self.save_dir, fn)
            with open(path, 'rb') as fin:
                size = pickle.load(fin)

            fns.append((path, int(match.group(1)), int(size)))

        return fns


    def delete_old_memories(self):
        fns = self.get_memory_file_list()

        for path, date, size in sorted(fns, key=lambda x: x[1]):
            self.log('deleting old memory: {}'.format(path))
            os.unlink(path)


    def get_memory_save_path(self):
        # now save
        path = os.path.join(self.save_dir, '{}-{:d}'.format(self.memory_file_prefix, int(time.time())))
        i = 0
        new_path = path + '-{:d}'.format(i)
        while os.path.exists(new_path):
            i += 1

            if i > 100:
                raise Exception('cannot find new save path')

            new_path = path + '-{:d}'.format(i)

        return new_path


    # def save_memory(self, memories, sess=None, batch_size=32):
    #     if sess is None:
    #         sess = tf.get_default_session()
    #
    #     # calculate priority value
    #     num_batches = int(math.ceil(len(memories) / batch_size))
    #
    #     memories_with_priority = []
    #     for i in range(num_batches):
    #         states, actions, rewards, next_states, continues, losses = self.make_batch(
    #                 memories[i * batch_size:(i + 1) * batch_size])
    #         # states, actions, rewards, next_states, continues = memories[i * batch_size:(i + 1) * batch_size]
    #
    #         target_max_q_values = self.get_target_max_q_values(sess, rewards, continues, next_states)
    #
    #         if self.testing_predict:
    #             for j in range(len(target_max_q_values)):
    #                 if continues[j] == 0:
    #                     assert (rewards[j] == target_max_q_values[j])
    #
    #         losses = sess.model.losses.eval(
    #                 feed_dict={sess.model.X_state: states,
    #                            sess.model.X_action: actions,
    #                            sess.model.y: target_max_q_values})
    #
    #         for row in zip(states, actions, rewards, next_states, continues, losses):
    #             memories_with_priority.append(row)
    #
    #         # for j in range(i * batch_size, (i + 1) * batch_size):
    #         #     memories.set_loss_abs(i * batch_size + j, losses[j])
    #
    #     # now save
    #     path = os.path.join(self.save_dir, '{}-{:d}'.format(self.memory_file_prefix, int(time.time())))
    #     i = 0
    #     new_path = path + '-{:d}'.format(i)
    #     while os.path.exists(new_path):
    #         i += 1
    #         new_path = path + '-{:d}'.format(i)
    #     path = new_path
    #
    #     self.log('saving memory to:', path, 'size:', len(memories_with_priority))
    #
    #     with open(path, 'wb') as f:
    #         pickle.dump(len(memories_with_priority), f)
    #
    #         for j in range(len(memories_with_priority)):
    #             pickle.dump(memories_with_priority[j], f)
    #
    #     self.log('save complete')
    #     self.memory_queue.put(path)


    def save_memory(self, memories, sess=None, batch_size=32):
        self.log('save memories')

        if sess is None:
            sess = tf.get_default_session()

        # calculate priority value
        num_batches = int(math.ceil(len(memories) / batch_size))

        for i in range(num_batches):
            states, actions, rewards, next_states, continues, losses = self.make_batch(
                    memories[i * batch_size:(i + 1) * batch_size])

            target_max_q_values = self.get_target_max_q_values(sess, rewards, continues, next_states)

            if self.testing_predict:
                for j in range(len(target_max_q_values)):
                    if continues[j] == 0:
                        assert (rewards[j] == target_max_q_values[j])

            losses = sess.model.losses.eval(
                    feed_dict={sess.model.X_state: states,
                               sess.model.X_action: actions,
                               sess.model.y: target_max_q_values})

            for j in range(len(losses)):
                # if j < 10:
                #     print('setting ', i * batch_size + j, 'loss to', losses[j])

                if i * batch_size + j >= len(memories):
                    break

                memories.set_loss_abs(i * batch_size + j, losses[j])

                # if j < 10:
                #     print('  check ', memories[i * batch_size + j][5], ' ?= ', losses[j])

                # assert memories[i * batch_size + j][5] == losses[j]

        memories.close()
        self.memory_queue.put(memories.filename)


    def load_memory(self, memory_fn, delete=True):
        try:
            self.log('loading memory from:', memory_fn)

            # with open(memory_fn, 'rb') as fin:
            #     size = pickle.load(fin)
            #
            #     for i in range(size):
            #         # self.add_memory(*pickle.load(fin))
            #         # row = pickle.load(fin)
            #         # print(row)
            #         # self.replay_memory.append(*row)
            #         self.replay_memory.append(*pickle.load(fin))
            #

            memories = ReplayMemoryDisk(memory_fn,
                                        cache_size=0)

            for i in range(len(memories)):
                self.replay_memory.append(*memories[i])

            self.log('load complete. loaded:', len(memories), 'total:', len(self.replay_memory))

            if delete:
                os.unlink(memory_fn)

        except EOFError:
            self.log('eoferror')
            pass
        except FileNotFoundError:
            self.log('file missing:', memory_fn, 'skipping')



    def get_session(parent, load_model=True, save_model=False, env=None):
        class Session:
            def __init__(self, env):
                self.env = env


            def __enter__(self):
                return self.open()


            def __exit__(self, ty, value, tb):
                self.close(ty, value, tb)


            def run(self, *args, **kwargs):
                return self._sess.run(*args, **kwargs)


            def save(self, save_path_prefix):
                parent.log('saving model: ', save_path_prefix)
                self.saver.save(self._sess, save_path_prefix)
                parent.log('saved model')


            def restore(self, save_path_prefix):
                if not os.path.exists(save_path_prefix + '.index'):
                    parent.log('model does not exist:', save_path_prefix)
                    return False

                parent.log('restoring model: ', save_path_prefix)
                self.saver.restore(self._sess, save_path_prefix)
                parent.log('restored model')

                return True


            def open(self):
                parent.log('creating new session. load_model: ', load_model, 'save_model:', save_model)

                tf.reset_default_graph()

                self.model = parent.model_class(self.env, options=parent.options)
                self.saver = tf.train.Saver()

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True

                self._sess = tf.Session(config=config)
                self._sess.as_default()
                self._sess.__enter__()

                loaded = False
                if load_model:
                    loaded = self.restore(parent.save_path_prefix)

                if not loaded:
                    self._sess.run(tf.global_variables_initializer())

                tf.get_default_graph().finalize()

                return self


            def close(self, ty=None, value=None, tb=None):
                if save_model:
                    self.save(parent.save_path_prefix)

                self._sess.__exit__(ty, value, tb)                    
                self._sess.close()
                self._sess = None


        return Session(env)


    def log(self, *mesg):
        # print(time_string(), self.header, ' '.join(mesg))
        print('{} [{}] {}'.format(time_string(), self.header, ' '.join([str(m) for m in mesg])))
