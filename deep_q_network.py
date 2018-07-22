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

from sum_tree import SumTree
from render import render_game
from replay_memory import ReplayMemory

def time_string():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


DEFAULT_OPTIONS = {
    'save_dir': './models',
    'eps_min': 0.1,
    'eps_max': 1.0,
    'eps_decay_steps': 2000000,
    'discount_rate': 0.99,
    'mem_save_size': 10000,
    'batch_size': 32,
    'testing_predict': False,
    'testing_train': False,
    'testing_weights': False,
    'model_save_prefix': None,
    'replay_max_memory_length': 500000,
    'max_num_training_steps': 20000000,
    'num_game_frames_before_training': 10000,
    'game_report_interval': 10
}


class DeepQNetwork:


    def __init__(self, 
                 game_id,
                 model_class, 
                 model_save_prefix=None,
                 options=None):
        self.model_class = model_class
        self.game_id = game_id

        if options:
            self.options = options
        else:
            self.options = {}
            self.options.update(DEFAULT_OPTIONS)

        self.save_dir = self.options['save_dir']

        if self.options['model_save_prefix'] is None:
            self.options['model_save_prefix'] = self.game_id

        self.file_prefix = '{}'.format(self.options['model_save_prefix'])
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

        # train settings
        self.max_num_training_steps = self.options['max_num_training_steps']
        self.replay_max_memory_length = self.options['replay_max_memory_length']
        self.num_game_frames_before_training = self.options['num_game_frames_before_training']
        self.memory_indices = []
        self.memory_last_i = -1
        self.batch_size = self.options['batch_size']

        # interprocess communication
        self.memory_queue = multiprocessing.Queue()
        self.save_count = multiprocessing.Value('i', 0)
        self.game_count = multiprocessing.Value('i', 0)

        self.testing_predict = self.options['testing_predict']
        self.testing_train = self.options['testing_train']
        self.testing_weights = self.options['testing_weights']

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        path = os.path.join(self.save_dir, '{}.conf'.format(self.file_prefix))
        if not os.path.exists(path):
            with open(path, 'w+') as fo:
                print(self.options)
                json.dump(self.options, fo, sort_keys=True, indent=4)


    def train(self):
        p_t = multiprocessing.Process(target=self.train_func)
        p_t.start()

        while True:
            g_t = multiprocessing.Process(target=self.run_game_func)
            g_t.start()

            while g_t.is_alive():
                if not p_t.is_alive():
                    print('train process not not alive!')
                    os.kill(g_t.pid, signal.SIGTERM)
                    raise Exception('train process not alive')
                g_t.join(5)

            if g_t.exitcode != 0:
                print('game process nonzero exit!')
                os.kill(p_t.pid, signal.SIGINT)
                break

        p_t.join()


    def train_func(self):
        env = gym.make(self.game_id)
        start_time = time.time()

        with self.get_session(load_model=True, save_model=True, env=env) as sess:
            # allocate memory
            self.replay_memory = ReplayMemory(self.replay_max_memory_length,
                                              sess.model.input_height,
                                              sess.model.input_width,
                                              sess.model.input_channels,
                                              state_type=sess.model.state_type)
            
            if self.options['use_priority']:
                replay_sum_tree = SumTree(self.replay_max_memory_length)
            else:
                replay_sum_tree = None

            save_steps = 10000
            copy_steps = 10000

            train_report_interval = 1000



            try:
                fns = self.get_memory_file_list()

                for path, date, size in sorted(fns, key=lambda x:x[1], reverse=True):
                    memories = []
                    self.load_memory(memories, path)
                    self.add_memories(memories, replay_sum_tree)
                    del memories

                    if self.options['use_priority']:
                        replay_memory_size = len(replay_sum_tree)
                    else:
                        replay_memory_size = len(self.replay_memory)

                    if replay_memory_size >= self.replay_max_memory_length:
                        break


                iteration = 0
                report_start_time = time.time()
                report_last_iteration = 0
                report_rate = 0
                step = sess.model.step.eval()
                report_last_step = step
                losses = []

                if self.options['use_priority']:
                    replay_memory_size = len(replay_sum_tree)
                else:
                    replay_memory_size = len(self.replay_memory)

                while replay_memory_size < self.num_game_frames_before_training:
                    print('waiting for memory', self.num_game_frames_before_training, 'cur size:', len(self.replay_memory))
                    fn = self.memory_queue.get()
                    memories = []
                    self.load_memory(memories, fn)
                    self.add_memories(memories, replay_sum_tree)
                    del memories

                    if self.options['use_priority']:
                        replay_memory_size = len(replay_sum_tree)
                    else:
                        replay_memory_size = len(self.replay_memory)


                while step < self.max_num_training_steps:
                    iteration += 1
                    step = sess.model.step.eval()

                    while self.memory_queue.qsize():
                        fn = self.memory_queue.get()

                        memories = []
                        self.load_memory(memories, fn)
                        self.add_memories(memories, replay_sum_tree)
                        del memories

                        if self.options['use_priority']:
                            replay_memory_size = len(replay_sum_tree)
                        else:
                            replay_memory_size = len(self.replay_memory)

                    # Sample memories and use the target DQN to produce the target Q-Value
                    # X_state_val, X_action_val, rewards, next_states, continues = self.sample_memories(replay_sum_tree, self.options['batch_size'])
                    if self.options['use_priority']:
                        states, actions, rewards, next_states, continues = self.sample_memories_sum_tree(replay_sum_tree, self.batch_size)
                    else:
                        states, actions, rewards, next_states, continues = self.sample_memories(self.replay_memory, self.batch_size)



                    target_max_q_values = self.get_target_max_q_values(sess, rewards, continues, next_states)


                    if self.testing_train:
                        # Train the online DQN
                        tr_res, \
                        online_q_values, \
                        online_max_q_values, \
                        losses, \
                        loss_val = sess.run([sess.model.training_op,
                                             sess.model.online_q_values,
                                             sess.model.online_max_q_values,
                                             sess.model.losses,
                                             sess.model.loss], 
                                            feed_dict={
                                                sess.model.X_state: states,
                                                sess.model.X_action: actions,
                                                sess.model.y: target_max_q_values
                                            })

                        for i in range(states.shape[0]):
                            print(i, 
                                  actions[i], 
                                  online_q_values[i], 
                                  online_max_q_values[i], 
                                  target_max_q_values[i], 
                                  losses[i])

                        
                    else:
                        # Train the online DQN
                        tr_res, loss_val = sess.run([sess.model.training_op,
                                                     sess.model.loss], 
                                                    feed_dict={
                                                        sess.model.X_state: states,
                                                        sess.model.X_action: actions,
                                                        sess.model.y: target_max_q_values
                                                    })



                    losses.append(loss_val)

                    # Regularly copy the online DQN to the target DQN
                    if step % copy_steps == 0:
                        if self.testing_weights:
                            for scope in ['q_networks/online']:
                                for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                             scope=scope):
                                    print(var.name)
                                    parts = var.name.split('/')
                                    target_name = '/'.join((parts[0], 'target', '/'.join(parts[2:])))
                                    print(target_name)

                                    online_weights, target_weights = sess.run([tf.get_default_graph().get_tensor_by_name(var.name),
                                                                               tf.get_default_graph().get_tensor_by_name(target_name)])
                                    print(np.min(online_weights), np.min(target_weights))
                                    print(np.max(online_weights), np.max(target_weights))
                                    print(np.average(online_weights), np.average(target_weights))

                        print('{} [train] copying to target network'.format(time_string()))
                        sess.model.copy_online_to_target.run()

                        # for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                        #     print(var.name)

                        if self.testing_weights:
                            for scope in ['q_networks/online']:
                                for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                             scope=scope):
                                    print(var.name)
                                    parts = var.name.split('/')
                                    target_name = '/'.join((parts[0], 'target', '/'.join(parts[2:])))
                                    print(target_name)

                                    online_weights, target_weights = sess.run([tf.get_default_graph().get_tensor_by_name(var.name),
                                                                               tf.get_default_graph().get_tensor_by_name(target_name)])

                                    print(np.min(online_weights), np.min(target_weights))
                                    print(np.max(online_weights), np.max(target_weights))
                                    print(np.average(online_weights), np.average(target_weights))

                                    # assert np.min(online_weights) == np.min(target_weights)
                                    # assert np.max(online_weights) == np.max(target_weights)
                                    # assert np.average(online_weights) == np.average(target_weights)

                                    # print('key:', var.name)
                                    # parts = var.name.split('/')
                                    # num_parts = len(parts)
                                    # found = False
                                    # for i in range(num_parts):
                                    #     try:
                                    #         name = '/'.join(var.name.split('/')[i:])
                                    #         print('  trying', name)
                                    #         weights = tf.get_default_graph().get_tensor_by_name(name)
                                    #         print('  found', name, weights)
                                    #         found = True
                                    #         break
                                    #     except (KeyError, ValueError):
                                    #         pass
                                    # if not found:
                                    #     print('did not find key')

                    # And save regularly
                    if step % save_steps == 0:
                        sess.save(self.save_path_prefix)
                        sess.model.game_count.load(self.game_count.value)
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

                        if len(losses) > 0:
                            avg_loss = sum(losses) / len(losses)
                        else:
                            avg_loss = 0

                        losses = []

                        epsilon = self.epsilon(step)

                        print('{} [train] step {} avg loss: {:0.3f} mem: {:d} fr: {:0.1f}'.format(
                            time_string(),
                            step,
                            avg_loss,
                            replay_memory_size,
                            frame_rate))
                            
            except KeyboardInterrupt:
                print('interrupted')

            sess.model.game_count.load(self.game_count.value)

        env.close()

        elapsed = time.time() - start_time 

        print('train finished in {:0.1f} seconds / {:0.1f} mins'.format(elapsed, elapsed / 60))


    def run_game_func(self):
        env = gym.make(self.game_id)

        print('run_game_func')

        with self.get_session(load_model=True, save_model=False, env=env) as sess:
            game_scores = deque(maxlen=1000)
            max_qs = deque(maxlen=1000)
            report_interval = 60
            max_game_length = 50000

            replay_memory = ReplayMemory(self.mem_save_size,
                                         sess.model.input_height,
                                         sess.model.input_width,
                                         sess.model.input_channels,
                                         state_type=sess.model.state_type)

            iteration = 0
            report_start_time = time.time()
            report_last_iteration = 0
            report_rate = 0
            step = sess.run([sess.model.step])[0]
            info = None

            if self.game_count.value <= 0:
                self.game_count.value = sess.run([sess.model.game_count])[0]
                print('game_count:', self.game_count.value)

            last_save_count = self.save_count.value

            try:
                while True:
                    epoch_start_time = time.time()
                    epoch_count = 0

                    total_max_q = 0.0
                    game_length = 0
                    game_score = 0
                    state_frames = deque(maxlen=sess.model.input_channels)
                    action = 0
                    reward = None
                    info = None
                    state = None
                    next_state = None
                    done = False
                    losses = []

                    if self.save_count.value > last_save_count:
                        print('save count changed reloading process')
                        break

                    self.game_count.value += 1

                    obs = env.reset()

                    if hasattr(sess.model, 'skip_steps'):
                        for skip in range(sess.model.skip_steps):
                            obs, reward, done, info = env.step(0)
                            sess.model.on_info(info)

                    obs = sess.model.preprocess_observation(obs)
                    state_frames.append(obs)

                    while not done:
                        iteration += 1
                        game_length += 1

                        if game_length > max_game_length:
                            print('game too long, breaking')
                            break

                        step = sess.run([sess.model.step])[0]

                        if len(state_frames) >= sess.model.input_channels:
                            next_state = self.make_state(state_frames)

                            if state is not None:
                                self.append_replay(replay_memory, state, action, reward, next_state, 1)

                                if len(replay_memory) % self.mem_save_size == 0:
                                    self.clear_old_memory()
                                    self.save_memory(replay_memory, sess=sess)
                                    replay_memory.clear()

                            state = next_state

                            # Online DQN evaluates what to do
                            q_values = sess.model.online_q_values.eval(feed_dict={sess.model.X_state: [next_state]})
                            total_max_q += q_values.max()

                            last_action = action

                            action = self.epsilon_greedy(q_values,
                                                         step)

                        action = sess.model.before_action(action, obs, reward, done, info)

                        # Online DQN plays
                        obs, reward, done, info = env.step(action)

                        game_score += reward

                        obs = sess.model.preprocess_observation(obs)
                        state_frames.append(obs)


                    # game done, save last step
                    next_state = self.make_state(state_frames)

                    self.append_replay(replay_memory, state, action, reward, next_state, 0)

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

                    if self.game_count.value % sess.model.game_report_interval == 0:
                        print('{} [game] step {} game {} len: {:d} max_q: {:0.3f}/{:0.3f} score: {:0.1f} avg: {:0.1f} mem: {:d} eps: {:0.3f} fr: {:0.1f}/{:0.1f}'.format(
                                   time_string(),
                                   step,
                                   self.game_count.value, 
                                   game_length, 
                                   mean_max_q,
                                   avg_max_q,
                                   game_score, 
                                   avg_score,
                                   len(replay_memory),
                                   epsilon,
                                   frame_rate,
                                   report_rate))

            except KeyboardInterrupt:
                print('run game interrupted')

            if len(replay_memory) > 0:
                self.clear_old_memory()
                self.save_memory(replay_memory, sess=sess)
                replay_memory.clear()

        env.close()

        return 0


    def play(self, num_games=1, use_epsilon=False, interval=60, no_display=False):
        env = gym.make(self.game_id)
        env._max_episode_steps = 1000

        with self.get_session(load_model=True, save_model=False, env=env) as sess:
            game_scores = []

            try:
                for i in range(num_games):
                    iteration = 0
                    step = sess.model.step.eval()

                    epoch_start_time = time.time()

                    total_max_q = 0.0
                    game_length = 0
                    game_score = 0
                    state_frames = deque(maxlen=sess.model.input_channels)
                    game_frames = []
                    actions = []
                    action = 0
                    state = None
                    next_state = None
                    done = False
                    num_lives = 0
                    reward = 0

                    info = None

                    obs = env.reset()

                    if hasattr(sess.model, 'skip_steps'):
                        for skip in range(sess.model.skip_steps):
                            obs, reward, done, info = env.step(0)

                    game_frames.append(obs)
                    actions.append(action)
                    obs = sess.model.preprocess_observation(obs)
                    state_frames.append(obs)

                    while not done:
                        iteration += 1
                        game_length += 1

                        if len(state_frames) >= sess.model.input_channels:
                            next_state = self.make_state(state_frames)

                            # Online DQN evaluates what to do
                            q_values = sess.model.online_q_values.eval(feed_dict={sess.model.X_state: [next_state]})
                            if use_epsilon:
                                action = self.epsilon_greedy(q_values,
                                                             step)
                            else:
                                action = np.argmax(q_values)

                        action = sess.model.before_action(action, obs, reward, done, info)

                        # Online DQN plays
                        obs, reward, done, info = env.step(action)

                        game_score += reward

                        game_frames.append(obs)
                        actions.append(action)
                        obs = sess.model.preprocess_observation(obs)
                        state_frames.append(obs)


                    # game done, save last step
                    next_state = self.make_state(state_frames)
                    state = next_state

                    if game_length > 0:
                        mean_max_q = total_max_q / game_length
                    else:
                        mean_max_q = 0

                    game_scores.append(game_score)

                    if len(game_scores) > 0:
                        avg_score = sum(game_scores) / len(game_scores)
                    else: 
                        avg_score = 0

                    max_score = max(game_scores)

                    min_score = min(game_scores)

                    std_dev = np.std(game_scores)

                    print('step: {:d} game {:d} len: {:d} max_q: {:0.3f} score: {:0.1f} avg: {:0.1f} max: {:0.1f} min: {:0.1f} std: {:0.2f}'.format(
                                                                                   step,
                                                                                   i,
                                                                                   game_length, 
                                                                                   mean_max_q,
                                                                                   game_score, 
                                                                                   avg_score,
                                                                                   max_score,
                                                                                   min_score, 
                                                                                   std_dev))
                    if not no_display:                                               
                        render_game(game_frames, actions, repeat=False, interval=interval)


                                
            except KeyboardInterrupt:
                print('interrupted')
                raise

        env.close()


    # def calculate_loss(self, ):
        # next_q_values = sess.model.target_max_q_values.eval(
        #     feed_dict={sess.model.X_state: self.convert_state(next_states)})
        # max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        # y_val = rewards + continues * discount_rate * max_next_q_values

        # online_actions, target_values = sess.run([sess.model.online_actions,
        #                                           sess.model.target_max_q_values],
        #                                          feed_dict={sess.model.X_state: next_states})

        # y_val = rewards + continues * self.discount_rate \
        #           * target_values[np.arange(target_values.shape[0]), 
        #                                     online_actions].reshape(-1, 1)



    def append_replay(self, replay_memory, state, action, reward, next_state, cont):
        replay_memory.append(state, action, np.array([reward]), next_state, np.array([cont]))


    def convert_state(self, state):
        return state        


    def sample_memories(self, replay_memory, batch_size, with_replacement=False):
        # if len(self.memory_indices) < len(replay_memory):
        #     print('resampling random memory list')
        #     new_size = min([int(len(replay_memory) * 2), self.replay_max_memory_length])
        #     self.memory_indices = np.random.permutation(new_size)
        #     self.memory_last_i = -1

        # memories = []
        # while len(memories) < batch_size:
        #     self.memory_last_i += 1

        #     if self.memory_last_i >= len(self.memory_indices):
        #         self.memory_last_i = 0

        #     i = self.memory_last_i
        #     idx = self.memory_indices[i]

        #     if idx >= len(replay_memory):
        #         continue

        #     memories.append((self.replay_memory.memory_states[idx], 
        #                      self.replay_memory.memory_actions[idx],
        #                      self.replay_memory.memory_rewards[idx],
        #                      self.replay_memory.memory_next_states[idx],
        #                      self.replay_memory.memory_continues[idx]))

        memories = []
        if with_replacement:
            period = len(self.replay_memory) / batch_size
            idx = random.randint(0, int(period)-1)

            for i in range(batch_size):
                memories.append(self.replay_memory[period * i + idx])
        else:
            for i in range(batch_size):
                idx = random.randint(0, len(self.replay_memory)-1)
                memories.append(self.replay_memory[idx])

        return self.make_batch(memories)


    def sample_memories_sum_tree(self, sum_tree, batch_size):
        memories = []
        num_tries = 0
        while len(memories) < batch_size:
            s = random.random() * sum_tree.total()
            idx, score, memory = sum_tree.get(s)

            if idx >= sum_tree.capacity:
                if num_tries > 10:
                    print('sample_memories exceeded max tries, breaking')
                    break
                print('warning: invalid index:', idx)
                continue

            memories.append((self.replay_memory.memory_states[idx], 
                             self.replay_memory.memory_actions[idx],
                             self.replay_memory.memory_rewards[idx],
                             self.replay_memory.memory_next_states[idx],
                             self.replay_memory.memory_continues[idx]))


            # # propagate train backward
            # extra_count = 0
            # idx -= 1

            # if idx <= 0:
            #     idx = len(sum_tree) - 1
            # while self.replay_memory.memory_continues[idx] != 0 \
            #         and len(memories) < batch_size \
            #         and extra_count < 4:
            #     memories.append((self.replay_memory.memory_states[idx], 
            #                      self.replay_memory.memory_actions[idx],
            #                      self.replay_memory.memory_rewards[idx],
            #                      self.replay_memory.memory_next_states[idx],
            #                      self.replay_memory.memory_continues[idx]))
            #     extra_count += 1
            #     idx -= 1
    
            #     if idx <= 0:
            #         idx = len(sum_tree) - 1

    
        return self.make_batch(memories)


    def epsilon(self, step):
        return max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)


    def epsilon_greedy(self, q_values, step):
        epsilon = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_outputs) # random action
        else:
            return np.argmax(q_values) # optimal action


    def save_memory(self, memories, sess=None, batch_size=5):
        if sess is None:
            sess = tf.get_default_session()

        count = 0

        # calculate priority value
        num_batches = int(math.ceil(len(memories) / batch_size))

        memories_with_priority = []
        for i in range(num_batches):
            states, actions, rewards, next_states, continues = self.make_batch(memories[i * batch_size:(i+1) * batch_size])

            target_max_q_values = self.get_target_max_q_values(sess, rewards, continues, next_states)

            if self.testing_predict:
                for j in range(len(target_max_q_values)):                    
                    if continues[j] == 0:
                        assert(rewards[j] == target_max_q_values[j])

            losses = sess.model.losses.eval(
                         feed_dict={sess.model.X_state: states,
                                    sess.model.X_action: actions,
                                    sess.model.y: target_max_q_values})


            for row in zip(states, actions, rewards, next_states, continues, losses):
                memories_with_priority.append(row)

        # now save
        path = os.path.join(self.save_dir, '{}-{:d}'.format(self.memory_file_prefix, int(time.time())))
        i = 0
        new_path = path + '-{:d}'.format(i)
        while os.path.exists(new_path):
            i += 1
            new_path = path + '-{:d}'.format(i)
        path = new_path
            

        print('saving memory to:', path, 'size:', len(memories_with_priority))

        with open(path, 'wb') as f:
            pickle.dump(len(memories_with_priority), f)

            for j in range(len(memories_with_priority)):
                pickle.dump(memories_with_priority[j], f)

        print('save complete')
        self.memory_queue.put(path)


    def make_state(self, frames):
        return np.concatenate(frames, axis=2)


    def make_batch(self, memories):
        cols = [[], [], [], [], []] # state, action, reward, next_state, continue

        for memory in memories:
            try:
                for col, value in zip(cols, memory):
                    col.append(value)
            except TypeError as e:
                print('TypeError: ', str(e), type(memory))
                pass

        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2], cols[3], cols[4]


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


    def clear_old_memory(self):
        fns = self.get_memory_file_list()

        max_memory_size = int(self.replay_max_memory_length * 3)
        cur_mem_size = sum([row[2] for row in fns])

        # print('cur_mem_size:', cur_mem_size, 'max_memory_size:', max_memory_size)

        if cur_mem_size > max_memory_size:
            for path, date, size in sorted(fns, key=lambda x: x[1]):
                print('{} [train] deleting old memory: {}'.format(time_string(), path))
                os.unlink(path)
                cur_mem_size -= size

                if cur_mem_size <= max_memory_size:
                    break


    def load_memory(self, memories, memory_fn):
        try:
            print('loading memory from:', memory_fn)

            with open(memory_fn, 'rb') as fin:
                size = pickle.load(fin)

                for i in range(size):
                    memories.append(pickle.load(fin))

            print('load complete. loaded:', len(memories))
        except EOFError:
            print('eoferror')
            pass
        except FileNotFoundError:
            print('file missing:', memory_fn, 'skipping')


    def load_saved_memory(self, memories):
        print('loaded replay memory length:', len(memories))

        fns = self.get_memory_file_list()

        for path, date, size in sorted(fns, key=lambda x:x[1]):
            self.load_memory(memories, path)


    def add_memories(self, memories, sum_tree=None):
        for state, action, reward, next_state, cont, loss in memories:
            if self.options['use_priority']:
                idx = sum_tree.add(loss+0.001, 0)

                self.replay_memory.memory_states[idx] = state
                self.replay_memory.memory_actions[idx] = action
                self.replay_memory.memory_rewards[idx] = reward
                self.replay_memory.memory_next_states[idx] = next_state
                self.replay_memory.memory_continues[idx] = cont
            else:
                self.replay_memory.append(state,
                                          action,
                                          reward,
                                          next_state,
                                          cont)


        # num_batches = int(math.ceil(len(memories) / batch_size))

        # num_rows_added = 0
        # print('num_batches: ', num_batches)
        # for i in range(num_batches):
        #     cols = [[], [], [], [], []] # state, action, reward, next_state, continue
        #     rows = memories[i * batch_size:(i+1) * batch_size]
        #     for memory in rows:
        #         for col, value in zip(cols, memory):
        #             col.append(value)

        #     cols = [np.array(col) for col in cols]
        #     states, actions, rewards, next_states, continues = cols

        #     online_actions, target_values = sess.run([sess.model.online_actions,
        #                                               sess.model.target_q_values],
        #                                              feed_dict={sess.model.X_state: next_states})

        #     y_val = rewards + continues * self.discount_rate * target_values[np.arange(target_values.shape[0]), online_actions].reshape(-1, 1)

        #     error = sess.run([sess.model.error], 
        #                         feed_dict={
        #                             sess.model.X_state: states,
        #                             sess.model.X_action: actions,
        #                             sess.model.y: y_val
        #                         })[0]


        #     # print(states.shape, actions.shape, y_val.shape)
        #     for loss, row in zip(error.reshape(-1), rows):
        #         num_rows_added += 1
        #         sum_tree.add(loss + 0.1, row)
        #         # print('adding:', len(row))
        #         # print(row[0].shape, row[1], row[2], row[3].shape)


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
                print('saving model: ', save_path_prefix)
                self.saver.save(self._sess, save_path_prefix)
                print('saved model')


            def restore(self, save_path_prefix):
                if not os.path.exists(save_path_prefix + '.index'):
                    print('model does not exist:', save_path_prefix)
                    return False

                print('restoring model: ', save_path_prefix)
                self.saver.restore(self._sess, save_path_prefix)
                print('restored model')

                return True


            def open(self):
                print('creating new session. load_model: ', load_model, 'save_model:', save_model)

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

