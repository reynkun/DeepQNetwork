import math
import re
import os
import time
import random
import datetime
import multiprocessing
import pickle
import signal

from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt

import gym
import tensorflow as tf
import numpy as np

from PIL import Image, ImageFont, ImageDraw
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, recall_score


from sum_tree import SumTree


def time_string():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def read_image(path):
    return np.array(Image.open(path)) / 255.0


def render_state(state, repeat=True, interval=800):
    def update_scene(num, frames, patch):
        patch.set_data(frames[num])
        return patch,


    frames = []
    for i in range(state.shape[2]):
        # print(state[:,:,i].shape)
        shape = state[:,:,i].shape
        frame = state[:,:,i].reshape(shape[0], shape[1], 1)

        img_dat = np.concatenate((frame, frame, frame), axis=2)
        img_dat = Image.fromarray((img_dat * 255).astype('uint8'))
        draw = ImageDraw.Draw(img_dat)
        draw.text((0,0), "fr {}".format(i))

        frames.append(img_dat)

    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('on')

    video = animation.FuncAnimation(fig, 
                                   update_scene, 
                                   fargs=(frames, patch), 
                                   frames=len(frames), 
                                   repeat=repeat, 
                                   interval=interval)
    plt.show()

    return video


def render_game(game_frames, actions, repeat=True, interval=30):
    def update_scene(num, frames, patch):
        patch.set_data(frames[num])
        return patch,


    frames = []
    for i in range(len(game_frames)):
        img_dat = Image.fromarray(game_frames[i])
        draw = ImageDraw.Draw(img_dat)
        draw.text((0,0), "fr {} act {:d}".format(i, actions[i]))

        frames.append(img_dat)


    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('on')

    video = animation.FuncAnimation(fig, 
                                    update_scene, 
                                    fargs=(frames, patch), 
                                    frames=len(frames), 
                                    repeat=repeat, 
                                    interval=interval)
    plt.show()

    return video


class DeepQNetwork:
    def __init__(self, 
                 game_id,
                 model_class, 
                 batch_size=32, 
                 save_dir='./models',
                 mem_save_size=10000,
                 num_training_start_steps=10000):
        self.model_class = model_class
        self.game_id = game_id
        self.batch_size = int(batch_size)
        self.save_dir = save_dir
        # self.save_path_prefix = os.path.join(self.save_dir, '{}-{}'.format(self.game_id, str(model_class)))
        self.file_prefix = '{}'.format(self.game_id)
        self.save_path_prefix = os.path.join(self.save_dir, self.file_prefix)
        self.memory_file_prefix = '{}.memory'.format(self.file_prefix)

        self.eps_min = 0.1
        self.eps_max = 1.0
        self.eps_decay_steps = 2000000
        self.discount_rate = 0.99

        self.num_state_frames = 4

        self.replay_max_memory_length = 500000
        self.max_num_run_game_replays = 10000
        self.mem_save_size = mem_save_size
        self.num_training_start_steps = num_training_start_steps

        self.memory_queue = multiprocessing.Queue()
        self.save_count = multiprocessing.Value('i', 0)
        self.game_count = multiprocessing.Value('i', 0)

        env = gym.make(self.game_id)
        self.num_outputs = env.action_space.n


    def train(self):
        p_t = multiprocessing.Process(target=self.train_func)
        p_t.start()

        while True:
            g_t = multiprocessing.Process(target=self.run_game_func)
            g_t.start()
            g_t.join()

            if g_t.exitcode != 0:
                print('game process nonzero exit')
                os.kill(p_t.pid, signal.SIGINT)
                break

        p_t.join()


    def train_func(self):
        # allocate memory
        env = gym.make(self.game_id)
        agent = self.model_class(env, initialize=False)

        print('allocating memory', self.replay_max_memory_length)
        self.memory_states = np.ndarray((self.replay_max_memory_length, agent.input_height, agent.input_width, agent.input_channels), dtype='uint8')
        self.memory_actions = np.ndarray((self.replay_max_memory_length), dtype='uint8')
        self.memory_rewards = np.ndarray((self.replay_max_memory_length, 1), dtype='uint32')
        self.memory_next_states = np.ndarray((self.replay_max_memory_length, agent.input_height, agent.input_width, agent.input_channels), dtype='uint8')
        self.memory_continues = np.ndarray((self.replay_max_memory_length, 1), dtype='uint8')
        self.memory_size = 0
        self.memory_index = 0
        print('finished allocating memory')
        
        replay_memory = SumTree(self.replay_max_memory_length)



        save_steps = 10000
        copy_steps = 10000

        max_num_training = 20000000

        train_report_interval = 1000

        # replay_memory = RingBuf(self.replay_max_memory_length)

        with self.get_session(load_model=True, save_model=True, env=env) as sess:
            try:
                fns = self.get_memory_file_list()

                for path, date, size in sorted(fns, key=lambda x:x[1], reverse=True):
                    memories = []
                    self.load_memory(memories, path)
                    self.add_sum_tree(memories, replay_memory)
                    del memories

                    if len(replay_memory) >= self.replay_max_memory_length: 
                        break


                iteration = 0
                report_start_time = time.time()
                report_last_iteration = 0
                report_rate = 0
                step = sess.model.step.eval()
                report_last_step = step
                losses = []

                while len(replay_memory) < self.num_training_start_steps:
                    print('waiting for memory', self.num_training_start_steps, 'cur size:', len(replay_memory))
                    fn = self.memory_queue.get()
                    memories = []
                    self.load_memory(memories, fn)
                    self.add_sum_tree(memories, replay_memory)
                    del memories


                while step < max_num_training:
                    iteration += 1
                    step = sess.model.step.eval()

                    while self.memory_queue.qsize():
                        fn = self.memory_queue.get()

                        memories = []
                        self.load_memory(memories, fn)
                        self.add_sum_tree(memories, replay_memory)
                        del memories

                    # Sample memories and use the target DQN to produce the target Q-Value
                    # X_state_val, X_action_val, rewards, next_states, continues = self.sample_memories(replay_memory, self.batch_size)
                    states, actions, rewards, next_states, continues = self.sample_memories_sum_tree(replay_memory, self.batch_size)

                    # next_q_values = sess.model.target_q_values.eval(
                    #     feed_dict={sess.model.X_state: self.convert_state(next_states)})
                    # max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
                    # 

                    # online_actions, target_values = sess.run([sess.model.online_actions,
                    #                                           sess.model.target_q_values],
                    #                                          feed_dict={sess.model.X_state: next_states})

                    # y_val = rewards + continues * self.discount_rate \
                    #           * target_values[np.arange(target_values.shape[0]), 
                    #                                     online_actions].reshape(-1, 1)

                    max_next_q_values = sess.model.double_max_q_values.eval(
                                            feed_dict={sess.model.X_state: next_states}) 
                    y_val = rewards + continues * self.discount_rate * max_next_q_values

                    # y_val = sess.model.q_cur_and_next_q_values.eval(feed_dict={
                    #                                     sess.model.X_rewards: rewards,
                    #                                     sess.model.X_continues: continues,
                    #                                     sess.model.X_state: next_states
                    #                                 })

                    # Train the online DQN
                    tr_res, loss_val = sess.run([sess.model.training_op, 
                                                 sess.model.loss], 
                                                feed_dict={
                                                    sess.model.X_state: states,
                                                    sess.model.X_action: actions,
                                                    sess.model.y: y_val
                                                })
                    losses.append(loss_val)

                    # Regularly copy the online DQN to the target DQN
                    if step % copy_steps == 0:
                        print('copying to target network')
                        sess.model.copy_online_to_target.run()

                    # And save regularly
                    if step % save_steps == 0:
                        print('saving tf model!!!')
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

                        epsilon = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)
                        print('{} [train] step {} avg loss: {:0.3f} mem: {:d} fr: {:0.1f}'.format(time_string(),
                                                                                              step,
                                                                                              avg_loss,
                                                                                              len(replay_memory),
                                                                                              frame_rate))
                            
            except KeyboardInterrupt:
                print('interrupted')

            sess.model.game_count.load(self.game_count.value)

        env.close()


    def run_game_func(self):
        game_scores = deque(maxlen=25)
        max_qs = deque(maxlen=25)
        report_interval = 60
        max_game_length = 50000

        replay_memory = []        

        env = gym.make(self.game_id)

        print('run_game_func')

        with self.get_session(load_model=True, save_model=False, env=env) as sess:
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
                    state_frames = deque(maxlen=self.num_state_frames)
                    action = 0
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

                        # if game_length % num_skip_frames == 0 and len(state_frames) >= num_state_frames:
                        if len(state_frames) >= self.num_state_frames:
                            next_state = np.concatenate(state_frames, axis=2)

                            if state is not None:
                                self.append_replay(replay_memory, state, action, reward, next_state, 1)

                                if len(replay_memory) % self.mem_save_size == 0:
                                    self.clear_old_memory()
                                    self.save_memory(replay_memory, sess=sess)
                                    replay_memory = []

                            state = next_state

                            # Online DQN evaluates what to do
                            q_values = sess.model.online_q_values.eval(feed_dict={sess.model.X_state: [next_state]})
                            total_max_q += q_values.max()

                            last_action = action

                            action = self.epsilon_greedy(q_values,
                                                         step)


                        # if info is not None and info['ale.lives'] != num_lives:
                        #     action = 1
                        #     num_lives = info['ale.lives']

                        # Online DQN plays
                        obs, reward, done, info = env.step(action)
                        sess.model.on_info(info)

                        game_score += reward

                        # if info is not None and info['ale.lives'] < num_lives:
                        #     reward = -5

                        obs = sess.model.preprocess_observation(obs)
                        state_frames.append(obs)


                    # game done, save last step
                    next_state = np.concatenate(state_frames, axis=2)

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

                    epsilon = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)
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

        env.close()

        return 0


    def play(self, use_epsilon=False, interval=60):
        env = gym.make(self.game_id)

        with self.get_session(load_model=True, save_model=False, env=env) as sess:
            try:
                iteration = 0
                step = sess.model.step.eval()

                epoch_start_time = time.time()

                total_max_q = 0.0
                game_length = 0
                game_score = 0
                state_frames = deque(maxlen=self.num_state_frames)
                game_frames = []
                game_scores = []
                actions = []
                action = 0
                state = None
                next_state = None
                done = False
                num_lives = 0
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

                    # if game_length % num_skip_frames == 0 and len(state_frames) >= self.num_state_frames:
                    if len(state_frames) >= self.num_state_frames:
                        next_state = np.concatenate(state_frames, axis=2)

                        # Online DQN evaluates what to do
                        q_values = sess.model.online_q_values.eval(feed_dict={sess.model.X_state: [next_state]})
                        if use_epsilon:
                            action = self.epsilon_greedy(q_values,
                                                         step)
                        else:
                            action = np.argmax(q_values)

                    if info is not None and info['ale.lives'] != num_lives:
                        action = 1
                        num_lives = info['ale.lives']

                    # Online DQN plays
                    obs, reward, done, info = env.step(action)

                    game_score += reward

                    game_frames.append(obs)
                    actions.append(action)
                    obs = sess.model.preprocess_observation(obs)
                    state_frames.append(obs)

                    if done: 
                        continue

                # game done, save last step
                next_state = np.concatenate(state_frames, axis=2)# 
                state = next_state
                mean_max_q = total_max_q / game_length

                game_scores.append(game_score)

                if len(game_scores) > 0:
                    avg_score = sum(game_scores) / len(game_scores)
                else: 
                    avg_score = 0

                max_score = max(game_scores)


                print('game len: {:d} max_q: {:0.3f} score: {:0.1f} avg: {:0.1f} max: {:0.1f}'.format(
                                                                               game_length, 
                                                                               mean_max_q,
                                                                               game_score, 
                                                                               avg_score,
                                                                               max_score))
                render_game(game_frames, actions, repeat=False, interval=interval)


                            
            except KeyboardInterrupt:
                print('interrupted')
                raise

        env.close()


    # def calculate_loss(self, ):
        # next_q_values = sess.model.target_q_values.eval(
        #     feed_dict={sess.model.X_state: self.convert_state(next_states)})
        # max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        # y_val = rewards + continues * discount_rate * max_next_q_values

        # online_actions, target_values = sess.run([sess.model.online_actions,
        #                                           sess.model.target_q_values],
        #                                          feed_dict={sess.model.X_state: next_states})

        # y_val = rewards + continues * self.discount_rate \
        #           * target_values[np.arange(target_values.shape[0]), 
        #                                     online_actions].reshape(-1, 1)



    def append_replay(self, replay_memory, state, action, reward, next_state, cont):
        replay_memory.append((state, action, np.array([reward]), next_state, np.array([cont])))


    def convert_state(self, state):
        return state        


    def sample_memories(self, replay_memory, batch_size):
        if len(self.memory_indices) < len(replay_memory):
            new_size = min([int(len(replay_memory) * 2), self.replay_max_memory_length])
            self.memory_indices = np.random.permutation(new_size)
            self.memory_last_i = -1

        memories = []
        while len(cols[0]) < batch_size:
            self.memory_last_i += 1

            if self.memory_last_i >= len(self.memory_indices):
                self.memory_last_i = 0

            i = self.memory_last_i
            idx = self.memory_indices[i]

            if idx >= len(replay_memory):
                continue

            memories.append(replay_memory[idx])

        return self.make_batch(memories)


    def sample_memories_sum_tree(self, sum_tree, batch_size):
        memories = []
        while len(memories) < batch_size:
            s = random.random() * sum_tree.total()
            idx, score, memory = sum_tree.get(s)

            memories.append((self.memory_states[idx], 
                             self.memory_actions[idx],
                             self.memory_rewards[idx],
                             self.memory_next_states[idx],
                             self.memory_continues[idx]))

        return self.make_batch(memories)


    def epsilon_greedy(self, q_values, step):
        epsilon = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_outputs) # random action
        else:
            return np.argmax(q_values) # optimal action


    def save_memory(self, memories, sess=None, batch_size=100):
        if sess is None:
            sess = tf.get_default_session()

        # calculate priority value
        num_batches = int(math.ceil(len(memories) / batch_size))

        memories_with_priority = []
        for i in range(num_batches):
            states, actions, rewards, next_states, continues = self.make_batch(memories[i * batch_size:(i+1) * batch_size])

            max_next_q_values = sess.model.double_max_q_values.eval(
                feed_dict={sess.model.X_state: next_states})

            y_val = rewards + continues * self.discount_rate * max_next_q_values

            losses = sess.model.losses.eval(
                         feed_dict={sess.model.X_state: states,
                                    sess.model.X_action: actions,
                                    sess.model.y: y_val})


            for row in zip(states, actions, rewards, next_states, continues, losses):
                memories_with_priority.append(row)

        # now save
        path = '{}-{:d}'.format(self.memory_file_prefix, int(time.time()))
        print('saving memory to:', path, 'size:', len(memories_with_priority))

        with open(path, 'wb') as f:
            pickle.dump(len(memories_with_priority), f)

            for j in range(len(memories_with_priority)):
                pickle.dump(memories_with_priority[j], f)

        print('save complete')
        self.memory_queue.put(path)


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


    def get_memory_file_list(self):
        fns = []
        for fn in os.listdir(self.save_dir):
            match = re.match(self.memory_file_prefix + '-(\\d+)', fn)
            if not match:
                continue
            path = os.path.join(self.save_dir, fn)
            with open(path, 'rb') as fin:
                size = pickle.load(fin)

            fns.append((path, int(match.group(1)), int(size)))

        return fns


    def clear_old_memory(self):
        fns = self.get_memory_file_list()

        max_memory_size = int(self.replay_max_memory_length * 1.5)
        cur_mem_size = sum([row[2] for row in fns])

        print('cur_mem_size:', cur_mem_size, 'max_memory_size:', max_memory_size)

        if cur_mem_size > max_memory_size:
            for path, date, size in sorted(fns, key=lambda x: x[1]):
                print('deleting old memory:', path)
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
            raise


    def load_saved_memory(self, memories):
        print('loaded replay memory length:', len(memories))

        fns = self.get_memory_file_list()

        for path, date, size in sorted(fns, key=lambda x:x[1]):
            self.load_memory(memories, path)


    def add_sum_tree(self, memories, sum_tree):
        for state, action, reward, next_state, cont, loss in memories:
            idx = sum_tree.add(loss+0.001, 0)

            self.memory_states[idx] = state
            self.memory_actions[idx] = action
            self.memory_rewards[idx] = reward
            self.memory_next_states[idx] = next_state
            self.memory_continues[idx] = cont


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

                self.model = parent.model_class(self.env)

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

