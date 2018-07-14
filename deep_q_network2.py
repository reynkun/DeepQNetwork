import math
import re
import os
import time
import random
import datetime
import multiprocessing
import pickle

from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt

import gym
import tensorflow as tf
import numpy as np

from PIL import Image, ImageFont, ImageDraw
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, recall_score


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


class RingBuf:
    def __init__(self, size):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        
    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
        
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    

class DeepQNetwork:

    def __init__(self, 
                 game_id='MsPacman-v0',
                 model_class=None, 
                 batch_size=32, 
                 num_epoch=1000,
                 report_ratio=0.1, 
                 save_path='./models',
                 verbose=True):
        self.model_class = model_class
        self.game_id = game_id
        self.num_epoch = num_epoch
        self.batch_size = int(batch_size)
        self.report_ratio = report_ratio
        self.save_path = save_path
        self.verbose = verbose

        self.eps_min = 0.1
        self.eps_max = 1.0
        self.eps_decay_steps = 1000000

        self.num_state_frames = 4

        self.memory_fn = '{}.memory'.format(self.game_id)
        self.memory_path = os.path.join(self.save_path, '{}.memory'.format(self.game_id))
        self.memory_dir = os.path.dirname(self.memory_path)
        self.memory_indices = np.random.permutation(10000)
        self.memory_last_i = -1
        self.replay_max_memory_length = 250000
        self.max_num_run_game_replays = 10000
        self.mem_save_steps = 10000

        self.memory_queue = multiprocessing.Queue()
        self.save_count = multiprocessing.Value('i', 0)


    def train(self):
        p_t = multiprocessing.Process(target=self.train_func)
        p_t.start()


        while True:
            g_t = multiprocessing.Process(target=self.run_game_func)
            g_t.start()
            g_t.join()

        p_t.join()


    def train_func(self):
        save_steps = 10000
        copy_steps = 10000

        num_training_start_steps = 10000
        max_num_training = 20000000

        discount_rate = 0.99
        train_report_interval = 1000

        replay_memory = RingBuf(self.replay_max_memory_length)

        self.load_saved_memory(replay_memory)
        env = gym.make(self.game_id)

        with self.get_session(load_model=True, env=env) as sess:
            try:
                iteration = 0
                report_start_time = time.time()
                report_last_iteration = 0
                report_rate = 0
                report_last_step = 0
                step = 0
                losses = []

                while len(replay_memory) < num_training_start_steps:
                    print('waiting for memory')
                    fn = self.memory_queue.get()
                    print('loading memory', fn)
                    self.load_memory(replay_memory, fn)


                while step < max_num_training:
                    iteration += 1
                    step = self.model.step.eval()

                    if self.memory_queue.qsize():
                        fn = self.memory_queue.get()

                        self.load_memory(replay_memory, fn)

                    # Sample memories and use the target DQN to produce the target Q-Value
                    X_state_val, X_action_val, rewards, X_next_state_val, continues = self.sample_memories(replay_memory, self.batch_size)

                    next_q_values = self.model.target_q_values.eval(
                        feed_dict={self.model.X_state: self.convert_state(X_next_state_val)})
                    max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
                    y_val = rewards + continues * discount_rate * max_next_q_values

                    # Train the online DQN
                    tr_res, loss_val = sess.run([self.model.training_op, 
                                                 self.model.loss], 
                                                feed_dict={
                                                    self.model.X_state: self.convert_state(X_state_val),
                                                    self.model.X_action: X_action_val,
                                                    self.model.y: y_val
                                                })
                    losses.append(loss_val)

                    # Regularly copy the online DQN to the target DQN
                    if step % copy_steps == 0:
                        print('copying to target network')
                        self.model.copy_online_to_target.run()

                    # And save regularly
                    if step % save_steps == 0:
                        sess.save(self.save_path)
                        self.save_count.value += 1

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
                        print('{} train {}/{} loss: {:0.3f} mem: {:d} fr: {:0.1f}'.format(time_string(),
                                                                                    iteration,
                                                                                    step,
                                                                                    avg_loss,
                                                                                    len(replay_memory),
                                                                                    frame_rate))
                            
            except KeyboardInterrupt:
                print('interrupted')
                pass

        env.close()


    def run_game_func(self):
        game_scores = deque(maxlen=25)
        report_interval = 60

        replay_memory = []        

        env = gym.make(self.game_id)
        self.num_outputs = env.action_space.n

        print('run_game_func')

        with self.get_session(load_model=True, save_model=False, env=env) as sess:
            iteration = 0
            report_start_time = time.time()
            report_last_iteration = 0
            report_rate = 0
            step = 0
            game_count = 0
            last_save_count = self.save_count.value

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
                game_count += 1

                if self.save_count.value > last_save_count:
                    print('save count changed reloading process')
                    break

                obs = env.reset()

                if hasattr(self.model, 'skip_steps'):
                    for skip in range(self.model.skip_steps):
                        obs, reward, done, info = env.step(0)

                obs = self.model.preprocess_observation(obs)
                state_frames.append(obs)

                while not done:
                    iteration += 1
                    game_length += 1
                    step = self.model.step.eval()

                    # if game_length % num_skip_frames == 0 and len(state_frames) >= num_state_frames:
                    if len(state_frames) >= self.num_state_frames:
                        next_state = np.concatenate(state_frames, axis=2)

                        if state is not None:
                            self.append_replay(replay_memory, state, action, reward, next_state, 1)

                        state = next_state

                        # Online DQN evaluates what to do
                        q_values = self.model.online_q_values.eval(feed_dict={self.model.X_state: [self.convert_state(next_state)]})
                        total_max_q += q_values.max()

                        last_action = action
                        action = self.epsilon_greedy(q_values,
                                                     step)

                    # Online DQN plays
                    obs, reward, done, info = env.step(action)

                    game_score += reward

                    obs = self.model.preprocess_observation(obs)
                    state_frames.append(obs)

                    if len(replay_memory) > 0 and len(replay_memory) % self.mem_save_steps == 0:
                        self.save_memory(replay_memory)
                        replay_memory = []
                        self.clear_old_memory()

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

                epsilon = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)
                print('{} game  {} game {} len: {:d} max_q: {:0.3f} score: {:0.1f} avg: {:0.1f} mem: {:d} fr: {:0.1f}/{:0.1f}'.format(
                                                                               time_string(),
                                                                               step,
                                                                               game_count, 
                                                                               game_length, 
                                                                               mean_max_q,
                                                                               game_score, 
                                                                               avg_score,
                                                                               len(replay_memory),
                                                                               frame_rate,
                                                                               report_rate))
                        


        env.close()


    def play_func(self):
        num_training_start_steps = 1000
        batch_size = 32

        discount_rate = 0.99

        print('game_id:', self.game_id)
        env = gym.make(self.game_id)
        self.num_outputs = env.action_space.n

        with self.get_session(load_model=True, save_model=False, env=env) as sess:
            try:
                iteration = 0

                for epoch in range(self.num_epoch):
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

                    obs = env.reset()

                    if hasattr(self.model, 'skip_steps'):
                        for skip in range(self.model.skip_steps):
                            obs, reward, done, info = env.step(0)

                    game_frames.append(obs)
                    actions.append(action)
                    obs = self.model.preprocess_observation(obs)
                    state_frames.append(obs)

                    while not done:
                        iteration += 1
                        game_length += 1

                        # if game_length % num_skip_frames == 0 and len(state_frames) >= self.num_state_frames:
                        if len(state_frames) >= self.num_state_frames:
                            next_state = np.concatenate(state_frames, axis=2)

                            # Online DQN evaluates what to do
                            q_values = self.model.online_q_values.eval(feed_dict={self.model.X_state: [next_state]})
                            # action = self.epsilon_greedy(q_values,
                            #                              step)
                            action = np.argmax(q_values)

                        # Online DQN plays
                        obs, reward, done, info = env.step(action)

                        game_score += reward

                        game_frames.append(obs)
                        actions.append(action)
                        obs = self.model.preprocess_observation(obs)
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


                    print('game {} len: {:d} max_q: {:0.3f} score: {:0.1f} avg: {:0.1f} max: {:0.1f}'.format(
                                                                                   epoch, 
                                                                                   game_length, 
                                                                                   mean_max_q,
                                                                                   game_score, 
                                                                                   avg_score,
                                                                                   max_score))
                    render_game(game_frames, actions, repeat=False)


                            
            except KeyboardInterrupt:
                print('interrupted')
                raise

        env.close()


    def append_replay(self, replay_memory, state, action, reward, next_state, cont):
        replay_memory.append((state, action, np.array([reward]), next_state, np.array([cont])))

        # if len(replay_memory) >= self.max_num_run_game_replays:
        #     self.save_memory(replay_memory)
        #     replay_memory.erase()


    def convert_state(self, state):
        return state        


    def sample_memories(self, replay_memory, batch_size):
        if len(self.memory_indices) < len(replay_memory):
            new_size = min([int(len(replay_memory) * 2), self.replay_max_memory_length])
            self.memory_indices = np.random.permutation(new_size)
            self.memory_last_i = -1


        cols = [[], [], [], [], []] # state, action, reward, next_state, continue

        while len(cols[0]) < batch_size:
            self.memory_last_i += 1

            if self.memory_last_i >= len(self.memory_indices):
                self.memory_last_i = 0

            i = self.memory_last_i
            idx = self.memory_indices[i]

            if idx >= len(replay_memory):
                continue

            memory = replay_memory[idx]
            for col, value in zip(cols, memory):
                col.append(value)


        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2], cols[3], cols[4]


    def epsilon_greedy(self, q_values, step):
        epsilon = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_outputs) # random action
        else:
            return np.argmax(q_values) # optimal action


    def save_memory(self, replay_memory):
        path = '{}-{:d}'.format(self.memory_path, int(time.time()))
        print('saving memory to:', path, len(replay_memory))

        with open(path, 'wb') as f:
            pickle.dump(len(replay_memory), f)

            for j in range(len(replay_memory)):
                pickle.dump(replay_memory[j], f)

        print('save complete')
        self.memory_queue.put(path)


    def clear_old_memory(self):
        fns = []
        for fn in os.listdir(self.memory_dir):
            match = re.match(self.memory_fn + '-(\\d+)', fn)
            if not match:
                continue
            fns.append((fn, int(match.group(1))))

        if len(fns) * self.mem_save_steps > self.replay_max_memory_length:
            num_to_delete = math.ceil((len(fns) * self.mem_save_steps - self.replay_max_memory_length) / self.mem_save_steps)
            print('#memory_files:', len(fns), self.replay_max_memory_length, 'deleting ', num_to_delete)

            fns = list(sorted(fns), key=lambda x: x[1])
            for i in range(num_to_delete):
                fn = os.path.join(self.memory_dir, fns[i][0])
                print('deleting', fn)
                os.unlink(fn)


    def load_memory(self, replay_memory, memory_fn):
        try:
            print('loading memory from: ', memory_fn)

            with open(memory_fn, 'rb') as fin:
                size = pickle.load(fin)

                for i in range(size):
                    replay_memory.append(pickle.load(fin))

            print('load complete. replay size:', len(replay_memory))
        except EOFError:
            print('eoferror')
            pass


    def load_saved_memory(self, replay_memory):
        print('loaded replay memory length:', len(replay_memory))

        fns = []
        for fn in os.listdir(self.memory_dir):
            match = re.match(self.memory_fn + '-(\\d+)', fn)
            if not match:
                continue
            fns.append((fn, int(match.group(1))))


        for fn, num in sorted(fns, key=lambda x:x[1], reverse=True):
            try:
                path = os.path.join(self.memory_dir, fn)
                print('loading memory from: ', path)

                with open(path, 'rb') as fin:
                    size = pickle.load(fin)

                    for i in range(size):
                        replay_memory.append(pickle.load(fin))

                if len(replay_memory) >= self.replay_max_memory_length:
                    break

            except EOFError:
                print('eoferror')
                pass


    def get_session(parent, load_model=True, save_model=True, env=None):
        class Session:
            def __init__(self, model_class, save_path, env):
                self.model_class = model_class
                self.save_path = save_path
                self.env = env


            def __enter__(self):
                return self.open()


            def __exit__(self, ty, value, tb):
                self.close()


            def run(self, *args, **kwargs):
                return self._sess.run(*args, **kwargs)


            def save(self, path):
                name = parent.game_id
                save_path = os.path.join(path, name)

                # for fn in os.listdir(path):
                #     if fn.startswith(name):
                #         fn_path = os.path.join(path, fn)
                #         print('deleting old file:', fn_path)
                #         os.unlink(fn_path)

                print('saving model: ', save_path)
                self.saver.save(self._sess, save_path)
                print('saved model')


            def restore(self, path):
                name = parent.game_id
                save_path = os.path.join(path, name)

                if not os.path.exists(save_path + '.index'):
                    print('model does not exist:', save_path)
                    return False

                print('restoring model: ', save_path)
                self.saver.restore(self._sess, save_path)
                print('restored model')

                return True


            def open(self):
                print('creating new session. load_model: ', load_model, 'save_model:', save_model)

                tf.reset_default_graph()

                parent.model = self.model_class(self.env)

                self.saver = tf.train.Saver()

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True

                self._sess = tf.Session(config=config)
                self._sess.as_default()
                self._sess.__enter__()

                loaded = False
                if load_model:
                    loaded = self.restore(self.save_path)

                if not loaded:
                    self._sess.run(tf.global_variables_initializer())

                tf.get_default_graph().finalize()

                return self


            def close(self):
                print('closing model')
                if save_model:
                    self.save(self.save_path)

                self._sess.__exit__(None, None, None)                    
                self._sess.close()
                self._sess = None


        return Session(parent.model_class, parent.save_path, env)



    def get_params(self, deep=False):
        return {
            'model_class': self.model_class,
            'num_epoch': self.num_epoch,
            'batch_size': self.batch_size,
            'report_ratio': self.report_ratio,
            'verbose': self.verbose,
            'save_path': self.save_path,
        }


    def set_params(self, params):
        self.model_class = params['model_class']
        self.num_epoch = params['num_epoch']
        self.batch_size = params['batch_size']
        self.report_ratio = params['report_ratio']
        self.verbose = params['verbose']
        self.save_path = params['save_path']
        self._sess = None        

