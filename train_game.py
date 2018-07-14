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
    

class GameTrainer(BaseEstimator):
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


    def fit(self):
        p = multiprocessing.Process(target=self.fit_func)
        p.start()
        p.join()


    def fit_func(self):
        max_acc = 0
        max_rec = 0

        save_steps = 10000
        copy_steps = 10000

        num_state_frames = 4
        num_training_start_steps = 10000
        max_num_training = 20000000

        discount_rate = 0.99

        game_scores = deque(maxlen=25)
        report_interval = 60

        self.replay_max_memory_length = 150000
        mem_save_steps = int(self.replay_max_memory_length / 2)
        self.replay_memory = RingBuf(self.replay_max_memory_length)        
        # self.replay_memory = deque(maxlen=self.replay_max_memory_length)

        self.memory_fn = '{}.memory'.format(self.game_id)
        self.memory_path = os.path.join(self.save_path, '{}.memory'.format(self.game_id))
        self.memory_dir = os.path.dirname(self.memory_path)

        self.memory_indices = np.random.permutation(num_training_start_steps)
        self.memory_last_i = -1
        # memory_fn = '{}.memory'.format(self.game_id)
        # memory_path = os.path.join(self.save_path, '{}.memory'.format(self.game_id))

        # memory_dir = os.path.dirname(memory_path)

        # for fn in os.listdir(memory_dir):
        #     try:
        #         if re.match(memory_fn + '-\\d+-\\d+', fn):
        #             path = os.path.join(memory_dir, fn)
        #             print('load memory: ', path)

        #             with open(path, 'rb') as fin:
        #                 size = pickle.load(fin)

        #                 for i in range(size):
        #                     self.replay_memory.append(pickle.load(fin))
        #     except EOFError:
        #         print('eoferror')
        #         pass

        self.load_memory()

        print('loaded replay memory length:', len(self.replay_memory))

        print('game_id:', self.game_id)
        self.env = gym.make(self.game_id)
        self.num_outputs = self.env.action_space.n


        with self.get_session(load_model=True) as sess:
            # print(sess.model.skip_steps)

            try:
                iteration = 0
                report_start_time = time.time()
                report_last_iteration = 0
                report_rate = 0
                step = 0

                while step < max_num_training:
                    epoch_start_time = time.time()
                    epoch_count = 0

                    total_max_q = 0.0
                    game_length = 0
                    game_score = 0
                    state_frames = deque(maxlen=num_state_frames)
                    action = 0
                    state = None
                    next_state = None
                    done = False
                    losses = []

                    obs = self.env.reset()

                    if hasattr(sess.model, 'skip_steps'):
                        for skip in range(sess.model.skip_steps):
                            obs, reward, done, info = self.env.step(0)

                    obs = sess.model.preprocess_observation(obs)
                    state_frames.append(obs)

                    while not done:
                        iteration += 1
                        game_length += 1
                        step = sess.model.step.eval()

                        # if game_length % num_skip_frames == 0 and len(state_frames) >= num_state_frames:
                        if len(state_frames) >= num_state_frames:
                            next_state = np.concatenate(state_frames, axis=2)

                            if state is not None:
                                # self.replay_memory.append((state, 
                                #                            action, 
                                #                            reward, 
                                #                            next_state, 
                                #                            1.0))
                                # self.replay_memory.append((state, action, np.array([reward]), next_state, np.array([1.0]m))
                                self.append_replay(state, action, reward, next_state, 1)

                            state = next_state

                            # Online DQN evaluates what to do
                            q_values = sess.model.online_q_values.eval(feed_dict={sess.model.X_state: [self.convert_state(next_state)]})
                            total_max_q += q_values.max()

                            last_action = action
                            action = self.epsilon_greedy(q_values,
                                                         step)

                        # Online DQN plays
                        obs, reward, done, info = self.env.step(action)

                        game_score += reward

                        obs = sess.model.preprocess_observation(obs)
                        state_frames.append(obs)

                        if done: 
                            continue

                        if len(self.replay_memory) < num_training_start_steps:
                            # only train after warmup period and at regular intervals
                            continue 

                        # Sample memories and use the target DQN to produce the target Q-Value
                        X_state_val, X_action_val, rewards, X_next_state_val, continues = self.sample_memories(self.batch_size)

                        next_q_values = sess.model.target_q_values.eval(
                            feed_dict={sess.model.X_state: self.convert_state(X_next_state_val)})
                        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
                        y_val = rewards + continues * discount_rate * max_next_q_values

                        # Train the online DQN
                        tr_res, loss_val = sess.run([sess.model.training_op, 
                                                     sess.model.loss], 
                                                    feed_dict={
                                                        sess.model.X_state: self.convert_state(X_state_val), 
                                                        sess.model.X_action: X_action_val, 
                                                        sess.model.y: y_val
                                                    })
                        losses.append(loss_val)

                        # Regularly copy the online DQN to the target DQN
                        if step % copy_steps == 0:
                            print('copying to target network')
                            sess.model.copy_online_to_target.run()

                        # And save regularly
                        if step % save_steps == 0:
                            sess.save(self.save_path)

                        if step % mem_save_steps == 0:
                            self.save_memory()

                    # game done, save last step
                    next_state = np.concatenate(state_frames, axis=2)# 
                    # self.replay_memory.append((state, action, np.array([reward]), next_state, np.array([0]))
                    self.append_replay(state, action, reward, next_state, 0)

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


                    if len(losses) > 0:
                        avg_loss = sum(losses) / len(losses)
                    else:
                        avg_loss = 0

                    _, game_count = sess.run([sess.model.game_count_op, sess.model.game_count])

                    game_scores.append(game_score)

                    if len(game_scores) > 0:
                        avg_score = sum(game_scores) / len(game_scores)
                    else: 
                        avg_score = 0

                    epsilon = max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)
                    print('{} {}/{} game {} len: {:d} max_q: {:0.3f} loss: {:0.3f} score: {:0.1f} avg: {:0.1f} mem: {:d} eps: {:0.3f} fr: {:0.1f}/{:0.1f}'.format(
                                                                                   time_string(),
                                                                                   iteration,
                                                                                   step,
                                                                                   game_count, 
                                                                                   game_length, 
                                                                                   mean_max_q,
                                                                                   avg_loss,
                                                                                   game_score, 
                                                                                   avg_score,
                                                                                   len(self.replay_memory),
                                                                                   epsilon,
                                                                                   frame_rate,
                                                                                   report_rate))
                            
            except KeyboardInterrupt:
                print('interrupted')
                pass

        self.env.close()
        self.save_memory()


    def play_func(self):
        num_state_frames = 4
        num_training_start_steps = 1000
        batch_size = 32

        discount_rate = 0.99

        print('game_id:', self.game_id)
        self.env = gym.make(self.game_id)
        self.num_outputs = self.env.action_space.n

        with self.get_session(load_model=True, save_model=False) as sess:
            try:
                iteration = 0

                for epoch in range(self.num_epoch):
                    epoch_start_time = time.time()

                    total_max_q = 0.0
                    game_length = 0
                    game_score = 0
                    state_frames = deque(maxlen=num_state_frames)
                    game_frames = []
                    game_scores = []
                    actions = []
                    action = 0
                    state = None
                    next_state = None
                    done = False

                    obs = self.env.reset()

                    if hasattr(sess.model, 'skip_steps'):
                        for skip in range(sess.model.skip_steps):
                            obs, reward, done, info = self.env.step(0)

                    game_frames.append(obs)
                    actions.append(action)
                    obs = sess.model.preprocess_observation(obs)
                    state_frames.append(obs)

                    while not done:
                        iteration += 1
                        game_length += 1

                        # if game_length % num_skip_frames == 0 and len(state_frames) >= num_state_frames:
                        if len(state_frames) >= num_state_frames:
                            next_state = np.concatenate(state_frames, axis=2)

                            # Online DQN evaluates what to do
                            q_values = sess.model.online_q_values.eval(feed_dict={sess.model.X_state: [next_state]})
                            # action = self.epsilon_greedy(q_values,
                            #                              step)
                            action = np.argmax(q_values)

                        # Online DQN plays
                        obs, reward, done, info = self.env.step(action)

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

        self.env.close()


    def append_replay(self, state, action, reward, next_state, cont):
        self.replay_memory.append((state, action, np.array([reward]), next_state, np.array([cont])))



    def convert_state(self, state):
        return state        


    def sample_memories(self, batch_size):
        if len(self.memory_indices) < len(self.replay_memory):
            new_size = min([int(len(self.memory_indices) * 2), self.replay_max_memory_length])
            self.memory_indices = np.random.permutation(new_size)
            self.memory_last_i = -1

        # cols = [[], [], [], [], []] # state, action, reward, next_state, continue

        # while len(cols[0]) < batch_size:
        #     self.memory_last_i += 1
        #     if self.memory_last_i >= len(self.memory_indices):
        #         self.memory_last_i = 0
        #     i = self.memory_last_i
        #     idx = self.memory_indices[i]
        #     if i > len(self.replay_memory):
        #         continue
        #     memory = self.replay_memory[idx]
        #     for col, value in zip(cols, memory):
        #         col.append(value)


        # cols = [np.array(col) for col in cols]
        # return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)


        cols = [[], [], [], [], []] # state, action, reward, next_state, continue

        while len(cols[0]) < batch_size:
            self.memory_last_i += 1

            if self.memory_last_i >= len(self.memory_indices):
                self.memory_last_i = 0

            i = self.memory_last_i
            idx = self.memory_indices[i]

            if idx >= len(self.replay_memory):
                continue

            memory = self.replay_memory[idx]
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


    def save_memory(self):
        dump_size = 10000
        num_files = int(math.ceil(len(self.replay_memory) / dump_size))

        print('saving memory. memory length:', len(self.replay_memory), num_files)
        for i in range(num_files):
            path = '{}-{:d}'.format(self.memory_path, i, num_files)
            print('saving memory to:', path)

            with open(path, 'wb') as f:
                if i < num_files - 1:
                    f_dump_size = dump_size
                else:
                    f_dump_size = len(self.replay_memory) % dump_size

                pickle.dump(f_dump_size, f)

                for j in range(f_dump_size):
                    pickle.dump(self.replay_memory[i*dump_size + j], f)


    def load_memory(self):
        fns = []
        for fn in os.listdir(self.memory_dir):
            match = re.match(self.memory_fn + '-(\\d+)(-\\d+)?', fn)
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
                        self.replay_memory.append(pickle.load(fin))

                if len(self.replay_memory) >= self.replay_max_memory_length:
                    break

            except EOFError:
                print('eoferror')
                pass


    def get_session(parent, load_model=True, save_model=True):
        class Session:
            def __init__(self, model_class, save_path):
                self.model_class = model_class
                self.save_path = save_path


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

                self.model = self.model_class(parent.env.action_space.n)
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


        return Session(parent.model_class, parent.save_path)



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


if __name__ == '__main__':
    from game_player import GamePlayer
    gt = GameTrainer('MsPacman-v0', GamePlayer)

    gt.fit_func()
