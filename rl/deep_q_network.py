import os
import time
import json
import logging
import importlib
import inspect

from collections import deque

import gym
import tensorflow as tf
import numpy as np

from .utils.session import Session
from .utils.logging import init_logging, log
from .game.render import render_game
from .data.replay_memory_disk import ReplayMemoryDisk
from .data.replay_memory import ReplayMemory
from .data.replay_sampler import ReplaySampler
from .data.replay_sampler_priority import ReplaySamplerPriority



def time_string():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


class DeepQNetwork:
    DEFAULT_OPTIONS = {
        'save_dir': './data',
        'eps_min': 0.1,
        'eps_max': 1.0,
        'eps_decay_steps': 2000000,
        'discount_rate': 0.99,
        'save_model_steps': 10000,
        'copy_network_steps': 10000,
        'batch_size': 32,
        'model_save_prefix': None,
        'replay_max_memory_length': 300000,
        'replay_cache_size': 300000,
        'max_num_training_steps': 2000000,
        'num_game_frames_before_training': 10000,
        'num_game_steps_per_train': 4,
        'num_train_steps_save_video': None,
        'game_report_interval': 10,
        'train_report_interval': 100,
        'use_episodes': True,
        'use_dueling': False,
        'use_double': False,
        'use_priority': False,
        'use_momentum': False,
        'use_memory': False,
        'use_log': True,
        'frame_skip': 1,
        'max_game_length': 50000,
        'tf_log_level': 3,
    }


    def __init__(self, conf, initialize=False):
        '''
        Constructor

        Will initialize if initialize is true, otherwise it 
        will search for a configuration file in 'save_dir' directory
        specified in the conf dict

        Values in conf will override the saved config if possible
        '''

        self.save_dir = conf['save_dir']

        if initialize:
            # create save dir
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            self.conf = self.DEFAULT_OPTIONS.copy()
        else:
            # go find conf file
            has_conf = False
            for fn in os.listdir(self.save_dir):
                if fn.endswith('.conf'):
                    with open(os.path.join(self.save_dir, fn)) as fin:
                        self.conf = json.load(fin)
                        has_conf = True
                        break

            if not has_conf:
                raise Exception('no conf file found')

        # override saved conf with parameters
        if conf is not None:
            for key, value in conf.items():
                if value is not None:
                    self.conf[key] = value

        self.game_id = self.conf['game_id']
        self.agent = self.conf['agent']

        file_prefix = self.game_id
        self.save_path_prefix = os.path.join(self.save_dir, file_prefix)
        self.conf['save_path_prefix'] = self.save_path_prefix

        if initialize:
            # now write final conf
            conf_path = self.save_path_prefix + '.conf'
            if not os.path.exists(conf_path):
                with open(conf_path, 'w+') as fo:
                    json.dump(self.conf, fo, sort_keys=True, indent=4)

        # init logging
        if self.conf.get('use_log', True):
            init_logging(self.save_path_prefix)

        log('config values:')
        for key, value in self.conf.items():
            log('  {}: {}'.format(key, value))

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(self.conf['tf_log_level'])



    def train(self):
        '''
        Runs training loop
        '''

        start_time = time.time()

        log('train start')

        with Session(self.conf,
                     init_env=True,
                     init_model=True,
                     load_model=True,
                     save_model=True) as self.sess:
            self.train_init()
            self.train_loop()
            self.train_finish()

        elapsed = time.time() - start_time 

        log('train finished in {:0.1f} seconds / {:0.1f} mins / {:0.1f} hours'.format(elapsed, elapsed / 60, elapsed / 60 / 60))


    def train_init(self, init_play_step=True, fill_memories=True):
        '''
        Initialize variables for training
        '''

        # train settings
        self.max_num_training_steps = self.conf['max_num_training_steps']
        self.replay_max_memory_length = self.conf['replay_max_memory_length']
        self.num_game_frames_before_training = self.conf['num_game_frames_before_training']
        self.batch_size = self.conf['batch_size']
        self.save_steps = self.conf['save_model_steps']
        self.copy_steps = self.conf['copy_network_steps']
        self.train_report_interval = self.conf['train_report_interval']
        self.num_game_steps_per_train = self.conf['num_game_steps_per_train']
        self.use_priority = self.conf['use_priority']
        self.num_train_steps_save_video = self.conf['num_train_steps_save_video']

        # other vars
        self.step = self.sess.model.step.eval()

        self.train_report_start_time = time.time()
        self.train_report_last_step = self.step
        self.total_losses = []

        # training batch
        self.train_batch = ReplayMemory(self.sess.model.input_height,
                                        self.sess.model.input_width,
                                        self.sess.model.input_channels,
                                        max_size=self.batch_size,
                                        state_type=self.sess.model.state_type)

        # allocate memory
        if self.conf['use_memory']:
            self.memories = ReplayMemory(self.sess.model.input_height,
                                         self.sess.model.input_width,
                                         self.sess.model.input_channels,
                                         state_type=self.sess.model.state_type,
                                         max_size=self.replay_max_memory_length)

            if os.path.exists(self.get_replay_memory_path()):
                log('loading old memories from', self.get_replay_memory_path())
                old_memories = ReplayMemoryDisk(self.get_replay_memory_path())
                log('found', len(old_memories))
                self.memories.extend(old_memories)
        else:
            self.memories = ReplayMemoryDisk(self.get_replay_memory_path(),
                                             self.sess.model.input_height,
                                             self.sess.model.input_width,
                                             self.sess.model.input_channels,
                                             state_type=self.sess.model.state_type,
                                             max_size=self.replay_max_memory_length,
                                             cache_size=self.conf['replay_cache_size'])

        # initialize memory sampler
        if self.use_priority:
            self.replay_sampler = ReplaySamplerPriority(self.memories)

            self.priority_min = 0.001
            self.per_a = 0.6
            self.per_b = 0.4
            self.per_b_increment_per_sampling = 0.001
            self.absolute_error_upper = 1

            self.tree_idxes = np.zeros((self.batch_size), dtype=int)
            self.priorities = np.zeros((self.batch_size), dtype=float)
        else:
            self.replay_sampler = ReplaySampler(self.memories)


        if init_play_step:
            # initialize run_game step generator
            self.play_step = self.play_generator(is_training=True)


        if fill_memories:
            # fill replay memory when first starting training
            if len(self.replay_sampler) < self.num_game_frames_before_training:
                log('filling memories until', self.num_game_frames_before_training)

            while len(self.replay_sampler) < self.num_game_frames_before_training:
                next(self.play_step)


    def train_loop(self):
        '''
        Main training loop
        '''

        try:
            log('start training')

            while self.step < self.max_num_training_steps:
                self.train_step()
        except (KeyboardInterrupt, StopIteration):
            log('train interrupted')


    def train_step(self):
        '''
        Run one training step
        '''

        # run game steps
        for _ in range(self.num_game_steps_per_train):
            next(self.play_step)

        # sample memories and use the target DQN to produce the target Q-Value
        if self.use_priority:
            self.per_b = np.min([1.0, self.per_b + self.per_b_increment_per_sampling])

            avg_weight = np.power(self.batch_size * (self.replay_sampler.total / len(self.replay_sampler)), -self.per_b)

            self.replay_sampler.sample_memories(self.train_batch,
                                                batch_size=self.batch_size,
                                                tree_idxes=self.tree_idxes,
                                                priorities=self.priorities)

            sampling_probs = self.priorities / self.replay_sampler.total + self.priority_min
            is_weights = np.power(self.batch_size * sampling_probs, -self.per_b) / avg_weight
        else:
            # sample randomly from each range
            self.replay_sampler.sample_memories(self.train_batch,
                                                batch_size=self.batch_size)
            is_weights = None

        # get max q value 
        target_max_q_values = self.get_target_max_q_values(self.train_batch.rewards,
                                                           self.train_batch.continues,
                                                           self.train_batch.next_states)


        # train the model
        self.step, losses, loss = self.sess.model.train(self.train_batch.states,
                                                        self.train_batch.actions,
                                                        target_max_q_values,
                                                        is_weights=is_weights)

        if self.use_priority:
            # update priority steps in sum tree
            losses += self.priority_min
            self.replay_sampler.update_sum_tree(self.tree_idxes, losses)

        self.total_losses.append(loss)

        # Regularly copy the online DQN to the target DQN
        if self.step % self.copy_steps == 0:
            log('copying online to target dqn')
            self.sess.model.copy_online_to_target.run()

        # And save regularly
        if self.step % self.save_steps == 0:
            self.sess.model.game_count.load(self.total_game_count)
            self.sess.save(self.save_path_prefix)

        # save video every so often
        if self.num_train_steps_save_video is not None and \
                self.step % self.num_train_steps_save_video == 0:

            log('saving video at step', self.step)
            for _ in self.run_game(is_training=False,
                                   num_games=1,
                                   use_epsilon=False,
                                   interval=60,
                                   display=False,
                                   save_video=True):
                pass

        # log info
        if self.step % self.train_report_interval == 0:
            elapsed = time.time() - self.train_report_start_time
            if elapsed > 0:
                frame_rate = (self.step - self.train_report_last_step) / elapsed
            else:
                frame_rate = 0.0

            self.train_report_last_step = self.step
            self.train_report_start_time = time.time()

            if len(self.total_losses) > 0:
                avg_loss = sum(self.total_losses) / len(self.total_losses)
            else:
                avg_loss = 0

            self.total_losses.clear()

            log('[train] step {} avg loss: {:0.5f} mem: {:d} fr: {:0.1f}'.format(self.step,
                                                                                      avg_loss,
                                                                                      len(self.replay_sampler),
                                                                                      frame_rate))


    def train_finish(self):
        '''
        Clean up training 
        '''

        log('closing replay memory')

        if self.conf['use_memory']:
            log('saving', len(self.memories), 'memories to disk')
            old_memories = ReplayMemoryDisk(self.get_replay_memory_path(),
                                            self.sess.model.input_height,
                                            self.sess.model.input_width,
                                            self.sess.model.input_channels,
                                            state_type=self.sess.model.state_type,
                                            max_size=self.replay_max_memory_length,
                                            cache_size=0)

            old_memories.extend(self.memories)
            old_memories.close()

            log('saved memory to disk')
        else:
            self.replay_sampler.close()

        # save game count
        self.sess.model.game_count.load(self.total_game_count)


    def predict(self,
                num_games=1,
                use_epsilon=False,
                interval=60,
                display=False, save_video=False):
        '''
        Runs game with the given model and calculates the scores
        '''

        with Session(self.conf,
                     init_env=True,
                     init_model=False,
                     load_model=True,
                     save_model=False) as self.sess:
            for _ in self.play_generator(is_training=False,
                                         num_games=num_games,
                                         use_epsilon=use_epsilon,
                                         display=display,
                                         save_video=save_video):
                pass


    def play_generator(self,
                       is_training=False,
                       num_games=None,
                       use_epsilon=False,
                       display=False,
                       save_video=False):
        '''
        Generator which will run the game one frame
        '''

        self.play_init()

        try:
            while num_games is None or self.cur_game_count < num_games:
                for _ in self.play_game_generator(is_training=is_training, 
                                                   use_epsilon=use_epsilon, 
                                                   display=display, 
                                                   save_video=save_video):
                    yield


        except KeyboardInterrupt:
            log('play interrupted')


    def play_init(self):
        '''
        Initialize play variables
        '''

        # run game settings
        self.eps_min = self.conf['eps_min']
        self.eps_max = self.conf['eps_max']
        self.eps_decay_steps = self.conf['eps_decay_steps']
        self.discount_rate = self.conf['discount_rate']
        self.use_episodes = self.conf['use_episodes']
        self.frame_skip = self.conf['frame_skip']
        self.max_game_length = self.conf['max_game_length']
        self.batch_size = self.conf['batch_size']

        self.total_game_count = self.sess.run([self.sess.model.game_count])[0]
        self.cur_game_count = 0

        self.step = self.sess.model.step.eval()

        # batch memory
        self.play_batch = ReplayMemory(self.sess.model.input_height,
                                       self.sess.model.input_width,
                                       self.sess.model.input_channels,
                                       max_size=self.batch_size,
                                       state_type=self.sess.model.state_type)


        # reporting stats
        self.play_game_scores = deque(maxlen=100)
        self.play_max_qs = deque(maxlen=100)


    def play_game_generator(self,
                            game_state=None,
                            is_training=True,
                            num_games=None,
                            use_epsilon=False,
                            display=False,
                            save_video=False):
        '''
        Generator which yields every frame and runs one game
        '''

        # reset game
        if game_state is None:
            game_state = self.make_game_state()

        while not game_state['game_done']:
            # play out episode
            for _ in self.play_episode_generator(game_state, 
                                                 is_training=is_training,
                                                 use_epsilon=use_epsilon,
                                                 display=display,
                                                 save_video=save_video):
                yield
            
            # update frame state of episode / game
            self.update_frame_state(game_state, 
                                    is_training=is_training, 
                                    cont=0)

            # return to training
            yield

        self.total_game_count += 1
        self.cur_game_count += 1

        self.play_game_scores.append(game_state['score'])
        self.play_max_qs.append(game_state['total_max_q'] / game_state['game_length'])

        self.report_play_stats(game_state, 
                               is_training=is_training)

        if save_video:
            save_path = os.path.join(self.save_dir,
                                     'video-{}-{}.mp4'.format(self.step,
                                                              self.total_game_count))
        else:
            save_path = None

        if save_video or display:
            render_game(game_state['frames_render'],
                        game_state['actions_render'],
                        repeat=False,
                        save_path=save_path,
                        display=display)   

        return game_state     


    def play_episode_generator(self, 
                               game_state, 
                               is_training=True, 
                               use_epsilon=False,
                               display=False,                     
                               save_video=False):
        '''
        Generator which yields every frame and runs one episode
        '''

        game_state['episode_length'] = 0
        game_state['episode_done'] = False

        while not game_state['episode_done'] and not game_state['game_done']:
            game_state['game_length'] += 1
            game_state['episode_length'] += 1

            if game_state['game_length'] > self.max_game_length:
                break

            if len(game_state['frames']) >= self.sess.model.num_frame_per_state:
                self.update_frame_state(game_state, 
                                        is_training=is_training, 
                                        cont=1)

                # return to training
                yield

                # Online DQN evaluates what to do
                q_values = self.sess.model.predict([game_state['state']])

                game_state['total_max_q'] += q_values.max()


                if is_training or use_epsilon:
                    game_state['action'] = self.epsilon_greedy(q_values, self.step)
                else:
                    game_state['action'] = np.argmax(q_values)

            game_state['action'] = self.sess.model.before_action(game_state['action'], 
                                                                 game_state['obs'], 
                                                                 game_state['reward'], 
                                                                 game_state['game_done'], 
                                                                 game_state['info'])

            # run action for frame_skip steps
            game_state['reward'] = 0
            for i in range(self.frame_skip):
                # online network plays
                game_state['obs'], step_reward, game_state['game_done'], game_state['info'] = self.sess.env.step(game_state['action'])

                if not is_training and (save_video or display):
                    game_state['actions_render'].append(game_state['action'])
                    game_state['frames_render'].append(self.sess.model.render_obs(game_state['obs']))

                game_state['reward'] += step_reward

                # check for episode change
                if self.use_episodes and 'ale.lives' in game_state['info']:
                    num_lives = game_state['info']['ale.lives']

                    if game_state['num_lives'] is not None and num_lives != game_state['num_lives']:
                        game_state['episode_done'] = True

                    if num_lives <= 0:
                        game_state['game_done'] = True

                    game_state['num_lives'] = num_lives

                if game_state['game_done']:
                    break

                if game_state['episode_done']:
                    break

            game_state['score'] += game_state['reward']
            game_state['obs'] = self.sess.model.preprocess_observation(game_state['obs'])
            game_state['frames'].append(game_state['obs'])    

        return game_state


    def make_game_state(self):
        '''
        Initialized game state variables
        '''

        game_state = {
            'game_start_time': time.time(),
            'game_done': False,
            'episode_done': False,
            'frames': deque(maxlen=self.sess.model.num_frame_per_state),
            'total_max_q': 0,
            'iteration': 0,
            'game_length': 0,
            'episode_length': 0,
            'score': 0,
            'obs': None,
            'state': None,
            'action': 0,
            'reward': None,
            'info': None,
            'num_lives': None,
            'actions_render': [],
            'frames_render': []
        }        

        game_state['obs'] = self.sess.model.preprocess_observation(self.sess.env.reset())
        game_state['frames'].append(game_state['obs'])

        return game_state


    def update_frame_state(self, game_state, is_training=True, cont=1):
        '''
        Makes a frame state from 4 sequential frames.
        Also saves memories to replay memory
        '''

        next_state = self.make_state(game_state['frames'])

        if is_training and game_state['state'] is not None:
            self.add_memories(state=game_state['state'], 
                              action=game_state['action'], 
                              reward=game_state['reward'], 
                              cont=cont, 
                              next_state=next_state)

        game_state['state'] = next_state        


    def add_memories(self, state, action, reward, cont, next_state):
        '''
        Add to replay memories
        '''
        self.play_batch.append(state=state,
                               action=action,
                               reward=reward,
                               cont=cont,
                               next_state=next_state)

        if len(self.play_batch) >= self.batch_size:
            if self.conf['use_priority']:
                target_max_q_values = self.get_target_max_q_values(self.play_batch.rewards,
                                                                   self.play_batch.continues,
                                                                   self.play_batch.next_states)

                losses = self.sess.model.get_losses(self.play_batch.states,
                                                    self.play_batch.actions,
                                                    target_max_q_values)
                losses += self.priority_min

                for i in range(len(self.play_batch)):
                    self.replay_sampler.append(state=self.play_batch.states[i],
                                               action=self.play_batch.actions[i],
                                               reward=self.play_batch.rewards[i],
                                               next_state=self.play_batch.next_states[i],
                                               cont=self.play_batch.continues[i],
                                               loss=losses[i])
            else:
                # print(self.play_batch.states.shape,
                #       self.play_batch.actions.shape)
                for i in range(len(self.play_batch)):
                    self.replay_sampler.append(state=self.play_batch.states[i],
                                               action=self.play_batch.actions[i],
                                               reward=self.play_batch.rewards[i],
                                               next_state=self.play_batch.next_states[i],
                                               cont=self.play_batch.continues[i],
                                               loss=0)

            self.play_batch.clear()


    def report_play_stats(self, game_state, is_training=True):
        '''
        Replay on play stats
        '''

        if not is_training or self.total_game_count % self.sess.model.game_report_interval == 0:
            if game_state['game_length'] > 0:
                mean_max_q = game_state['total_max_q'] / game_state['game_length']
            else:
                mean_max_q = 0

            elapsed = time.time() - game_state['game_start_time']
            if elapsed > 0:
                frame_rate = game_state['game_length'] / (time.time() - game_state['game_start_time'])
            else:
                frame_rate = 0.0

            if len(self.play_game_scores) > 0:
                avg_score = sum(self.play_game_scores) / len(self.play_game_scores)
            else:
                avg_score = 0

            min_score = None
            max_score = None

            for score in self.play_game_scores:
                if max_score is None or score > max_score:
                    max_score = score

                if min_score is None or score < min_score:
                    min_score = score

            if len(self.play_max_qs) > 0:
                avg_max_q = sum(self.play_max_qs) / len(self.play_max_qs)
            else:
                avg_max_q = 0

            epsilon = self.epsilon(self.step)

            if is_training:
                mem_len = len(self.replay_sampler)
            else:
                mem_len = 0

            log('[play] step {} game {}/{} len: {:d} max_q: {:0.3f}/{:0.3f} score: {:0.1f}/{:0.1f}/{:0.1f} mem: {:d} eps: {:0.3f} fr: {:0.1f}'.format(
                       self.step,
                       self.cur_game_count,
                       self.total_game_count,
                       game_state['game_length'],
                       mean_max_q,
                       avg_max_q,
                       avg_score,
                       min_score,
                       max_score,
                       mem_len,
                       epsilon,
                       frame_rate))


    def epsilon(self, step):
        return max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)


    def epsilon_greedy(self, q_values, step):
        epsilon = self.epsilon(step)
        if np.random.rand() < epsilon:
            return np.random.randint(self.sess.model.num_outputs) # random action
        else:
            return np.argmax(q_values) # optimal action


    def make_state(self, frames):
        return np.concatenate(frames, axis=2)


    def get_target_max_q_values(self, rewards, continues, next_states):
        max_next_q_values = self.sess.model.get_max_q_value(next_states)

        return rewards + continues * self.discount_rate * max_next_q_values


    # def get_losses(self, states, actions, max_q_values):
    #     return self.sess.model.losses.eval(feed_dict={
    #                                     self.sess.model.X_state: states,
    #                                     self.sess.model.X_action: actions,
    #                                     self.sess.model.y: max_q_values
    #                                   })


    def get_replay_memory_path(self):
        return '{}_replay_memory.hdf5'.format(self.save_path_prefix)


