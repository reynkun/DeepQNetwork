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

from .game_runner import GameRunner
from .utils.logging import init_logging, log
from .game.render import render_game
from .data.replay_memory_disk import ReplayMemoryDisk
from .data.replay_memory import ReplayMemory
from .data.replay_sampler import ReplaySampler
from .data.replay_sampler_priority import ReplaySamplerPriority


class DeepQNetwork:
    DEFAULT_OPTIONS = {
        'save_dir': './data',
        'game_id': None,
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
        'train_report_interval': 100,
        'use_episodes': True,
        'use_dueling': False,
        'use_double': False,
        'use_per': False,
        'use_per_annealing': False,
        'use_momentum': False,
        'use_memory': True,
        'use_log': True,
        'frame_skip': 1,
        'max_game_length': 50000,
        'tf_log_level': 3,
        'per_a': 0.6,
        'per_b_start': 0.4,
        'per_b_end': 1,
        'per_anneal_steps': 2000000
    }


    MAX_MEMORY_BATCH_SIZE = 128
    MIN_ERROR_PRIORITY = 0.01
    MAX_ERROR_PRIORITY = 1.0


    def __init__(self, conf, initialize=False):
        '''
        Constructor

        Will initialize if initialize is true, otherwise it 
        will search for a configuration file in 'save_dir' directory
        specified in the conf dict

        Values in conf will override the saved config if possible
        '''

        self._init_conf(conf, initialize=initialize)
        self._save_conf()
        self._init_logging()
        self._init_env()
        self._init_agent()


    def _init_conf(self, conf=None, initialize=False):
        '''
        Will create a conf if initialize is true, otherwise it 
        will search for a configuration file in 'save_dir' directory
        specified in the conf dict

        Finally, values in conf will override the saved config 
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
        for key, value in conf.items():
            if value is not None:
                self.conf[key] = value


    def _save_conf(self):
        '''
        Save final conf for subsequent runs
        '''
        # save final config to directory
        file_prefix = self.conf['environment']
        self.save_path_prefix = os.path.join(self.save_dir, file_prefix)
        self.conf['save_path_prefix'] = self.save_path_prefix

        conf_path = self.save_path_prefix + '.conf'
        if not os.path.exists(conf_path):
            with open(conf_path, 'w+') as fo:
                json.dump(self.conf, fo, sort_keys=True, indent=4)


    def _init_logging(self):
        '''
        Initialize logging and output conf
        '''

        # init logging
        if self.conf.get('use_log', True):
            init_logging(self.save_path_prefix)

        log('config values:')
        for key, value in self.conf.items():
            log('  {}: {}'.format(key, value))

        # reduce log out for tensorflow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(self.conf['tf_log_level'])


    def _init_env(self):
        '''
        Initialize game env
        '''

        # make environment
        if isinstance(self.conf['environment'], str):
            mod_env_str, cl_env_str = self.conf['environment'].rsplit('.', 1)
            mod_ag = importlib.import_module(mod_env_str)
            env_class = getattr(mod_ag, cl_env_str)
        elif inspect.isclass(self.conf['2']):
            env_class = self.conf['environment']
        else:
            raise Exception('invalid env class')

        self.env = env_class(self.conf['game_id'])
        self.conf['action_space'] = self.env.get_action_space()


    def _init_agent(self):
        '''
        Initialize game agent
        '''

        # make agent
        if isinstance(self.conf['agent'], str):
            mod_agent_str, cl_agent_str = self.conf['agent'].rsplit('.', 1)
            mod_ag = importlib.import_module(mod_agent_str)

            agent_class = getattr(mod_ag, cl_agent_str)
        elif inspect.isclass(self.conf['agent']):
            agent_class = self.conf['agent']
        else:
            raise Exception('invalid agent class')

        self.agent = agent_class(conf=self.conf)


    def train(self):
        '''
        Runs training loop
        '''

        start_time = time.time()

        log('train start')

        with self.agent:
            self._train_init()
            self._train_loop()
            self._train_finish()

        elapsed = time.time() - start_time 

        log('train finished in {:0.1f} seconds / {:0.1f} mins / {:0.1f} hours'.format(elapsed, elapsed / 60, elapsed / 60 / 60))


    def _train_init(self):
        '''
        Initialize variables for training
        '''

        # pull some settings from config
        self.max_num_training_steps = self.conf['max_num_training_steps']
        self.replay_max_memory_length = self.conf['replay_max_memory_length']
        self.num_game_frames_before_training = self.conf['num_game_frames_before_training']
        self.batch_size = self.conf['batch_size']
        self.save_steps = self.conf['save_model_steps']
        self.copy_steps = self.conf['copy_network_steps']
        self.train_report_interval = self.conf['train_report_interval']
        self.num_game_steps_per_train = self.conf['num_game_steps_per_train']
        self.use_per = self.conf['use_per']
        self.num_train_steps_save_video = self.conf['num_train_steps_save_video']
        self.discount_rate = self.conf['discount_rate']

        self.step = self.agent.get_training_step()
        self.train_report_start_time = time.time()
        self.train_report_last_step = self.step
        self.total_losses = []

        # training batch
        self.train_batch = ReplayMemory(self.agent.INPUT_HEIGHT,
                                        self.agent.INPUT_WIDTH,
                                        self.agent.INPUT_CHANNELS,
                                        max_size=self.batch_size,
                                        state_type=self.agent.STATE_TYPE)

        if self.use_per:
            # batch memory
            self.memory_batch = ReplayMemory(self.agent.INPUT_HEIGHT,
                                             self.agent.INPUT_WIDTH,
                                             self.agent.INPUT_CHANNELS,
                                             max_size=self.MAX_MEMORY_BATCH_SIZE,
                                             state_type=self.agent.STATE_TYPE)

        # allocate replay memory
        if self.conf['use_memory']:
            self.memories = ReplayMemory(self.agent.INPUT_HEIGHT,
                                         self.agent.INPUT_WIDTH,
                                         self.agent.INPUT_CHANNELS,
                                         state_type=self.agent.STATE_TYPE,
                                         max_size=self.replay_max_memory_length)

            if os.path.exists(self._get_replay_memory_path()):
                log('loading old memories from', self._get_replay_memory_path())
                old_memories = ReplayMemoryDisk(self._get_replay_memory_path())
                log('found', len(old_memories))
                self.memories.extend(old_memories)
        else:
            self.memories = ReplayMemoryDisk(self._get_replay_memory_path(),
                                             self.agent.INPUT_HEIGHT,
                                             self.agent.INPUT_WIDTH,
                                             self.agent.INPUT_CHANNELS,
                                             state_type=self.agent.STATE_TYPE,
                                             max_size=self.replay_max_memory_length,
                                             cache_size=self.conf['replay_cache_size'])

        # initialize memory sampler
        if self.use_per:
            self._init_per()
        else:
            self.replay_sampler = ReplaySampler(self.memories)


        # initialize run_game step generator
        self.game_runner = GameRunner(self.conf, self.env, self.agent)
        self.play_step = self.game_runner.play_generator()


        # fill replay memory when first starting training
        if len(self.replay_sampler) < self.num_game_frames_before_training:
            log('filling memories until', self.num_game_frames_before_training)

            while len(self.replay_sampler) < self.num_game_frames_before_training:
                self._game_step()


    def _init_per(self):
        '''
        Initialize prioritized experience replay variables
        '''

        self.replay_sampler = ReplaySamplerPriority(self.memories)

        self.per_a = self.conf['per_a']

        self.per_b_start = self.conf['per_b_start']
        self.per_b_end = self.conf['per_b_end']
        self.per_b = self.per_b_start

        self.per_anneal_steps = self.conf['per_anneal_steps']

        self.last_min_loss = None
        self.last_max_loss = None
        self.last_max_weight = None

        self.tree_idxes = np.zeros((self.batch_size), dtype=int)
        self.priorities = np.zeros((self.batch_size), dtype=float)


    def _train_loop(self):
        '''
        Main training loop
        '''

        try:
            log('start training')

            while self.step < self.max_num_training_steps:
                self._train_step()
        except (KeyboardInterrupt, StopIteration):
            log('train interrupted')


    def _train_step(self):
        '''
        Run one training step
        '''

        self._train_run_and_train()        
        self._train_update_model()
        self._train_report_stats()


    def _train_run_and_train(self):
        '''
        Run game steps and train model
        '''
        # run game steps
        for _ in range(self.num_game_steps_per_train):
            self._game_step()

        
        self._sample_memories()

        # get max q value 
        target_max_q_values = self._get_target_max_q_values(self.train_batch.rewards,
                                                            self.train_batch.continues,
                                                            self.train_batch.next_states)


        # train the model
        self.step, losses, loss = self.agent.train(self.train_batch.states,
                                                   self.train_batch.actions,
                                                   target_max_q_values,
                                                   is_weights=self.is_weights)

        if self.use_per:
            # update priority steps in sum tree
            losses = self._make_losses(losses)
                    
            self.replay_sampler.update_sum_tree(self.tree_idxes, losses)

        self.total_losses.append(loss)


    def _train_update_model(self):
        '''
        Update target model and save model to disk
        '''
        # Regularly copy the online DQN to the target DQN
        if self.step % self.copy_steps == 0:
            log('copying online to target dqn')
            self.agent.copy_network()


        # And save regularly
        if self.step % self.save_steps == 0:
            self.agent.set_game_count(self.game_runner.total_game_count)
            self.agent.save(self.save_path_prefix)


    def _train_report_stats(self):
        '''
        Train and report stats to the log
        '''

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

            if self.use_per:
                per_str = ' per_ab: {:0.2f}/{:0.2f} avg/min/max loss: {:0.3f}/{:0.3f}/{:0.3f} max_weight: {:0.3f} sum total: {:0.1f} '.format(self.per_a, self.per_b, self.last_avg_loss, self.last_min_loss, self.last_max_loss, self.last_max_weight, self.replay_sampler.total)
            else:
                per_str = ''

            log('[train] step {} avg loss: {:0.5f}{}mem: {:d} fr: {:0.1f}'.format(self.step,
                                                                                    avg_loss,
                                                                                    per_str,
                                                                                    len(self.replay_sampler),
                                                                                    frame_rate))


    def _train_finish(self):
        '''
        Clean up training 
        '''

        log('closing replay memory')

        if self.conf['use_memory']:
            # save memories to disk if using ram 
            log('saving', len(self.memories), 'memories to disk')
            old_memories = ReplayMemoryDisk(self._get_replay_memory_path(),
                                            self.agent.INPUT_HEIGHT,
                                            self.agent.INPUT_WIDTH,
                                            self.agent.INPUT_CHANNELS,
                                            state_type=self.agent.STATE_TYPE,
                                            max_size=self.replay_max_memory_length,
                                            cache_size=0)

            old_memories.extend(self.memories)
            old_memories.close()

            log('saved memory to disk')
        else:
            self.replay_sampler.close()

        # save game count
        self.agent.set_game_count(self.game_runner.total_game_count)
        self.agent.save(self.save_path_prefix)


    def _game_step(self):
        '''
        Run a game step
        '''

        self._update_priority_sampling()

        next(self.play_step)

        if self.game_runner.game_state['old_state'] is not None:
            self._add_memories(state=self.game_runner.game_state['old_state'], 
                               action=self.game_runner.game_state['action'], 
                               reward=self.game_runner.game_state['reward'], 
                               cont=self.game_runner.game_state['cont'], 
                               next_state=self.game_runner.game_state['state'])                


    def _update_priority_sampling(self):
        '''
        Update the strength of priority sampling if set
        '''

        if self.use_per and self.conf['use_per_annealing']:
            self.per_b = min(self.per_b_end, self.per_b_start + self.step * ((self.per_b_end - self.per_b_start) / self.per_anneal_steps))


    def _add_memories(self, state, action, reward, cont, next_state):
        '''
        Add to replay memories
        '''

        if self.use_per:
            self._add_priority_memory(state, action, reward, cont, next_state)
        else:
            self.replay_sampler.append(state=state,
                                       action=action,
                                       reward=reward,
                                       next_state=next_state,
                                       cont=cont)


    def _add_priority_memory(self, state, action, reward, cont, next_state):
        self.memory_batch.append(state=state,
                                 action=action,
                                 reward=reward,
                                 cont=cont,
                                 next_state=next_state)


        if len(self.memory_batch) >= self.MAX_MEMORY_BATCH_SIZE:                                    
            target_max_q_values = self._get_target_max_q_values(self.memory_batch.rewards,
                                                                self.memory_batch.continues,
                                                                self.memory_batch.next_states)

            losses = self.agent.get_losses(self.memory_batch.states,
                                           self.memory_batch.actions,
                                           target_max_q_values)
            losses = self._make_losses(losses)

            for i in range(len(self.memory_batch)):
                self.replay_sampler.append(state=self.memory_batch.states[i],
                                           action=self.memory_batch.actions[i],
                                           reward=self.memory_batch.rewards[i],
                                           next_state=self.memory_batch.next_states[i],
                                           cont=self.memory_batch.continues[i],
                                           loss=losses[i])
            self.memory_batch.clear()

            # if len(self.replay_sampler) <= 0:
            #     max_loss = self.MAX_ERROR_PRIORITY
            # else:
            #     max_loss = self.replay_sampler.get_max()
            #     log('max loss:', max_loss)


            # for i in range(len(self.memory_batch)):
            #     self.replay_sampler.append(state=self.memory_batch.states[i],
            #                                action=self.memory_batch.actions[i],
            #                                reward=self.memory_batch.rewards[i],
            #                                next_state=self.memory_batch.next_states[i],
            #                                cont=self.memory_batch.continues[i],
            #                                loss=max_loss)
            # self.memory_batch.clear()


    def _sample_memories(self):
        '''
        Sample game steps and write to train batch
        '''

        if self.use_per:
            if self.step % 5000 == 0 or self.last_max_weight is None:
                self.last_avg_loss = self.replay_sampler.get_average()
                self.last_min_loss = max(self.replay_sampler.get_min(), self.MIN_ERROR_PRIORITY)
                self.last_max_loss = min(self.replay_sampler.get_max(), self.MAX_ERROR_PRIORITY)
                self.last_max_weight = pow(self.batch_size * (self.last_min_loss / self.replay_sampler.total), -self.per_b)

            # max_weight = pow(self.batch_size * (self.replay_sampler.get_min() + self.MIN_ERROR_PRIORITY), -self.per_b)

            # sample memories from sum tree
            self.replay_sampler.sample_memories(self.train_batch,
                                                batch_size=self.batch_size,
                                                tree_idxes=self.tree_idxes,
                                                priorities=self.priorities)

            # log(len(self.replay_sampler.sum_tree), self.tree_idxes, self.priorities)
            sampling_probs = self.priorities / self.replay_sampler.total
            # log(sampling_probs)
            self.is_weights = np.power(self.batch_size * sampling_probs, -self.per_b) / self.last_max_weight 
        else:
            # sample randomly from each range
            self.replay_sampler.sample_memories(self.train_batch,
                                                batch_size=self.batch_size)
            self.is_weights = None


    def _get_target_max_q_values(self, rewards, continues, next_states):
        '''
        Get max q_values with discount
        '''

        max_next_q_values = self.agent.get_max_q_value(next_states)

        return rewards + continues * self.discount_rate * max_next_q_values


    def _make_losses(self, losses):
        '''
        Calculates priority based on losses for priority queue
        '''
        return np.power(np.minimum(losses + self.MIN_ERROR_PRIORITY, 
                                   self.MAX_ERROR_PRIORITY), 
                        self.per_a)


    def _get_replay_memory_path(self):
        '''
        Get replay memory path
        '''
        return '{}_replay_memory.hdf5'.format(self.save_path_prefix)


    def predict(self):
        '''
        Runs game with the given model and calculates the scores
        '''

        self._predict_init()
        
        with self.agent:
            try:
                for game_state in self.play_step:
                    pass

            except KeyboardInterrupt:
                log('play interrupted')


    def _predict_init(self):
        '''
        Initialize play variables
        '''

        # batch memory
        self.train_batch = ReplayMemory(self.agent.INPUT_HEIGHT,
                                        self.agent.INPUT_WIDTH,
                                        self.agent.INPUT_CHANNELS,
                                        max_size=self.conf['batch_size'],
                                        state_type=self.agent.STATE_TYPE)


        # reporting stats
        self.play_game_scores = deque(maxlen=100)
        self.play_max_qs = deque(maxlen=100)

        self.conf.update({
            'is_training': False
        })

        self.game_runner = GameRunner(self.conf, self.env, self.agent)
        self.play_step = self.game_runner.play_generator()
