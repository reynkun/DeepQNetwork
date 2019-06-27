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

from .agent.game_agent import GameAgent
from .utils.import_class import import_class
from .utils.logging import init_logging, log
from .game.render import render_game
from .data.replay_memory_disk import ReplayMemoryDisk
from .data.replay_memory import ReplayMemory
from .data.replay_sampler import ReplaySampler
from .data.replay_sampler_priority import ReplaySamplerPriority


class DeepQAgent(GameAgent):
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
        'per_anneal_steps': 2000000,
        'per_calculate_steps': 5000
    }


    MAX_MEMORY_BATCH_SIZE = 128
    MIN_ERROR_PRIORITY = 0.01
    MAX_ERROR_PRIORITY = 1.0


    def __init__(self, conf):
        '''
        Constructor

        Will initialize if initialize is true, otherwise it 
        will search for a configuration file in 'save_dir' directory
        specified in the conf dict

        Values in conf will override the saved config if possible
        '''

        self._init_conf(conf)
        self._init_logging()
        self._init_model()

        self.is_training = self.conf['is_training']


    def _init_conf(self, conf=None):
        '''
        Will create a conf if initialize is true, otherwise it 
        will search for a configuration file in 'save_dir' directory
        specified in the conf dict

        Finally, values in conf will override the saved config 
        '''

        self.save_dir = conf['save_dir']

        overwrite_data = False
        if conf is not None:
            if 'overwrite_data' in conf:
                overwrite_data = True


        if not os.path.exists(self.save_dir) or overwrite_data:
            # create data dir if it doesn't exist and set default options
            os.mkdir(self.save_dir)

            self.conf = self.DEFAULT_OPTIONS.copy()
        else:
            # go find conf file
            has_conf = False
            try:
                for fn in os.listdir(self.save_dir):
                    if fn.endswith('.conf'):
                        with open(os.path.join(self.save_dir, fn)) as fin:
                            self.conf = json.load(fin)
                            has_conf = True
                            break
            except FileNotFoundError:
                raise

            if not has_conf:
                raise Exception('no conf file found')        


        # override saved conf with parameters
        if conf is not None:
            for key, value in conf.items():
                if value is not None:
                    self.conf[key] = value

        self._save_conf()


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


    # def _init_env(self):
    #     '''
    #     Initialize game env
    #     '''

    #     # make environment
    #     if isinstance(self.conf['environment'], str):
    #         mod_env_str, cl_env_str = self.conf['environment'].rsplit('.', 1)
    #         mod_ag = importlib.import_module(mod_env_str)
    #         env_class = getattr(mod_ag, cl_env_str)
    #     elif inspect.isclass(self.conf['2']):
    #         env_class = self.conf['environment']
    #     else:
    #         raise Exception('invalid env class')

    #     self.env = env_class(self.conf['game_id'])
    #     self.conf['action_space'] = self.env.get_action_space()


    def _init_model(self):
        '''
        Initialize game agent
        '''

        # make agent
        # if isinstance(self.conf['model'], str):
        #     mod_model_str, cl_model_str = self.conf['model'].rsplit('.', 1)
        #     mod_ag = importlib.import_module(mod_model_str)

        #     model_class = getattr(mod_ag, cl_model_str)
        # elif inspect.isclass(self.conf['model']):
        #     model_class = self.conf['model']
        # else:
        #     raise Exception('invalid model class')

        # self.model = model_class(conf=self.conf)

        model_class = import_class(self.conf['model'])
        self.model = model_class(conf=self.conf)


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

        self.eps_min = self.conf['eps_min']
        self.eps_max = self.conf['eps_max']
        self.eps_decay_steps = self.conf['eps_decay_steps']


        self.train_start_time = time.time()
        self.game_step_count = 0
        self.step = self.model.get_training_step()

        self.train_report_start_time = time.time()
        self.train_report_last_step = self.step
        self.total_losses = []

        # training batch
        self.train_batch = ReplayMemory(self.model.INPUT_HEIGHT,
                                        self.model.INPUT_WIDTH,
                                        self.model.INPUT_CHANNELS,
                                        max_size=self.batch_size,
                                        state_type=self.model.STATE_TYPE)


        # allocate replay memory
        if self.conf['use_memory']:
            self.memories = ReplayMemory(self.model.INPUT_HEIGHT,
                                         self.model.INPUT_WIDTH,
                                         self.model.INPUT_CHANNELS,
                                         state_type=self.model.STATE_TYPE,
                                         max_size=self.replay_max_memory_length)

            if os.path.exists(self._get_replay_memory_path()):
                log('loading old memories from', self._get_replay_memory_path())
                old_memories = ReplayMemoryDisk(self._get_replay_memory_path())
                log('found', len(old_memories))
                self.memories.extend(old_memories)
        else:
            self.memories = ReplayMemoryDisk(self._get_replay_memory_path(),
                                             self.model.INPUT_HEIGHT,
                                             self.model.INPUT_WIDTH,
                                             self.model.INPUT_CHANNELS,
                                             state_type=self.model.STATE_TYPE,
                                             max_size=self.replay_max_memory_length,
                                             cache_size=self.conf['replay_cache_size'])

        # initialize memory sampler
        if self.use_per:
            self._init_per()
        else:
            self.replay_sampler = ReplaySampler(self.memories)


        # # initialize run_game step generator
        # self.game_runner = GameRunner(self.conf, self.env, self.model)
        # self.play_step = self.game_runner.play_generator()


        # # fill replay memory when first starting training
        # if len(self.replay_sampler) < self.num_game_frames_before_training:
        #     log('filling memories until', self.num_game_frames_before_training)

        #     while len(self.replay_sampler) < self.num_game_frames_before_training:
        #         self._game_step()






    # def train(self):
    #     '''
    #     Runs training loop
    #     '''

    #     start_time = time.time()

    #     log('train start')

    #     with self.model:
    #         self._train_init()
    #         self._train_loop()
    #         self._train_finish()

    #     elapsed = time.time() - start_time 


    def _init_per(self):
        '''
        Initialize prioritized experience replay variables
        '''

        # temporary memory for priority
        # we can add to memory in batches instead of one at time
        self.per_memory_batch = ReplayMemory(self.model.INPUT_HEIGHT,
                                                self.model.INPUT_WIDTH,
                                                self.model.INPUT_CHANNELS,
                                                max_size=self.MAX_MEMORY_BATCH_SIZE,
                                                state_type=self.model.STATE_TYPE)

        self.replay_sampler = ReplaySamplerPriority(self.memories)

        self.per_a = self.conf['per_a']

        self.per_b_start = self.conf['per_b_start']
        self.per_b_end = self.conf['per_b_end']
        self.per_b = self.per_b_start

        self.per_anneal_steps = self.conf['per_anneal_steps']
        self.per_calculate_steps = self.conf['per_calculate_steps']

        self.last_min_loss = None
        self.last_max_loss = None
        self.last_max_weight = None

        self.tree_idxes = np.zeros((self.batch_size), dtype=int)
        self.priorities = np.zeros((self.batch_size), dtype=float)


    # def _train_loop(self):
    #     '''
    #     Main training loop
    #     '''

    #     try:
    #         log('start training')

    #         while self.step < self.max_num_training_steps:
    #             self._train_step()
    #     except (KeyboardInterrupt, StopIteration):
    #         log('train interrupted')


    def train(self, game_state):
        '''
        train with current game state
        '''

        if not self.is_training:
            return True


        if self.is_training_done():
            return False


        if game_state['game_done']:
            self.model.set_game_count(self.model.get_game_count() + 1)

        if game_state['old_state'] is not None:
            self._add_memories(state=game_state['old_state'], 
                               action=game_state['action'], 
                               reward=game_state['reward'], 
                               cont=game_state['cont'], 
                               next_state=game_state['state'])                

        # log('memories: ', len(self.replay_sampler), self.num_game_frames_before_training)
        # fill replay memory when first starting training
        if len(self.replay_sampler) < self.num_game_frames_before_training:
            return True

        if self.game_step_count == 0:
            log('[train] train start')

        # only train every N steps
        self.game_step_count += 1
        if self.game_step_count % self.num_game_steps_per_train != 0:
            return True
        

        self._train_step()

        return True


    def is_training_done(self):
        return self.step >= self.max_num_training_steps


    def _train_step(self):
        '''
        Run one training step
        '''

        self._train_run_train()        
        self._train_save_model()
        self._train_report_stats()


    def _train_run_train(self):
        '''
        Run game steps and train model
        '''
        # # run game steps
        # for _ in range(self.num_game_steps_per_train):
        #     self._game_step()

        
        self._sample_memories()

        # # get max q value 
        # target_max_q_values = self._get_target_max_q_values(self.train_batch.rewards,
        #                                                     self.train_batch.continues,
        #                                                     self.train_batch.next_states)


        # train the model
        self.step, losses, loss = self.model.train(self.train_batch.states,
                                                   self.train_batch.actions,
                                                   self.train_batch.rewards,
                                                   self.train_batch.continues,
                                                   self.train_batch.next_states,
                                                #    target_max_q_values,
                                                   is_weights=self.is_weights)

        if self.use_per:
            # update priority steps in sum tree
            losses = self._make_losses(losses)
                    
            self.replay_sampler.update_sum_tree(self.tree_idxes, losses)

        self.total_losses.append(loss)


    # def _decide_action(self):
    #     '''
    #     Decide which action to do.  Uses epsilon if still training, 
    #     otherwise return the action with max q value
    #     '''

    #     # Online DQN evaluates what to do
    #     q_values = self.brain.predict([self.game_state['state']])

    #     # self.game_state['total_max_q'] += q_values.max()


    #     if self.is_training or self.use_epsilon:
    #         self.game_runner.set_action(self.epsilon_greedy(q_values, self.training_step))
    #         # self.game_state['action'] = self.model.epsilon_greedy(q_values, self.training_step)
    #     else:
    #         self.game_runner.set_action(np.argmax(q_values))

    #         # self.game_state['action'] = np.argmax(q_values)


    def _train_save_model(self):
        '''
        Update target model and save model to disk
        '''

        # Regularly copy the online DQN to the target DQN
        if self.step % self.copy_steps == 0:
            log('copying online to target dqn')
            self.model.copy_network()


        # And save regularly
        if self.step % self.save_steps == 0:
            self.model.save(self.save_path_prefix)


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
                per_str = 'per_ab: {:0.2f}/{:0.2f} avg/min/max loss: {:0.3f}/{:0.3f}/{:0.3f} max_weight: {:0.3f} sum total: {:0.1f} '.format(self.per_a, self.per_b, self.last_avg_loss, self.last_min_loss, self.last_max_loss, self.last_max_weight, self.replay_sampler.total)
            else:
                per_str = ''

            log('[train] step {} game {}/{} avg loss: {:0.5f} {}mem: {:d} fr: {:0.1f} eps: {:0.2f} '.format(self.step,
                                                                                    self.game_step_count,
                                                                                    self.model.get_game_count(),
                                                                                    avg_loss,
                                                                                    per_str,
                                                                                    len(self.replay_sampler),
                                                                                    frame_rate,
                                                                                    self._get_epsilon(self.step)))


    def _train_finish(self):
        '''
        Clean up training, saving replays, model, and other variables
        '''

        log('train finish')
        log('closing replay memory')

        if self.conf['use_memory']:
            # save memories to disk if using ram 
            log('saving', len(self.memories), 'memories to disk')
            old_memories = ReplayMemoryDisk(self._get_replay_memory_path(),
                                            self.model.INPUT_HEIGHT,
                                            self.model.INPUT_WIDTH,
                                            self.model.INPUT_CHANNELS,
                                            state_type=self.model.STATE_TYPE,
                                            max_size=self.replay_max_memory_length,
                                            cache_size=0)

            old_memories.extend(self.memories)
            old_memories.close()

            log('saved memory to disk')
        else:
            self.replay_sampler.close()

        # save game count
        # self.model.set_game_count(self.game_runner.total_game_count)
        self.model.save(self.save_path_prefix)

        elapsed = time.time() - self.train_start_time

        log('train finished in {:0.1f} seconds / {:0.1f} mins / {:0.1f} hours'.format(elapsed, elapsed / 60, elapsed / 60 / 60))



    # def _game_step(self):
    #     '''
    #     Run a game step
    #     '''



    #     next(self.play_step)

    #     self._update_priority_sampling()
    #     self._decide_action()

    #     if self.game_runner.game_state['old_state'] is not None:
    #         self._add_memories(state=self.game_runner.game_state['old_state'], 
    #                            action=self.game_runner.game_state['action'], 
    #                            reward=self.game_runner.game_state['reward'], 
    #                            cont=self.game_runner.game_state['cont'], 
    #                            next_state=self.game_runner.game_state['state'])                



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
        self.per_memory_batch.append(state=state,
                                 action=action,
                                 reward=reward,
                                 cont=cont,
                                 next_state=next_state)


        # append only every after MAX_MEMORY_BATCH_SIZE memories are added
        if len(self.per_memory_batch) >= self.MAX_MEMORY_BATCH_SIZE:

            # calculate losses
            losses = self.model.get_losses(self.per_memory_batch.states,
                                           self.per_memory_batch.actions,
                                           self.per_memory_batch.rewards,
                                           self.per_memory_batch.continues,
                                           self.per_memory_batch.next_states)

            losses = self._make_losses(losses)

            for i in range(len(self.per_memory_batch)):
                self.replay_sampler.append(state=self.per_memory_batch.states[i],
                                           action=self.per_memory_batch.actions[i],
                                           reward=self.per_memory_batch.rewards[i],
                                           next_state=self.per_memory_batch.next_states[i],
                                           cont=self.per_memory_batch.continues[i],
                                           loss=losses[i])
            self.per_memory_batch.clear()


    def _sample_memories(self):
        '''
        Sample game steps and write to train batch
        '''

        if self.use_per:
            if self.step % self.per_calculate_steps == 0 or self.last_max_weight is None:
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


    # def _get_target_max_q_values(self, rewards, continues, next_states):
    #     '''
    #     Get max q_values with discount
    #     '''

    #     max_next_q_values = self.model.get_max_q_value(next_states)

    #     return rewards + continues * self.discount_rate * max_next_q_values


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


    # def play(self):
    #     '''
    #     Runs game with the given model and calculates the scores
    #     '''

    #     self._play_init()
        
    #     with self.model:
    #         try:
    #             for game_state in self.play_step:
    #                 self._decide_action()
    #                 pass

    #         except KeyboardInterrupt:
    #             log('play interrupted')


    def _play_init(self):
        '''
        Initialize play variables
        '''

        self.use_epsilon = self.conf['use_epsilon']
        # self.is_training = False
        # self.game_runner = GameRunner(self.conf, self.env, self.model)
        # self.play_step = self.game_runner.play_generator()

    def _play_finish(self):
        pass


    def get_action(self, state):
        '''
        Decide which action to do.  Uses epsilon if still training, 
        otherwise return the action with max q value
        '''

        # Online DQN evaluates what to do
        q_values = self.model.predict([state])

        # self.game_state['total_max_q'] += q_values.max()

        if self.is_training or self.use_epsilon:
            return self._epsilon_greedy(q_values, self.step)
            # self.game_state['action'] = self.model.epsilon_greedy(q_values, self.training_step)
        else:
            return np.argmax(q_values)
            # self.game_state['action'] = np.argmax(q_values)


    def _get_epsilon(self, step):
        '''
        Gets current epsilon based on what step and the epsilon range
        '''
        return max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)


    def _epsilon_greedy(self, q_values, step):
        '''
        Returns the optimal value if over epsilon, other wise returns the argmax action
        '''
        epsilon = self._get_epsilon(step)

        if np.random.rand() < epsilon:
            # random action
            return np.random.randint(self.model.num_outputs) 
        else:
            # optimal action
            return np.argmax(q_values)


    def open(self):
        self.model.open()

        if self.is_training:
            self._train_init()
        else:
            self._play_init()



    def close(self, ty=None, value=None, tb=None):
        if self.is_training:
            self._train_finish()
        else:
            self._play_finish()

        self.model.close()


    def __enter__(self):
        '''
        Allow for with statement
        '''
        return self.open()


    def __exit__(self, ty, value, tb):
        '''
        Allow for with statement
        '''
        self.close(ty, value, tb)
