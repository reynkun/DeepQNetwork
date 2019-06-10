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

from .game_generator import GameGenerator
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
        'use_memory': True,
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

        # self.save_dir = conf['save_dir']

        # if initialize:
        #     # create save dir
        #     if not os.path.exists(self.save_dir):
        #         os.mkdir(self.save_dir)

        #     self.conf = self.DEFAULT_OPTIONS.copy()
        # else:
        #     # go find conf file
        #     has_conf = False
        #     for fn in os.listdir(self.save_dir):
        #         if fn.endswith('.conf'):
        #             with open(os.path.join(self.save_dir, fn)) as fin:
        #                 self.conf = json.load(fin)
        #                 has_conf = True
        #                 break

        #     if not has_conf:
        #         raise Exception('no conf file found')

        # # override saved conf with parameters
        # if conf is not None:
        #     for key, value in conf.items():
        #         if value is not None:
        #             self.conf[key] = value

        self.init_conf(conf, initialize=initialize)


        # self.game_id = self.conf['game_id']
        # self.agent = self.conf['agent']

        # file_prefix = self.game_id
        # self.save_path_prefix = os.path.join(self.save_dir, file_prefix)
        # self.conf['save_path_prefix'] = self.save_path_prefix

        # if initialize:
        #     # now write final conf
        #     conf_path = self.save_path_prefix + '.conf'
        #     if not os.path.exists(conf_path):
        #         with open(conf_path, 'w+') as fo:
        #             json.dump(self.conf, fo, sort_keys=True, indent=4)

        # # init logging
        # if self.conf.get('use_log', True):
        #     init_logging(self.save_path_prefix)

        # log('config values:')
        # for key, value in self.conf.items():
        #     log('  {}: {}'.format(key, value))

        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(self.conf['tf_log_level'])

        self.init_conf_values()
        self.init_env()
        self.init_agent()


    def init_conf(self, conf=None, initialize=False):
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


    def init_conf_values(self):
        '''
        Set values to instance variables for easier access
        '''

        self.game_id = self.conf['game_id']

        file_prefix = self.game_id
        self.save_path_prefix = os.path.join(self.save_dir, file_prefix)
        self.conf['save_path_prefix'] = self.save_path_prefix

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


    def init_env(self):
        '''
        Init open ai gym
        '''

        self.env = gym.make(self.conf['game_id'])
        self.env.seed(int(time.time()))
        self.conf['action_space'] = self.env.action_space.n        


    def init_agent(self):
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
            self.train_init()
            self.train_loop()
            self.train_finish()

        elapsed = time.time() - start_time 

        log('train finished in {:0.1f} seconds / {:0.1f} mins / {:0.1f} hours'.format(elapsed, elapsed / 60, elapsed / 60 / 60))


    def train_init(self):
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
        self.discount_rate = self.conf['discount_rate']

        # other vars
        self.step = self.agent.get_step()

        # # other vars
        # self.step = self.agent.step.eval()

        self.train_report_start_time = time.time()
        self.train_report_last_step = self.step
        self.total_losses = []

        # training batch
        self.train_batch = ReplayMemory(self.agent.input_height,
                                        self.agent.input_width,
                                        self.agent.input_channels,
                                        max_size=self.batch_size,
                                        state_type=self.agent.state_type)

        # allocate memory
        if self.conf['use_memory']:
            self.memories = ReplayMemory(self.agent.input_height,
                                         self.agent.input_width,
                                         self.agent.input_channels,
                                         state_type=self.agent.state_type,
                                         max_size=self.replay_max_memory_length)

            if os.path.exists(self.get_replay_memory_path()):
                log('loading old memories from', self.get_replay_memory_path())
                old_memories = ReplayMemoryDisk(self.get_replay_memory_path())
                log('found', len(old_memories))
                self.memories.extend(old_memories)
        else:
            self.memories = ReplayMemoryDisk(self.get_replay_memory_path(),
                                             self.agent.input_height,
                                             self.agent.input_width,
                                             self.agent.input_channels,
                                             state_type=self.agent.state_type,
                                             max_size=self.replay_max_memory_length,
                                             cache_size=self.conf['replay_cache_size'])

        self.replay_sampler = ReplaySampler(self.memories)


        # initialize run_game step generator
        self.game_generator = GameGenerator(self.conf, self.env, self.agent)
        self.game_state = self.game_generator.reset_game_state()
        self.play_step = self.game_generator.play_generator(game_state=self.game_state,
                                                            is_training=True)

        # # batch memory
        # self.train_batch = ReplayMemory(self.agent.input_height,
        #                                self.agent.input_width,
        #                                self.agent.input_channels,
        #                                max_size=self.batch_size,
        #                                state_type=self.agent.state_type)

        # fill replay memory when first starting training
        if len(self.replay_sampler) < self.num_game_frames_before_training:
            log('filling memories until', self.num_game_frames_before_training)

        while len(self.replay_sampler) < self.num_game_frames_before_training:
            self.game_step()


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

        self.train_run_and_train()        
        self.train_update_model()
        # self.train_save_video()
        self.train_report_stats()


    def train_run_and_train(self):
        # run game steps
        for _ in range(self.num_game_steps_per_train):
            self.game_step()

        
        self.sample_memories()

        # get max q value 
        target_max_q_values = self.get_target_max_q_values(self.train_batch.rewards,
                                                           self.train_batch.continues,
                                                           self.train_batch.next_states)


        # train the model
        # self.step, losses, loss = self.agent.train(self.train_batch.states,
        #                                                 self.train_batch.actions,
        #                                                 target_max_q_values,
        #                                                 is_weights=is_weights)
        self.step, losses, loss = self.agent.train(self.train_batch.states,
                                                   self.train_batch.actions,
                                                   target_max_q_values)

        self.total_losses.append(loss)


    def train_update_model(self):
        # Regularly copy the online DQN to the target DQN
        if self.step % self.copy_steps == 0:
            log('copying online to target dqn')
            self.agent.copy_network()


        # And save regularly
        if self.step % self.save_steps == 0:
            self.agent.set_game_count(self.game_generator.total_game_count)
            self.agent.save(self.save_path_prefix)


    def train_save_video(self):
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
                # pass
                self.add_memories(state=game_state['state'], 
                                  action=game_state['action'], 
                                  reward=game_state['reward'], 
                                  cont=cont, 
                                  next_state=next_state)


    def train_report_stats(self):
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
                                            self.agent.input_height,
                                            self.agent.input_width,
                                            self.agent.input_channels,
                                            state_type=self.agent.state_type,
                                            max_size=self.replay_max_memory_length,
                                            cache_size=0)

            old_memories.extend(self.memories)
            old_memories.close()

            log('saved memory to disk')
        else:
            self.replay_sampler.close()

        # save game count
        self.agent.set_game_count(self.game_generator.total_game_count)
        self.agent.save(self.save_path_prefix)


    def game_step(self):
        next(self.play_step)

        if self.game_state['old_state'] is not None:
            self.add_memories(state=self.game_state['old_state'], 
                              action=self.game_state['action'], 
                              reward=self.game_state['reward'], 
                              cont=self.game_state['cont'], 
                              next_state=self.game_state['state'])                



    def add_memories(self, state, action, reward, cont, next_state):
        '''
        Add to replay memories
        '''
        self.train_batch.append(state=state,
                                action=action,
                                reward=reward,
                                cont=cont,
                                next_state=next_state)

        if len(self.train_batch) >= self.batch_size:
            for i in range(len(self.train_batch)):
                self.replay_sampler.append(state=self.train_batch.states[i],
                                           action=self.train_batch.actions[i],
                                           reward=self.train_batch.rewards[i],
                                           next_state=self.train_batch.next_states[i],
                                           cont=self.train_batch.continues[i],
                                           loss=0)

            self.train_batch.clear()


    def sample_memories(self):
        # sample randomly from each range
        self.replay_sampler.sample_memories(self.train_batch,
                                            batch_size=self.batch_size)
        is_weights = None


    def get_target_max_q_values(self, rewards, continues, next_states):
        # max_next_q_values = self.agent.get_max_q_value(next_states)
        max_next_q_values = self.agent.get_max_q_value(next_states)

        return rewards + continues * self.discount_rate * max_next_q_values


    def get_replay_memory_path(self):
        return '{}_replay_memory.hdf5'.format(self.save_path_prefix)


    def predict(self,
                num_games=1,
                use_epsilon=False,
                interval=60,
                display=False, 
                save_video=False):
        '''
        Runs game with the given model and calculates the scores
        '''

        self.play_init(num_games=num_games,
                       use_epsilon=use_epsilon,
                       display=display,
                       save_video=save_video)

        
        with self.agent:
            try:
                for game_state in self.play_step:
                    pass
                    # self.add_memories(state=game_state['old_state'], 
                    #                   action=game_state['action'], 
                    #                   reward=game_state['reward'], 
                    #                   cont=game_state['cont'], 
                    #                   next_state=game_state['state'])  


            except KeyboardInterrupt:
                log('play interrupted2')


    def play_init(self, num_games=10, use_epsilon=False, display=False, save_video=False):
        '''
        Initialize play variables
        '''

        # batch memory
        self.train_batch = ReplayMemory(self.agent.input_height,
                                        self.agent.input_width,
                                        self.agent.input_channels,
                                        max_size=self.conf['batch_size'],
                                        state_type=self.agent.state_type)


        # reporting stats
        self.play_game_scores = deque(maxlen=100)
        self.play_max_qs = deque(maxlen=100)
        self.game_generator = GameGenerator(self.conf, self.env, self.agent)
        self.game_step = self.game_generator.make_game_state()
        self.play_step = self.game_generator.play_generator(is_training=False,
                                                            num_games=num_games,
                                                            use_epsilon=use_epsilon, 
                                                            display=display, 
                                                            save_video=save_video)

