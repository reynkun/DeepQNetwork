import os
import time
import json
import logging
import importlib

from collections import deque
from logging.handlers import TimedRotatingFileHandler

import gym
import tensorflow as tf
import numpy as np

from .game.render import render_game
from .data.replay_memory_disk import ReplayMemoryDisk
from .data.replay_memory import ReplayMemory
from .data.replay_sampler import ReplaySampler
from .data.replay_sampler_priority import ReplaySamplerPriority




def time_string():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


class DeepQNetwork:
    DEFAULT_OPTIONS = {
        'save_dir': './models',
        'eps_min': 0.1,
        'eps_max': 1.0,
        'eps_decay_steps': 2000000,
        'discount_rate': 0.99,
        'save_model_steps': 10000,
        'copy_network_steps': 10000,
        'batch_size': 32,
        'model_save_prefix': None,
        'replay_max_memory_length': 1000000,
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
        'tf_log_level': 3,
    }


    def __init__(self, conf, initialize=False):
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

        file_prefix = self.conf['game_id']
        self.save_path_prefix = os.path.join(self.save_dir, file_prefix)

        if initialize:
            # now write final conf
            conf_path = self.save_path_prefix + '.conf'
            if not os.path.exists(conf_path):
                with open(conf_path, 'w+') as fo:
                    json.dump(self.conf, fo, sort_keys=True, indent=4)

        # init logging
        if self.conf.get('use_log', True):
            self.init_logging()
            self.use_log = True
        else:
            self.use_log = False

        self.log('config values:')
        for key, value in self.conf.items():
            self.log('  {}: {}'.format(key, value))

        # make agent
        mod_agent_str, cl_agent_str = self.conf['agent'].rsplit('.', 1)
        mod_ag = importlib.import_module(mod_agent_str)

        self.agent_class = getattr(mod_ag, cl_agent_str)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(self.conf['tf_log_level'])


        # set variables from settings for convenience
        self.game_id = self.conf['game_id']

        # run game settings
        self.eps_min = self.conf['eps_min']
        self.eps_max = self.conf['eps_max']
        self.eps_decay_steps = self.conf['eps_decay_steps']
        self.discount_rate = self.conf['discount_rate']
        self.use_episodes = self.conf['use_episodes']
        self.frame_skip = self.conf['frame_skip']

        # train settings
        self.max_num_training_steps = self.conf['max_num_training_steps']
        self.replay_max_memory_length = self.conf['replay_max_memory_length']
        self.num_game_frames_before_training = self.conf['num_game_frames_before_training']
        self.batch_size = self.conf['batch_size']
        self.save_steps = self.conf['save_model_steps']
        self.copy_steps = self.conf['copy_network_steps']


    def train(self):
        '''
        Runs training loop
        '''

        start_time = time.time()

        self.log('train start')

        with self.get_session(init_env=True,
                              init_model=True,
                              load_model=True,
                              save_model=True) as self.sess:
            self._init_train()
            self._train_loop()
            self._finish_train()

        elapsed = time.time() - start_time 

        self.log('train finished in {:0.1f} seconds / {:0.1f} mins / {:0.1f} hours'.format(elapsed, elapsed / 60, elapsed / 60 / 60))


    def _init_train(self):
        self.game_count = 0
        self.train_report_interval = self.conf['train_report_interval']
        self.report_start_time = time.time()
        self.step = self.sess.model.step.eval()
        self.report_last_step = self.step
        self.total_losses = []

        # training batch
        self.batch = ReplayMemory(self.sess.model.input_height,
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
                self.log('loading old memories from', self.get_replay_memory_path())
                old_memories = ReplayMemoryDisk(self.get_replay_memory_path())
                self.log('found', len(old_memories))
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
        if self.conf['use_priority']:
            self.replay_sampler = ReplaySamplerPriority(self.memories)
        else:
            self.replay_sampler = ReplaySampler(self.memories)

        # initialize run_game step generator
        self.run_step = self.run_game(is_training=True)


        # fill replay memory when first starting training
        while len(self.replay_sampler) < self.num_game_frames_before_training:
            next(self.run_step)


    def _train_loop(self):
        try:
            self.log('start training')

            while self.step < self.max_num_training_steps:
                self._train_step()
        except (KeyboardInterrupt, StopIteration):
            self.log('train interrupted')


    def _train_step(self):
        self.step = self.sess.model.step.eval()

        # run game steps
        for _ in range(self.conf['num_game_steps_per_train']):
            next(self.run_step)

        # sample memories and use the target DQN to produce the target Q-Value
        if self.conf['use_priority']:
            tree_idxes = []
            self.replay_sampler.sample_memories(self.batch,
                                                batch_size=self.batch_size,
                                                tree_idxes=tree_idxes)
        else:
            self.replay_sampler.sample_memories(self.batch,
                                                batch_size=self.batch_size)

        target_max_q_values = self.get_target_max_q_values(self.batch.rewards,
                                                           self.batch.continues,
                                                           self.batch.next_states)

        # Train the online DQN
        tr_res, new_losses, loss_val = self.sess.run([self.sess.model.training_op,
                                                      self.sess.model.losses,
                                                      self.sess.model.loss],
                                                     feed_dict={
                                                         self.sess.model.X_state: self.batch.states,
                                                         self.sess.model.X_action: self.batch.actions,
                                                         self.sess.model.y: target_max_q_values
                                                     })

        if self.conf['use_priority']:
            self.replay_sampler.update_sum_tree(tree_idxes, new_losses)

        self.total_losses.append(loss_val)

        # Regularly copy the online DQN to the target DQN
        if self.step % self.copy_steps == 0:
            self.log('copying online to target dqn')
            self.sess.model.copy_online_to_target.run()

        # And save regularly
        if self.step % self.save_steps == 0:
            self.sess.model.game_count.load(self.game_count)
            self.sess.save(self.save_path_prefix)

        # save video every so often
        if self.conf['num_train_steps_save_video'] is not None and \
                self.step % self.conf['num_train_steps_save_video'] == 0:

            self.log('saving video at step', self.step)
            for _ in self.run_game(is_training=False,
                                   num_games=1,
                                   use_epsilon=False,
                                   interval=60,
                                   display=False,
                                   save_video=True):
                pass

        # report
        if self.step % self.train_report_interval == 0:
            elapsed = time.time() - self.report_start_time
            if elapsed > 0:
                frame_rate = (self.step - self.report_last_step) / elapsed
            else:
                frame_rate = 0.0

            self.report_last_step = self.step
            self.report_start_time = time.time()

            if len(self.total_losses) > 0:
                avg_loss = sum(self.total_losses) / len(self.total_losses)
            else:
                avg_loss = 0

            self.total_losses.clear()

            self.log('[train] step {} avg loss: {:0.5f} mem: {:d} fr: {:0.1f}'.format(self.step,
                                                                                      avg_loss,
                                                                                      len(self.replay_sampler),
                                                                                      frame_rate))


    def _finish_train(self):
        self.log('closing replay memory')

        if self.conf['use_memory']:
            self.log('saving', len(self.memories), 'memories to disk')
            old_memories = ReplayMemoryDisk(self.get_replay_memory_path(),
                                            self.sess.model.input_height,
                                            self.sess.model.input_width,
                                            self.sess.model.input_channels,
                                            state_type=self.sess.model.state_type,
                                            max_size=self.replay_max_memory_length,
                                            cache_size=0)

            old_memories.extend(self.memories)
            old_memories.close()

            self.log('saved memory to disk')
        else:
            self.replay_sampler.close()

        # save game count
        self.sess.model.game_count.load(self.game_count)


    def play(self, num_games=1, use_epsilon=False, interval=60, display=False, save_video=False):
        with self.get_session(init_env=True,
                              init_model=False,
                              load_model=True,
                              save_model=False) as self.sess:
            for _ in self.run_game(is_training=False,
                                   num_games=num_games,
                                   use_epsilon=use_epsilon,
                                   interval=interval,
                                   display=display,
                                   save_video=save_video):
                pass


    def run_game(self,
                 is_training=False,
                 num_games=None,
                 use_epsilon=False,
                 interval=60,
                 display=False,
                 save_video=False):
        game_scores = deque(maxlen=1000)
        max_qs = deque(maxlen=1000)
        report_interval = 60
        max_game_length = 50000

        iteration = 0
        report_start_time = time.time()
        report_last_iteration = 0
        report_rate = 0
        step = self.sess.run([self.sess.model.step])[0]
        num_episodes = 0

        batch = ReplayMemory(self.sess.model.input_height,
                             self.sess.model.input_width,
                             self.sess.model.input_channels,
                             max_size=self.batch_size,
                             state_type=self.sess.model.state_type)

        if is_training and self.game_count <= 0:
            self.game_count = self.sess.run([self.sess.model.game_count])[0]
            self.log('game_count:', self.game_count)
        else:
            game_count = 0

        try:
            while True:
                epoch_start_time = time.time()

                total_max_q = 0.0
                game_length = 0
                game_score = 0
                state_frames = deque(maxlen=self.sess.model.num_frame_per_state)
                action = 0
                reward = None
                info = None
                game_done = False
                state = None
                num_lives = 0

                # for not training only
                if is_training:
                    self.game_count += 1
                else:
                    actions = []
                    game_frames = []

                    if game_count >= num_games:
                        break

                    game_count += 1

                obs = self.sess.env.reset()
                obs = self.sess.model.preprocess_observation(obs)

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

                        step = self.sess.run([self.sess.model.step])[0]

                        if len(state_frames) >= self.sess.model.num_frame_per_state:
                            next_state = self.make_state(state_frames)

                            if is_training and state is not None:
                                batch.append(state=state,
                                             action=action,
                                             reward=reward,
                                             cont=1,
                                             next_state=next_state)

                                if len(batch) >= self.batch_size:
                                    target_max_q_values = self.get_target_max_q_values(batch.rewards,
                                                                                       batch.continues,
                                                                                       batch.next_states)

                                    losses = self.get_losses(batch.states,
                                                             batch.actions,
                                                             target_max_q_values)

                                    for i in range(len(batch)):
                                        self.replay_sampler.append(state=batch.states[i],
                                                                  action=batch.actions[i],
                                                                  reward=batch.rewards[i],
                                                                  next_state=batch.next_states[i],
                                                                  cont=1,
                                                                  loss=losses[i])
                                    batch.clear()

                                yield


                            state = next_state

                            # Online DQN evaluates what to do
                            # print('next_state:', next_state.shape)
                            q_values = self.sess.model.online_q_values.eval(feed_dict={self.sess.model.X_state: [next_state]})
                            total_max_q += q_values.max()

                            if is_training or use_epsilon:
                                action = self.epsilon_greedy(q_values,
                                                             step)
                            else:
                                action = np.argmax(q_values)

                        action = self.sess.model.before_action(action, obs, reward, game_done, info)

                        # run action for frame_skip steps
                        reward = 0
                        for i in range(self.frame_skip):
                            # Online DQN plays
                            obs, step_reward, game_done, info = self.sess.env.step(action)

                            if not is_training and (save_video or display):
                                actions.append(action)
                                game_frames.append(self.sess.model.render_obs(obs))

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

                        obs = self.sess.model.preprocess_observation(obs)
                        state_frames.append(obs)

                    num_episodes += 1

                    # game / episode done, save last step
                    next_state = self.make_state(state_frames)

                    if is_training and state is not None:
                        batch.append(state=state,
                                     action=action,
                                     reward=reward,
                                     cont=0,
                                     next_state=next_state)

                        if len(batch) >= self.batch_size:
                            target_max_q_values = self.get_target_max_q_values(batch.rewards,
                                                                               batch.continues,
                                                                               batch.next_states)

                            losses = self.get_losses(batch.states,
                                                     batch.actions,
                                                     target_max_q_values)

                            for i in range(len(batch)):
                                self.replay_sampler.append(state=batch.states[i],
                                                          action=batch.actions[i],
                                                          reward=batch.rewards[i],
                                                          next_state=batch.next_states[i],
                                                          cont=1,
                                                          loss=losses[i])
                            batch.clear()

                        # return to training
                        yield

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
                    mem_len = len(self.replay_sampler)
                else:
                    mem_len = 0

                if is_training:
                    if self.game_count % self.sess.model.game_report_interval == 0:
                        self.log('[play] step {} game {} epi {} len: {:d} max_q: {:0.3f}/{:0.3f} score: {:0.1f} avg: {:0.2f} mem: {:d} eps: {:0.3f} fr: {:0.1f}/{:0.1f}'.format(
                                   step,
                                   self.game_count,
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

                else:
                    self.log('[play] step {} game {} epi {} len: {:d} max_q: {:0.3f}/{:0.3f} score: {:0.1f} avg: {:0.2f} mem: {:d} eps: {:0.3f} fr: {:0.1f}/{:0.1f}'.format(
                                    step,
                                    game_count,
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

                    if save_video:
                        save_path = os.path.join(self.save_dir,
                                                 'video-{}-{}.mp4'.format(step,
                                                                          self.game_count))
                    else:
                        save_path = None

                    if save_video or display:
                        render_game(game_frames,
                                    actions,
                                    repeat=False,
                                    interval=interval,
                                    save_path=save_path,
                                    display=display)


        except KeyboardInterrupt:
            self.log('play interrupted')


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
        if self.conf['use_double']:
            max_next_q_values = self.sess.model.double_max_q_values.eval(
                feed_dict={self.sess.model.X_state: next_states})
        else:
            max_next_q_values = self.sess.model.max_q_values.eval(
                feed_dict={self.sess.model.X_state: next_states})


        return rewards + continues * self.discount_rate * max_next_q_values


    def get_losses(self, states, actions, max_q_values):
        return self.sess.model.losses.eval(feed_dict={
                                        self.sess.model.X_state: states,
                                        self.sess.model.X_action: actions,
                                        self.sess.model.y: max_q_values
                                      })


    def get_replay_memory_path(self):
        return '{}_replay_memory.hdf5'.format(self.save_path_prefix)


    def get_session(parent, init_env=True, init_model=False, load_model=True, save_model=False):
        class Session:
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
                    parent.log('  model does not exist:', save_path_prefix)
                    return False

                parent.log('  restoring model: ', save_path_prefix)
                self.saver.restore(self._sess, save_path_prefix)
                parent.log('  restored model')

                return True


            def open(self):
                parent.log('creating new session. init_env:', init_env, 'load_model: ', load_model, 'save_model:', save_model)

                if init_env:
                    self.env = gym.make(parent.game_id)
                    self.env.seed(int(time.time()))
                    parent.conf['action_space'] = self.env.action_space.n

                tf.reset_default_graph()

                self.model = parent.agent_class(conf=parent.conf)
                self.saver = tf.train.Saver()

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True

                self._sess = tf.Session(config=config)
                self._sess.as_default()
                self._sess.__enter__()

                loaded = False
                if load_model:
                    loaded = self.restore(parent.save_path_prefix)

                if not loaded and not init_model:
                    raise Exception('cannot load existing model')

                if not loaded:
                    self._sess.run(tf.global_variables_initializer())

                tf.get_default_graph().finalize()

                return self


            def close(self, ty=None, value=None, tb=None):
                if init_env:
                    self.env.close()


                if save_model:
                    self.save(parent.save_path_prefix)

                self._sess.__exit__(ty, value, tb)                    
                self._sess.close()
                self._sess = None


        return Session()


    def init_logging(self, add_std_err=True):
        self.logger = logging.getLogger('rl')
        self.logger.setLevel(logging.DEBUG)

        log_path = self.save_path_prefix + '.log'

        hdlr = TimedRotatingFileHandler(log_path, when='D')
        formatter = logging.Formatter('%(asctime).19s [%(levelname)s] %(message)s')
        hdlr.setFormatter(formatter)

        self.logger.addHandler(hdlr)

        if add_std_err:
            hdlr = logging.StreamHandler()
            hdlr.setFormatter(formatter)
            self.logger.addHandler(hdlr)

        return self.logger


    def log(self, *mesg):
        if self.use_log:
            self.logger.info(' '.join([str(m) for m in mesg]))
        else:
            print(' '.join([str(m) for m in mesg]))
