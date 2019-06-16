import os
import time
import numpy as np

from collections import deque

from .utils.logging import log
from .utils.render import render_game, render_state


class GameRunner:
    '''
    Runs the game with given model and environment.  Has generator style
    accessors for steping through the game.
    '''

    DEFAULT_OPTIONS = {
        'frame_skip': 1,
        'use_episodes': True,
        'max_game_length': 50000,
        'num_games': None,
        'is_training': True,
        'use_epsilon': False,
        'display': False,
        'save_video': False,
        'save_dir': None
    }

    GAME_STATE_RESET = {
        'game_start_time': time.time(),
        'game_done': False,
        'episode_done': False,
        'total_max_q': 0,
        'iteration': 0,
        'game_length': 0,
        'episode_length': 0,
        'score': 0,
        'observation': None,
        'old_state': None,
        'state': None,
        'action': 0,
        'reward': None,
        'cont': 1,
        'info': None,
        'num_lives': None,
        'frames_render': [],
        'infos_render': [],
    }   


    def __init__(self, conf, env, model):
        self.env = env
        self.model = model

        # set up default values
        self.conf = self.DEFAULT_OPTIONS.copy()
        self.game_state = {}

        # override saved conf with parameters
        for key, value in conf.items():
            if value is not None:
                self.conf[key] = value


    def play_generator(self):
        '''
        Generator which will run the game one frame
        '''

        self._play_init()

        try:
            while self.conf['num_games'] is None or self.cur_game_count < self.conf['num_games']:
                for _ in self.play_game_generator():
                    yield self.game_state

        except KeyboardInterrupt:
            log('play interrupted')


    def _play_init(self):
        '''
        Initialize play variables
        '''

        # run game settings
        self.use_episodes = self.conf['use_episodes']
        self.frame_skip = self.conf['frame_skip']
        self.max_game_length = self.conf['max_game_length']
        self.is_training = self.conf['is_training']
        self.use_epsilon = self.conf['use_epsilon']
        self.total_game_count = self.model.get_game_count()
        self.cur_game_count = 0
        self.training_step = self.model.get_training_step()


        # reporting stats
        self.play_game_scores = deque(maxlen=100)
        self.play_max_qs = deque(maxlen=100)

        self._reset_game_state()


    def play_game_generator(self):
        '''
        Generator which yields every frame and runs one game
        '''

        # reset game
        self.training_step = self.model.get_training_step()
        self._reset_game_state()

        while not self.game_state['game_done']:
            # play out episode
            for _ in self.play_episode_generator():
                yield self.game_state
            
            # update frame final state of episode / game
            self.game_state['cont'] = 0
            self._update_frame_state()

            # return to training
            yield self.game_state

        self._update_and_report_play_stats()
        self._render_game()


    def _update_and_report_play_stats(self):
        '''
        Replay on play stats
        '''

        self.total_game_count += 1
        self.cur_game_count += 1

        self.play_game_scores.append(self.game_state['score'])
        self.play_max_qs.append(self.game_state['total_max_q'] / self.game_state['game_length'])

        if not self.is_training or self.total_game_count % self.env.GAME_REPORT_INTERVAL == 0:
            if self.game_state['game_length'] > 0:
                mean_max_q = self.game_state['total_max_q'] / self.game_state['game_length']
            else:
                mean_max_q = 0

            elapsed = time.time() - self.game_state['game_start_time']
            if elapsed > 0:
                frame_rate = self.game_state['game_length'] / (time.time() - self.game_state['game_start_time'])
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

            epsilon = self.model.epsilon(self.training_step)

            log('[play] step {} game {}/{} len: {:d} max_q: {:0.3f}/{:0.3f} score: {:0.1f}/{:0.1f}/{:0.1f}/{:0.1f} eps: {:0.3f} fr: {:0.1f}'.format(
                       self.training_step,
                       self.cur_game_count,
                       self.total_game_count,
                       self.game_state['game_length'],
                       mean_max_q,
                       avg_max_q,
                       self.game_state['score'],
                       avg_score,
                       min_score,
                       max_score,
                       epsilon,
                       frame_rate))
          


    def play_episode_generator(self):
        '''
        Generator which yields every frame and runs one episode
        '''

        # reset game state related to episode
        self.game_state['episode_length'] = 0
        self.game_state['episode_done'] = False

        while not self.game_state['episode_done'] and not self.game_state['game_done']:
            self.game_state['game_length'] += 1
            self.game_state['episode_length'] += 1

            if self.game_state['game_length'] > self.max_game_length:
                break

            # get next action 
            if len(self.game_state['frames']) >= self.model.NUM_FRAMES_PER_STATE:
                self.game_state['cont'] = 1
                self._update_frame_state()

                # yield to caller
                yield self.game_state

                self._decide_action()

            # run action for frame_skip steps
            self.game_state['reward'] = 0
            for _ in range(self.frame_skip):
                if not self._run_game_step():
                    break

            self.game_state['score'] += self.game_state['reward']
            self.game_state['frames'].append(self.game_state['observation'])


    def _decide_action(self):
        '''
        Decide which action to do.  Uses epsilon if still training, 
        otherwise return the action with max q value
        '''

        # Online DQN evaluates what to do
        q_values = self.model.predict([self.game_state['state']])

        self.game_state['total_max_q'] += q_values.max()


        if self.is_training or self.use_epsilon:
            self.game_state['action'] = self.model.epsilon_greedy(q_values, self.training_step)
        else:
            self.game_state['action'] = np.argmax(q_values)


    def _run_game_step(self):
        '''
        Run one game step and add up rewards.
        Checks for when episodes are finished
        '''

        # run step
        self.game_state['observation'], step_reward, self.game_state['game_done'], self.game_state['info'] = self.env.step(self.game_state['action'])

        self.game_state['reward'] += step_reward

        # check for episode change
        if self.use_episodes and 'ale.lives' in self.game_state['info']:
            num_lives = self.game_state['info']['ale.lives']

            if self.game_state['num_lives'] is not None and num_lives != self.game_state['num_lives']:
                self.game_state['episode_done'] = True

            if num_lives <= 0:
                self.game_state['game_done'] = True

            self.game_state['num_lives'] = num_lives

        
        if not self.is_training and (self.conf['save_video'] or self.conf['display']):
            self.game_state['frames_render'].append(self.env.render_observation(self.env.raw_obs))
            self.game_state['infos_render'].append({
                                                      'a': self.game_state['action'],
                                                      'c': not (self.game_state['episode_done'] or self.game_state['game_done']),
                                                      'r': self.game_state['reward']
                                                  })

        if self.game_state['game_done']:
            return False

        if self.game_state['episode_done']:
            return False        

        return True


    def _render_game(self):
        '''
        Renders the game to either video or to the display 
        '''
        if self.conf['save_video']:
            save_path = os.path.join(self.conf['save_dir'],
                                     'video-{}-{}.mp4'.format(self.training_step,
                                                              self.total_game_count))
        else:
            save_path = None

        if self.conf['save_video'] or self.conf['display']:
            render_game(self.game_state['frames_render'],
                        self.game_state['infos_render'],
                        repeat=False,
                        interval=self.conf['interval'],
                        save_path=save_path,
                        display=self.conf['display'])         


    def _reset_game_state(self):
        '''
        Initialized game state variables
        '''

        self.game_state.update(self.GAME_STATE_RESET)

        self.game_state['observation'] = self.env.reset()
        self.game_state['frames'] = deque(maxlen=self.model.NUM_FRAMES_PER_STATE)
        self.game_state['frames'].append(self.game_state['observation'])
        


    def _update_frame_state(self):
        '''
        Makes a frame state from 4 sequential frames.
        Also saves memories to replay memory
        '''

        self.game_state['old_state'] = self.game_state['state']
        self.game_state['state'] = np.concatenate(self.game_state['frames'], axis=2)


    def _render_state_action(self):
        '''
        For debug purposes
        '''

        try:
            self.counter
        except AttributeError:
            self.counter = 0
        
        if self.counter % 20 == 0:
            log(self.counter, self.game_state['action'])
            render_state(self.game_state['state'])
            log(q_values)

        self.counter += 1
