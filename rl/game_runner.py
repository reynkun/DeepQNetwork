import os
import time
import numpy as np

from collections import deque

from .utils.logging import log
from .utils.render import render_game, render_state


class GameRunner:
    '''
    Runs the game with given environment and agent
    '''

    DEFAULT_OPTIONS = {
        'frame_skip': 1,
        'use_episodes': True,
        'max_game_length': 50000,
        'num_games': None,
        'is_training': True,
        'display': False,
        'save_video': False,
        'save_dir': None,
        'num_frames_per_state': 4
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


    def __init__(self, conf, env, agent):
        self.env = env
        self.agent = agent

        # set up default values
        self.conf = self.DEFAULT_OPTIONS.copy()
        self.game_state = {}

        # override saved conf with parameters
        for key, value in conf.items():
            if value is not None:
                self.conf[key] = value


    def run(self):
        '''
        Generator which will run the game one frame
        '''

        self._run_init()

        try:
            while self.conf['num_games'] is None or self.cur_game_count < self.conf['num_games']:
                if self.is_training and self.agent.is_training_done():
                    break

                self._run_game()

        except KeyboardInterrupt:
            log('play interrupted')


    def _run_init(self):
        '''
        Initialize play variables
        '''

        # run game settings
        self.use_episodes = self.conf['use_episodes']
        self.frame_skip = self.conf['frame_skip']
        self.max_game_length = self.conf['max_game_length']
        self.is_training = self.conf['is_training']
        self.num_frames_per_state = self.conf['num_frames_per_state']
        self.cur_game_count = 0


        # reporting stats
        self.play_game_scores = deque(maxlen=100)
        self.play_max_qs = deque(maxlen=100)

        self._reset_game_state()


    def _run_game(self):
        '''
        Runs one game
        '''

        # reset game
        self._reset_game_state()

        while not self.game_state['game_done']:
            # play out episode
            self._run_episode()
            
            # update frame final state of episode / game
            self.game_state['cont'] = 0
            self._update_frame_state()

            # return to training
            if self.is_training:
                self.agent.train(self.game_state)

        self._update_and_report_play_stats()
        self._render_game()


    def _reset_game_state(self):
        '''
        Initialized game state variables
        '''

        self.game_state.update(self.GAME_STATE_RESET)

        self.game_state['observation'] = self.env.reset()
        self.game_state['frames'] = deque(maxlen=self.num_frames_per_state)
        self.game_state['frames'].append(self.game_state['observation'])


    def _run_episode(self):
        '''
        Runs one episode
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
            if len(self.game_state['frames']) >= self.num_frames_per_state:
                self.game_state['cont'] = 1
                self._update_frame_state()

                # yield to caller
                if self.is_training:
                    # log('train episode')
                    self.agent.train(self.game_state)

                # self.game_state['action'] = self.agent.get_action(self.game_state['state'])
                self.game_state['action'] = self.agent.get_action(self.game_state['state'])

            # run action for frame_skip steps
            self.game_state['reward'] = 0
            for _ in range(self.frame_skip):
                if not self._run_game_step():
                    break

            self.game_state['score'] += self.game_state['reward']
            self.game_state['frames'].append(self.game_state['observation'])

        # log('end episode', not self.game_state['episode_done'], not self.game_state['game_done'])


    def _run_game_step(self):
        '''
        Run one game step and add up rewards.
        Checks for when episodes are finished
        '''

        # run step
        self.game_state['observation'], step_reward, self.game_state['game_done'], self.game_state['info'] = self.env.step(self.game_state['action'])

        # add to reward
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
            # log('breaking game done')
            return False

        if self.game_state['episode_done']:
            # log('breaking episode done')
            return False        

        return True


    def _update_frame_state(self):
        '''
        Makes a frame state from 4 sequential frames.
        Also saves memories to replay memory
        '''

        if len(self.game_state['frames']) >= self.num_frames_per_state:
            self.game_state['old_state'] = self.game_state['state']
            self.game_state['state'] = np.concatenate(self.game_state['frames'], axis=2)

        # log(self.game_state['state'].shape)


    def _update_and_report_play_stats(self):
        '''
        Replay on play stats
        '''

        self.cur_game_count += 1

        self.play_game_scores.append(self.game_state['score'])
        self.play_max_qs.append(self.game_state['total_max_q'] / self.game_state['game_length'])

        if not self.is_training or self.cur_game_count % self.env.GAME_REPORT_INTERVAL == 0:
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

            log('[play] game {} len: {:d} score: {:0.1f}/{:0.1f}/{:0.1f}/{:0.1f} fr: {:0.1f}'.format(
                       self.cur_game_count,
                       self.game_state['game_length'],
                       self.game_state['score'],
                       avg_score,
                       min_score,
                       max_score,
                       frame_rate))


    def _render_game(self):
        '''
        Renders the game to either video or to the display 
        '''
        if self.conf['save_video']:
            save_path = os.path.join(self.conf['save_dir'],
                                     'video-{}.mp4'.format(self.cur_game_count))
        else:
            save_path = None

        if self.conf['save_video'] or self.conf['display']:
            render_game(self.game_state['frames_render'],
                        self.game_state['infos_render'],
                        repeat=False,
                        interval=self.conf['interval'],
                        save_path=save_path,
                        display=self.conf['display'])         
