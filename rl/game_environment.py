'''

GameEnvironment wraps the game environment.  

Preprocesses the images for efficiency sake and also does some game-specific
actions to speed up training. 

Additionally, uses the 'deterministic' version of each game to speed up training.

To add new games, extend GameEnvironment and overload the appropriate methods

'''

import time

import gym
import numpy as np


class GameEnvironment:
    '''
    Wraps the environment class.  
    Provides hooks for game specific tweaks.
    '''

    GAME_ID = None

    def __init__(self, game_id=None, do_preprocess=True, do_before_action=True):
        if game_id is not None:
            self.env = gym.make(game_id)
        else:
            self.env = gym.make(self.GAME_ID)

        self.env.seed(int(time.time()))

        self.do_preprocess = do_preprocess
        self.do_before_action = do_before_action

        self.raw_obs = None
        self.reward = None
        self.done = None
        self.info = None


    def get_action_space(self):
        '''
        Get the available actions for environment
        '''
        return self.env.action_space.n


    def step(self, action):
        '''
        Run one step.  Also check before_action if possible
        '''
        if self.do_before_action:
            action = self.before_action(
                action, self.raw_obs, self.reward, self.done, self.info)

        self.raw_obs, self.reward, self.done, self.info = self.env.step(action)

        if self.do_preprocess:
            return self.preprocess_observation(self.raw_obs), self.reward, self.done, self.info
        else:
            return self.raw_obs, self.reward, self.done, self.info


    def reset(self):
        '''
        Resets game
        '''
        if self.do_preprocess:
            return self.preprocess_observation(self.env.reset())
        else:
            return self.env.reset()


    def preprocess_observation(self, obs):
        '''
        Preprocess the observation.  Use to save space in memory
        '''
        return obs


    def render_observation(self, obs):
        '''
        Render an image of observation.  
        Only used for non-image inputs like cartpole.
        '''
        return obs


    def before_action(self, action, obs, reward, done, info):
        '''
        Run before action to possibly override action based on the
        current game state
        '''
        return action


class BreakoutEnvironment(GameEnvironment):
    GAME_ID = 'BreakoutDeterministic-v4'
    COMPRESS_RATIO = 2
    GAME_REPORT_INTERVAL = 10
    INPUT_HEIGHT = 89
    INPUT_WIDTH = 80

    num_lives = 0

    def preprocess_observation(self, obs):
        # crop and cut off score and bottom blank space
        obs = obs[16:-16:self.COMPRESS_RATIO,
                  ::self.COMPRESS_RATIO]  # crop and downsize
        obs = np.dot(obs[..., :3], [0.299, 0.587, 0.144])

        return obs.astype('uint8').reshape(self.INPUT_HEIGHT, self.INPUT_WIDTH, 1)

    def before_action(self, action, obs, reward, done, info):
        # start episode with action 1
        if info is not None and info['ale.lives'] != self.num_lives:
            action = 1
            self.num_lives = info['ale.lives']

        return action


class SpaceInvadersEnvironment(GameEnvironment):
    GAME_ID = 'SpaceInvadersDeterministic-v4'
    COMPRESS_RATIO = 2
    GAME_REPORT_INTERVAL = 10
    INPUT_HEIGHT = 89
    INPUT_WIDTH = 80

    def preprocess_observation(self, obs):
        # crop and cut off score and bottom blank space
        obs = obs[20:-12:self.COMPRESS_RATIO,
                  ::self.COMPRESS_RATIO]  # crop and downsize
        obs = np.dot(obs[..., :3], [0.299, 0.587, 0.144])

        return obs.astype('uint8').reshape(self.INPUT_HEIGHT, self.INPUT_WIDTH, 1)


class MsPacmanEnvironment(GameEnvironment):
    GAME_ID = 'MsPacmanDeterministic-v4'
    COMPRESS_RATIO = 2
    GAME_REPORT_INTERVAL = 10
    INPUT_HEIGHT = 86
    INPUT_WIDTH = 80

    ACTION_NOTHING = 0

    num_lives = 0

    def preprocess_observation(self, obs):
        # cut off score and icons from bottom
        obs = obs[0:-38:self.COMPRESS_RATIO,
                  ::self.COMPRESS_RATIO]  # crop and downsize
        obs = np.dot(obs[..., :3], [0.299, 0.587, 0.144])

        return obs.astype('uint8').reshape(self.INPUT_HEIGHT, self.INPUT_WIDTH, 1)


    def before_action(self, action, obs, reward, done, info):
        # skip time in intro and between episodes
        if info is not None and info['ale.lives'] != self.num_lives:
            self.num_lives = info['ale.lives']

            action = self.ACTION_NOTHING

            for _ in range(30):
                self.env.step(self.ACTION_NOTHING)

        return action
