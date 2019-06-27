import gym
import time
import numpy as np

from PIL import Image, ImageFont, ImageDraw


class GameEnvironment:
    '''
    Wraps the environment class.  Provides hooks
    for game specific tweaks
    '''

    def __init__(self, game_id=None, do_preprocess=True, do_before_action=True):
    # def __init__(self, game_id=None, do_preprocess=True, do_before_action=True):
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
        # return self.env.step(*args, **kwargs)
        if self.do_before_action:
            action = self.before_action(action, self.raw_obs, self.reward, self.done, self.info)

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
            return env.reset()


    def preprocess_observation(obs):
        '''
        Preprocess the observation.  Use to save space in memory
        '''
        return obs


    def render_observation(self, obs):
        '''
        Render an image of observation.  Only used for non-image inputs 
        like cartpole.
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


    def preprocess_observation(self, img):
        # crop and cut off score and bottom blank space
        img = img[16:-16:self.COMPRESS_RATIO, ::self.COMPRESS_RATIO] # crop and downsize
        img = np.dot(img[...,:3], [0.299, 0.587, 0.144])

        return img.astype('uint8').reshape(self.INPUT_HEIGHT, self.INPUT_WIDTH, 1)


    def before_action(self, action, obs, reward, done, info):
        # start episode with action 1
        if info is not None and info['ale.lives'] != self.num_lives:
            action = 1
            self.num_lives = info['ale.lives']        

        return action


class CartPoleEnvironment(GameEnvironment):
    GAME_ID = 'Cartpole-v0'
    GAME_REPORT_INTERVAL = 100


    def render_observation(self, obs):
        # rendering for the cart pole environment (in case OpenAI gym can't do it)
        img_w = 600
        img_h = 400
        cart_w = img_w // 12
        cart_h = img_h // 15
        pole_len = img_h // 3.5
        pole_w = img_w // 80 + 1
        x_width = 2
        max_ang = 0.2
        bg_col = (255, 255, 255)
        cart_col = 0x000000 # Blue Green Red
        pole_col = 0x669acc # Blue Green Red

        pos, vel, ang, ang_vel = obs
        img = Image.new('RGB', (img_w, img_h), bg_col)
        draw = ImageDraw.Draw(img)
        cart_x = pos * img_w // x_width + img_w // x_width
        cart_y = img_h * 95 // 100
        top_pole_x = cart_x + pole_len * np.sin(ang)
        top_pole_y = cart_y - cart_h // 2 - pole_len * np.cos(ang)
        draw.line((0, cart_y, img_w, cart_y), fill=0)
        draw.rectangle((cart_x - cart_w // 2, cart_y - cart_h // 2, cart_x + cart_w // 2, cart_y + cart_h // 2), fill=cart_col) # draw cart
        draw.line((cart_x, cart_y - cart_h // 2, top_pole_x, top_pole_y), fill=pole_col, width=pole_w) # draw pole
        
        return np.array(img)


class SpaceInvadersEnvironment(GameEnvironment):
    GAME_ID = 'SpaceInvadersDeterministic-v4'
    COMPRESS_RATIO = 2
    GAME_REPORT_INTERVAL = 10
    INPUT_HEIGHT = 89
    INPUT_WIDTH = 80


    def preprocess_observation(self, img):
        # crop and cut off score and bottom blank space
        img = img[20:-12:self.COMPRESS_RATIO, ::self.COMPRESS_RATIO] # crop and downsize
        img = np.dot(img[...,:3], [0.299, 0.587, 0.144])

        return img.astype('uint8').reshape(self.INPUT_HEIGHT, self.INPUT_WIDTH, 1)


class MsPacmanEnvironment(GameEnvironment):
    GAME_ID = 'MsPacmanDeterministic-v4'
    COMPRESS_RATIO = 2
    GAME_REPORT_INTERVAL = 10
    INPUT_HEIGHT = 86
    INPUT_WIDTH = 80

    ACTION_NOTHING = 1

    num_lives = 0


    def preprocess_observation(self, img):
        # cut off score and icons from bottom
        img = img[0:-38:self.COMPRESS_RATIO, ::self.COMPRESS_RATIO] # crop and downsize
        img = np.dot(img[...,:3], [0.299, 0.587, 0.144])

        return img.astype('uint8').reshape(self.INPUT_HEIGHT, self.INPUT_WIDTH, 1)


    def before_action(self, action, obs, reward, done, info):
        # start episode with action 0
        if info is not None and info['ale.lives'] != self.num_lives:
            self.num_lives = info['ale.lives']        

            action = self.ACTION_NOTHING

            # skip intro time at start of game
            if self.num_lives == 3:
                for i in range(90):
                    self.env.step(self.ACTION_NOTHING)
            else:
                # skip time between death
                for i in range(30):
                    self.env.step(self.ACTION_NOTHING)

        return action
