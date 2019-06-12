import gym
import time
import numpy as np


class GameEnvironment:
    def __init__(self, game_id=None, preprocess=True):
        if game_id is not None:
            self.env = gym.make(game_id)
        else:
            self.env = gym.make(self.GAME_ID)

        self.env.seed(int(time.time()))


    def get_action_space(self):
        return self.env.action_space.n

    
    def step(self, *args, **kwargs):
        # return self.env.step(*args, **kwargs)
        self.raw_obs, reward, done, info = self.env.step(*args, **kwargs)

        return self.preprocess_observation(self.raw_obs), reward, done, info


    def reset(self):
        # return self.env.reset()
        return self.preprocess_observation(self.env.reset())


    def preprocess_observation(obs):
        return obs


    def render_observation(self, obs):
        '''
        Render an image of observation.  Only used for non-image inputs 
        like cartpole.
        '''
        return obs


    def before_action(self, action, obs, reward, done, info):
        return action


class BreakoutEnvironment(GameEnvironment):
    GAME_ID = 'Breakout-v0'
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


class MsPacmanEnvironment(GameEnvironment):
    GAME_ID = 'Cartpole-v0'
    COMPRESS_RATIO = 2
    GAME_REPORT_INTERVAL = 10
    INPUT_HEIGHT = 86
    INPUT_WIDTH = 80


    def preprocess_observation(self, img):
        # cut off score and icons from bottom
        img = img[0:-38:self.COMPRESS_RATIO, ::self.COMPRESS_RATIO] # crop and downsize
        img = np.dot(img[...,:3], [0.299, 0.587, 0.144])

        return img.astype('uint8').reshape(self.INPUT_HEIGHT, self.INPUT_WIDTH, 1)


