from collections import deque


import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw

from .model.model import Model


class GameAgent(Model):
    '''
    Creates the Game Agent and its tf network
    '''

    DEFAULT_OPTIONS = {
        # use dueling architecture
        'use_dueling': False,
        # learning rate for adam / momentum optmizer
        'learning_rate': 0.00025,
        # use momentum instead of default adam
        'use_momentum': False,
        # momentum for momentum optimizer
        'momentum': 0.95,
    }

    # class variables.  override for each game
    
    # game screen size / channels
    INPUT_HEIGHT = 110
    INPUT_WIDTH = 80
    INPUT_CHANNELS = 4

    # num of (time) sequential frames to use for learning
    num_frames_per_state = 4

    # use convolutional filters. only needs to be false for non-image input
    USE_CONV = True

    # experimental use of auto encoder to compress images
    USE_ENCODER = False

    # dtype for input
    STATE_TYPE = 'uint8'

    # num hidden nodes in dense layer
    NUM_HIDDEN = 512


    def __init__(self, conf, initialize=True):
        super().__init__(self, conf)
        
        self.conf = self.DEFAULT_OPTIONS.copy()

        for key, value in conf.items():
            if key not in self.conf:
                self.conf[key] = value

        self.num_outputs = self.conf['action_space']
        self.eps_min = self.conf['eps_min']
        self.eps_max = self.conf['eps_max']
        self.eps_decay_steps = self.conf['eps_decay_steps']


    def train(self, X_states, X_actions, y, is_weights=None):
        '''
        trains network
        '''

        feed_dict = {
            self.X_state: X_states,
            self.X_action: X_actions,
            self.y: y
        }


        step, _, losses, loss = self.run([self.training_step,
                                          self.training_op,
                                          self.losses,
                                          self.loss],
                                          feed_dict=feed_dict)
        return step, losses, loss


    def predict(self, X_states, use_target=False):
        '''
        Return q values for given states
        '''

        if use_target:
            q_values = self.target_q_values
        else:
            q_values = self.online_q_values

        values = self.run([q_values],
                                  feed_dict={self.X_state: X_states})

        return values[0]


    def get_action(self, X_states, use_target=False):
        '''
        Get top action for given states
        '''

        values = self.predict(X_states, use_target=use_target)

        return np.argmax(values, axis=1)


    def get_max_q_value(self, X_states):
        '''
        Get max q value
        '''

        return self.run([self.max_q_values],
                        feed_dict={self.X_state: X_states})[0]


    def get_losses(self, X_states, actions, max_q_values):
        '''
        Get losses
        '''

        return self.run([self.abs_losses],
                                feed_dict={
                                    self.X_state: X_states,
                                    self.X_action: actions,
                                    self.y: max_q_values
                                })[0]

    def copy_network(self):
        self.copy_online_to_target.run()        


    def get_training_step(self):
        '''
        Increments step and returns new value
        '''

        return self.training_step.eval()
        

    def set_game_count(self, count):
        '''
        sets game count
        '''
        self.game_count.load(count)


    def get_game_count(self):
        '''
        gets game count
        '''

        return self.run([self.game_count])[0]


    def epsilon(self, step):
        '''
        Gets current epsilon based on what step and the epsilon range
        '''
        return max(self.eps_min, self.eps_max - (self.eps_max-self.eps_min) * step/self.eps_decay_steps)


    def epsilon_greedy(self, q_values, step):
        '''
        Returns the optimal value if over epsilon, other wise returns the argmax action
        '''
        epsilon = self.epsilon(step)
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_outputs) # random action
        else:
            return np.argmax(q_values) # optimal action


    def make_model(self):
        '''
        Construct the tensorflow model
        '''

        # set placeholders
        self.X_action = tf.placeholder(tf.uint8, shape=[None], name='action')

        # target Q
        self.y = tf.placeholder(tf.float32, shape=[None], name='y')

        if self.conf['use_priority']:
            self.is_weights = tf.placeholder(tf.float32, [None], name='is_weights')
        else:
            self.is_weights = None

        # save how many games we've played
        self.game_count = tf.Variable(0, trainable=False, name='game_count')

        if self.USE_ENCODER:
            self.X_state = tf.placeholder(tf.float32, shape=[None,
                                                             self.INPUT_HEIGHT,
                                                             self.INPUT_WIDTH,
                                                             self.INPUT_CHANNELS])
            last = self.X_state
        elif self.USE_CONV:
            self.X_state = tf.placeholder(tf.uint8, shape=[None, 
                                                           self.INPUT_HEIGHT,
                                                           self.INPUT_WIDTH,
                                                           self.INPUT_CHANNELS])
            # convert rgb int (0-255) to floats
            last = tf.cast(self.X_state, tf.float32)
            last = tf.divide(last, 255)
        else:
            self.X_state = tf.placeholder(tf.float32, shape=[None, 
                                                             self.INPUT_HEIGHT,
                                                             self.INPUT_WIDTH,
                                                             self.INPUT_CHANNELS])
            last = self.X_state


        # make online and target q networks
        self.online_q_values, self.online_actions, online_vars = self.make_q_network(last, name='q_networks/online')
        self.target_q_values, self.target_actions, target_vars = self.make_q_network(last, name='q_networks/target')




        if self.conf['use_double']:
            # use online to select action and target to get max q value
            self.max_q_values = tf.reduce_max(self.target_q_values * tf.one_hot(tf.argmax(self.online_q_values, 
                                                                                          axis=1), 
                                                                                self.num_outputs), 
                                                     axis=1)
        else:
            # use target network to get max q value
            self.max_q_values = tf.reduce_max(self.target_q_values, axis=1)


        # make copy settings action
        copy_ops = []
        for var_name, target_var in target_vars.items():
            copy_ops.append(target_var.assign(online_vars[var_name]))

        self.copy_online_to_target = tf.group(*copy_ops)

        self.make_train()


    def make_q_network(self, X_input, name):
        '''
        Makes the core q network 
        '''

        last = X_input

        hidden_initializer = tf.contrib.layers.variance_scaling_initializer()


        if self.USE_CONV:
            # make convolutional network 
            conv_num_maps = [32, 64, 64]
            # conv_kernel_sizes = [8, 4, 3]
            conv_kernel_sizes = [8, 4, 4]
            conv_strides = [4, 2, 1]
            conv_paddings = ['same'] * 3
            conv_activations = [tf.nn.relu] * 3
            num_hidden = self.NUM_HIDDEN
        elif self.USE_ENCODER:
            # experimental autoencoder network
            conv_num_maps = [64, 64]
            conv_kernel_sizes = [4, 4]
            conv_strides = [2, 1]
            conv_paddings = ['same'] * len(conv_num_maps)
            conv_activations = [tf.nn.relu] * len(conv_num_maps)
            num_hidden = self.NUM_HIDDEN

        with tf.variable_scope(name) as scope:
            if self.USE_CONV or self.USE_ENCODER:
                # conv layers
                for num_maps, kernel_size, stride, padding, act_func in zip(conv_num_maps,
                                                                            conv_kernel_sizes,
                                                                            conv_strides,
                                                                            conv_paddings,
                                                                            conv_activations):
                    last = tf.layers.conv2d(last, 
                                            num_maps, 
                                            kernel_size=kernel_size, 
                                            strides=stride, 
                                            padding=padding,
                                            activation=act_func)

            input_layer = tf.layers.flatten(last)


            if self.conf['use_dueling']:
                # action output
                last = tf.layers.dense(input_layer,
                                       num_hidden,
                                       activation=tf.nn.relu,
                                       kernel_initializer=hidden_initializer)
                advantage = tf.layers.dense(last, 
                                            self.num_outputs,
                                            kernel_initializer=hidden_initializer)

                # value output
                last = tf.layers.dense(input_layer,
                                       num_hidden,
                                       activation=tf.nn.relu,
                                       kernel_initializer=hidden_initializer)
                value = tf.layers.dense(last, 
                                        1,
                                        kernel_initializer=hidden_initializer)

                # combine
                outputs = value + tf.subtract(advantage, tf.reduce_mean(advantage, 
                                                                        axis=1, 
                                                                        keepdims=True))
            else:
                # standard non-dueling architecture
                last = tf.layers.dense(input_layer,
                                       num_hidden,
                                       activation=tf.nn.relu,
                                       kernel_initializer=hidden_initializer)
                outputs = tf.layers.dense(last, 
                                          self.num_outputs,
                                          kernel_initializer=hidden_initializer)

            actions = tf.argmax(outputs, axis=1)


        # get var names for copy
        var_dict = {}
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=scope.name):
            var_dict[var.name[len(scope.name):]] = var

        return outputs, actions, var_dict



    def make_train(self):
        '''
        Make training tensors
        '''

        with tf.variable_scope("train"):
            self.online_max_q_values = tf.reduce_sum(self.online_q_values * tf.one_hot(self.X_action, self.num_outputs, dtype=tf.float32),
                                          axis=1)
            self.abs_losses = tf.abs(self.y - self.online_max_q_values,
                                     name='abs_losses')
            self.huber_losses = tf.losses.huber_loss(self.y,
                                                     self.online_max_q_values,
                                                     reduction=tf.losses.Reduction.NONE)

            if self.conf['use_priority']:
                self.loss = tf.reduce_mean(self.is_weights * self.huber_losses)
                self.losses = self.abs_losses
            else:
                self.loss = tf.reduce_mean(self.huber_losses)
                self.losses = self.huber_losses

            self.training_step = tf.Variable(0, 
                                    trainable=False, 
                                    name='step')

            if self.conf['use_momentum']:
                optimizer = tf.train.MomentumOptimizer(self.conf['learning_rate'],
                                                       self.conf['momentum'],
                                                       use_nesterov=True)
            else:
                optimizer = tf.train.AdamOptimizer(self.conf['learning_rate'])

            self.training_op = optimizer.minimize(self.loss, 
                                                  global_step=self.training_step)


class BreakoutAgent(GameAgent):
    INPUT_HEIGHT = 89
    INPUT_WIDTH = 80
    INPUT_CHANNELS = 4
    USE_CONV = True
    # compress_ratio = 2
    # game_report_interval = 10
    # num_lives = 0


    # def preprocess_observation(self, img):
    #     # crop and cut off score and bottom blank space
    #     img = img[16:-16:self.compress_ratio, ::self.compress_ratio] # crop and downsize
    #     img = np.dot(img[...,:3], [0.299, 0.587, 0.144])

    #     return img.astype('uint8').reshape(self.INPUT_HEIGHT, self.INPUT_WIDTH, 1)


    # def before_action(self, action, obs, reward, done, info):
    #     # start episode with action 1
    #     if info is not None and info['ale.lives'] != self.num_lives:
    #         action = 1
    #         self.num_lives = info['ale.lives']        

    #     return action


class CartPoleAgent(GameAgent):
    INPUT_HEIGHT = 2
    INPUT_WIDTH = 2
    INPUT_CHANNELS = 4
    USE_CONV = False
    # game_report_interval = 100
    # STATE_TYPE = 'float32'


    # def render_observation(self, obs):
    #     # rendering for the cart pole environment (in case OpenAI gym can't do it)
    #     img_w = 600
    #     img_h = 400
    #     cart_w = img_w // 12
    #     cart_h = img_h // 15
    #     pole_len = img_h // 3.5
    #     pole_w = img_w // 80 + 1
    #     x_width = 2
    #     max_ang = 0.2
    #     bg_col = (255, 255, 255)
    #     cart_col = 0x000000 # Blue Green Red
    #     pole_col = 0x669acc # Blue Green Red

    #     pos, vel, ang, ang_vel = obs
    #     img = Image.new('RGB', (img_w, img_h), bg_col)
    #     draw = ImageDraw.Draw(img)
    #     cart_x = pos * img_w // x_width + img_w // x_width
    #     cart_y = img_h * 95 // 100
    #     top_pole_x = cart_x + pole_len * np.sin(ang)
    #     top_pole_y = cart_y - cart_h // 2 - pole_len * np.cos(ang)
    #     draw.line((0, cart_y, img_w, cart_y), fill=0)
    #     draw.rectangle((cart_x - cart_w // 2, cart_y - cart_h // 2, cart_x + cart_w // 2, cart_y + cart_h // 2), fill=cart_col) # draw cart
    #     draw.line((cart_x, cart_y - cart_h // 2, top_pole_x, top_pole_y), fill=pole_col, width=pole_w) # draw pole
        
    #     return np.array(img)


class MsPacmanAgent(GameAgent):
    INPUT_HEIGHT = 86
    INPUT_WIDTH = 80
    INPUT_CHANNELS = 4
    USE_CONV = True
    # compress_ratio = 2
    # game_report_interval = 10
    # num_lives = 0


    # def preprocess_observation(self, img):
    #     # cut off score and icons from bottom
    #     img = img[0:-38:self.compress_ratio, ::self.compress_ratio] # crop and downsize
    #     img = np.dot(img[...,:3], [0.299, 0.587, 0.144])

    #     return img.astype('uint8').reshape(self.INPUT_HEIGHT, self.INPUT_WIDTH, 1)


