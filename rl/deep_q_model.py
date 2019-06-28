from collections import deque


import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw

from .model.model import Model


class DeepQModel(Model):
    '''
    Builds the tensorflow model for the game agent.
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
        # use priority experience replay
        'use_per': False,
        # q value discount rate between states
        'discount_rate': 0.99
    }

    # class variables.  override for each game
    
    # game screen size / channels
    INPUT_HEIGHT = 110
    INPUT_WIDTH = 80
    INPUT_CHANNELS = 4

    # use convolutional filters. only needs to be false for non-image input
    USE_CONV = True

    # experimental use of auto encoder to compress images
    USE_ENCODER = False

    # dtype for input
    STATE_TYPE = 'uint8'

    # num hidden nodes in dense layer
    NUM_HIDDEN = 512

    # num of game frames per game state
    NUM_FRAMES_PER_STATE = 4


    def __init__(self, conf):
        super().__init__(self, conf)
        
        self.conf = self.DEFAULT_OPTIONS.copy()

        for key, value in conf.items():
            if key not in self.conf:
                self.conf[key] = value

        self.num_outputs = self.conf['action_space']
        self.discount_rate = self.conf['discount_rate']


    def train(self, X_states, X_actions, rewards, continues, next_states, is_weights=None):
        '''
        trains network
        '''

        target_q_values = self._get_target_q_values(rewards, continues, next_states)


        if self.conf['use_per']:
            feed_dict = {
                self.X_state: X_states,
                self.X_action: X_actions,
                self.y: target_q_values,
                self.is_weights: is_weights
            }
        else:
            feed_dict = {
                self.X_state: X_states,
                self.X_action: X_actions,
                self.y: target_q_values
            }


        step, _, losses, loss = self.run([self.training_step,
                                          self.training_op,
                                          self.losses,
                                          self.loss],
                                          feed_dict=feed_dict)
        return step, losses, loss


    def _get_target_q_values(self, rewards, continues, next_states):
        '''
        Get max q value
        '''

        max_q_values = self.run([self.max_q_values],
                                feed_dict={self.X_state: next_states})[0]

        return rewards + continues * self.discount_rate * max_q_values



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


    def get_gasses(self, X_states, actions, rewards, continues, next_states):
        '''
        Get losses
        '''

        target_q_values = self._get_target_q_values(rewards, continues, next_states)


        return self.run([self.losses],
                        feed_dict={
                            self.X_state: X_states,
                            self.X_action: actions,
                            self.y: target_q_values
                        })[0]


    def copy_network(self):
        '''
        copy online network to the target network
        '''

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


    def make_model(self):
        '''
        Construct the tensorflow model
        '''

        self._make_inputs()
        self._make_network()
        self._make_train()


    def _make_inputs(self):
        '''
        Make inputs 
        '''        
        # set placeholders
        self.X_action = tf.placeholder(tf.uint8, shape=[None], name='action')

        # target Q
        self.y = tf.placeholder(tf.float32, shape=[None], name='y')

        if self.conf['use_per']:
            self.is_weights = tf.placeholder(tf.float32, [None], name='is_weights')
        else:
            self.is_weights = None

        # save how many games we've played
        self.game_count = tf.Variable(0, trainable=False, name='game_count')

        if self.USE_CONV:
            # regular image input
            self.X_state = tf.placeholder(tf.uint8, shape=[None, 
                                                           self.INPUT_HEIGHT,
                                                           self.INPUT_WIDTH,
                                                           self.INPUT_CHANNELS])
            # convert rgb int (0-255) to floats
            last = tf.cast(self.X_state, tf.float32)
            self.input = tf.divide(last, 255)
        else:
            # allow for float input 
            self.X_state = tf.placeholder(tf.float32, shape=[None, 
                                                             self.INPUT_HEIGHT,
                                                             self.INPUT_WIDTH,
                                                             self.INPUT_CHANNELS])
            self.input = self.X_state


    def _make_network(self):
        '''
        Make main network 
        '''

        # make online and target q networks
        self.online_q_values, self.online_actions, online_vars = self._make_q_network(self.input, name='q_networks/online')
        self.target_q_values, self.target_actions, target_vars = self._make_q_network(self.input, name='q_networks/target')


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


    def _make_q_network(self, X_input, name):
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



    def _make_train(self):
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

            if self.conf['use_per']:
                # self.loss = tf.reduce_mean(self.is_weights * self.huber_losses)
                self.loss = tf.reduce_mean(self.is_weights * tf.squared_difference(self.y, self.online_max_q_values))
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


class BreakoutModel(DeepQModel):
    INPUT_HEIGHT = 89
    INPUT_WIDTH = 80
    INPUT_CHANNELS = 4
    USE_CONV = True


class SpaceInvadersModel(DeepQModel):
    INPUT_HEIGHT = 89
    INPUT_WIDTH = 80
    INPUT_CHANNELS = 4
    USE_CONV = True


class MsPacmanModel(DeepQModel):
    INPUT_HEIGHT = 86
    INPUT_WIDTH = 80
    INPUT_CHANNELS = 4
    USE_CONV = True
 