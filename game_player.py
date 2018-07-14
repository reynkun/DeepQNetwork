from collections import deque

import time
import gym
import os

import numpy as np
import tensorflow as tf

import matplotlib.animation as animation
import matplotlib.pyplot as plt


MSPACMAN_COLOR = np.array([210, 164, 74]).mean()



class GamePlayer2:
    def __init__(self, env, compress_ratio=2):
        self.num_outputs = env.action_space.n
        self.compress_ratio = compress_ratio

        obs = env.reset()

        self.input_height = int(obs.shape[0] / compress_ratio)
        self.input_width = int(obs.shape[1] / compress_ratio)
        self.input_channels = 4
        self.skip_steps = 90

        self.make_network()


    def get_q_values(self, state, online=True):
        if online:
            return self.online_q_values.eval(feed_dict={sess.model.X_state: [next_state]})
        else:
            return self.target_q_values.eval(feed_dict={sess.model.X_state: [next_state]})


    def train_q_values(self, states, actions, y_vals):
        tr_res, loss_val = sess.run([sess.model.training_op, 
                                     sess.model.loss], 
                                    feed_dict={
                                        sess.model.X_state: self.convert_state(X_state_val),
                                        sess.model.X_action: X_action_val,
                                        sess.model.y: y_val
                                    })


    def make_network(self):
        self.X_state = tf.placeholder(tf.uint8, shape=[None, 
                                                       self.input_height, 
                                                       self.input_width, 
                                                       self.input_channels])
        last = tf.cast(self.X_state, tf.float16)
        last = tf.divide(last, 255)

        self.online_q_values, online_vars = self.make_q_network(last, name="q_networks/online")
        self.target_q_values, target_vars = self.make_q_network(last, name="q_networks/target")

        copy_ops = [target_var.assign(online_vars[var_name]) 
                    for var_name, target_var in target_vars.items()]
        self.copy_online_to_target = tf.group(*copy_ops)

        self.make_train()

        replay_size = 100000
        self.replay_memory = deque([], maxlen=replay_size)


    def make_q_network(self, X_input, name):
        last = X_input

        conv_num_maps = [32, 64, 64]
        conv_kernel_sizes = [8, 4, 3]
        conv_strides = [4, 2, 1]
        conv_paddings = ['same'] * 3
        conv_activations = [tf.nn.relu] * 3
        num_hidden = 512
        hidden_initializer = tf.contrib.layers.variance_scaling_initializer()

        with tf.variable_scope(name) as scope:
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

            last = tf.reshape(last, 
                              shape=[-1, last.shape[1] * last.shape[2] * last.shape[3]])
            last = tf.layers.dense(last,
                                   num_hidden,
                                   activation=tf.nn.relu,
                                   kernel_initializer=hidden_initializer)
            outputs = tf.layers.dense(last, 
                                      self.num_outputs,
                                      kernel_initializer=hidden_initializer)

        var_dict = {}
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=scope.name):
            var_dict[var.name[len(scope.name):]] = var

        # trainable_vars_by_name = {var.name[len(scope.name):]: var
        #                           for var in trainable_vars}
        return outputs, var_dict


    def make_train(self):
        learning_rate = 0.00025
        momentum = 0.95

        print('learning rate: {:0.3f}, momentum: {:0.3f}'.format(learning_rate, momentum))

        with tf.variable_scope("train"):
            self.X_action = tf.placeholder(tf.int32, shape=[None])
            self.y = tf.placeholder(tf.float16, shape=[None, 1])

            q_value = tf.reduce_sum(self.online_q_values * tf.one_hot(self.X_action, self.num_outputs, dtype=tf.float16),
                                    axis=1, 
                                    keepdims=True)

            # error = tf.abs(self.y - q_value)
            # clipped_error = tf.clip_by_value(error, 0.0, 1.0)
            # linear_error = 2 * (error - clipped_error)

            # # self.loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)
            # self.loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

            error = tf.abs(self.y - q_value)
            clipped_error = tf.clip_by_value(error, 0.0, 1.0)
            linear_error = (error - clipped_error)
            
            # self.loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)
            self.loss = tf.reduce_mean(tf.multiply(tf.square(clipped_error), 0.5) + linear_error)


            self.step = tf.Variable(0, 
                                    trainable=False, 
                                    name='step')
            self.game_count = tf.Variable(0, trainable=False, name='game_count')
            self.game_count_op = self.game_count.assign(tf.add(self.game_count, 1))

            optimizer = tf.train.MomentumOptimizer(learning_rate, 
                                                   momentum, 
                                                   use_nesterov=True)
            self.training_op = optimizer.minimize(self.loss, 
                                                  global_step=self.step)



    def preprocess_observation(self, img):
        img = img[::self.compress_ratio, ::self.compress_ratio] # crop and downsize
        img = img.mean(axis=2) # to grayscale
        img[img==MSPACMAN_COLOR] = 0 # improve contrast
        img = (img * 255).astype('uint8')

        return img.reshape(self.input_height, self.input_width, 1)


class GamePlayer():
    def __init__(self, num_outputs):
        self.num_outputs = num_outputs
        self.input_height = 105 
        self.input_width = 80
        # self.input_height = 210 
        # self.input_width = 160
        self.input_channels = 4
        self.skip_steps = 90

        self.make_network()


    def make_network(self):
        self.X_state = tf.placeholder(tf.float16, shape=[None, 
                                                         self.input_height, 
                                                         self.input_width, 
                                                         self.input_channels])

        self.online_q_values, online_vars = self.make_q_network(self.X_state, name="q_networks/online")
        self.target_q_values, target_vars = self.make_q_network(self.X_state, name="q_networks/target")



        copy_ops = [target_var.assign(online_vars[var_name]) 
                    for var_name, target_var in target_vars.items()]
        self.copy_online_to_target = tf.group(*copy_ops)

        self.make_train()

        replay_size = 100000
        self.replay_memory = deque([], maxlen=replay_size)


    def make_q_network(self, X_input, name):
        last = X_input

        conv_num_maps = [32, 64, 64]
        conv_kernel_sizes = [8, 4, 3]
        conv_strides = [4, 2, 1]
        conv_paddings = ['same'] * 3
        conv_activations = [tf.nn.relu] * 3
        num_hidden = 512
        hidden_initializer = tf.contrib.layers.variance_scaling_initializer()

        with tf.variable_scope(name) as scope:
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

            last = tf.reshape(last, 
                              shape=[-1, last.shape[1] * last.shape[2] * last.shape[3]])
            last = tf.layers.dense(last,
                                   num_hidden,
                                   activation=tf.nn.relu,
                                   kernel_initializer=hidden_initializer)
            outputs = tf.layers.dense(last, 
                                      self.num_outputs,
                                      kernel_initializer=hidden_initializer)

        var_dict = {}
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=scope.name):
            var_dict[var.name[len(scope.name):]] = var

        # trainable_vars_by_name = {var.name[len(scope.name):]: var
        #                           for var in trainable_vars}
        return outputs, var_dict


    def make_train(self):
        learning_rate = 0.00025
        momentum = 0.95

        print('learning rate: {:0.3f}, momentum: {:0.3f}'.format(learning_rate, momentum))

        with tf.variable_scope("train"):
            self.X_action = tf.placeholder(tf.int32, shape=[None])
            self.y = tf.placeholder(tf.float16, shape=[None, 1])

            q_value = tf.reduce_sum(self.online_q_values * tf.one_hot(self.X_action, self.num_outputs, dtype=tf.float16),
                                    axis=1, 
                                    keepdims=True)

            # error = tf.abs(self.y - q_value)
            # clipped_error = tf.clip_by_value(error, 0.0, 1.0)
            # linear_error = 2 * (error - clipped_error)

            # # self.loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)
            # self.loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

            error = tf.abs(self.y - q_value)
            clipped_error = tf.clip_by_value(error, 0.0, 1.0)
            linear_error = (error - clipped_error)
            
            # self.loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)
            self.loss = tf.reduce_mean(tf.multiply(tf.square(clipped_error), 0.5) + linear_error)


            self.step = tf.Variable(0, 
                                    trainable=False, 
                                    name='step')
            self.game_count = tf.Variable(0, trainable=False, name='game_count')
            self.game_count_op = self.game_count.assign(tf.add(self.game_count, 1))

            optimizer = tf.train.MomentumOptimizer(learning_rate, 
                                                   momentum, 
                                                   use_nesterov=True)
            self.training_op = optimizer.minimize(self.loss, 
                                                  global_step=self.step)



    def preprocess_observation(self, img):
        img = img[::2, ::2] # crop and downsize
        img = img.mean(axis=2) # to grayscale
        img[img==MSPACMAN_COLOR] = 0 # improve contrast

        return img.reshape(self.input_height, self.input_width, 1) / 255

