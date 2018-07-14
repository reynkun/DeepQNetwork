from collections import deque

import time
import gym
import os

import numpy as np
import tensorflow as tf

import matplotlib.animation as animation
import matplotlib.pyplot as plt



class GamePlayer():
    def __init__(self, env):
        self.env = env
        self.num_outputs = env.action_space.n
        self.make_network()


    def make_network():
        self.X = tf.placeholder(tf.float32, shape=[None, input_height, input_width,
                                                    input_channels])
        online_q_values, online_vars = self.make_q_network(X_state, name="q_networks/online")
        target_q_values, target_vars = self.make_q_network(X_state, name="q_networks/target")



        copy_ops = [target_var.assign(online_vars[var_name]) 
                    for var_name, target_var in target_vars.items()]
        copy_online_to_target = tf.group(*copy_ops)

        self.make_train()

        replay_size = 100000
        self.replay_memory = deque([], maxlen=replay_size)


    def make_q_network(self, X_input, scope):
        last = X_input

        conv_num_maps = [32, 64, 64]
        conv_kernel_sizes = [8, 4, 3]
        conv_strides = [4, 2, 1]
        conv_paddings = ['same'] * 3
        conv_activations = [tf.nn.relu] * 3
        num_hidden = 512
        hidden_initializer = tf.contrib.layers.variance_scaling_initializer()

        with tf.variable_scope('q_network') as scope:
            for num_maps, 
                kernel_size,
                stride,
                padding, 
                act_func in zip(conv_num_maps,
                               conv_kernel_sizes,
                               conv_paddings,
                               conv_activations):
                last = tf.layers.conv2d(last, 
                                        num_maps, 
                                        kernel_size=kernel_size, 
                                        strides=stride, 
                                        padding=padding,
                                        activation=act_func)

            print(last.shape)
            last = tf.reshape(last, 
                              shape=[-1, last.shape[1] * last.shape[2] * last.shape[3]])
            last = tf.layers.dense(last,
                                   num_hidden,
                                   activation=tf.nn.relu,
                                   kernel_initializer=hidden initializer)
        var_dict = {}
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=scope.name):
            var_dict[var.name[len(scope.name):]] = var

        # trainable_vars_by_name = {var.name[len(scope.name):]: var
        #                           for var in trainable_vars}
        return outputs, var_dict


    def make_train(self):
        learning_rate = 0.01
        momentum = 0.95

        with tf.variable_scope("train"):
            self.X_action = tf.placeholder(tf.int32, shape=[None])
            self.y = tf.placeholder(tf.float32, shape=[None, 1])
            q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, self.num_outputs),
                                    axis=1, 
                                    keep_dims=True)
            error = tf.abs(y - q_value)
            clipped_error = tf.clip_by_value(error, 0.0, 1.0)
            linear_error = 2 * (error - clipped_error)
            loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

            global_step = tf.Variable(0, 
                                      trainable=False, 
                                      name='global_step')
            optimizer = tf.train.MomentumOptimizer(learning_rate, 
                                                   momentum, 
                                                   use_nesterov=True)
            self.training_op = optimizer.minimize(loss, 
                                                  global_step=global_step)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()


    def sample_memories(self, batch_size):
        indices = np.random.permutation(len(replay_memory))[:batch_size]
        cols = [[], [], [], [], []] # state, action, reward, next_state, continue
        for idx in indices:
            memory = replay_memory[idx]
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)


    def epsilon_greedy(q_values, step):
        epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
        if np.random.rand() < epsilon:
            return np.random.randint(n_outputs) # random action
        else:
            return np.argmax(q_values) # optimal action

