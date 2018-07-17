from collections import deque


import numpy as np
import tensorflow as tf


class GameAgent:
    def __init__(self, env, compress_ratio=2, initialize=True):
        self.num_outputs = env.action_space.n
        self.compress_ratio = compress_ratio

        obs = env.reset()

        self.input_height = int(obs.shape[0] / compress_ratio)
        self.input_width = int(obs.shape[1] / compress_ratio)
        self.input_channels = 4
        self.skip_steps = 0

        self.lives = 0

        if initialize:
            self.make_model()


    def best_action(self, X_states, sess=None, use_target=False):
        if sess is None:
            sess = tf.get_default_session()

        values = self.predict(X_states, sess=sess, use_target=use_target)

        return np.argmax(values, axis=1)


    def train(self, X_states, X_actions, y, sess=None):
        if sess is None:
            sess = tf.get_default_session()

        train, loss = sess.run([self.model.training_op, 
                                self.model.loss], 
                               feed_dict={
                                   self.model.X_state: X_states,
                                   self.model.X_action: X_actions,
                                   self.model.y: y
                               })

        return loss


    def predict(self, X_states, sess=None, use_target=False):
        if sess is None:
            sess = tf.get_default_session()

        if use_target:
            q_values = self.target_q_values
        else:
            q_values = self.online_q_values

        values = sess.run([q_values], 
                          feed_dict={self.model.X_state: self.convert_state(X_next_state_val)})

        return values


    def make_model(self):
        # set placeholders
        self.X_state = tf.placeholder(tf.uint8, shape=[None, 
                                                       self.input_height, 
                                                       self.input_width, 
                                                       self.input_channels])
        self.X_action = tf.placeholder(tf.int32, shape=[None])
        self.X_rewards = tf.placeholder(tf.int32, shape=[None])
        self.X_continues = tf.placeholder(tf.int8, shape=[None])

        self.y = tf.placeholder(tf.float16, shape=[None, 1])
        self.discount_rate = tf.constant(0.99, dtype=tf.float16)

        # save how many games we've played
        self.game_count = tf.Variable(0, trainable=False, name='game_count')

        # convert rgb int (0-255) to floats
        last = tf.cast(self.X_state, tf.float16)
        last = tf.divide(last, 255)

        # make online and target q networks
        self.online_q_values, self.online_actions, online_vars = self.make_q_network(last, name="q_networks/online")
        self.target_q_values, self.target_actions, target_vars = self.make_q_network(last, name="q_networks/target")

        # self.max_online_q_values = tf.reduce_sum(self.online_q_values * tf.one_hot(self.X_action, self.num_outputs, dtype=tf.float16),
        #                                          axis=1, 
        #                                          keepdims=True)
        self.double_max_q_values = tf.reduce_sum(self.target_q_values * tf.one_hot(self.online_actions, self.num_outputs, dtype=tf.float16),
                                                 axis=1,
                                                 keepdims=True)
        self.max_q_values = tf.reduce_sum(self.target_q_values * tf.one_hot(self.X_action, self.num_outputs, dtype=tf.float16),
                                          axis=1,
                                          keepdims=True)

        # print('dd:', self.double_max_q_values.shape)
        # self.q_cur_and_next_q_values = tf.add(tf.cast(tf.reshape(self.X_rewards, (-1, 1)), tf.float16),
        #                                       tf.multiply(tf.cast(tf.reshape(self.X_continues, (-1, 1)), tf.float16),
        #                                                   tf.multiply(self.discount_rate,
        #                                                               self.double_max_q_values)))
        # print('qnext', self.q_cur_and_next_q_values.shape)
        # self.max_target_q_value = tf.reduce_sum(self.target_q_values * tf.one_hot(self.X_action, self.num_outputs, dtype=tf.float16),
        #                                         axis=1, 
        #                                         keepdims=True)
        # make copy online to target op
        copy_ops = []
        for var_name, target_var in target_vars.items():
            copy_ops.append(target_var.assign(online_vars[var_name]))

        self.copy_online_to_target = tf.group(*copy_ops)

        self.make_train()


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

            reshape_layer = tf.reshape(last, 
                              shape=[-1, last.shape[1] * last.shape[2] * last.shape[3]])
            # action output
            last = tf.layers.dense(reshape_layer,
                                   num_hidden,
                                   activation=tf.nn.relu,
                                   kernel_initializer=hidden_initializer)
            advantage = tf.layers.dense(last, 
                                      self.num_outputs,
                                      kernel_initializer=hidden_initializer)

            # value output
            last = tf.layers.dense(reshape_layer,
                                   num_hidden,
                                   activation=tf.nn.relu,
                                   kernel_initializer=hidden_initializer)
            value = tf.layers.dense(last, 
                                         1,
                                         kernel_initializer=hidden_initializer)

            # combine
            outputs = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))
            actions = tf.argmax(outputs, axis=1)

        var_dict = {}
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=scope.name):
            var_dict[var.name[len(scope.name):]] = var

        return outputs, actions, var_dict


    def make_train(self):
        learning_rate = 0.00025
        momentum = 0.95

        print('learning rate: {:0.3f}, momentum: {:0.3f}'.format(learning_rate, momentum))

        with tf.variable_scope("train"):
            q_values = tf.reduce_sum(self.online_q_values * tf.one_hot(self.X_action, self.num_outputs, dtype=tf.float16),
                                     axis=1, 
                                     keepdims=True)

            # self.error = tf.abs(self.y - q_value)
            # clipped_error = tf.clip_by_value(self.error, 0.0, 1.0)
            # linear_error = (self.error - clipped_error)
            # self.loss_action = tf.multiply(tf.square(clipped_error), 0.5) + linear_error
            # self.loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)
            # self.loss = tf.reduce_mean(tf.multiply(tf.square(clipped_error), 0.5) + linear_error)
            self.losses = tf.losses.huber_loss(self.y, q_values, reduction=tf.losses.Reduction.NONE)
            self.loss = tf.reduce_mean(self.losses)

            self.step = tf.Variable(0, 
                                    trainable=False, 
                                    name='step')

            optimizer = tf.train.MomentumOptimizer(learning_rate, 
                                                   momentum, 
                                                   use_nesterov=True)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, 
            #                                       momentum=momentum)
            self.training_op = optimizer.minimize(self.loss, 
                                                  global_step=self.step)


    def preprocess_observation(self, img):
        img = img[::self.compress_ratio, ::self.compress_ratio] # crop and downsize
        img = np.dot(img[...,:3], [0.299, 0.587, 0.144])

        return img.astype('uint8').reshape(self.input_height, self.input_width, 1)


    def on_info(self, info):
        pass


class PacmanGameAgent(GameAgent):
    skip_steps = 90