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
        self.X_state = tf.placeholder(tf.uint8, shape=[None, 
                                                       self.input_height, 
                                                       self.input_width, 
                                                       self.input_channels])
        last = tf.cast(self.X_state, tf.float16)
        last = tf.divide(last, 255)

        self.online_q_values, self.online_actions, online_vars = self.make_q_network(last, name="q_networks/online")
        self.target_q_values, self.target_actions, target_vars = self.make_q_network(last, name="q_networks/target")

        # all_range = tf.range(0, tf.shape(self.target_q_values, out_type='int64')[0])

        # print(self.online_actions.shape, all_range.shape, self.target_q_values.shape)
        # double_q_values = self.target_q_values[all_range,
        #                                        self.online_actions]
        # self.double_q_values = tf.reshape(double_q_values, -1, 1)

        copy_ops = [target_var.assign(online_vars[var_name]) 
                    for var_name, target_var in target_vars.items()]
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

            dense_input = tf.reshape(last, 
                              shape=[-1, last.shape[1] * last.shape[2] * last.shape[3]])
            # action output
            last = tf.layers.dense(dense_input,
                                   num_hidden,
                                   activation=tf.nn.relu,
                                   kernel_initializer=hidden_initializer)
            advantage = tf.layers.dense(last, 
                                      self.num_outputs,
                                      kernel_initializer=hidden_initializer)

            # value output
            last = tf.layers.dense(dense_input,
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
            self.X_action = tf.placeholder(tf.int32, shape=[None])
            self.y = tf.placeholder(tf.float16, shape=[None, 1])

            q_value = tf.reduce_sum(self.online_q_values * tf.one_hot(self.X_action, self.num_outputs, dtype=tf.float16),
                                    axis=1, 
                                    keepdims=True)

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
        img = np.dot(img[...,:3], [0.299, 0.587, 0.144])

        return img.astype('uint8').reshape(self.input_height, self.input_width, 1)


    def on_info(self, info):
        pass
