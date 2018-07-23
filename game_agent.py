from collections import deque


import numpy as np
import tensorflow as tf


class GameAgent:
    skip_steps = 0
    input_height = 2
    input_width = 2
    input_channels = 4
    use_conv = False
    state_type='uint8'


    def __init__(self, env, initialize=True, options=None):
        self.num_outputs = env.action_space.n

        if options:
            self.options = options
        else:
            self.options = {}

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
        print('input_channels:', self.input_channels)
        self.X_action = tf.placeholder(tf.uint8, shape=[None])
        # target Q
        self.y = tf.placeholder(tf.float16, shape=[None, 1])

        # save how many games we've played
        self.game_count = tf.Variable(0, trainable=False, name='game_count')

        if self.use_conv:
            self.X_state = tf.placeholder(tf.uint8, shape=[None, 
                                                           self.input_height,
                                                           self.input_width,
                                                           self.input_channels])
            # convert rgb int (0-255) to floats
            last = tf.cast(self.X_state, tf.float16)
            last = tf.divide(last, 255)
        else:
            self.X_state = tf.placeholder(tf.float16, shape=[None, 
                                                             self.input_height,
                                                             self.input_width,
                                                             self.input_channels])
            last = self.X_state


        # make online and target q networks
        self.online_q_values, self.online_actions, online_vars = self.make_q_network(last, name='q_networks/online')
        self.target_q_values, self.target_actions, target_vars = self.make_q_network(last, name='q_networks/target')

        self.double_max_q_values = tf.reduce_sum(self.target_q_values * tf.one_hot(self.online_actions, self.num_outputs, dtype=tf.float16),
                                                 axis=1,
                                                 keepdims=True)
        self.max_q_values = tf.reduce_sum(self.target_q_values * tf.one_hot(self.X_action, self.num_outputs, dtype=tf.float16),
                                          axis=1,
                                          keepdims=True)
        copy_ops = []
        for var_name, target_var in target_vars.items():
            copy_ops.append(target_var.assign(online_vars[var_name]))

        self.copy_online_to_target = tf.group(*copy_ops)

        self.make_train()


    def make_q_network(self, X_input, name):
        last = X_input

        num_hidden = 512
        hidden_initializer = tf.contrib.layers.variance_scaling_initializer()


        if self.use_conv:
            conv_num_maps = [32, 64, 64]
            conv_kernel_sizes = [8, 4, 3]
            conv_strides = [4, 2, 1]
            conv_paddings = ['same'] * 3
            conv_activations = [tf.nn.relu] * 3
            num_hidden = 512


        with tf.variable_scope(name) as scope:
            if self.use_conv:
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

            input_layer = tf.reshape(last, 
                                     shape=[-1, last.shape[1] * last.shape[2] * last.shape[3]])



            if self.options['use_dueling']:
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
        learning_rate = 0.00025
        momentum = 0.95

        print('learning rate: {:0.3f}, momentum: {:0.3f}'.format(learning_rate, momentum))

        with tf.variable_scope("train"):
            self.online_max_q_values = tf.reduce_sum(self.online_q_values * tf.one_hot(self.X_action, self.num_outputs, dtype=tf.float16),
                                          axis=1, 
                                          keepdims=True)
            print(self.online_q_values.shape, self.online_max_q_values.shape)

            self.losses = tf.losses.huber_loss(self.y, self.online_max_q_values, reduction=tf.losses.Reduction.NONE)
            self.loss = tf.reduce_mean(self.losses)

            self.step = tf.Variable(0, 
                                    trainable=False, 
                                    name='step')

            optimizer = tf.train.MomentumOptimizer(learning_rate, 
                                                   momentum, 
                                                   use_nesterov=True)

            self.training_op = optimizer.minimize(self.loss, 
                                                  global_step=self.step)


    def preprocess_observation(self, obs):
        return obs.reshape(self.input_height, self.input_width, 1)


    def render_obs(self, obs):
        return obs


class BreakoutAgent(GameAgent):
    input_height = 89
    input_width = 80
    input_channels = 4
    use_conv = True
    compress_ratio = 2
    game_report_interval = 10
    num_lives = 0


    def preprocess_observation(self, img):
        img = img[16:-16:self.compress_ratio, ::self.compress_ratio] # crop and downsize
        img = np.dot(img[...,:3], [0.299, 0.587, 0.144])

        return img.astype('uint8').reshape(self.input_height, self.input_width, 1)


    def before_action(self, action, obs, reward, done, info):
        if info is not None and info['ale.lives'] != self.num_lives:
            action = 1
            self.num_lives = info['ale.lives']        

        return action


class CartPoleAgent(GameAgent):
    input_height = 2
    input_width = 2
    input_channels = 4
    use_conv = False
    game_report_interval = 100
    state_type='float16'


    def render_obs(obs):
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
