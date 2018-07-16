from collections import deque

import time
import numpy as np
import tensorflow as tf
import gym
import os
import pickle

import matplotlib.animation as animation
import matplotlib.pyplot as plt


mspacman_color = np.array([210, 164, 74]).mean()

# def render_policy_net(q_values_op, X, n_max_steps = 1000):
#     frames = []
#     env = gym.make("MsPacman-v0")
#     obs = env.reset()

#     for step in range(n_max_steps):
#         frames.append(obs)
#         state = preprocess_observation(obs)
#         q_values = q_values_op.eval(feed_dict={X_state: [state]})
#         action = np.argmax(q_values)
#         obs, reward, done, info = env.step(action)

#         if done:
#           break

#     env.close()
#     return frames  


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,


def plot_animation(frames, repeat=False, interval=40):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(fig, 
                                   update_scene, 
                                   fargs=(frames, patch), 
                                   frames=len(frames), 
                                   repeat=repeat, 
                                   interval=interval)


def run_game(env, q_values_op, n_max_steps=10000):
    info = None
    frames = []
    total_rewards = 0
    last_lives = 0
    obs = env.reset()
    # img = env.render(mode="rgb_array")
    frames.append(obs)

    for step in range(n_max_steps):
        if info is not None and info['ale.lives'] != last_lives:
            last_lives = info['ale.lives']
            action = 1
        else:
            state = preprocess_observation(obs)

            # Online DQN evaluates what to do
            q_values = q_values_op.eval(feed_dict={X_state: [state]})
            action = np.argmax(q_values)

        # Online DQN plays
        obs, reward, done, info = env.step(action)
        total_rewards += reward

        # img = env.render(mode="rgb_array")
        frames.append(obs)

        if done:
            print('done at {} steps. frame {} score {}'.format(step, len(frames), total_rewards))
            break

    video = plot_animation(frames)
    plt.show()    


def preprocess_observation(obs):
    img = obs[40:200:2, ::2] # crop and downsize
    img = img.mean(axis=2) # to grayscale
    img[img==mspacman_color] = 0 # improve contrast
    img = (img - 128) / 128 - 1 # normalize from -1. to 1.
    return img.reshape(80, 80, 1)


game_id = "Breakout-v0"
env = gym.make(game_id)

input_height = 80
input_width = 80
input_channels = 1
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3 
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 10 * 10  # conv3 has 64 maps of 11x10 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n  # 9 discrete actions are available
initializer = tf.contrib.layers.variance_scaling_initializer()

def q_network(X_state, name):
    prev_layer = X_state / 128.0 # scale pixel intensities to the [-1.0, 1.0] range.
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
                conv_n_maps, conv_kernel_sizes, conv_strides,
                conv_paddings, conv_activation):
            prev_layer = tf.layers.conv2d(
                prev_layer, filters=n_maps, kernel_size=kernel_size,
                strides=strides, padding=padding, activation=activation,
                kernel_initializer=initializer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                 activation=hidden_activation,
                                 kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden, n_outputs,
                                  kernel_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name


X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width,
                                            input_channels])
online_q_values, online_vars = q_network(X_state, name="q_networks/online")
target_q_values, target_vars = q_network(X_state, name="q_networks/target")

copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

# X_action = tf.placeholder(tf.int32, shape=[None])
# q_value = tf.reduce_sum(target_q_values * tf.one_hot(X_action, n_outputs),
#                         axis=1, keep_dims=True)

# y = tf.placeholder(tf.float32, shape=[None, 1])
# error = tf.abs(y - q_value)
# clipped_error = tf.clip_by_value(error, 0.0, 1.0)
# linear_error = 2 * (error - clipped_error)
# loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

# learning_rate = 0.001
# momentum = 0.95

# global_step = tf.Variable(0, trainable=False, name='global_step')
# optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
# training_op = optimizer.minimize(loss, global_step=global_step)

# init = tf.global_variables_initializer()
# saver = tf.train.Saver()

learning_rate = 0.001
momentum = 0.95

with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                            axis=1, keep_dims=True)
    error = tf.abs(y - q_value)
    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
    linear_error = 2 * (error - clipped_error)
    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()





from collections import deque

replay_memory_size = 100000
replay_memory = deque([], maxlen=replay_memory_size)

def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)


eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 2000000

def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action




n_steps = 4000000  # total number of training steps
training_start = 10000  # start training after 10,000 game iterations
training_interval = 4  # run a training step every 4 game iterations
save_steps = 1000  # save the model every 1,000 training steps
copy_steps = 10000  # copy online DQN to target DQN every 10,000 training steps
discount_rate = 0.99
skip_start = 90  # Skip the start of every game (it's just waiting time).
batch_size = 50
iteration = 0  # game iterations
checkpoint_path = "./{}".format(game_id)
memory_fn = checkpoint_path + '.memory'
done = True # env needs to be reset


loss_val = np.infty
game_length = 0
total_max_q = 0
mean_max_q = 0.0


with tf.Session() as sess:
    if os.path.isfile(checkpoint_path + ".index"):
        print('restoring model')
        saver.restore(sess, checkpoint_path)

        if os.path.exists(memory_fn):
            print('loading memory file')
            try:
                with open(memory_fn, 'rb') as f:
                    replay_memory = pickle.load(f)
            except EOFError:
                print('error while loading memory file')
                pass

    else:
        print('creating new model')
        init.run()
        copy_online_to_target.run()

    run_game(env, online_q_values)

    obs = env.reset()

    next_q_values = None
    first_run = True
    game_scores = []
    num_games_played = -1

    try:
        while True:
            step = global_step.eval()
            if step >= n_steps:
                break
            iteration += 1
            if iteration % 1000 == 0:
                if next_q_values is not None:
                    sh = next_q_values.shape
                else:
                    sh = ''
      
                t = time.strftime('%H:%M:%S', time.localtime(time.time()))

                if len(game_scores) > 0:
                    avg_score = sum(game_scores) / len(game_scores)
                else:
                    avg_score = 0.0

                print("{} Iteration {} Training step {}/{} ({:.1f})% Loss {:5f} Mean Max-Q {:5f} {} games played, average score: {:.3f} / {}".format(
                      t, iteration, step, n_steps, step * 100 / n_steps, loss_val, mean_max_q, num_games_played, avg_score, len(game_scores)))
                game_scores = []

            if done: # game over, start again
                if not first_run:
                    game_scores.append(game_score)
                first_run = False
                game_score = 0
                last_lives = 0
                num_games_played += 1
                info = None

                obs = env.reset()
                state = preprocess_observation(obs)

            if info is not None and last_lives != info['ale.lives']:
                last_lives = info['ale.lives']                
                obs, reward, done, info = env.step(1)                

            # Online DQN evaluates what to do
            q_values = online_q_values.eval(feed_dict={X_state: [state]})
            action = epsilon_greedy(q_values, step)

            # Online DQN plays
            obs, reward, done, info = env.step(action)
            game_score += reward
            next_state = preprocess_observation(obs)

            # Let's memorize what happened
            replay_memory.append((state, action, reward, next_state, 1.0 - done))
            state = next_state

            # Compute statistics for tracking progress (not shown in the book)
            total_max_q += q_values.max()
            game_length += 1
            if done:
                mean_max_q = total_max_q / game_length
                total_max_q = 0.0
                game_length = 0

            if len(replay_memory) < 10000 or iteration % training_interval != 0:
                continue # only train after warmup period and at regular intervals
            
            # Sample memories and use the target DQN to produce the target Q-Value
            X_state_val, X_action_val, rewards, X_next_state_val, continues = (
                sample_memories(batch_size))
            next_q_values = target_q_values.eval(
                feed_dict={X_state: X_next_state_val})
            max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
            y_val = rewards + continues * discount_rate * max_next_q_values

            # Train the online DQN
            _, loss_val = sess.run([training_op, loss], feed_dict={
                X_state: X_state_val, X_action: X_action_val, y: y_val})

            # Regularly copy the online DQN to the target DQN
            if step % copy_steps == 0:
                copy_online_to_target.run()

            # And save regularly
            if step % save_steps == 0:
                saver.save(sess, checkpoint_path)
    except KeyboardInterrupt:
        pass


    print('dumping memory to ', memory_fn)
    while True:
        try:
            with open(memory_fn, 'wb') as f:
                pickle.dump(replay_memory, f)
            break
        except MemoryError:
            print('memory error!')
            size = int(len(replay_memory) * 0.80)
            for i in range(size):
                replay_memory.popleft()
            continue

    print('dumped memory')

