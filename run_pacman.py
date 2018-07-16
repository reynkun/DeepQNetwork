import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import gym
from PIL import Image, ImageFont, ImageDraw

env = gym.make('MsPacman-v0')
obs = env.reset()


def preprocess_observation(obs):

    img = obs[1:176:2, ::2] # crop and downsize
    img = img.mean(axis=2) # to grayscale
    # img = img / 255 
    img = img.reshape(88, 80, 1) 
    return np.concatenate((img, img, img), axis=2).astype('uint8')
    # return img

def update_scene(num, frames, patch):
    print('f', end='')
    patch.set_data(frames[num])
    return patch,


def plot_animation(frames, repeat=False, interval=20):
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

def run_game(env, max_steps=10000):
    frames = []
    total_rewards = 0
    obs = env.reset()
    # img = env.render(mode="rgb_array")
    frames.append(obs)

    env.step(1)
    last_lives = 5
    info = None
    action = 0

    for step in range(max_steps):
        # Online DQN plays
        print('.', end='')

        obs, reward, done, info = env.step(action)

        if reward > 0:
            print('+', end='')

        obs2 = preprocess_observation(obs)

        # obs2 = np.concatenate((obs2, obs2, obs2), axis=2)
        # print(obs2)
        # print(obs)
        # print(obs2.shape)
        img = Image.fromarray(obs2)
        draw = ImageDraw.Draw(img)
        draw.text((0,0), "fr {}".format(step))

        obs = np.array(img)

        frames.append(obs)

        if done:
            print('done at {} steps. frame {} score {}'.format(step, len(frames), total_rewards))
            break

    video = plot_animation(frames)
    plt.show()


print('actions', env.action_space.n)


run_game(env, max_steps=10000)