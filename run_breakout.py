import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import gym
from PIL import Image, ImageFont, ImageDraw

env = gym.make('Breakout-v0')
obs = env.reset()

print('actions', env.action_space.n)

def update_scene(num, frames, patch):
    print('f', end='')
    patch.set_data(frames[num])
    return patch,


def plot_animation(frames, repeat=False, interval=500):
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

    for step in range(max_steps):
        # Online DQN plays
        print('.', end='')

        if info is not None and info['ale.lives'] != last_lives:
            last_lives = info['ale.lives']
            action = 1
        else:
            action = 0

        obs, reward, done, info = env.step(action)

        if reward > 0:
            print('+', end='')

        img = Image.fromarray(obs)
        draw = ImageDraw.Draw(img)
        draw.text((0,0), "fr {}".format(step))

        obs = np.array(img)

        frames.append(obs)

        if done:
            print('done at {} steps. frame {} score {}'.format(step, len(frames), total_rewards))
            break

    video = plot_animation(frames)
    plt.show()


run_game(env, max_steps=10000)