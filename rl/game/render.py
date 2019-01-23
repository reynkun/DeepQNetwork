import matplotlib.animation as animation
import matplotlib.pyplot as plt

from PIL import Image, ImageFont, ImageDraw

import numpy as np


def render_state(state, repeat=True, interval=800):
    def update_scene(num, frames, patch):
        patch.set_data(frames[num])
        return patch,


    frames = []
    for i in range(state.shape[2]):
        # print(state[:,:,i].shape)
        shape = state[:,:,i].shape
        frame = state[:,:,i].reshape(shape[0], shape[1], 1)

        img_dat = np.concatenate((frame, frame, frame), axis=2)
        img_dat = Image.fromarray((img_dat * 255).astype('uint8'))
        draw = ImageDraw.Draw(img_dat)
        draw.text((0,0), "fr {}".format(i))

        frames.append(img_dat)

    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('on')

    video = animation.FuncAnimation(fig, 
                                   update_scene, 
                                   fargs=(frames, patch), 
                                   frames=len(frames), 
                                   repeat=repeat, 
                                   interval=interval)
    plt.show()

    return video


def render_cart_pole(obs):
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


def render_game(game_frames, actions, render_func=None, repeat=True, interval=300, save_path=None, display=False):
    def update_scene(num, frames, patch):
        patch.set_data(frames[num])
        return patch,


    frames = []
    for i in range(len(game_frames)):
        if render_func:
            obs = render_func(game_frames[i])
        else:
            obs = game_frames[i]

        img_dat = Image.fromarray(obs)
        draw = ImageDraw.Draw(img_dat)
        draw.text((0,0), "fr {} act {:d}".format(i, actions[i]))

        frames.append(img_dat)


    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('on')

    print('interval:', interval)


    video = animation.FuncAnimation(fig, 
                                    update_scene, 
                                    fargs=(frames, patch), 
                                    frames=len(frames), 
                                    repeat=repeat, 
                                    interval=interval)

    if save_path is not None:
        video.save(save_path)

    if display:
        plt.show()

    return video

