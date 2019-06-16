#
# Play game(s) using the deep q network and optionally 
# displays and saves the video
#
# To play with display (need X11 server)
#
# python play.py -O ./data/data-breakout/ --display
#

import argparse


from rl.deep_q_network import DeepQNetwork


parser = argparse.ArgumentParser()

# read config file from target save dir
parser.add_argument('-O', '--dir', '--save-dir', dest='save_dir', default='./data', help='read config and saved network from save dir')

# play options
parser.add_argument('-i', '--interval', dest='interval', type=int, default=60, help='frame rate for video / display')
parser.add_argument('-e', '--use-epsilon', dest='use_epsilon', action='store_true', default=False, help='use current epsilon setting from network')
parser.add_argument('--ng', '--num-games', dest='num_games', type=int, default=1, help='number of games to play')

# render game to x server
parser.add_argument('--display', dest='display', action='store_true', default=False, help='render game and display with x server')

# save game to mp4
parser.add_argument('--save', dest='save_video', action='store_true', default=False, help='save video to mp4 into save dir')

args = parser.parse_args()

conf = {}
conf.update(vars(args))

network = DeepQNetwork(conf, initialize=False)

network.predict()
