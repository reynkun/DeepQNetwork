#
# Plays game using the deep q network
#

import argparse


from rl.deep_q_network import DeepQNetwork


parser = argparse.ArgumentParser()

# read config file from target save dir
parser.add_argument('-O', '--dir', '--save-dir', dest='save_dir', default='./data', help='read config and saved network from save dir')

parser.add_argument('-i', '--interval', dest='interval', type=int, default=60, 'frame rate for video / display')
parser.add_argument('-e', '--use-epsilon', dest='use_epsilon', action='store_true', default=False, help='use current epsilon setting from network')
parser.add_argument('--ng', '--num-games', dest='num_games', type=int, default=1, help='number of games to play')

# render game to x server
parser.add_argument('--display', dest='display', action='store_true', default=False, help='render game and display with x server')

# save game to mp4
parser.add_argument('--save', dest='save_video', action='store_true', default=False, help='save video to mp4 into save dir')

args = parser.parse_args()

conf = {}
conf.update(vars(args))

qn = DeepQNetwork(conf, initialize=False)

qn.predict(use_epsilon=args.use_epsilon, 
           interval=args.interval, 
           num_games=args.num_games, 
           display=args.display,
           save_video=args.save_video)
