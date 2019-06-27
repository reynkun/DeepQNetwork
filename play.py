#
# Play game(s) using the deep q network and optionally 
# displays and saves the video
#
# To play with display (need X11 server)
#
# python play.py -O ./data/data-breakout/ --display
#

import argparse


from rl.utils.import_class import import_class
from rl.game_runner import GameRunner
from rl.utils.logging import init_logging


parser = argparse.ArgumentParser()

parser.add_argument('-a', '--agent', dest='agent', default='rl.deep_q_agent.DeepQAgent')
parser.add_argument('--env', '--environment', dest='environment', default='rl.game_environment.BreakoutEnvironment')

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

# args = parser.parse_args()

# conf = {}
# conf.update(vars(args))

# # network = DeepQNetwork(conf, initialize=False)
# # network.predict()


args = parser.parse_args()

conf = {
    'is_training': False
}

conf.update(vars(args))

env = import_class(conf['environment'])()
conf['action_space'] = env.get_action_space()
agent = import_class(conf['agent'])(conf)

with agent:
    runner = GameRunner(conf, env, agent)
    runner.run()