#
# Trains the deep q network
#

import argparse

from rl.deep_q_network import DeepQNetwork


parser = argparse.ArgumentParser()

# game agent
parser.add_argument('-a', '--agent', dest='agent', default='rl.game_agent.BreakoutAgent')
# game environment
parser.add_argument('-e', '--env', dest='environment', default='rl.game_environment.BreakoutEnvironment')
parser.add_argument('-m', '--model-save-prefix', dest='model_save_prefix', default=None)
parser.add_argument('-O', '--dir', '--save-dir', dest='save_dir', default='./data')

# network options
parser.add_argument('--double', '--use-double', dest='use_double', action='store_true', help='use double network')
parser.add_argument('--dueling', '--use-dueling', dest='use_dueling', action='store_true', help='use dueling network')
parser.add_argument('--priority', '--use-priority', dest='use_priority', action='store_true', help='[broken] does not converge correctly')

# train options
parser.add_argument('--max-train', dest='max_num_training_steps', type=int)
parser.add_argument('--eps', '--eps-steps', dest='eps_decay_steps', type=int)
parser.add_argument('--mss', '--mem-save-size', dest='mem_save_size', type=int)
parser.add_argument('--sms', '--save-model-steps', dest='save_model_steps', type=int)
parser.add_argument('--cns', '--copy-network-steps', dest='copy_network_steps', type=int)

parser.add_argument('--svs', '--save-video-steps', dest='num_train_steps_save_video', type=int)
parser.add_argument('--fbt', '--frames-before-training', dest='num_game_frames_before_training', type=int)
parser.add_argument('--fs', '--frame-skip', dest='frame_skip', type=int)
parser.add_argument('--rs', '--replay-size', dest='replay_max_memory_length', type=int)
parser.add_argument('--bs', '--batch-size', dest='batch_size', type=int)
parser.add_argument('--memory', '--use-memory', dest='use_memory', action='store_true')
parser.add_argument('--disk', '--use-disk', dest='use_memory', action='store_false')


args = parser.parse_args()

conf = {}
conf.update(vars(args))

network = DeepQNetwork(conf, initialize=True)

network.train()