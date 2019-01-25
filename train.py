import argparse

from rl.deep_q_network import DeepQNetwork


parser = argparse.ArgumentParser()

parser.add_argument('-a', '--agent', dest='agent', default='rl.game_agent.BreakoutAgent')
parser.add_argument('-g', '--game-id', dest='game_id', default='BreakoutDeterministic-v4')
parser.add_argument('-m', '--model-save-prefix', dest='model_save_prefix', default=None)
parser.add_argument('-O', '--dir', '--save-dir', dest='save_dir', default='./data')

# network options
parser.add_argument('--double', dest='use_double', action='store_true')
parser.add_argument('--dueling', '--dueling', dest='use_dueling', action='store_true')
parser.add_argument('--priority', dest='use_priority', action='store_true')

# train options
parser.add_argument('--max-train', dest='max_num_training_steps', type=int)
parser.add_argument('--eps', '--eps-steps', dest='eps_decay_steps', type=int)
parser.add_argument('--mss', '--mem-save-size', dest='mem_save_size', type=int)
parser.add_argument('--svs', '--save-video-steps', dest='num_train_steps_save_video', type=int)
parser.add_argument('--fbt', '--frames-before-training', dest='num_game_frames_before_training', type=int)
parser.add_argument('--fs', '--frame-skip', dest='frame_skip', type=int)
parser.add_argument('--rs', '--replay-size', dest='replay_max_memory_length', type=int)
parser.add_argument('--use-memory', dest='use_memory', action='store_true')


args = parser.parse_args()

options = {}
options.update(vars(args))

qn = DeepQNetwork(options, initialize=True)

qn.train()
