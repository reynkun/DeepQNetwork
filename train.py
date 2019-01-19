import argparse
import importlib

parser = argparse.ArgumentParser()

parser.add_argument('-a', '--agent', dest='agent', default='rl.game_agent.BreakoutAgent')
parser.add_argument('-n', '--network', dest='network', default='rl.deep_q_network.DeepQNetwork')
parser.add_argument('-g', '--game-id', dest='game_id', default='BreakoutDeterministic-v4')
parser.add_argument('-m', '--model-save-prefix', dest='model_save_prefix', default=None)
parser.add_argument('-O', '--dir', '--save-dir', dest='save_dir', default='./data')

# arch options
parser.add_argument('--double', dest='use_double', action='store_true')
parser.add_argument('--dueling', '--dueling', dest='use_dueling', action='store_true')
parser.add_argument('--priority', dest='use_priority', action='store_true')
parser.add_argument('--train-backward', '--ntb', dest='train_backward', action='store_false')
parser.add_argument('--frame-skip', '--fs', dest='frame_skip', type=int)

# train options
parser.add_argument('--max-train', dest='max_num_training_steps', type=int)
parser.add_argument('--eps', '--eps-steps', dest='eps_decay_steps', type=int)
parser.add_argument('--mss', '--mem-save-size', dest='mem_save_size', type=int)
parser.add_argument('--fbt', '--frames-before-training', dest='num_game_frames_before_training', type=int)
parser.add_argument('--rs', '--replay-size', dest='replay_max_memory_length', type=int)
parser.add_argument('--use-memory', dest='use_memory', action='store_true')

args = parser.parse_args()


mod_net_str, cl_net_str = args.network.rsplit('.', 1)
mod_agent_str, cl_agent_str = args.agent.rsplit('.', 1)

mod_net = importlib.import_module(mod_net_str)
net_cl = getattr(mod_net, cl_net_str)

# print(mod_agent_str, cl_agent_str)
mod_ag = importlib.import_module(mod_agent_str)
ag_cl = getattr(mod_ag, cl_agent_str)

options = {}
options.update(vars(args))

qn = net_cl(args.game_id,
            ag_cl,
            options=options)

qn.train()
