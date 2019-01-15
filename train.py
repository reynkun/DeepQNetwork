import argparse
import importlib

parser = argparse.ArgumentParser()

parser.add_argument('-a', '--agent', dest='agent', default='rl.game_agent.BreakoutAgent')
parser.add_argument('-n', '--network', dest='network', default='rl.deep_q_network.DeepQNetwork')
parser.add_argument('-g', '--game-id', dest='game_id', default='BreakoutDeterministic-v4')
parser.add_argument('-m', '--model-save-prefix', dest='model_save_prefix', default=None)

parser.add_argument('--no-double', dest='use_double', action='store_false')
parser.add_argument('--nd', '--no-dueling', dest='use_dueling', action='store_false')
parser.add_argument('--p', '--priority', dest='use_priority', default=False, action='store_true')
parser.add_argument('--dir', '--save-dir', dest='save_dir', default='./data')
parser.add_argument('--train-backward', '--ntb', dest='train_backward', action='store_false')
parser.add_argument('--frame-skip', '--fs', dest='frame_skip', type=int)

parser.add_argument('--max-train', dest='max_num_training_steps', type=int)
parser.add_argument('--eps', '--eps-steps', dest='eps_decay_steps', type=int)
parser.add_argument('--mss', '--mem-save-size', dest='mem_save_size', default=10000, type=int)

args = parser.parse_args()


mod_net_str, cl_net_str = args.network.rsplit('.', 1)
mod_agent_str, cl_agent_str = args.agent.rsplit('.', 1)

mod_net = importlib.import_module(mod_net_str)
net_cl = getattr(mod_net, cl_net_str)

# print(mod_agent_str, cl_agent_str)
mod_ag = importlib.import_module(mod_agent_str)
ag_cl = getattr(mod_ag, cl_agent_str)

print('args.game_id', args.game_id)
print('args.save_prefix', args.model_save_prefix)

options = {}
options.update(vars(args))

qn = net_cl(args.game_id,
            ag_cl,
            options=options)

qn.train()
