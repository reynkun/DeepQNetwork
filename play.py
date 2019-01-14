import argparse
import importlib

parser = argparse.ArgumentParser()

parser.add_argument('-a', '--agent', dest='agent', default='rl.game_agent.BreakoutAgent')
parser.add_argument('-n', '--network', dest='network', default='rl.deep_q_network.DeepQNetwork')
parser.add_argument('-g', '--game-id', dest='game_id', default='BreakoutDeterministic-v4')
parser.add_argument('-m', '--model-save-prefix', dest='model_save_prefix', default=None)

parser.add_argument('-i', '--interval', dest='interval', type=int, default=50)
parser.add_argument('-e', '--use-epsilon', dest='use_epsilon', action='store_true', default=False)
parser.add_argument('--ng', '--num-games', dest='num_games', type=int, default=1)
parser.add_argument('--display', dest='no_display', action='store_false', default=True)
parser.add_argument('--dir', '--save-dir', dest='save_dir', default='./models')


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

qn.play(use_epsilon=args.use_epsilon, 
        interval=args.interval, 
        num_games=args.num_games, 
        no_display=args.no_display)