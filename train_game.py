import argparse
import importlib

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--network', dest='network', default='deep_q_network_multi.DeepQNetwork')
parser.add_argument('-a', '--agent', dest='agent', default='game_agent.GameAgent')
parser.add_argument('-g', '--game-id', dest='game_id', default='MsPacman-v0')

args = parser.parse_args()


net_mod, net_cl_str = args.network.split('.')

ag_mod, ag_cl_str = args.agent.split('.')

mod = importlib.import_module(net_mod)
net_cl = getattr(mod, net_cl_str)

mod = importlib.import_module(ag_mod)
ag_cl = getattr(mod, ag_cl_str)

print('args.game_id', args.game_id)
qn = net_cl(args.game_id, ag_cl)

qn.train()