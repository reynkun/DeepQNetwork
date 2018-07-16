import argparse
import importlib

parser = argparse.ArgumentParser()

parser.add_argument('network')
parser.add_argument('agent')

args = parser.parse_args()


net_mod, net_cl_str = args.network.split('.')
# net_cl = __import__(args.network)

# net_cl = globals()[net_cl_str]

ag_mod, ag_cl_str = args.agent.split('.')

# ag_cl = __import__(args.agent)

# ag_cl = globals()[ag_cl_str]

mod = importlib.import_module(net_mod)
net_cl = getattr(mod, net_cl_str)

mod = importlib.import_module(ag_mod)
ag_cl = getattr(mod, ag_cl_str)

qn = net_cl('MsPacman-v0', ag_cl)

qn.play()