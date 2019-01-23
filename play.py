import argparse
import importlib
import os
import json

parser = argparse.ArgumentParser()

# parser.add_argument('-a', '--agent', dest='agent', default='rl.game_agent.BreakoutAgent')
# parser.add_argument('-n', '--network', dest='network', default='rl.deep_q_network.DeepQNetwork')
# parser.add_argument('-g', '--game-id', dest='game_id', default='BreakoutDeterministic-v4')
# parser.add_argument('-m', '--model-save-prefix', dest='model_save_prefix', default=None)
parser.add_argument('-O', '--dir', '--save-dir', dest='save_dir', default='./data')
# parser.add_argument('--encoder-save-path', dest='encoder_save_path', default='./data/')

parser.add_argument('-i', '--interval', dest='interval', type=int, default=50)
parser.add_argument('-e', '--use-epsilon', dest='use_epsilon', action='store_true', default=False)
parser.add_argument('--ng', '--num-games', dest='num_games', type=int, default=1)
parser.add_argument('--display', dest='display', action='store_true', default=False)
parser.add_argument('--save', dest='save_video', action='store_true', default=False)


args = parser.parse_args()

for fn in os.listdir(args.save_dir):
    if fn.endswith('.conf'):
        with open(os.path.join(args.save_dir, fn)) as f:
            options = json.loads(f.read())


mod_net_str, cl_net_str = options['network'].rsplit('.', 1)
mod_agent_str, cl_agent_str = options['agent'].rsplit('.', 1)

mod_net = importlib.import_module(mod_net_str)
net_cl = getattr(mod_net, cl_net_str)

print(mod_agent_str, cl_agent_str)
mod_ag = importlib.import_module(mod_agent_str)
ag_cl = getattr(mod_ag, cl_agent_str)


qn = net_cl(options['game_id'],
            ag_cl, 
            options=options)

qn.play(use_epsilon=args.use_epsilon, 
        interval=args.interval, 
        num_games=args.num_games, 
        display=args.display,
        save_video=args.save_video)
