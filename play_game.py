import argparse
import importlib

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--network', dest='network', default='deep_q_network.DeepQNetwork')
parser.add_argument('-a', '--agent', dest='agent', default='cart_pole_agent.CartPoleAgent')
parser.add_argument('-g', '--game-id', dest='game_id', default='CartPole-v1')
parser.add_argument('-m', '--model-save-prefix', dest='model_save_prefix', default=None)
parser.add_argument('-i', '--interval', dest='interval', type=int, default=30)
parser.add_argument('-e', '--use-epsilon', dest='use_epsilon', action='store_true', default=False)
parser.add_argument('--ng', '--num-games', dest='num_games', type=int, default=1)
parser.add_argument('--display', dest='no_display', action='store_false', default=True)

parser.add_argument('--no-double', dest='use_double', action='store_false', default=True)
parser.add_argument('--nd', '--no-dueling', dest='use_dueling', action='store_false', default=True)
parser.add_argument('--np', '--no-priority', dest='use_priority', action='store_false', default=True)
parser.add_argument('--dir', '--save-dir', dest='save_dir', default='./models')

parser.add_argument('--max-train', dest='max_num_training_steps', type=int, default=10000000)
parser.add_argument('--eps-steps', dest='eps_decay_steps', type=int, default=2000000)

args = parser.parse_args()


net_mod, net_cl_str = args.network.split('.')

ag_mod, ag_cl_str = args.agent.split('.')

mod_net = importlib.import_module(net_mod)
net_cl = getattr(mod_net, net_cl_str)

mod_ag = importlib.import_module(ag_mod)
ag_cl = getattr(mod_ag, ag_cl_str)

print('args.game_id', args.game_id)
print('args.save_prefix', args.model_save_prefix)

options = mod_net.DEFAULT_OPTIONS
options.update(vars(args))


qn = net_cl(args.game_id, 
            ag_cl, 
            model_save_prefix=args.model_save_prefix, 
            options=options)

qn.play(use_epsilon=args.use_epsilon, 
        interval=args.interval, 
        num_games=args.num_games, 
        no_display=args.no_display)