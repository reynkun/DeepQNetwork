import argparse


from rl.deep_q_network import DeepQNetwork


parser = argparse.ArgumentParser()

parser.add_argument('-O', '--dir', '--save-dir', dest='save_dir', default='./data')

parser.add_argument('-i', '--interval', dest='interval', type=int, default=50)
parser.add_argument('-e', '--use-epsilon', dest='use_epsilon', action='store_true', default=False)
parser.add_argument('--ng', '--num-games', dest='num_games', type=int, default=1)
parser.add_argument('--display', dest='display', action='store_true', default=False)
parser.add_argument('--save', dest='save_video', action='store_true', default=False)
parser.add_argument('--init', dest='initialize', action='store_true', default=False)

args = parser.parse_args()

options = {}
options.update(vars(args))

qn = DeepQNetwork(options, initialize=False)

qn.play(use_epsilon=args.use_epsilon, 
        interval=args.interval, 
        num_games=args.num_games, 
        display=args.display,
        save_video=args.save_video)
