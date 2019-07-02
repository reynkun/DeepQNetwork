'''
Trains the deep q network

To run with BreakoutAgent and BreakoutEnvironment and
save to data/data-breakout:

```
pipenv run python train.py --modelenv=Breakout -O data/data-breakout
```
To run with less or more steps (Breakout shows ok performance around 1000000)

```
pipenv run python train.py --max-train=2000000 --modelenv=Breakout -O data/data-breakout
```
Add double, dueling, priority enhancements:

```
pipenv run python train.py --double --dueling --per --modelenv=Breakout -O data/data-breakout
```

'''

import argparse

from rl.utils.import_class import import_class
from rl.game_runner import GameRunner


def main():
    '''
    Main function to train agent
    '''

    parser = argparse.ArgumentParser()

    # game agent / environment
    parser.add_argument('-a'
                        '--agent',
                        dest='agent',
                        default='rl.deep_q_agent.DeepQAgent')
    parser.add_argument('-m',
                        '--model',
                        dest='model',
                        default='rl.deep_q_model.BreakoutModel')
    parser.add_argument('-e',
                        '--env',
                        dest='environment',
                        default='rl.game_environment.BreakoutEnvironment')
    parser.add_argument('--me',
                        '--modelenv',
                        dest='model_environment',
                        default=None)

    # save options
    parser.add_argument('--model-save-prefix',
                        dest='model_save_prefix',
                        default=None)
    parser.add_argument('-O',
                        '--dir',
                        '--save-dir',
                        dest='save_dir',
                        default='./data')

    # network options
    parser.add_argument('--double',
                        '--use-double',
                        dest='use_double',
                        action='store_true',
                        help='use double network')
    parser.add_argument('--dueling',
                        '--use-dueling',
                        dest='use_dueling',
                        action='store_true',
                        help='use dueling network')

    # priority experience replay settings
    parser.add_argument('--per',
                        '--use-per',
                        dest='use_per',
                        action='store_true',
                        help='use priority experience replay')
    parser.add_argument('--per-anneal',
                        '--use-per-anneal',
                        dest='use_per_annealing',
                        action='store_true',
                        help='use per annealing')
    parser.add_argument('--per-b-start',
                        dest='per_b_start',
                        type=float)
    parser.add_argument('--per-b-end',
                        dest='per_b_end',
                        type=float)
    parser.add_argument('--per-anneal-steps',
                        dest='per_anneal_steps',
                        type=int)

    # train options
    parser.add_argument('--max-train',
                        dest='max_num_training_steps',
                        type=int)
    parser.add_argument('--eps', '--eps-steps',
                        dest='eps_decay_steps',
                        type=int)
    parser.add_argument('--mss', '--mem-save-size',
                        dest='mem_save_size',
                        type=int)
    parser.add_argument('--sms',
                        '--save-model-steps',
                        dest='save_model_steps',
                        type=int)
    parser.add_argument('--cns',
                        '--copy-network-steps',
                        dest='copy_network_steps',
                        type=int)

    parser.add_argument('--fbt',
                        '--frames-before-training',
                        dest='num_game_frames_before_training',
                        type=int)
    parser.add_argument('--fs',
                        '--frame-skip',
                        dest='frame_skip',
                        type=int)
    parser.add_argument('--rs',
                        '--replay-size',
                        dest='replay_max_memory_length',
                        type=int)
    parser.add_argument('--bs',
                        '--batch-size',
                        dest='batch_size',
                        type=int)
    parser.add_argument('--memory',
                        '--use-memory',
                        dest='use_memory',
                        action='store_true')
    parser.add_argument('--disk', '--use-disk',
                        dest='use_memory',
                        action='store_false')

    args = parser.parse_args()

    conf = {
        'is_training': True
    }

    conf.update(vars(args))

    if args.model_environment is not None:
        # override agent and environment with model_environment option
        conf['model'] = 'rl.deep_q_model.{}Model'.format(
            args.model_environment)
        conf['environment'] = 'rl.game_environment.{}Environment'.format(
            args.model_environment)

    env = import_class(conf['environment'])()
    conf['action_space'] = env.get_action_space()
    agent = import_class(conf['agent'])(conf)

    with agent:
        runner = GameRunner(conf, env, agent)
        runner.run()


if __name__ == '__main__':
    main()
