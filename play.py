'''

Play game(s) using the deep q network and optionally
displays and saves the video


To play using the deep q network stored in ./data/data-breakout/:


```
python play.py -O ./data/data-breakout/
```

To play and display game:

```
python play.py -O ./data/data-breakout/ --display
```

To play and save video:

```
python play.py -O ./data/data-breakout/ --save
```

To play using random agent:

```
python play.py -a rl.agent.game_agent.RandomAgent --env rl.game_environment.BreakoutEnvironment
```

'''


import argparse


from rl.utils.import_class import import_class
from rl.utils.config import get_conf
from rl.game_runner import GameRunner

def main():
    '''
    Main function to play game
    '''
    parser = argparse.ArgumentParser()

    # set agent and game environment
    parser.add_argument('-a',
                        '--agent',
                        dest='agent',
                        default='rl.deep_q_agent.DeepQAgent',
                        help='game agent to run')
    parser.add_argument('--env',
                        '--environment',
                        dest='environment',
                        default='rl.game_environment.BreakoutEnvironment',
                        help='game environment to run')

    # read config file from target save dir
    parser.add_argument('-O',
                        '--dir',
                        '--save-dir',
                        dest='save_dir',
                        default='./data',
                        help='read config and saved network from save dir')

    # play options
    parser.add_argument('-i',
                        '--interval',
                        dest='interval',
                        type=int,
                        default=60,
                        help='frame rate for video / display')
    parser.add_argument('-e',
                        '--use-epsilon',
                        dest='use_epsilon',
                        action='store_true',
                        default=False,
                        help='use current epsilon setting from network')
    parser.add_argument('--ng',
                        '--num-games',
                        dest='num_games',
                        type=int,
                        default=1,
                        help='number of games to play')

    # render game to x server
    parser.add_argument('--display',
                        dest='display',
                        action='store_true',
                        default=False,
                        help='render game and display with x server')

    # save game to mp4
    parser.add_argument('--save',
                        dest='save_video',
                        action='store_true',
                        default=False,
                        help='save video to mp4 into save dir')

    args = parser.parse_args()

    conf = {
        'is_training': False
    }

    conf.update(vars(args))

    
    if 'save_dir' in conf:
        # pull environment from save dir if specified
        temp_conf = get_conf(conf['save_dir'])

        if temp_conf is not None and 'environment' in temp_conf:
            conf['environment'] = temp_conf['environment']

    env = import_class(conf['environment'])()
    conf['action_space'] = env.get_action_space()
    agent = import_class(conf['agent'])(conf)

    with agent:
        runner = GameRunner(conf, env, agent)
        runner.run()


if __name__ == '__main__':
    main()
