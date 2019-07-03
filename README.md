# Deep Q Reinforcement Learning


Implements a deep q reinforcement learning as specified by [Deep Mind's 2015 paper](https://deepmind.com/research/dqn/).  Using the raw pixels and score, the system is able to learn how to play a simple atari games.

Includes double dueling dqn and also prioritized experience replay improvements.

Implemented in python 3.6 and tensorflow 1.12.

Sample Videos:

- [Breakout - 2 million steps](https://www.youtube.com/watch?v=K1WTUuAyDY8)
- [MsPacman - 10 million steps](https://www.youtube.com/watch?v=xMP6TSwPmPE)

Written by Reyn Nakamoto

## System Requirements

While it should run on any computer, a tensorflow supported GPU will greatly speed up the training process by orders.  Additionally, at large amount of memory helps a bit (>24GB) when storing replay memory on memory.  

## Installation

To install necessary python packages:

```
pipenv install
```


## Training

Training can take a long time.  For reasonable performance, it may take at least half a day and 1m steps of training.  

For optimal performance include the `--dueling`, `--double`, and `--per` options.  

### Running training

To running a deep q network with all the enhancements on for the Breakout game and save data to the data/data-breakout directory, use the following command:

```
pipenv run python train.py --double --dueling --per --modelenv=Breakout -O data/data-breakout
```

Training may be stopped at anytime using ctrl-c and can be restarted by the running the same command.  As long as the correct data directory is specified with `-O` option, the program will read the options from the config file and continue training with the existing model.

You may also override certain options on subsequent training runs by specifying different options.  For example, options like `--max-train` or `--eps` can be changed on the subsequent training runs.  Network architecture options cannot be changed and will result in an error.  Options like `--double` and `--dueling` will cause an error.

### Example training options:

- To run with BreakoutAgent and BreakoutEnvironment and
save to data/data-breakout:

```
pipenv run python train.py --modelenv=Breakout -O data/data-breakout
```
- To run with less or more steps (Breakout shows ok performance around 1000000)

```
pipenv run python train.py --max-train=10000000 --modelenv=Breakout -O data/data-breakout
```
- Add double, dueling, priority enhancements:

```
pipenv run python train.py --double --dueling --per --modelenv=Breakout -O data/data-breakout
```

### Available Environments

Currently, three environments are ready to go with this package:

- Breakout (BreakoutEnvironment)
- Space Invaders (SpaceInvaders)
- Ms Pacman (MsPacman)

However, you may add new environments by extending GameEnvironment as in `rl/game_environment.py`.  You must also provide a matching model as in `rl/deep_q_model.py`.  The input height and width must match.


## Playing

Once you've trained a model, you may now play the game using the deep q agent.  Specify the `-O save_dir` 
to point to the path of the saved model.

```
pipenv run python play.py -O ./data/data-breakout/
```

### Example options

- To play and display game:

```
pipenv run python play.py -O ./data/data-breakout/ --display
```

- To play and save video:

```
pipenv run python play.py -O ./data/data-breakout/ --save
```

- To play using random agent:

```
pipenv python play.py -a rl.agent.game_agent.RandomAgent --env rl.game_environment.BreakoutEnvironment
```

