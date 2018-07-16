# from deep_q_network import DeepQNetwork
# from game_player import GamePlayer

# qn = DeepQNetwork('MsPacman-v0', GamePlayer)

# qn.train()


from deep_q_network2 import DeepQNetwork
from ddqn_game_player import GamePlayer

qn = DeepQNetwork('Breakout-v0', GamePlayer)

qn.train()