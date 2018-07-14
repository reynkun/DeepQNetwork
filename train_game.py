from deep_q_network import DeepQNetwork
from game_player import GamePlayer

qn = DeepQNetwork('MsPacman-v0', GamePlayer)

qn.train()
