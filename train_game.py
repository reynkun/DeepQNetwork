from deep_q_network import DeepQNetwork
from game_player import GamePlayer2


qn = DeepQNetwork('MsPacman-v0', GamePlayer2)

qn.fit_func()
