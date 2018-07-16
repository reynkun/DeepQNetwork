# from train_game import GameTrainer
# from game_player import GamePlayer

# gt = GameTrainer('MsPacman-v0', GamePlayer)

# gt.play_func()
# 
# 
# from deep_q_network2 import DeepQNetwork
# from game_player import GamePlayer2

# qn = DeepQNetwork('MsPacman-v0', GamePlayer2)

# qn.play()
# 
# 
from deep_q_network2 import DeepQNetwork
from ddqn_game_player import GamePlayer

qn = DeepQNetwork('MsPacman-v0', GamePlayer)

qn.play()