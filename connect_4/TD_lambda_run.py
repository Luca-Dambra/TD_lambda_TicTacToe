# Imports:
import functools
import numpy as np

from NN_base import load_network, save_network, create_network
from connect4 import Connect4GameSpec, play_game
from TD_lambda import TD_train
from benchmark import benchmark_player
 
NETWORK_FILE_PATH = None
NUMBER_OF_GAMES_TO_RUN = 5000
NUMBER_OF_TEST = 200
NUMBER_OF_ROUNDS = 200
EPSILON = 0.1
TAU = 0.9
LAMBDA = 0.2
DECAY_RATE = 0.95
DECAY_STEP = 20000
ALPHA = 0.05 ## starting learning rate
 
game_spec = Connect4GameSpec()
create_network_func = functools.partial(create_network, input_nodes=42, hidden_nodes=(300,243), output_nodes=1, output_softmax=False)
 
 
results = TD_train(game_spec,
         create_network_func,
         network_file_path = NETWORK_FILE_PATH,
         opp_func = benchmark_player,
         number_of_games = NUMBER_OF_GAMES_TO_RUN,
         number_of_test = NUMBER_OF_TEST,
         number_of_rounds = NUMBER_OF_ROUNDS,
         epsilon = EPSILON,
         tau = TAU,
         lamda = LAMBDA,
         decay_rate = DECAY_RATE,
         decay_steps = DECAY_STEP,
         alpha_start = ALPHA)