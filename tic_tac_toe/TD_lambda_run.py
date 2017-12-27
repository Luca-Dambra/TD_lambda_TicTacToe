import functools
import numpy as np
import pandas as pd


from NN_base import load_network, save_network, create_network
from tic_tac_toe import TicTacToeGameSpec, play_game
from TD_lambda import TD_train

NETWORK_FILE_PATH = None
NUMBER_OF_GAMES_TO_RUN = 500
NUMBER_OF_TEST = 200
NUMBER_OF_ROUNDS = 100
EPSILON = 0.1
TAU = 0.8
LAMBDA = 0.3
DECAY_RATE = 0.95
DECAY_STEP = 1000
ALPHA = 0.04 ## starting learning rate

game_spec = TicTacToeGameSpec()
create_network_func = functools.partial(create_network, input_nodes=9, hidden_nodes=(20,30), output_nodes=1, output_softmax=False)

results = TD_train(game_spec,
          create_network_func,
          network_file_path = None,
          opp_func = None,
          number_of_games = NUMBER_OF_GAMES_TO_RUN,
          number_of_test = NUMBER_OF_TEST,
          number_of_rounds = NUMBER_OF_ROUNDS,
          epsilon = EPSILON,
          tau = TAU,
          lamda = LAMBDA,
          decay_rate = DECAY_RATE,
          decay_steps = DECAY_STEP,
          alpha_start = ALPHA)
