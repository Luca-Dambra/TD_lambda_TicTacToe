# Imports:
import collections
import os
import random
import numpy as np
import tensorflow as tf
import pickle


from NN_base import load_network, save_network, create_network, make_ep_greedy_move, make_opp_greedy_move, sample
from tic_tac_toe import TicTacToeGameSpec, play_game

##### Inputs:
### game_spec
### create_network_func
### network_file_path
### opp_func
### number_of_rounds --> number of rounds of the experiment
### number_of_games --> number of training games in each round
### number_of_test --> number of test games in each round
### epsilon --> exploration rate
### tau --> discount parameter
### lamda --> decay parameter
### alpha_start --> starting learning rate
### decay_rate --> decay rate parameter of the Exponential decay function
### decay_steps --> decay step parameter of the Exponential decay function
##### Output:
### variables --> variables of the trained network
def TD_train(game_spec,
             create_network_func,
             network_file_path,
             opp_func = None,
             number_of_games= 1000,
             number_of_test = 1000,
             number_of_rounds = 100,
             epsilon = 0.1,
             tau = 1,
             lamda = 0.7,
             decay_rate = 0.96,
             decay_steps = 1000,
             alpha_start = 0.01):
    ## Load some parameters:              
    save_network_file_path =  network_file_path
    opp_func = opp_func or game_spec.get_random_player_func()
    
    ## Create the network
    input_layer, output_layer, variables = create_network_func()
    ## Target place holder:
    target_placeholder = tf.placeholder("float", shape=(None,))
    ## Learning rate parameters:
    alpha_placeholder = tf.placeholder("float", shape=(None,)) 
    learn_rate = alpha_start
    global_step = 0
    
    delta_op = tf.reduce_sum(target_placeholder - output_layer, name='delta')
    tvars = tf.trainable_variables()
    grads = tf.gradients(output_layer, tvars)
    
    apply_gradients = []
    reset_trace = []
    results = []
    
    for grad, var in zip(grads, tvars):
        trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False)
        trace_op = trace.assign((tau * lamda * trace) + grad)
        grad_trace = alpha_placeholder * delta_op * trace_op
        grad_apply = var.assign(var + grad_trace)
        apply_gradients.append(grad_apply)
        ##### reset op:
        trace_reset = tf.Variable(tf.zeros(grad.get_shape()), trainable=False)
        trace_reset_op = trace.assign(trace_reset)
        reset_trace.append(trace_reset_op)
    
    # define single operation to apply all gradient updates
    train_op = tf.group(*apply_gradients)
    reset_op = tf.group(*reset_trace)
    ## Start the session:
    with tf.Session() as session:
        ## Initialize the network
        session.run(tf.global_variables_initializer())
        
        ## Load the network if specified
        if network_file_path and os.path.isfile(network_file_path):
            print("loading pre-existing network")
            load_network(session, variables, network_file_path)
        
        ## Plus player function:
        def plus_player_function(board_state,side):
            move, _, _ = make_ep_greedy_move(session, input_layer, output_layer,
                                                board_state, side, game_spec, ep = 0)
            return move
        ## Minus player function:
        def minus_player_function(board_state,side):
            move, _, _ = make_opp_greedy_move(session, input_layer, output_layer,
                                                board_state, side, game_spec, ep = 0)
            return move
        ## Training function:
        def train_game(game_spec, session, input_layer, output_layer, alpha):
            ## Reset the trace:
            session.run(reset_op)
            ## Create an empty board_state
            board_state = game_spec.new_board()
            player_turn = 1  ## The first player start
            X_player_move_count = 0  ## counter of X's moves
            O_player_move_count = 0  ## counter of O's moves
            while True:
                if player_turn > 0:
                    ## Save the previous after state:
                    if X_player_move_count > 0:
                        prev_after_state_X = after_state_X
                    ## Make an epsilon greedy move storing move, value of the after_state and after_state
                    move, value_X,after_state_X = make_ep_greedy_move(session, input_layer, output_layer, board_state, 1, game_spec, ep = epsilon)
                    X_player_move_count += 1  ## Increase the counter
                else:  ## First player tourn:
                    if O_player_move_count > 0:
                        prev_after_state_O = after_state_O
                    ## Make an epsilon greedy move storing move, value of the after_state and after_state
                    move, value_O, after_state_O = make_opp_greedy_move(session, input_layer, output_layer, board_state, -1, game_spec, ep = epsilon)
                    O_player_move_count += 1 ## Increase the counter
                ## Apply the move:
                board_state = game_spec.apply_move(board_state, move, player_turn)
                ## Check if someone has won the game:
                winner = game_spec.has_winner_TD_train(board_state)
                if winner == 0:
                    if player_turn > 0 and X_player_move_count > 1:
                        np_prev_after_state_X = np.ravel(prev_after_state_X)
                        np_prev_after_state_X = np.array(np_prev_after_state_X) \
                                .reshape(1, *input_layer.get_shape().as_list()[1:])
                        target_X = [tau*value_X]
                        session.run(train_op, feed_dict={input_layer: np_prev_after_state_X,
                                                         target_placeholder: target_X,
                                                         alpha_placeholder: [alpha]})
                    if player_turn < 0 and O_player_move_count > 1:
                        np_prev_after_state_O = np.ravel(prev_after_state_O)
                        np_prev_after_state_O = np.array(np_prev_after_state_O) \
                            .reshape(1, *input_layer.get_shape().as_list()[1:])
                        target_O = [tau*value_O]
                        session.run(train_op, feed_dict={input_layer: np_prev_after_state_O,
                                                         target_placeholder: target_O,
                                                         alpha_placeholder: [alpha]})
                if winner == 1:
                    np_prev_after_state_X = np.ravel(after_state_X)
                    np_prev_after_state_X = np.array(np_prev_after_state_X) \
                            .reshape(1, *input_layer.get_shape().as_list()[1:])
                    target_X = [1]
                    session.run(train_op, feed_dict={input_layer: np_prev_after_state_X,
                                                     target_placeholder: target_X,
                                                     alpha_placeholder: [alpha]})
                    np_prev_after_state_O = np.ravel(after_state_O)
                    np_prev_after_state_O = np.array(np_prev_after_state_O) \
                            .reshape(1, *input_layer.get_shape().as_list()[1:])
                    target_O = [1]
                    session.run(train_op, feed_dict={input_layer: np_prev_after_state_O,
                                                     target_placeholder: target_O,
                                                     alpha_placeholder: [alpha]})
                    return winner
                if winner == -1:
                    np_prev_after_state_X = np.ravel(after_state_X)
                    np_prev_after_state_X = np.array(np_prev_after_state_X) \
                                .reshape(1, *input_layer.get_shape().as_list()[1:])
                    target_X = [-1]
                    session.run(train_op, feed_dict={input_layer: np_prev_after_state_X,
                                                     target_placeholder: target_X,
                                                     alpha_placeholder: [alpha]})
                    np_prev_after_state_O = np.ravel(after_state_O)
                    np_prev_after_state_O = np.array(np_prev_after_state_O) \
                                .reshape(1, *input_layer.get_shape().as_list()[1:])
                    target_O = [-1]
                    session.run(train_op, feed_dict={input_layer: np_prev_after_state_O,
                                                     target_placeholder: target_O,
                                                     alpha_placeholder: [alpha]})
                    return winner
                if winner == 20:
                    np_prev_after_state_X = np.ravel(after_state_X)
                    np_prev_after_state_X = np.array(np_prev_after_state_X) \
                                .reshape(1, *input_layer.get_shape().as_list()[1:])
                    target_X = [0]
                    session.run(train_op, feed_dict={input_layer: np_prev_after_state_X,
                                                      target_placeholder: target_X,
                                                      alpha_placeholder: [alpha]})
                    np_prev_after_state_O = np.ravel(after_state_O)
                    np_prev_after_state_O = np.array(np_prev_after_state_O) \
                        .reshape(1, *input_layer.get_shape().as_list()[1:])
                    target_O = [0]
                    session.run(train_op, feed_dict={input_layer: np_prev_after_state_O,
                                                     target_placeholder: target_O,
                                                     alpha_placeholder: [alpha]})
                    return 0
                player_turn = -player_turn
                    
        # wins, losses and draws
        wins = 0
        losses = 0
        draws = 0
        
        ## Un giro per ogni round
        for i in range(number_of_rounds):
            ### TRAIN PHASE:
            ## Run the games:
            for episode_number in range(1, number_of_games):
                global_step += 1
                learn_rate = alpha_start*decay_rate**(global_step/decay_steps)   ## decrease the learning rate
                train_game(game_spec, session, input_layer, output_layer, learn_rate)
            ### TEST PHASE:
            ## Run the games:
            for episode in range(number_of_test):
                starting_player = sample(2,[1/2]*2)
                if starting_player == 1:
                    winner_player = play_game(plus_player_function, opp_func)
                else:
                    winner_player = -play_game(opp_func, minus_player_function)
                if winner_player == 1:
                    wins += +1
                if winner_player == 0:
                    draws += +1
                if winner_player == -1:
                    losses += +1
            ## Print the results:
            print("Round: %s Wins: %s Losses: %s Draws: %s learn_rate: %s" % (i, wins , losses, draws, learn_rate))
            risultato = [wins, losses, draws, learn_rate]
            results.append(risultato)
            ## Clear the vectors:
            wins = 0
            losses = 0
            draws = 0
        ## Save network:
        if network_file_path:
            save_network(session, variables, save_network_file_path)
    return results
