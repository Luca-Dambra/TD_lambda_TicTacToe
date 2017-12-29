# Import:
import operator
import pickle
from functools import reduce
import numpy as np
import tensorflow as tf

from connect4 import Connect4GameSpec, apply_move


game_spec = Connect4GameSpec()


## Create a Neural Network:
    ## Inputs: input_nodes (number of input nodes)
    ##         hidden_nodes (number of hidden nodes in each hidden layer)
    ##         output_nodes (number of output nodes)
    ##         output_softmax (True if softmax is used in the final layer)
    ## Output: input_layer ....
    ##         output_layer ....
    ##         [variables] list containing all the parameters, wieghts and biases of NN 
def create_network(input_nodes, hidden_nodes, output_nodes=None, output_softmax=True):
    
    output_nodes = output_nodes or input_nodes
    variables = []
    with tf.name_scope('network'):
        if isinstance(input_nodes, tuple):
            input_layer = tf.placeholder("float", (None,) + input_nodes)
            flat_size = reduce(operator.mul, input_nodes, 1)
            current_layer = tf.reshape(input_layer, (-1, flat_size))
        else:
            input_layer = tf.placeholder("float", (None, input_nodes))
            current_layer = input_layer

        for hidden_nodes in hidden_nodes:
            last_layer_nodes = int(current_layer.get_shape()[-1])
            hidden_weights = tf.Variable(
                tf.truncated_normal((last_layer_nodes, hidden_nodes), stddev=1. / np.sqrt(last_layer_nodes)),
                name='weights')
            hidden_bias = tf.Variable(tf.constant(0.01, shape=(hidden_nodes,)), name='biases')

            variables.append(hidden_weights)
            variables.append(hidden_bias)

            current_layer = tf.sigmoid(
                tf.matmul(current_layer, hidden_weights) + hidden_bias)

        if isinstance(output_nodes, tuple):
            output_nodes = reduce(operator.mul, input_nodes, 1)

        output_weights = tf.Variable(
            tf.truncated_normal((hidden_nodes, output_nodes), stddev=1. / np.sqrt(output_nodes)), name="output_weights")
        output_bias = tf.Variable(tf.constant(0.01, shape=(output_nodes,)), name="output_bias")

        variables.append(output_weights)
        variables.append(output_bias)

        output_layer = tf.matmul(current_layer, output_weights) + output_bias
        if output_softmax:
            output_layer = tf.nn.softmax(output_layer)
    return input_layer, output_layer, variables



## Save the given set of variables
    ## Inputs: session
    ##         tf.variables: list of vatiables which will be saved to the file
    ##         file_path: path of the file we want to save
def save_network(session, tf_variables, file_path):
    variable_values = session.run(tf_variables)
    with open(file_path, mode='wb') as f:
        pickle.dump(variable_values, f)



## Load the given set of variables:
    ## Inputs: session
    ##         tf_variables: list of variables that will be load from the filter
    ##         file_path: path of the file we want to load
def load_network(session, tf_variables, file_path):
    with open(file_path, mode='rb') as f:
        variable_values = pickle.load(f)
    try:
        if len(variable_values) != len(tf_variables):
            raise ValueError("Network in file had different structure, variables in file: %s variables in memeory: %s"
                             % (len(variable_values), len(tf_variables)))
        for value, tf_variable in zip(variable_values, tf_variables):
            session.run(tf_variable.assign(value))
    except ValueError as ex:
        raise ValueError("""Tried to load network file %s with different architecture from the in memory network.
Error was %s
Either delete the network file to train a new network from scratch or change the in memory network to match that dimensions of the one in the file""" % (file_path, ex))


## Sample an uniform between 0 and N with the given probability

def sample(N ,p):
    aus = np.random.uniform()
    cumulative = []
    out = 0
    for d in range(N):
        cumulative.append(sum(p[0:d+1]))
        if cumulative[d] > aus:
            out = d
            break
    return out



## Choose an epsylon greedy move for the player 1
    ## Inputs: session
    ##         input_layer: tf.Placeholder to the network used to feed in the state
    ##         output_layer: tf.Tensor that will output the probability of moves
    ##         state: current state,
    ##         game_spec
    ##         side: The side that is making the move
    ##         ep: exploration rate epsylon
    ## Output: move, value of the after_state, after_state
def make_ep_greedy_move(session, input_layer, output_layer, state,
                                side, game_spec= None, ep = 0.1):
    available = list(game_spec.available_moves(state))
    values = []
    prob = [0,0,0,0,0,0,0]
    moves = [0,1,2,3,4,5,6]
    for x in range(7):
        if x in available:
            np_new_state = np.array(game_spec.apply_move(state, x, side))
            new_state_input = np_new_state.reshape(1, *input_layer.get_shape().as_list()[1:])
            aus = float(session.run(output_layer, feed_dict={input_layer: new_state_input})[0])
            values.append(aus)
        else:
            values.append(-10000)
    for y in range(7):
        if y in available:
            if values[y] == max(values):
                prob[y] = 1 - ep + ep/len(available)
            else:
                prob[y] = ep/len(available)
            
    aus1 = sample(7,prob)
    move = moves[aus1]
    after_state = apply_move(state, move, side)
    out = [move,values[aus1],after_state]
    return out

## Choose an epsylon greedy move for the player 2 
    ## Inputs: session
    ##         input_layer: tf.Placeholder to the network used to feed in the state
    ##         output_layer: tf.Tensor that will output the probability of moves
    ##         state: current state,
    ##         game_spec
    ##         side: The side that is making the move
    ##         ep: exploration rate epsylon
    ## Output: move, value of the after_state, after_state
def make_opp_greedy_move(session, input_layer, output_layer, state,
                                side, game_spec= None, ep = 0.1):
    available = list(game_spec.available_moves(state))
    values = []
    prob = [0,0,0,0,0,0,0]
    moves = [0,1,2,3,4,5,6]
    for q in range(7):
        if q in available:
            np_new_state = np.array(game_spec.apply_move(state, q, side))
            new_state_input = np_new_state.reshape(1, *input_layer.get_shape().as_list()[1:])
            aus = float(session.run(output_layer, feed_dict={input_layer: new_state_input})[0])
            values.append(aus)
        else:
            values.append(+10000)
    for w in range(7):
        if w in available:
            if values[w] == min(values):
                prob[w] = 1 - ep + ep/len(available)
            else:
                prob[w] = ep/len(available)
    aus1 = sample(7,prob)
    move = moves[aus1]
    after_state = apply_move(state, move, side)
    out = [move,values[aus1],after_state]
    return out
