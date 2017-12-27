import itertools
import random
import sys

from base_game import BaseGameSpec

### Create a 3x3 grid of zeros
def _new_board():
    return ((0, 0, 0),
            (0, 0, 0),
            (0, 0, 0))

##### Apply a move to the current board:
##### Inputs:
### board_state: 3x3 grid
### move: an integer between 0 and 8 representing the selected move
### side: 1 if the move is made by the X player, -1 id the move is made by the O player
##### Outputs:
### 3x3 grid
def apply_move(board_state, move, side):
    move_x, move_y = move

    def get_tuples():
        for x in range(3):
            if move_x == x:
                temp = list(board_state[x])
                temp[move_y] = side
                yield tuple(temp)
            else:
                yield board_state[x]
    return tuple(get_tuples())

### Return all the available moves from a given board_state
def available_moves(board_state):
    for x, y in itertools.product(range(3), range(3)):
        if board_state[x][y] == 0:
            yield (x, y)


def _has_3_in_a_line(line):
    return all(x == -1 for x in line) | all(x == 1 for x in line)

### Check if someone has won:
def has_winner(board_state):
    # check rows
    for x in range(3):
        if _has_3_in_a_line(board_state[x]):
            return board_state[x][0]
    # check columns
    for y in range(3):
        if _has_3_in_a_line([i[y] for i in board_state]):
            return board_state[0][y]

    # check diagonals
    if _has_3_in_a_line([board_state[i][i] for i in range(3)]):
        return board_state[0][0]
    if _has_3_in_a_line([board_state[2 - i][i] for i in range(3)]):
        return board_state[0][2]

    return 0  # no one has won, return 0 for a draw


## Check if someone has won (return 1 if the X player has won, -1 if the O player has won, 20 if the game ends in a TIE and 0 if the game has not
## ended yet)
def has_winner_TD_train(board_state):
    # check rows
    for x in range(3):
        if _has_3_in_a_line(board_state[x]):
            return board_state[x][0]
    # check columns
    for y in range(3):
        if _has_3_in_a_line([i[y] for i in board_state]):
            return board_state[0][y]
    # check diagonals
    if _has_3_in_a_line([board_state[i][i] for i in range(3)]):
        return board_state[0][0]
    if _has_3_in_a_line([board_state[2 - i][i] for i in range(3)]):
        return board_state[0][2]
    # check tie:
    if len(list(available_moves(board_state))) == 0:
        return 20
    return 0

### Play a game selecting the moves according to plus_player_func and minus_player_func. Return 1 if the X player wins, -1 if the O player wins
### and 0 if the game ends in a draw.
def play_game(plus_player_func, minus_player_func, log=False):
    board_state = _new_board()
    player_turn = 1
    while True:
        _available_moves = list(available_moves(board_state))
        if len(_available_moves) == 0:
            # draw
            if log:
                print("no moves left, game ended a draw")
            return 0.
        if player_turn > 0:
            move = plus_player_func(board_state, 1)
        else:
            move = minus_player_func(board_state, -1)
        if move not in _available_moves:
            # if a player makes an invalid move the other player wins
            if log:
                print("illegal move ", move)
            return -player_turn

        board_state = apply_move(board_state, move, player_turn)
        if log:
            print(board_state)

        winner = has_winner(board_state)
        if winner != 0:
            if log:
                print("we have a winner, side: %s" % player_turn)
            return winner
        player_turn = -player_turn

## Random player function (selects a move randomly from the available ones)
def random_player(board_state, _):
    moves = list(available_moves(board_state))
    return random.choice(moves)


class TicTacToeGameSpec(BaseGameSpec):
    def __init__(self):
        self.available_moves = available_moves
        self.has_winner = has_winner
        self.has_winner_TD_train = has_winner_TD_train
        self.new_board = _new_board
        self.apply_move = apply_move

    def board_dimensions(self):
        return 3, 3

if __name__ == '__main__':
    # example of playing a game
    play_game(random_player, random_player, log=True)

