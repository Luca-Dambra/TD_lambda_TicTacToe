import random

from base_game import BaseGameSpec

## Create a new board:
    ## Input: board_width 
    ##        board_height
    ## Output: board_width x board_height grid of zeros
    
def _new_board(board_width = 7, board_height = 6):
    return tuple(tuple(0 for _ in range(board_height)) for _ in range(board_width))

## Apply the move:
    ## Inputs: state (tuple)
    ##         move_x (column of the move)
    ##         side (player who is playing the move)
    ## Output: new state (tuple)
def apply_move(board_state, move_x, side):
    move_y = 0
    for e in board_state[move_x]:
        if e == 0:
            break
        else:
            move_y += 1
    def get_tuples():
        for r in range(len(board_state)):
            if move_x == r:
                temp = list(board_state[r])
                temp[move_y] = side
                yield tuple(temp)
            else:
                yield board_state[r]
    return tuple(get_tuples())


## Get all legal moves for the current state
    ## Input: state (tuple)
    ## Output: all valid move (generator of int)

def available_moves(board_state):
    for t in range(len(board_state)):
        if any(y == 0 for y in board_state[t]):
            yield t


## Choose a move randomly from the available ones:
    ## Input: state 
    ##        _ (side of the player who is playing the move)
    ## Output: move
def random_player(board_state, _):
    moves = list(available_moves(board_state))
    return random.choice(moves)


# Useful function:
def _has_winning_line(line, winning_length):
    count = 0
    last_side = 0
    for x in line:
        if x == last_side:
            count += 1
            if count == winning_length:
                return last_side
        else:
            count = 1
            last_side = x
    return 0

## Determine if a player has won
    ## Input: state 
    ##        winning length
    ## Output: 1 if plus player won, -1 if minus player won, 0 if no winning position
def has_winner(board_state, winning_length=4):
    board_width = len(board_state)
    board_height = len(board_state[0])
    # check rows
    for u in range(board_width):
        winner = _has_winning_line(board_state[u], winning_length)
        if winner != 0:
            return winner
    # check columns
    for o in range(board_height):
        winner = _has_winning_line((i[o] for i in board_state), winning_length)
        if winner != 0:
            return winner

    # check diagonals
    diagonals_start = -(board_width - winning_length)
    diagonals_end = (board_width - winning_length)
    for a in range(diagonals_start, diagonals_end+1):
        winner = _has_winning_line(
            (board_state[j][j + a] for j in range(max(-a, 0), min(board_width, board_height - a))),
            winning_length)
        if winner != 0:
            return winner
    for s in range(diagonals_start, diagonals_end+1):
        winner = _has_winning_line(
            (board_state[k][board_height - k - s - 1] for k in range(max(-s, 0), min(board_width, board_height - s))),
            winning_length)
        if winner != 0:
            return winner
    return 0  # no one has won, return 0 for a draw

## Determine if a player has won
    ## Input: state 
    ##        winning length
    ## Output: 1 if plus player won, -1 if minus player won, 0 if no winning position and 20 if the game has ended in a draw
def has_winner_TD_train(board_state, winning_length=4):
    board_width = len(board_state)
    board_height = len(board_state[0])
    # check rows
    for u in range(board_width):
        winner = _has_winning_line(board_state[u], winning_length)
        if winner != 0:
            return winner
    # check columns
    for o in range(board_height):
        winner = _has_winning_line((i[o] for i in board_state), winning_length)
        if winner != 0:
            return winner

    # check diagonals
    diagonals_start = -(board_width - winning_length)
    diagonals_end = (board_width - winning_length)
    for a in range(diagonals_start, diagonals_end+1):
        winner = _has_winning_line(
            (board_state[j][j + a] for j in range(max(-a, 0), min(board_width, board_height - a))),
            winning_length)
        if winner != 0:
            return winner
    for s in range(diagonals_start, diagonals_end+1):
        winner = _has_winning_line(
            (board_state[k][board_height - k - s - 1] for k in range(max(-s, 0), min(board_width, board_height - s))),
            winning_length)
        if winner != 0:
            return winner
    if len(list(available_moves(board_state))) == 0:
        return 20
    return 0  # no one has won, return 0 for a draw


## Play a single game
    ## Input: plus player func
    ##        minus player func
    ##        board_width
    ##        board_height
    ##        winning_length
    ##        log
    ## Output: 1 if plus player win, -1 if minus player win, 0 if tie
def play_game(plus_player_func, minus_player_func, board_width=7, board_height=6, winning_length=4, log=False):
    board_state = _new_board(board_width, board_height)
    player_turn = 1

    while True:
        _avialable_moves = list(available_moves(board_state))
        if len(_avialable_moves) == 0:
            # draw
            if log:
                print("no moves left, game ended a draw")
            return 0.
        if player_turn > 0:
            move = plus_player_func(board_state, 1)
        else:
            move = minus_player_func(board_state, -1)

        if move not in _avialable_moves:
            # if a player makes an invalid move the other player wins
            if log:
                print("illegal move ", move)
            return -player_turn

        board_state = apply_move(board_state, move, player_turn)
        if log:
            print(board_state)

        winner = has_winner(board_state, winning_length)
        if winner != 0:
            if log:
                print("we have a winner, side: %s" % player_turn)
            return winner
        player_turn = -player_turn



class Connect4GameSpec(BaseGameSpec):
    def __init__(self, board_width=7, board_height=6, winning_length=4):
        self._board_height = board_height
        self._board_width = board_width
        self._winning_length = winning_length
        self.available_moves = available_moves
        self.apply_move = apply_move

    def new_board(self):
        return _new_board(self._board_width, self._board_height)

    def has_winner(self, board_state):
        return has_winner(board_state, self._winning_length)

    def has_winner_TD_train(self, board_state):
        return has_winner_TD_train(board_state, self._winning_length)
    
    def board_dimensions(self):
        return self._board_width, self._board_height

    def outputs(self):
        return self._board_width


if __name__ == '__main__':
    # example of playing a game
    play_game(random_player, random_player, log=True, board_width=7, board_height=6, winning_length=4)


