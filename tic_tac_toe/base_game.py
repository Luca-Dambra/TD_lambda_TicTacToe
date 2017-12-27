import operator
import random
from functools import reduce


class BaseGameSpec(object):
    def __init__(self):
        raise NotImplementedError('This is an abstract base class')

    def new_board(self):
        raise NotImplementedError()

    def apply_move(self, board_state, move, side):
        raise NotImplementedError()

    def available_moves(self, board_state):
        raise NotImplementedError()

    def has_winner(self, board_state):
        raise NotImplementedError()
    
    def has_winner_TD_train(self, board_state):
        raise NotImplementedError()

    def board_dimensions(self):
        raise NotImplementedError()

    def board_squares(self):
        return reduce(operator.mul, self.board_dimensions(), 1)

    def outputs(self):
        return self.board_squares()

    def flat_move_to_tuple(self, move_index):
        if len(self.board_dimensions()) == 1:
            return move_index

        board_x = self.board_dimensions()[0]
        return int(move_index / board_x), move_index % board_x

    def tuple_move_to_flat(self, tuple_move):
        if len(self.board_dimensions()) == 1:
            return tuple_move[0]
        else:
            return tuple_move[0] * self.board_dimensions()[0] + tuple_move[1]

    def play_game(self, plus_player_func, minus_player_func, log=False, board_state=None):
        board_state = board_state or self.new_board()
        player_turn = 1

        while True:
            _available_moves = list(self.available_moves(board_state))

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

            board_state = self.apply_move(board_state, move, player_turn)
            if log:
                print(board_state)

            winner = self.has_winner(board_state)
            if winner != 0:
                if log:
                    print("we have a winner, side: %s" % player_turn)
                return winner
            player_turn = -player_turn

    def get_random_player_func(self):
        return lambda board_state, side: random.choice(list(self.available_moves(board_state)))
