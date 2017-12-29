## Imports:
import random
from connect4 import Connect4GameSpec


def benchmark(game_spec, state, side):
    ## 1) Get all possible moves in random order
    available = list(game_spec.available_moves(state))
    random.shuffle(available)    
    ## 2) If there is a move that would win the game, do the first one that does.
    for t in available:
        new_state = game_spec.apply_move(state, t, side)
        aus = game_spec.has_winner(new_state)
        if aus == side:
            return t
    ## 3) Otherwise, if the opponent has any way to win next turn, block the first one found.    
    for b in available:    
        new_state = game_spec.apply_move(state, b, -side)
        aus_opp = game_spec.has_winner(new_state)
        if aus_opp == -side:
            return b
    ## 4) Otherwise, do the first possible move which does not allow the opponent to win the game
    ## by placing a piece on top of it
    for c in available:
        new_state = game_spec.apply_move(state, c, side)
        new_state = game_spec.apply_move(state, c, -side)
        aus2 = game_spec.has_winner(new_state)
        if aus2 == 0:
            return c


game_spec = Connect4GameSpec()

def benchmark_player(state, side):
    return benchmark(game_spec, state, side)
