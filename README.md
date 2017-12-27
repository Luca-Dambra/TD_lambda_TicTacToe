# TD_lambda_TicTacToe
Train an agent that plays Tic Tac Toe using the TD-lambda algorithm.


# Experiment Setup
The experiment is divided in rounds. Each round is composed by two phases (Train and Test phase):
### Train phase:
the program is trained for a certain number of games.

the exploration rate epsilon is grater than 0.

the learning rate alpha is decreased according to an exponential decay rule, starting from a given value.
### Test phase:
the agent is testes against a fixed opponent (which choose moves randomly).

the exploration rate is set equal to 0 and no updates is performed.



Most of the code in this reposity is based on the works of Daniel Slater ([AlphaToe](https://github.com/DanielSlater/AlphaToe)) and
Fomoro Company Organization ([TD-gammon](https://github.com/fomorians/td-gammon)).


WORK IN PROGRESS...
