## Problem Statement

Implement and train an RL agent to solve Killer Sudoku puzzles. 

## Agent Design

* Observations provide the current state of the puzzle.
* Actions are the numbers to be placed in the cells: 1-9.
* A done signal is set when the puzzle is solved or when the agent makes an illegal move.
* Rewards are binary and sparse: the immediate reward is always 0, except when the puzzle is solved, in which case the reward is 1.