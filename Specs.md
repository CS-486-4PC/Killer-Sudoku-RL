## Problem Statement

Implement and train an RL agent to solve Killer Sudoku puzzles. 

## Agent Design

### State Representation

* Observations provide the current state of the entire 9x9x10 space: 9x9 grid and 0-9 values.
* Action space has the size of 9x9x9: Action is the location value (9x9) and the entry value (1-9).

### Reward Structure

#### 1st Approach: Binary Rewards

* A done signal is set when the puzzle is solved or when the agent makes an illegal move.
* Rewards are binary and sparse: the immediate reward is always 0, except when the puzzle is solved, in which case the reward is 1.

#### 2nd Approach: Allow for corrections

4 components of the reward structure:

1. no_progress (-1), when a picked entry was filled (in the initial setup or by the agent)
2. progress (+1), if an entry satisfies the Sudoku constraints
3. partial_loss (-2), if an entry does not satisfy the Sudoku constraints
4. loss (-20) if the number of moves per episode passes a threshold.

## Killer Sudoku

### Rules
* The objective is to fill the grid with numbers from 1 to 9
* Usual Sudoku rules apply: Each row, column, and box contains each number exactly once.
* Additionally, the grid is divided into cages, and the value of each cage is given. The sum of the values in each cage must equal the value given for the cage.
* No digit can be repeated within a cage.

## Killer Sudoku Generator

* Generate a complete Sudoku grid
* Randomly select a subset of cells to remove, resulting in a masked grid
* Define cages:
  * 