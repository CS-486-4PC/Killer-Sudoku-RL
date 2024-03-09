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

### Cage Generation Algorithm

1. Select the starting point and size for a cage.

   Top priority for a starting point is a cell that is surrounded on three or four sides, otherwise the cell is chosen at random from the list of unused cells.s

   A cell surrounded on four sides must become a single-cell cage, with a pre-determined value and no operator. Choosing a cell surrounded on three sides allows the cage to occupy and grow out of tight corners, avoiding an excess of small and single-cell cages.

   The chosen size is initially 1 (needed to control the difficulty of the puzzle) and later a random number from 2 to the maximum cage-size. The maximum cage-size also affects difficulty.

2. Use the function `makeOneCage()` to make a cage of the required size.

   The `makeOneCage()` function keeps adding unused neighbouring cells until the required size is reached or it runs out of space to grow the cage further. It updates the lists of used cells and neighbours as it goes. 

    A neighbour that would otherwise become surrounded on all four sides is always added to the cage as it grows, but normally the next cell is chosen randomly.

3. Use the function `setCageTarget()` to choose an operator (+*-/)

    Calculate the cage's value from the cell-values in the puzzle's solution and find all the possible combinations of values that cells in the cage *might* have, as seen by the user.

   The possible combinations are used when solving the generated puzzle, using the DLX algorithm, to check that the puzzle has a unique solution. Many generated puzzles have multiple solutions and have to be discarded.

4. Validate the cage, using function `isCageOK()`.

   A cage can be rejected if it might make the puzzle too hard or too easy. If so, discard the cage, back up and repeat steps 1 to 4.

5. Repeat steps 1 to 4 until all cells have been assigned to cages.
