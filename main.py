from sudoku import generateGrid
import random

mask_rate = random.uniform(0.1, 0.7)
print("mask_rate:", mask_rate)

# generate a sudoku grid
g, s = generateGrid(mask_rate, seed=42)
for row in g:
    print(row)

print()

for row in s:
    print(row)
