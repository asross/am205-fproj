from modular_matrix import *
from transition_helpers import *
import numpy as np

def each_possible_grid(rows, cols, mod):
    # each grid can be thought of as a number in base mod
    for i in range(mod**(rows*cols)):
        grid = np.zeros(rows*cols)
        j = len(grid)-1
        while i > 0:
            grid[j] = i % mod
            i = i // mod
            j -= 1
        yield grid.reshape((rows, cols))

def cycle_information(A, grid):
    try:
        b = grid.ravel()
        seen = [str(b)]
        while True:
            b = A.solve(b)
            sight = str(b)
            if sight in seen:
                return len(seen), seen.index(sight)
            else:
                seen.append(sight)
    except SingularMatrixError:
        return 0, 0

if __name__ == '__main__':
    (n,m) = 2, 2
    mod = 3
    lengths = []
    indexes = []

    singulars = 0
    adj = lambda i, j: [[i,j,1],[i+1,j,1],[i-1,j,1],[i,j+1,1],[i,j-1,1]]
    A = transition_matrix(adj, n, m, mod)

    if A.nullity() > 0:
      raise SingularMatrixError("Matrix is singular")

    for grid in each_possible_grid(n, m, mod):
      length, idx = cycle_information(A, grid)
      lengths.append(length)
      indexes.append(idx)

    from collections import Counter
    counts = Counter(lengths)
    print(counts)
