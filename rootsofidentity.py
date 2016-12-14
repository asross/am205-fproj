#!/usr/bin/python
#iterates through non-singular cases to find the cycle length for a matrix to return to the identity
from modular_matrix import *
from transition_helpers import *
import numpy as np

transition_fn = lambda i, j: [[i,j,1],[i+1,j,1],[i-1,j,1],[i,j+1,1],[i,j-1,1]]

def find_cycle(M, max_iter=100000):
  Mp = M.copy()
  I = np.identity(M.array.shape[0])
  for i in range(max_iter):
    if np.array_equal(Mp.array, I):
      return i + 1
      break
    Mp = Mp * M
  return -1

n = 10
results = np.zeros((n, n), dtype=np.int)
for i in range(n):
  for j in range(n):
    A = transition_matrix(transition_fn, i+1, j+1, 2)
    if A.nullity() == 0:
      results[i, j] = find_cycle(A)

print "0 means singular matrix, -1 means length > 100000"
print results