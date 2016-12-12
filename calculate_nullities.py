from modular_matrix import *
from transition_helpers import *
import numpy as np

if __name__ == '__main__':
  adj = lambda i, j: [[i,j,1],[i+1,j,1],[i-1,j,1],[i,j+1,1],[i,j-1,1]]
  print('mod,n,m,nullity')
  for mod in range(2, 18):
    for n in range(1, 17):
      for m in range(1, 17):
        A = transition_matrix(adj, n, m, mod)
        print('{},{},{},{}'.format(mod,n,m,A.nullity()))
