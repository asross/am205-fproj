#!/usr/bin/python
#Shawn Pan
#AM205
import numpy as np
from hw2_binary import *

#part c: nullspace dimensions of matrix
max_dim = 9
nullspace_dim = np.zeros((max_dim, max_dim))
for i in xrange(max_dim):
  for j in xrange(max_dim):
    nullspace_dim[i, j] = nullity(transition_matrix(i + 1, j + 1))
print nullspace_dim