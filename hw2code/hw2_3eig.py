#!/usr/bin/python
#Shawn Pan
#AM205
import numpy as np
import matplotlib.pyplot as plt
from hw2_binary import *

A = transition_matrix(7, 7)
I = np.identity(49, dtype=np.int8)
Z = np.zeros(49, dtype=np.int8)

P, L, U = lufactor(A - I)

print "Rank of A - I:", rank(U)

#make U full rank diagonal
U1 = U.copy()
for i in xrange(49):
  U1[i, i] = 1

#find linearly independent eigenvectors
for i in xrange(rank(U), 49):
  Z1 = Z.copy()
  Z1[i] = 1
  y = fsolve(L, Z1)
  x = rsolve(U1, y)
  fig = plt.figure(figsize=(2, 2))
  eig = lusolve(A, x)
  plt.spy(eig.reshape(7, 7))
  plt.savefig("eigbasis" + str(i) + ".png", bbox_inches="tight")
  #print np.array_equal(bin_mul(A, x).flatten(), x)

#find all 127 eigenvectors + 1 empty board
for i in xrange(128):
  Z1 = Z.copy()
  b = i
  j = 48
  while b > 0:
    Z1[j] = b & 1
    j -= 1
    b >>= 1
  y = fsolve(L, Z1)
  x = rsolve(U1, y)
  fig = plt.figure(figsize=(2, 2))
  eig = lusolve(A, x)
  plt.spy(eig.reshape(7, 7))
  plt.savefig("eig" + str(i) + ".png", bbox_inches="tight")
  #print np.array_equal(bin_mul(A, x).flatten(), x)