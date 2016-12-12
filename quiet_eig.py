#!/usr/bin/python
from modular_matrix import *
from transition_helpers import *
import numpy as np
np.set_printoptions(linewidth=160)

transition_fn = lambda i, j: [[i,j,1],[i+1,j,1],[i-1,j,1],[i,j+1,1],[i,j-1,1]]

A7 = transition_matrix(transition_fn, 7, 7, 2)
A7eig = A7 - np.identity(49, dtype=np.int8)
A5 = transition_matrix(transition_fn, 5, 5, 2)

print "Quiet States for 5x5"
print "Any Mode"
print A5.solve(np.zeros(25), singular_mode="any")
print "Basis Mode"
print A5.solve(np.zeros(25), singular_mode="basis")
print "All Mode"
print A5.solve(np.zeros(25), singular_mode="all")
print

print "Eigenvectors for 7x7"
print "Any Mode"
print A7eig.solve(np.zeros(49), singular_mode="any")
print "Basis Mode"
print A7eig.solve(np.zeros(49), singular_mode="basis")