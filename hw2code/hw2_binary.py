#!/usr/bin/python
#Shawn Pan
#AM205
import numpy as np

#calculate binary matrix multiplication, adapted from bin_mul.py
def bin_mul(c, d):
  # convert 1D arrays to row or column vectors as appropriate
  if len(c.shape) == 1:
    c = np.reshape(c, (1, c.shape[0]))
  if len(d.shape) == 1:
    d = np.reshape(d, (d.shape[0], 1))

  # Check that the dimensions of the matrices are compatible
  (m, n) = c.shape
  (nn, p) = d.shape
  if n != nn:
    raise ValueError("Matrix size mismatch")

  # Initailize blank matrix of integer zeros
  result = np.zeros((m, p), dtype=np.int8)

  # Calculate each term, using "&" instead of "*" and "^" instead of "+"
  for i in xrange(m):
      for j in xrange(p):
          for k in xrange(n):
              result[i,j] ^= (c[i, k] & d[k, j])
  return result

#2a forward substitution
def fsolve(L, b):
  if not np.all(np.diag(L)):
    raise ValueError("Matrix is singular")
  n = L.shape[0]
  x = np.empty(n, dtype=np.int8)
  for i in xrange(n):
    s = b[i]
    for j in xrange(0, i):
      s ^= (L[i, j] & x[j])
    x[i] = s
  return x

#2b back substitution
def rsolve(U, b):
  if not np.all(np.diag(U)):
    raise ValueError("Matrix is singular")
  n = U.shape[0]
  x = np.empty(n, dtype=np.int8)
  for i in xrange(n - 1, -1, -1):
    s = b[i]
    for j in xrange(i + 1, n):
      s ^= (U[i, j] & x[j])
    x[i] = s
  return x

#2c LU factor
def lufactor(A):
  n = A.shape[0]
  U = np.copy(A)
  L = np.identity(n, dtype=np.int8)
  P = np.identity(n, dtype=np.int8)
  for j in xrange(0, n - 1):
    #select row with a 1
    for i in xrange(j, n):
      if U[i, j] == 1:
        #swap rows of U
        temp = np.array(U[j, j:n])
        U[j, j:n] = U[i, j:n]
        U[i, j:n] = temp
        #swap rows of L
        temp = np.array(L[j, 0:j])
        L[j, 0:j] = L[i, 0:j]
        L[i, 0:j] = temp
        #swap rows of P
        temp = np.array(P[j, :])
        P[j, :] = P[i, :]
        P[i, :] = temp
        break
    for i in xrange(j + 1, n):
      L[i, j] = U[i, j]
      #eliminate column
      for k in xrange(j, n):
        U[i, k] ^= (L[i, j] & U[j, k])

  return P, L, U

#2d solve with LU factorization
def lusolve(A, b):
  P, L, U = lufactor(A)
  y = fsolve(L, bin_mul(P, b))
  x = rsolve(U, y)
  return x

#calculate rank of square matrix
def rank(A):
  P, L, U = lufactor(A)
  return np.sum(np.diag(U))

#calculate nullity of square matrix
def nullity(A):
  return A.shape[0] - rank(A)

#3a get lights transition matrix for r rows and c columns
def transition_matrix(r, c):
  #number of buttons
  n = r * c
  #convert row and column to index
  index = lambda i, j: c * i + j
  A = np.zeros((n, n), dtype=np.int8)
  for i in xrange(r):
    for j in xrange(c):
      k = index(i, j)
      #add pressed button
      A[k, k] = 1
      #add button above
      if i > 0:
        A[index(i - 1, j), k] = 1
      #add button below
      if i < r - 1:
        A[index(i + 1, j), k] = 1
      #add button left
      if j > 0:
        A[index(i, j - 1), k] = 1
      #add button right
      if j < c - 1:
        A[index(i, j + 1), k] = 1
  return A

#Plot transtion matrix for 7 by 7
# A = transition_matrix(7, 7)
# plt.figure(figsize=(4, 4))
# plt.spy(A)
# plt.savefig("transition.png", bbox_inches="tight")