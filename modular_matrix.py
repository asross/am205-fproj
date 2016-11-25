#!/usr/bin/python
import numpy as np
from collections import defaultdict

def modmat_mult(C, D, mod):
  (m, n) = C.shape
  (_, p) = D.shape
  E = np.zeros((m,p), dtype=np.int8)
  for i in range(m):
    for j in range(p):
      for k in range(n):
        E[i,j] = (E[i,j] + C[i,k] * D[k,j]) % mod
  return E

def modmat_dot(A, x, mod):
  return modmat_mult(A, np.array([x], dtype=np.int8).T, mod).T[0]

moddiv_cache = defaultdict(lambda: defaultdict(dict))
def mod_divide(a, b, mod):
  if mod not in moddiv_cache:
    for i in range(mod):
      for j in range(mod):
        moddiv_cache[mod][(i * j) % mod][i] = j
        moddiv_cache[mod][(i * j) % mod][j] = i
  return moddiv_cache[mod][a][b]

assert(mod_divide(1, 2, 3) == 2)
assert(mod_divide(2, 1, 3) == 2)

def modmat_fsolve(L, b, mod):
  if not np.all(np.diag(L)):
    raise ValueError("Matrix is singular")
  n = L.shape[0]
  x = np.empty(n, dtype=np.int8)
  for j in range(n):
    x[j] = b[j]
    for k in range(j):
      x[j] -= L[j, k] * x[k]
    x[j] = x[j] % mod
    x[j] = mod_divide(x[j], L[j, j], mod)
  return x

def modmat_rsolve(U, b, mod):
  if not np.all(np.diag(U)):
    raise ValueError("Matrix is singular")
  n = U.shape[0]
  x = np.empty(n, dtype=np.int8)
  for j in reversed(range(n)):
    x[j] = b[j]
    for k in range(j+1, n):
      x[j] -= U[j, k] * x[k]
    x[j] = x[j] % mod
    x[j] = mod_divide(x[j], U[j, j], mod)
  return x

def modmat_lu(A, mod):
  (m,n) = A.shape
  if m != n: raise ValueError("must pass a square matrix")
  U = np.copy(A)
  L = np.identity(n, dtype=np.int8)
  P = np.identity(n, dtype=np.int8)
  for j in range(n-1):
    #select non-zero row
    for i in range(j, n):
      if U[i, j] != 0:
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
    # Now apply the normal LU operations
    for i in range(j+1, n):
      L[i,j] = mod_divide(U[i,j], U[j,j], mod)
      for k in range(j, n):
        U[i,k] = (U[i,k] - (L[i,j] * U[j,k])) % mod

  return P, L, U

def modmat_solve(A, b, mod):
  P, L, U = modmat_lu(A, mod)
  y = modmat_fsolve(L, modmat_dot(P, b, mod), mod)
  x = modmat_rsolve(U, y, mod)
  return x

def modmat_rank(A, mod):
  P, L, U = modmat_lu(A, mod)
  return np.sum(np.diag(U))

def modmat_nullity(A, mod):
  return A.shape[0] - modmat_rank(A, mod)

def transition_matrix(adj_fn, rows, cols):
  A = np.zeros((rows*cols, rows*cols), dtype=np.int8)
  for row in range(rows):
    for col in range(cols):
      for i, j, k in adj_fn(row, col):
        if 0 <= i < rows and 0 <= j < cols:
          A[row+col*rows][i+j*rows] = k
  return A

if __name__ == '__main__':
  print('running tests...')

  # solving with identity should echo b
  L = np.array([[1, 0],
                [0, 1]], dtype=np.int8)
  b = np.array([1, 1], dtype=np.int8)
  np.testing.assert_array_equal(modmat_fsolve(L, b, 2), b)

  # another case where we should echo the same b
  L = np.array([[1, 0],
                [1, 1]], dtype=np.int8)
  b = np.array([0, 1], dtype=np.int8)
  np.testing.assert_array_equal(modmat_fsolve(L, b, 2), b)

  # singular
  failed = False
  L = np.array([[1, 0],
                [1, 0]], dtype=np.int8)
  try: modmat_fsolve(L, b, 2)
  except ValueError: failed = True
  assert(failed)

  # for I, should return 3 Is
  I =  np.array([[1, 0],
                 [0, 1]], dtype=np.int8)
  P, L, U = modmat_lu(I, 2)
  np.testing.assert_array_equal(P, I)
  np.testing.assert_array_equal(L, I)
  np.testing.assert_array_equal(U, I)

  # Solution should echo
  x = np.array([1, 0], dtype=np.int8)
  np.testing.assert_array_equal(x, modmat_solve(I, x, 2))

  # on an upper triangular matrix U, should return U + 2Is
  A = np.array([[1, 1],
                [0, 1]], dtype=np.int8)
  P, L, U = modmat_lu(A, 2)
  np.testing.assert_array_equal(P, I)
  np.testing.assert_array_equal(L, I)
  np.testing.assert_array_equal(U, A)

  # on a lower triangular matrix L, should return L + 2Is
  A = np.array([[1, 0],
                [1, 1]], dtype=np.int8)
  P, L, U = modmat_lu(A, 2)
  np.testing.assert_array_equal(P, I)
  np.testing.assert_array_equal(L, A)
  np.testing.assert_array_equal(U, I)

  # on a permutation P, it should return P^T + 2Is
  A = np.array([[0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]], dtype=np.int8)
  P, L, U = modmat_lu(A, 2)
  np.testing.assert_array_equal(P, A.T)
  np.testing.assert_array_equal(L, np.identity(3, dtype=np.int8))
  np.testing.assert_array_equal(U, np.identity(3, dtype=np.int8))

  # should work for mod 3
  A = np.array([[0, 1, 2],
                [0, 1, 1],
                [0, 2, 0]], dtype=np.int8)
  x = np.array([1, 2, 0], dtype=np.int8)
  b = np.array([2, 2, 1], dtype=np.int8)
  np.testing.assert_array_equal(modmat_dot(A, x, 3), b)

  #note this A matrix is singular
  #row1 = 2 X row2 + row3 (mod 3)
  try: modmat_solve(A, b, 3)
  except ValueError: failed = True
  assert(failed)

  # test solve 2x2 matrix mod 7
  A = np.array( [[3, 1],
                 [0, 2]], dtype=np.int8)
  x = np.array([6, 4], dtype=np.int8)
  b = modmat_dot(A, x, 7)
  np.testing.assert_array_equal(modmat_solve(A, b, 7), x)

  # test solve 5x5 matrix mod 7
  A = np.array( [[3, 1, 0, 6, 2],
                 [0, 2, 2, 1, 4],
                 [4, 1, 5, 3, 4],
                 [5, 3, 2, 5, 1],
                 [1, 1, 4, 3, 1]], dtype=np.int8)
  x = np.array([6, 4, 3, 3, 3], dtype=np.int8)
  b = modmat_dot(A, x, 7)
  np.testing.assert_array_equal(modmat_solve(A, b, 7), x)

  # create a mod 3 light array from blank array
  transition_fn = lambda i, j: [[i,j,2],[i+1,j,1],[i-1,j,1],[i,j+1,1],[i,j-1,1]]
  A = transition_matrix(transition_fn, 3, 3)
  grid = np.array([[0,1,1],[1,0,0],[0,1,1]])
  presses = modmat_solve(A, grid.ravel(), 3)
  print(grid)
  print(modmat_dot(A, presses, 3).reshape(grid.shape))

  # state = grid.ravel()
  # print(state.reshape(grid.shape))
  # for i, n in enumerate(presses):
    # for _ in range(n):
      # press = np.zeros(len(presses), dtype=np.int8)
      # press[i] = 1
      # effect = modmat_dot(A, press, 3)
      # state = (state + effect) % 3
      # print(state.reshape(grid.shape))

  print('done!')
