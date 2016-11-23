import numpy as np

def modmat_mult(C, D, mod):
  (m, n) = C.shape
  (_, p) = D.shape
  E = np.zeros((m,p), dtype=np.int8)
  for i in range(m):
    for j in range(p):
      for k in range(n):
        E[i,j] = (E[i,j] + ((C[i,k] * D[k,j]) % mod)) % mod
  return E

def modmat_dot(A, x, mod):
  return modmat_mult(A, np.array([x], dtype=np.int8).T, mod)

def modmat_fsolve(L, b, mod):
  if not np.all(np.diag(L)):
    raise ValueError("Matrix is singular")
  n = L.shape[0]
  x = np.empty(n, dtype=np.int8)
  for i in range(n):
    s = b[i]
    for j in range(0, i):
      s = (s + ((L[i, j] * x[j]) % mod)) % mod
    x[i] = s
  return x

def modmat_rsolve(U, b, mod):
  if not np.all(np.diag(U)):
    raise ValueError("Matrix is singular")
  n = U.shape[0]
  x = np.empty(n, dtype=np.int8)
  for i in range(n - 1, -1, -1):
    s = b[i]
    for j in range(i + 1, n):
      s = (s + ((U[i, j] * x[j]) % mod)) % mod
    x[i] = s
  return x

def modmat_lu(A, mod):
  n = A.shape[0]
  U = np.copy(A)
  L = np.identity(n, dtype=np.int8)
  P = np.identity(n, dtype=np.int8)
  for j in range(0, n - 1):
    #select row with a 1
    for i in range(j, n):
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
    for i in range(j + 1, n):
      L[i, j] = U[i, j]
      #eliminate column
      for k in range(j, n):
        U[i, k] = (U[i, k] + ((L[i, j] * U[j, k]) % mod)) % mod

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
  np.testing.assert_array_equal(modmat_dot(A, x, 3), np.array([b]).T)

  print('done!')
