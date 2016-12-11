#!/usr/bin/python
import numpy as np

class ModularMatrix():
  def __init__(self, array, modulus):
    self.array = np.array(array) % modulus
    self.modulus = modulus
    self.lu_cache = None

  def __getitem__(self, *args, **kwargs):
    return self.array.__getitem__(*args, **kwargs)

  def __setitem__(self, *args, **kwargs):
    self.lu_cache = None
    self.array.__setitem__(*args, **kwargs)
    self.array = self.array % self.modulus

  def __str__(self):
    return "ModularMatrix {} %{}\n".format(self.array.shape, self.modulus) + self.array.__str__()

  def _array(self, o):
    return o.array if isinstance(o, self.__class__) else o

  def new(self, array):
    return self.__class__(array, self.modulus)

  def __add__(self, other):
    return self.new(np.add(self.array, self._array(other)))

  def __mul__(self, other):
    return self.new(np.dot(self.array, self._array(other)))

  def dot(self, other):
    return np.dot(self.array, self._array(other)) % self.modulus

  def solve(self, b):
    P, L, U = self.lu_factorization()
    return U.rsolve(L.fsolve(P.dot(b)))

  def rank(self):
    P, L, U = self.lu_factorization()
    return np.count_nonzero(np.diag(U.array))

  def nullity(self):
    return self.array.shape[0] - self.rank()

  def fsolve(self, b):
    L = self.array
    if not np.all(np.diag(L)):
      raise ValueError("Matrix is singular")
    n = L.shape[0]
    x = np.empty(n, dtype=self.array.dtype)
    for j in range(n):
      x[j] = b[j]
      for k in range(j):
        x[j] -= L[j, k] * x[k]
      x[j] = x[j] % self.modulus
      x[j] = mod_divide(x[j], L[j, j], self.modulus)
    return x

  def rsolve(self, b):
    U = self.array.copy()
    n = U.shape[0]
    x = np.empty(n, dtype=self.array.dtype)
    for j in reversed(range(n)):
      x[j] = b[j]
      for k in range(j+1, n):
        x[j] -= U[j, k] * x[k]
      x[j] = x[j] % self.modulus
      if U[j, j] == 0: #singular row
        if x[j] == 0: #0 = 0 free parameter, many solutions
          print "Warning: matrix is singular, returning one of many solutions"
          #arbitrarily set x_j = 0
          x[j] = 0
          U[j, j] = 1
        else: #no solutions
          raise ValueError("Matrix is singular, no solutions")
      else:
        x[j] = mod_divide(x[j], U[j, j], self.modulus)
    return x

  def lu_factorization(self):
    if self.lu_cache:
      return self.lu_cache
    A = self.array
    (m,n) = A.shape
    if m != n: raise ValueError("must pass a square matrix")
    U = np.copy(A)
    L = np.identity(n, dtype=self.array.dtype)
    P = np.identity(n, dtype=self.array.dtype)
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
        L[i,j] = mod_divide(U[i,j], U[j,j], self.modulus)
        for k in range(j, n):
          U[i,k] = (U[i,k] - (L[i,j] * U[j,k])) % self.modulus
    self.lu_cache = (self.new(P), self.new(L), self.new(U))
    return self.lu_cache

moddiv_cache = {}
def mod_divide(a, b, mod):
  if mod not in moddiv_cache:
    moddiv_cache[mod] = np.zeros((mod, mod), dtype=np.int8)
    for i in range(1, mod):
      for j in range(i, mod):
        dividend = (i * j) % mod
        moddiv_cache[mod][dividend, i] = j
        moddiv_cache[mod][dividend, j] = i
  return moddiv_cache[mod][a][b]

if __name__ == '__main__':
  print('running tests...')

  #test division
  assert(mod_divide(1, 2, 3) == 2)
  assert(mod_divide(2, 1, 3) == 2)

  # test operations
  mm = ModularMatrix(np.identity(3), 3)
  np.testing.assert_array_equal((mm+3).array, mm.array)
  np.testing.assert_array_equal((mm+mm+mm+mm).array, mm.array)
  np.testing.assert_array_equal((mm*mm).array, mm.array)
  assert(mm[0][0] == 1)
  mm[0][1] = 1
  assert(mm[0][1] == 1)

  # solving with identity should echo b
  L = ModularMatrix(
      np.array([[1, 0],
                [0, 1]], dtype=np.int8), 2)
  b = np.array([1, 1], dtype=np.int8)
  np.testing.assert_array_equal(L.fsolve(b), b)

  # another case where we should echo the same b
  L = ModularMatrix(
      np.array([[1, 0],
                [1, 1]], dtype=np.int8), 2)
  b = np.array([0, 1], dtype=np.int8)
  np.testing.assert_array_equal(L.fsolve(b), b)

  # singular
  failed = False
  L = ModularMatrix(
      np.array([[1, 0],
                [1, 0]], dtype=np.int8), 2)
  try:
    L.fsolve(b)
    assert(False)
  except ValueError:
    pass

  # for I, should return 3 Is
  I = ModularMatrix(
      np.array([[1, 0],
                [0, 1]], dtype=np.int8), 2)
  P, L, U = I.lu_factorization()
  np.testing.assert_array_equal(P.array, I.array)
  np.testing.assert_array_equal(L.array, I.array)
  np.testing.assert_array_equal(U.array, I.array)

  # Solution should echo
  x = np.array([1, 0], dtype=np.int8)
  np.testing.assert_array_equal(x, I.solve(x))

  # on an upper triangular matrix U, should return U + 2Is
  A = ModularMatrix(
      np.array([[1, 1],
                [0, 1]], dtype=np.int8), 2)
  P, L, U = A.lu_factorization()
  np.testing.assert_array_equal(P.array, I.array)
  np.testing.assert_array_equal(L.array, I.array)
  np.testing.assert_array_equal(U.array, A.array)

  # on a lower triangular matrix L, should return L + 2Is
  A = ModularMatrix(
      np.array([[1, 0],
                [1, 1]], dtype=np.int8), 2)
  P, L, U = A.lu_factorization()
  np.testing.assert_array_equal(P.array, I.array)
  np.testing.assert_array_equal(L.array, A.array)
  np.testing.assert_array_equal(U.array, I.array)

  # on a permutation P, it should return P^T + 2Is
  A = ModularMatrix(
      np.array([[0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]], dtype=np.int8), 2)
  P, L, U = A.lu_factorization()
  np.testing.assert_array_equal(P.array, A.array.T)
  np.testing.assert_array_equal(L.array, np.identity(3, dtype=np.int8))
  np.testing.assert_array_equal(U.array, np.identity(3, dtype=np.int8))

  # should work for mod 3
  A = ModularMatrix(
      np.array([[0, 1, 2],
                [0, 1, 1],
                [0, 2, 0]], dtype=np.int8), 3)
  x = np.array([1, 2, 0], dtype=np.int8)
  b = np.array([2, 2, 1], dtype=np.int8)
  np.testing.assert_array_equal(A.dot(x), b)

  #note this A matrix is singular
  #row1 = 2 X row2 + row3 (mod 3)
  assert A.rank() == 2
  assert A.nullity() == 1
  x2 = A.solve(b)
  np.testing.assert_array_equal(A.dot(x2), b)

  #test no solution vs many solutions
  A = ModularMatrix(
      np.array([[3, 4, 0],
                [3, 4, 0],
                [1, 2, 2]], dtype=np.int8), 5)
  b = np.array([2, 2, 3], dtype=np.int8)
  x = A.solve(b)
  np.testing.assert_array_equal(A.dot(x), b)

  b_no_sol = np.array([1, 2, 3], dtype=np.int8)
  try:
    A.solve(b_no_sol)
    assert(False)
  except ValueError:
    pass

  # test solve 2x2 matrix mod 7
  A = ModularMatrix(
      np.array([[3, 1],
                [0, 2]], dtype=np.int8), 7)
  x = np.array([6, 4], dtype=np.int8)
  b = A.dot(x)
  np.testing.assert_array_equal(A.solve(b), x)

  # test solve 5x5 matrix mod 7
  A = ModularMatrix(
      np.array( [[3, 1, 0, 6, 2],
                 [0, 2, 2, 1, 4],
                 [4, 1, 5, 3, 4],
                 [5, 3, 2, 5, 1],
                 [1, 1, 4, 3, 1]], dtype=np.int8), 7)
  x = np.array([6, 4, 3, 3, 3], dtype=np.int8)
  b = A.dot(x)
  np.testing.assert_array_equal(A.solve(b), x)
  assert A.rank() == 5
  assert A.nullity() == 0

  # test caching behavior
  assert A.lu_cache
  A[0, 0] = 5
  assert not A.lu_cache
  b = A.dot(x)
  np.testing.assert_array_equal(A.solve(b), x)
  assert A.lu_cache

  print('done!')
