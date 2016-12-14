#!/usr/bin/python
import numpy as np

class SingularMatrixError(ValueError):
    pass

class ModularMatrix():
  def __init__(self, array, modulus):
    self.array = np.array(array) % modulus
    self.modulus = modulus
    self.lu_cache = None
    self.relatively_prime = [gcd(i, self.modulus) == 1 for i in range(self.modulus)]

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

  def copy(self):
    return self.new(self.array)

  def __add__(self, other):
    return self.new(np.add(self.array, self._array(other)))

  def __sub__(self, other):
    return self.new(np.add(self.array, -1 * self._array(other)))

  def __mul__(self, other):
    return self.new(np.dot(self.array, self._array(other)))

  def dot(self, other):
    return np.dot(self.array, self._array(other)) % self.modulus

  #solve Ax = b with LU factorization and forward/backward substitution
  #takes in a keyword to decide what to do if the matrix is singular an has many solutions
  #error (default): throw error
  #any: return arbitrary solution
  #basis: find a basis of the solution space
  #all: find all solutions
  def solve(self, b, singular_mode="error"):
    P, L, U = self.lu_factorization()
    ndim = self.nullity()
    #matrix is full rank
    if ndim == 0:
      return U.rsolve(L.fsolve(P.dot(b)))
    #matrix is singular
    if singular_mode == "error":
      raise SingularMatrixError("Matrix is singular")
    #handle multiple solutions
    Ub = L.fsolve(P.dot(b))
    if singular_mode == "any":
      return U.rsolve(Ub, np.zeros(ndim))
    elif singular_mode == "basis":
      solutions = np.empty((ndim, self.array.shape[0]), dtype=self.array.dtype)
      for i in range(ndim):
        freeParam = np.zeros(ndim)
        freeParam[i] = 1
        solutions[i, :] = U.rsolve(Ub, freeParam)
      return solutions
    elif singular_mode == "all":
      solution_count = self.modulus ** ndim
      solutions = np.empty((solution_count, self.array.shape[0]), dtype=self.array.dtype)
      for i in range(solution_count):
        freeParam = np.empty(ndim)
        solution_id = i
        for p in range(ndim):
          freeParam[p] = solution_id % self.modulus
          solution_id /= self.modulus
        solutions[i, :] = U.rsolve(Ub, freeParam)
      return solutions
    else:
      raise ValueError("Invalid singular mode: " + singular_mode)

  def rank(self):
    P, L, U = self.lu_factorization()
    return np.count_nonzero([self.relatively_prime[d] for d in np.diag(U.array)])

  def nullity(self):
    return self.array.shape[0] - self.rank()

  #find inverse by solving basis columns
  def inverse(self):
    n = self.array.shape[0]
    I = np.identity(n, dtype=self.array.dtype)
    inv = np.zeros((n, n), dtype=self.array.dtype)
    for c in range(n):
      inv[:, c] = self.solve(I[:, c])
    return self.new(inv)

  def fsolve(self, b):
    L = self.array
    if not np.all(np.diag(L)):
      raise SingularMatrixError("Matrix is singular")
    n = L.shape[0]
    x = np.empty(n, dtype=self.array.dtype)
    for j in range(n):
      #dot product is faster than looping, extra % is to prevent overflows
      x[j] = b[j] - np.dot(L[j, :j], x[:j]) % self.modulus
      x[j] = x[j] % self.modulus
      x[j] = mod_divide(x[j], L[j, j], self.modulus)
    return x

  def rsolve(self, b, freeParam=[]):
    U = self.array.copy()
    n = U.shape[0]
    x = np.empty(n, dtype=self.array.dtype)
    freeParamCount = 0
    for j in reversed(range(n)):
      #dot product is faster than looping, extra % is to prevent overflows
      x[j] = b[j] - np.dot(U[j, j+1:n], x[j+1:n]) % self.modulus
      x[j] = x[j] % self.modulus
      if U[j, j] == 0: #singular row
        if x[j] == 0: #0 = 0 free parameter, many solutions
          #set x_j = specified free parameter
          x[j] = freeParam[freeParamCount]
          U[j, j] = 1
          freeParamCount += 1
        else: #no solutions
          raise SingularMatrixError("No solutions")
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
      self.find_pivot(P, L, U, j, n)
      # Now apply the normal LU operations
      for i in range(j+1, n):
        L[i,j] = mod_divide(U[i,j], U[j,j], self.modulus)
        #update row with numpy vector operation
        U[i, j:n] = (U[i, j:n] - (L[i,j] * U[j, j:n])) % self.modulus
    self.lu_cache = (self.new(P), self.new(L), self.new(U))
    return self.lu_cache

  def find_pivot(self, P, L, U, j, n):
    for i in range(j, n):
      #select non-zero row
      #in composite case, want a relatively prime pivot for division
      if self.relatively_prime[U[i, j]]:
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
        return
    #no relatively prime pivot available, try random linear combinations
    #warning, this is not yet a complete check of all linear combinations
    #give up after 1000 trails, matrix probably singular
    np.random.seed(0)
    trials = 0
    while trials < 1000 and not self.relatively_prime[U[j, j]]:
      i = np.random.randint(j+1, n)
      U[j, j:n] = (U[j, j:n] + U[i, j:n]) % self.modulus
      L[j, 0:j] = (L[j, 0:j] + L[i, 0:j]) % self.modulus
      P[j, :] = (P[j, :] + P[i, :]) % self.modulus
      trials += 1
    return

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

#gcd for python 2 support
def gcd(a, b):
  while b > 0:
    temp = b
    b = a % b
    a = temp
  return a

### Unit Tests ###
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
  except SingularMatrixError:
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
  try:
    A.solve(b)
    assert False
  except SingularMatrixError:
    pass

  #test no solution vs many solutions
  A = ModularMatrix(
      np.array([[3, 4, 0],
                [3, 4, 0],
                [1, 2, 2]], dtype=np.int8), 5)
  b = np.array([2, 2, 3], dtype=np.int8)
  x = A.solve(b, singular_mode="any")
  np.testing.assert_array_equal(A.dot(x), b)

  b_no_sol = np.array([1, 2, 3], dtype=np.int8)
  try:
    A.solve(b_no_sol, singular_mode="any")
    assert False
  except SingularMatrixError:
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

  # matrix inverse
  Ainv = A.inverse()
  I = np.identity(5, dtype=np.int8)
  np.testing.assert_array_equal((A * Ainv).array, I)
  np.testing.assert_array_equal((Ainv * A).array, I)

  # test caching behavior
  assert A.lu_cache
  A[0, 0] = 5
  assert not A.lu_cache
  b = A.dot(x)
  np.testing.assert_array_equal(A.solve(b), x)
  assert A.lu_cache

  # gcd
  assert gcd(10, 2) == 2
  assert gcd(24, 42) == 6
  assert gcd(7, 1) == 1
  assert gcd(35, 128) == 1

  # composite modulus
  # standard transition matrix for 3x3 %6
  A = ModularMatrix(
      np.array( [[1,1,0,1,0,0,0,0,0],
                 [1,1,1,0,1,0,0,0,0],
                 [0,1,1,0,0,1,0,0,0],
                 [1,0,0,1,1,0,1,0,0],
                 [0,1,0,1,1,1,0,1,0],
                 [0,0,1,0,1,1,0,0,1],
                 [0,0,0,1,0,0,1,1,0],
                 [0,0,0,0,1,0,1,1,1],
                 [0,0,0,0,0,1,0,1,1]], dtype=np.int8), 6)
  x = np.array([3,4,5,1,2,2,3,2,1])
  b = (A * x).array
  np.testing.assert_array_equal(A.solve(b), x)

  # standard transition matrix for 2x2 %6
  A = ModularMatrix(
      np.array( [[1, 1, 1, 0],
                 [1, 1, 0, 1],
                 [1, 0, 1, 1],
                 [0, 1, 1, 1]], dtype=np.int8), 6)
  try:
    A.solve(np.ones(4, dtype=np.int8))
    assert False
  except SingularMatrixError:
    pass

  assert A.nullity() == 1
  assert A.rank() == 3

  # edge case of linearly combination being nonsingular
  A = ModularMatrix(
    np.array( [[2, 3, 0],
               [3, 1, 1],
               [3, 0, 2]], dtype=np.int8), 6)

  assert A.nullity() == 0
  I3 = np.identity(3, dtype=np.int8)
  Icalc = (A * A.inverse()).array
  np.testing.assert_array_equal(Icalc, I3)

  print('done!')