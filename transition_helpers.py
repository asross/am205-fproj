from modular_matrix import *

def transition_matrix(adj_fn, rows, cols, mod):
  A = np.zeros((rows*cols, rows*cols), dtype=np.int8)
  for row in range(rows):
    for col in range(cols):
      for i, j, k in adj_fn(row, col):
        if 0 <= i < rows and 0 <= j < cols:
          A[row+col*rows][i+j*rows] = k
  return ModularMatrix(A, mod)

def intermediate_states(A, presses, initial_state):
  state = initial_state.copy()
  for i, n in enumerate(presses):
    press = np.zeros(len(presses), dtype=presses.dtype)
    press[i] = 1
    effect = A.dot(press)
    for _ in range(n):
      state = (state + effect) % A.modulus
      yield state

def all_solution_states(grid, A):
  presses = -1 * A.solve(grid.ravel()) % A.modulus
  yield grid
  for state in intermediate_states(A, presses, grid.ravel()):
    yield state.reshape(grid.shape)

if __name__ == '__main__':
  print('running tests...')

  # create a mod 3 light array from blank array
  transition_fn = lambda i, j: [[i,j,2],[i+1,j,1],[i-1,j,1],[i,j+1,1],[i,j-1,1]]
  A = transition_matrix(transition_fn, 3, 3, 3)
  grid = np.array([[0,1,1],[1,0,0],[0,1,1]]).ravel()
  presses = A.solve(grid)
  np.testing.assert_array_equal(A.dot(presses), grid)

  print('done!')
