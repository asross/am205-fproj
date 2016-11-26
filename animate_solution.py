from modular_matrix import *
import os
import matplotlib.pyplot as plt

def animate_solution(grid, transitions, mod, fname='images/img'):
  os.system('mkdir -p $(dirname {})'.format(fname))
  (n, m) = grid.shape
  b = grid.ravel()
  x = modmat_solve(transitions, b, mod)
  x = (mod - x) % mod
  i = 0
  plt.imshow(b.reshape(grid.shape), cmap='summer', interpolation='none')
  plt.title('Step 0')
  plt.savefig('{}{:05d}'.format(fname, i))
  for j, val in enumerate(x):
    for k in range(val):
      press = np.zeros(len(x), dtype=np.int8)
      press[j] = 1
      effect = modmat_dot(transitions, press, mod)
      b = (b + effect) % mod
      i += 1
      plt.imshow(b.reshape(grid.shape), cmap='summer', interpolation='none')
      plt.title('Step {}'.format(i))
      plt.savefig('{}{:05d}'.format(fname, i))
  if os.system('which ffmpeg') == 0:
    os.system('ffmpeg -framerate 5 -i {}%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p $(dirname {})/solution.mp4'.format(fname, fname))

if __name__ == '__main__':
  grid = np.array([
    [0, 1, 0, 1, 0, 2],
    [1, 0, 1, 0, 2, 0],
    [0, 1, 0, 2, 0, 1],
    [1, 0, 2, 0, 1, 0],
    [0, 2, 0, 1, 0, 1],
    [2, 0, 1, 0, 1, 0]
  ])
  adj = lambda i, j: [[i,j,2],[i+1,j,1],[i-1,j,1],[i,j+1,1],[i,j-1,1]]
  animate_solution(grid, transition_matrix(adj, *grid.shape), 3)
