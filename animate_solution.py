from modular_matrix import *
from transition_helpers import *
import os
import matplotlib.pyplot as plt
import scipy.misc

def animate_solution(grid, A, fname='images/img', framerate=5):
  os.system('mkdir -p $(dirname {})'.format(fname))
  i = 0
  for state in all_solution_states(grid, A):
    plt.imshow(state, cmap='summer', interpolation='none')
    plt.title('Step {}'.format(i))
    plt.savefig('{}{:05d}'.format(fname, i))
    i += 1
  if os.system('which ffmpeg') == 0:
    os.system('ffmpeg -framerate {} -i {}%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p $(dirname {})/solution.mp4'.format(framerate, fname, fname))

def image_grid(filename, resolution):
    return (scipy.misc.imread(filename, mode='L') * ((resolution-1) / 255.0)).round().astype(np.int8)

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
  animate_solution(grid, transition_matrix(adj, *grid.shape, 3))

  # grid = image_grid('harvard-icon.png', 8)
  # adj = lambda i, j: [[i,j,7],[i+1,j,3],[i-1,j,2],[i,j+1,3],[i,j-1,4]]
  # animate_solution(grid, transition_matrix(adj, *grid.shape, 8), fname='harvard/img', framerate=50)
