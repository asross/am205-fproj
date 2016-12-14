from modular_matrix import *
from transition_helpers import *
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def each_possible_grid(mod, rows, cols):
  # each grid can be thought of as a number in base mod
  for i in range(mod**(rows*cols)):
    grid = np.zeros(rows*cols)
    j = len(grid)-1
    while i > 0:
      grid[j] = i % mod
      i = i // mod
      j -= 1
    yield grid.reshape((rows, cols))

def cycle_for(transition_fn, grid, return_states=False):
  try:
    b = grid.ravel()
    seen = [tuple(b.astype(int))]
    while True:
      b = transition_fn(b)
      sight = tuple(b.astype(int))
      if sight in seen:
        if return_states:
          return len(seen), seen.index(sight), seen
        else:
          return len(seen), seen.index(sight)
      else:
        seen.append(sight)
  except (SingularMatrixError, ValueError):
    return ((0, 0, []) if return_states else (0, 0))

def forward_cycle(A, grid, **kwargs):
  return cycle_for(lambda b: A.dot(b),
      grid, **kwargs)
  
def inverse_cycle(A, grid, **kwargs):
  return cycle_for(lambda b: A.solve(b, singular_mode="any"),
      grid, **kwargs)

def all_forward_cycles(k, n, m):
  A = standard_A(k, n, m)
  return Counter(forward_cycle(A, grid) for grid in each_possible_grid(k, n, m))

def all_inverse_cycles(k, n, m):
  A = standard_A(k, n, m)
  return Counter(inverse_cycle(A, grid) for grid in each_possible_grid(k, n, m))

def is_identity(arr):
  return np.array_equal(arr, np.identity(len(arr), dtype=arr.dtype))

def identity_root(A, maxiters=10000):
  i = 1
  arr = A.new(A.array)
  while not is_identity(arr.array):
    arr = arr * A
    i += 1
    if i > maxiters:
      return None
  return i

def plot_state(s,t=0.1,fs=30):
  plt.axis('off')
  for i in range(len(s)):
    for j in range(len(s[0])):
      plt.gca().text(i-t, j+t, int(s[j][i]), fontsize=fs)
  plt.imshow(s, cmap='summer', interpolation='none')

def draw_state_group_border(wadj):
  sub = plt.gca()
  ax = sub.axis()
  width = ax[1]-ax[0]
  height = ax[3]-ax[2]
  rec = plt.Rectangle(
    (ax[0]-wadj*width, ax[2]-0.1*height),
    (wadj+1.1)*width,
    1.2*height,
    fill=False,lw=1)
  rec = sub.add_patch(rec)
  rec.set_clip_on(False)

def group_into_chains(states, transition_fn):
  chains = []
  for s in states:
    stuple = tuple(s.ravel())
    if any(stuple in chain for chain in chains):
      continue
    chain = []
    while stuple not in chain:
      s = transition_fn(s.ravel()).reshape(s.shape)
      chain.append(tuple(s.ravel()))
    chains.append(chain)
  return chains
