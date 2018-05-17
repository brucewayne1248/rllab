from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import time

def mypause(interval):
   """ Pause function to be used in plotting loop to keep plot window in background and not lose focus at every frame. """
   backend = plt.rcParams['backend']
   if backend in matplotlib.rcsetup.interactive_bk:
      figManager = matplotlib._pylab_helpers.Gcf.get_active()
      if figManager is not None:
         canvas = figManager.canvas
         if canvas.figure.stale:
            canvas.draw()
         canvas.start_event_loop(interval)
         return

class PointEnv(Env):
   def __init__(self):
      self.fig = None

   @property
   def observation_space(self):
      return Box(low=-np.inf, high=np.inf, shape=(2,))

   @property
   def action_space(self):
      return Box(low=-0.1, high=0.1, shape=(2,))

   def reset(self):
      self._state = np.random.uniform(-1, 1, size=(2,))
      observation = np.copy(self._state)
      return observation

   def step(self, action):
      self._state = self._state + action
      x, y = self._state
      reward = -(x**2 + y**2) ** 0.5
      done = abs(x) < 0.01 and abs(y) < 0.01
      if done: print("#"*20+"DONE"+"#"*20)
      next_observation = np.copy(self._state)
      return Step(observation=next_observation, reward=reward, done=done)

   def render(self, pause=0.000000001):
#      if self.fig == None:
#         self.init_render()

#      while self.ax.lines:
#         self.ax.lines.pop()
#
#      self.ax.plot(self._state[0], self._state[1], c="red", marker="*", markersize=5)
      t = time.time()
      print(self._state)
#      mypause(pause)
      print("pause time", time.time() -t )

   def init_render(self):
      plt.ion()
#      self.fig = plt.figure()
#      self.ax = self.fig.add_subplot(111)
      self.fig, self.ax = plt.subplots()
      self.ax.set(xlabel="x", ylabel="y", title="Point Environment")
      self.ax.grid()
      self.ax.set_xlim([-1, 1])
      self.ax.set_ylim([-1, 1])
      plt.show()
      self.fig.tight_layout()
