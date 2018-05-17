from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box

#import time
import numpy as np
from numpy.linalg import norm
from math import sqrt, asin, atan2, cos, sin
from rllab.utils.plot_utils import Arrow3D, plot_loop_pause

"""
new reward fn
gamma_t = 1-(self.steps/self.max_steps) # encourages fast completion of task
self.reward = gamma_t * (1-(self.new_dist_euclid/self.dist_start)**0.4)
max steps ~ 150
allowing for greater action space 0.005 m
resets continuum robot completely random in workspace
"""

class TendonEnvOneSegment(Env):
   """
   class handling the forward kinematics of a single segment tendon driven continuum robot
   :lmin, lmax: min and max tendon length [m]:
   :d pitch distance to cable guides [m]:

   n:          number of units (spacer discs) within one segment
   """
   precision_digits = 16 # rounding precision needed for handling the singularity at l1=l2=l3
   total_episodes = 0
   total_goals_reached = 0

   def __init__(self):
      self.lmin = 0.075
      self.lmax = 0.125
      self.d = 0.01
      self.n = 5

      self.l1 = None; self.l2 = None; self.l3 = None; # tendon lengths
      self.lengths = None # [l1, l2, l3]
      self.base = np.array([0.0, 0.0, 0.0, 1.0]) # base vector used for transformations
      self.kappa = None # curvature kappa [m^(-1)]
      self.phi = None # angle rotating arc out of x-z plane [rad]
      self.seg_len = None # total arc length [m]

      self.T01 = None # transformation matrix either Bishop or Frenet frames
      self.normal_vec = None # Frenet: pointing towards center point of arc radius # Bishop: aligned with the base frame
      self.binormal_vec = None # tangent_vec x normal_vec
      self.tangent_vec = None # tangent vector of arc
      self.tip_vec = None # robot's tip vector [m] [x, y, z]
      self.old_dist_vec = None # goal-tip_vec in vector form, at time step t
      self.new_dist_vec = None # goal-tip_vec in vector form, at time step t+1
      self.old_dist_euclid = None # euclid dist to goal at time step t
      self.new_dist_euclid = None # euclid dist to goal at time step t+1

      self.fig = None # fig variable used for plotting

      # variables needed in episodic reinforcement learning
      self._state = None # state vector containing l1, l2, l3, tip position, and goal position
      self.info = {}
#      self.reward = None # current reward
#      self.done = None # indicates that episode is progress or over
#      self.info = None # additional info returned by stepping the environment, indicating goal reached
      self.steps = None # current step the episode is in
      self.goal = None # goal to be reached by the robot's tip [x, y, z] [m]
      self.tangent_vec_goal = None # tangent vector of goal position
#      self.state_dim = 10
#      self.action_dim = 3 # number of actions per timestep
      self.delta_l = 0.001 # max tendon length change per timestep
      self.max_steps = 150 # max steps per episode
      self.eps = 1e-3 # distance tolerance to reach goal
      self.dist_start = None # start distance to goal
      self.dist = None # current distance to goal

   @property
   def observation_space(self):
      """allowed value ranges for states"""
      return Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

   @property
   def action_space(self):
      """allowed value ranges for actions"""
      return Box(low=-self.delta_l, high=self.delta_l, shape=(3,), dtype=np.float32)

   def reset(self):
      """ Resets the environment and updates other variables accordingly. Returns state of new episode. """
      self.l1 = np.random.uniform(self.lmin, self.lmax)
      self.l2 = np.random.uniform(self.lmin, self.lmax)
      self.l3 = np.random.uniform(self.lmin, self.lmax)
      self.update_workspace()
      # create goal with a little distance away from tip-vetor
      self.goal = self.tip_vec
      while norm(self.goal-self.tip_vec) < 2*self.eps:
         self.set_goal() # set a new goal for the episode
      self._state = self.get_state()
      self.dist_start = norm(self.goal-self.tip_vec)
      self.steps = 0
      return self._state

   def set_goal(self):
      """ Sets the goal to a random point of the robot's workspace [x, y, z] in [m]
      and sets the tangent vector accordingly."""
      l1goal = np.random.uniform(self.lmin, self.lmax)
      l2goal = np.random.uniform(self.lmin, self.lmax)
      l3goal = np.random.uniform(self.lmin, self.lmax)
      kappa, phi, seg_len = self.configuration_space(l1goal, l2goal, l3goal)
      T01 = self.transformation_matrix(kappa, phi, seg_len)
      self.goal = np.matmul(T01, self.base)[0:3]
      self.tangent_vec_goal =  T01[0:3, 2]

   def step(self, action):
      """Steps the environment and returns new state, reward, done, info."""
      self.steps += 1
      done = False

      self.l1 += action[0]; self.l2 += action[1]; self.l3 += action[2]
      # make sure tendon lengths are within min, max
      lengths = [self.l1, self.l2, self.l3]
      for i in range(len(lengths)):
         if lengths[i] < self.lmin:
            lengths[i] = self.lmin
         elif lengths[i] > self.lmax:
            lengths[i] = self.lmax
      self.l1 = lengths[0]; self.l2 = lengths[1]; self.l3 = lengths[2]

      old_dist = self.goal-self.tip_vec; self.old_dist = old_dist
      old_dist_euclid = norm(old_dist); self.old_dist_euclid = old_dist_euclid
      self.update_workspace()
      # handling regular step
      new_dist = self.goal-self.tip_vec; self.new_dist = new_dist
      new_dist_euclid = norm(new_dist); self.new_dist_euclid = new_dist_euclid
      self._state = self.get_state()

      # first term punishing being far away from goal and rewarding being close to goal
      # second term punishing/rewarding moving away from/towards goal
      reward = (1-(self.new_dist_euclid/self.dist_start)**0.4) \
               -100*(self.new_dist_euclid-self.old_dist_euclid)
      info = "EPISODE RUNNING @STEP {} DISTANCE: {:5.2f}mm".format(self.steps, 1000*new_dist_euclid)

      # moving too far away from goal
      if norm(self.tip_vec-self.goal) > self.dist_start + 10*self.eps:
         done = True
         reward = -100
         self.dist_end = norm(self.tip_vec-self.goal)
         info = "Moving too far away from goal @step {}, start_dist: {:5.2f}mm, end_dist: {:5.2f}mm".format(self.steps, 1000*self.dist_start, 1000*norm(self.goal-self.tip_vec))

      # handling case when max steps are exceeded
      if self.steps >= self.max_steps:
         self.dist_end = norm(self.goal-self.tip_vec)
         done = True
         info = "MAX STEPS {} REACHED, DISTANCE {:5.2f}mm COVERED {:5.2f}mm." \
                .format(self.max_steps, 1000*self.dist_end, 1000*(self.dist_start-self.dist_end))

      # handling goal reaching case
      if norm(self.tip_vec-self.goal) < self.eps:
         reward = 100
         done = True
         self.dist_end = norm(self.goal-self.tip_vec)
         info = "GOAL!!! DISPLACEMENT {:.2f}mm @step {}, COVERED {:.2f}" \
                .format(1000*norm(self.goal-self.tip_vec), self.steps, 1000*norm(self.dist_start-self.dist_end))

      gamma_t = 1-(self.steps/self.max_steps) # encourages fast completion of tasks
      reward = gamma_t * reward

      if done == True:
         self.total_episodes += 1
         if "GOAL" in info:
            self.total_goals_reached += 1
            print("{}/{} goals reached.".format(self.total_goals_reached, self.total_episodes))


      next_observeration = np.copy(self._state)
      return Step(observation=next_observeration, reward=reward, done=done)

   def get_state(self):
      return np.array([self.l1, self.l2, self.l3, self.tip_vec[0], self.tip_vec[1], self.tip_vec[2],
                       self.goal[0], self.goal[1], self.goal[2], norm(self.goal-self.tip_vec)])

   def update_workspace(self):
      """ updates configuration and work space variables after changing tendon lengths """
      self.lengths = np.array([self.l1, self.l2, self.l3])
      self.kappa, self.phi, self.seg_len = self.configuration_space(self.l1, self.l2, self.l3)
      self.T01 = self.transformation_matrix(self.kappa, self.phi, self.seg_len, frame="bishop")
      self.normal_vec = self.T01[0:3, 0]
      self.binormal_vec = self.T01[0:3, 1]
      self.tangent_vec = self.T01[0:3, 2]
      self.tip_vec = np.matmul(self.T01, self.base)[0:3]

   def configuration_space(self, l1, l2, l3):
      # useful expressions to shorten formulas below
      lsum = l1+l2+l3
      expr = l1**2+l2**2+l3**2-l1*l2-l1*l3-l2*l3
      # in rare cases expr ~ +-1e-17 when l1~l2~l3 due to floating point operations
      # in these cases expr has to be set to 0.0 in order to handle the singularity
      if round(abs(expr), self.precision_digits) == 0:
         expr = 0.0
      kappa = 2*sqrt(expr) / (self.d*lsum)
      phi = atan2(sqrt(3)*(l2+l3-2*l1), 3*(l2-l3))
      # calculate total segment length
      if l1 == l2 == l3 or expr == 0.0: # handling the singularity
         seg_len = lsum / 3
      else:
         seg_len = self.n*self.d*lsum / sqrt(expr) * asin(sqrt(expr)/(3*self.n*self.d))
      return kappa, phi, seg_len

   def transformation_matrix(self, kappa, phi, s, frame="bishop"):
      if round(kappa, self.precision_digits) == 0.0: #handling singularity
         T = np.identity(4)
         T[2, 3] = s
      else:
         if frame == "bishop":
            T = np.array([[cos(phi)**2*(cos(kappa*s)-1)+1, sin(phi)*cos(phi)*(cos(kappa*s)-1), cos(phi)*sin(kappa*s), cos(phi)*(1-cos(kappa*s))/kappa],
                          [sin(phi)*cos(phi)*(cos(kappa*s)-1), cos(phi)**2*(1-cos(kappa*s))+cos(kappa*s), sin(phi)*sin(kappa*s), sin(phi)*(1-cos(kappa*s))/kappa],
                          [-cos(phi)*sin(kappa*s), -sin(phi)*sin(kappa*s), cos(kappa*s), sin(kappa*s)/kappa],
                          [0, 0, 0, 1]])
         elif frame == "frenet":
            T = np.array([[cos(phi)*cos(kappa*s), -sin(phi), cos(phi)*sin(kappa*s), cos(phi)*(1-cos(kappa*s))/kappa],
                          [sin(phi)*cos(kappa*s),  cos(phi), sin(phi)*sin(kappa*s), sin(phi)*(1-cos(kappa*s))/kappa],
                          [-sin(kappa*s), 0, cos(kappa*s), sin(kappa*s)/kappa],
                          [0, 0, 0, 1]])
         else:
            raise NotImplementedError('Use frame="bishop" or frame="frenet"')
      return T

   def render(self, pause=0.0000001, frame="bishop", save_frames=False):
      """ renders the 3d plot of the robot's arc, pause (float) determines how long each frame is shown
          when save frames is set to True each frame of the plot is saved in an png file"""
      if self.steps % 5 == 0:
         print("STEP{:3d}\tDISTANCE: {:5.2f}mm".format(self.steps, 1000*norm(self.goal-self.tip_vec)))
