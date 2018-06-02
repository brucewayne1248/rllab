from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
from rllab.utils.plot_utils import Arrow3D, plot_loop_pause

import numpy as np
from numpy.linalg import norm
from math import sqrt, asin, atan2, cos, sin, acos
import matplotlib.pyplot as plt

"""
new reward fn
gamma_t = 1-(self.steps/self.max_steps) # encourages fast completion of task
self.reward = gamma_t * (1-(self.new_dist_euclid/self.dist_start)**0.4)
max steps ~ 150
allowing for greater action space 0.005 m
resets continuum robot completely random in workspace
"""

class TendonOneSegmentEnv(Env):
   """
   class handling the forward kinematics of a single segment tendon driven continuum robot
   lmin, lmax: min and max tendon length [m]:
   d: pitch distance to cable guides [m]:
   n: number of units (spacer discs) within one segment
   """
   precision_digits = 16 # rounding precision needed for handling the singularity at l1=l2=l3
   total_episodes = 0
   total_goals_reached = 0

   def __init__(self):
      self.lmin = 0.085
      self.lmax = 0.115
      self.d = 0.01
      self.n = 10

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
      self.new_dist_vec = None # goal-tip_vec in vector form, at time step t+1 after taking action a_t
      self.old_dist_euclid = None # euclid dist to goal at time step t
      self.new_dist_euclid = None # euclid dist to goal at time step t+1 after taking action a_t

      self.fig = None # fig variable used for plotting
      self.frame = 10000 # used to save frames

      # variables needed in episodic reinforcement learning
      self._state = None # state vector containing l1, l2, l3, tip position, and goal position
      self.info = {} # used to store adaditional environment data
      self.steps = None # current step the episode is in
      self.goal = None # goal to be reached by the robot's tip [x, y, z] [m]
      self.tangent_vec_goal = None # tangent vector of goal position
      self.delta_l = 0.001 # max tendon length change per timestep
      self.max_steps = 100 # max steps per episode
      self.eps = 1e-3 # distance tolerance to reach goal
      self.dist_start = None # start distance to goal
      self.dist_end = None # end distance to goal of episode
      self.dist_min = None # min distance to goal reached within each episode
      self.dist = None # current distance to goal

   @property
   def observation_space(self):
      """allowed value range for states"""
      return Box(low=-np.inf, high=np.inf, shape=(10,))

   @property
   def action_space(self):
      """allowed value range for actions"""
      return Box(low=-self.delta_l, high=self.delta_l, shape=(3,))

   def reset(self, lengths=None, goal=None, tangent_vec_goal=None):
      """ Resets the environment and updates other variables accordingly. Returns state of new episode. """
      self.l1 = np.random.uniform(self.lmin, self.lmax) if lengths is None else lengths[0]
      self.l2 = np.random.uniform(self.lmin, self.lmax) if lengths is None else lengths[1]
      self.l3 = np.random.uniform(self.lmin, self.lmax) if lengths is None else lengths[2]
      self.update_workspace()
      # create goal with a little distance away from tip-vetor
      if goal is None and tangent_vec_goal is None:
         self.goal = self.tip_vec
         while norm(self.goal-self.tip_vec) < 2*self.eps:
            self.set_goal() # set a new goal for the episode
      elif goal is not None and tangent_vec_goal is not None:
         self.goal = goal
         self.tangent_vec_goal = tangent_vec_goal
      else:
         raise NotImplementedError

      self._state = self.get_state()
      self.dist_start = norm(self.goal-self.tip_vec)
      self.dist_min = self.dist_start
      self.steps = 0
      self.info["str"] = "Reset the environment."
      self.info["goal"] = False
      return self._state

   def set_goal(self):
      """ Sets the goal to a random point of the robot's workspace [x, y, z] in [m]
      and sets the tangent vector accordingly."""
      l1goal = np.random.uniform(self.lmin, self.lmax)
      l2goal = np.random.uniform(self.lmin, self.lmax)
      l3goal = np.random.uniform(self.lmin, self.lmax)
      kappa, phi, seg_len = self.arc_params(l1goal, l2goal, l3goal)
      T01 = self.transformation_matrix(kappa, phi, seg_len)
      self.goal = np.matmul(T01, self.base)[0:3]
      self.tangent_vec_goal =  T01[0:3, 2]

   def step(self, action):
      """Steps the environment and returns new state, reward, done, info."""
      self.steps += 1

      self.take_action(action) # changes tendon lengths and updates configuration/work space

      self._state = self.get_state()

      reward, done, self.info = self.get_reward_done_info(self.new_dist_euclid, self.old_dist_euclid, self.steps)

      if done == True:
         self.total_episodes += 1
         if "GOAL" in self.info['str']:
            self.total_goals_reached += 1
         if self.total_episodes % 100 == 0:
            print("-->{}/{} goals reached.".format(self.total_goals_reached, self.total_episodes))

      next_observeration = np.copy(self._state)
      return Step(observation=next_observeration, reward=reward, done=done, info=self.info)

   def get_state(self):
      """returns state consisting of tendon lengths, workspace coordinates,
      goal coordinates, euclidean distance to goal"""
      return np.array([self.l1, self.l2, self.l3, self.tip_vec[0], self.tip_vec[1], self.tip_vec[2],
                       self.goal[0], self.goal[1], self.goal[2], norm(self.goal-self.tip_vec)])
#      return np.array([self.l1, self.l2, self.l3, self.goal[0], self.goal[1], self.goal[2], norm(self.goal-self.tip_vec)])

   def update_workspace(self):
      """updates configuration and work space variables after changing tendon lengths"""
      self.lengths = np.array([self.l1, self.l2, self.l3])
      self.kappa, self.phi, self.seg_len = self.arc_params(self.l1, self.l2, self.l3)
      self.T01 = self.transformation_matrix(self.kappa, self.phi, self.seg_len)
      self.normal_vec = self.T01[0:3, 0]
      self.binormal_vec = self.T01[0:3, 1]
      self.tangent_vec = self.T01[0:3, 2]
      self.tip_vec = np.matmul(self.T01, self.base)[0:3]

   def take_action(self, action):
      """executes action at timestep t, and updates configuration/work space
      as well as distance variables"""
      self.l1 += action[0]; self.l2 += action[1]; self.l3 += action[2]
      # make sure tendon lengths are within min, max
      lengths = [self.l1, self.l2, self.l3]
      for i in range(len(lengths)):
         if lengths[i] < self.lmin:
            lengths[i] = self.lmin
         elif lengths[i] > self.lmax:
            lengths[i] = self.lmax
      self.l1 = lengths[0]; self.l2 = lengths[1]; self.l3 = lengths[2]

      self.old_dist_vec = self.goal-self.tip_vec
      self.old_dist_euclid = norm(self.old_dist_vec)
      self.update_workspace()
      self.new_dist_vec = self.goal-self.tip_vec
      self.new_dist_euclid = norm(self.new_dist_vec)
      if self.new_dist_euclid < self.dist_min:
         self.dist_min = self.new_dist_euclid

   def get_reward_done_info(self, new_dist_euclid, old_dist_euclid, steps):
      """returns reward, done, info dict after taking action"""
      done = False
      alpha = 0.4; c1 = 1; c2 = 100; gamma=0.99 # reward function params
      # regular step without terminating episode
      """R1"""
#      reward = c1*(1-(new_dist_euclid/self.dist_start)**alpha) \
#               -c2*(new_dist_euclid-old_dist_euclid)
      """R2"""
#      reward = c1*(1-(new_dist_euclid/self.dist_start)**alpha)
      """R3"""
      reward = -c1*((new_dist_euclid/self.dist_start)**alpha)
      """R4"""
#      reward = -gamma*((new_dist_euclid/self.dist_start)**alpha) + (old_dist_euclid/self.dist_start)**alpha
      """R5"""
#      reward = -((new_dist_euclid/self.dist_start)**alpha) + (old_dist_euclid/self.dist_start)**alpha
      """R6 like R5 leave out gamma_t"""

      self.info["str"] = "Regular step @ {:3d}, dist covered: {:5.2f}" \
                         .format(self.steps, 1000*(new_dist_euclid-old_dist_euclid))

      #terminate episode, when
      # 1. moving too far from goal
      # 2. reaching goal
      # 3. exceeding max steps of environment
      if (new_dist_euclid > self.dist_start + 10*self.eps) or \
         (new_dist_euclid < self.eps) or \
         self.steps >= self.max_steps:

         done = True
         self.dist_end = new_dist_euclid

         if new_dist_euclid > self.dist_start+10*self.eps:
            reward = -100
            self.info["str"] = "Moving too far away from goal @step {:3d}, start_dist: {:5.2f}mm, end_dist: {:5.2f}mm." \
                               .format(self.steps, 1000*self.dist_start, 1000*norm(self.goal-self.tip_vec))
         elif new_dist_euclid < self.eps:
            reward = 100
            self.info["goal"] = True
            self.info["str"] = "GOAL! Distance {:.2f}mm @step {:3d}, total distance covered {:.2f}mm." \
                               .format(1000*norm(self.goal-self.tip_vec), self.steps, 1000*norm(self.dist_start-self.dist_end))
         elif self.steps >= self.max_steps:
            self.info["str"] = "Max steps {}, distance to goal {:5.2f}mm, total distance covered {:5.2f}mm." \
                               .format(self.max_steps, 1000*self.dist_end, 1000*(self.dist_start-self.dist_end))

      gamma_t = 1-(self.steps/(self.max_steps+1))
      reward = gamma_t*reward
      return reward, done, self.info

   def arc_params(self, l1, l2, l3):
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

   def transformation_matrix(self, kappa, phi, s):
      if round(kappa, self.precision_digits) == 0.0: #handling singularity
         T = np.identity(4)
         T[2, 3] = s
      else:
         T = np.array([[cos(phi)**2*(cos(kappa*s)-1)+1, sin(phi)*cos(phi)*(cos(kappa*s)-1), cos(phi)*sin(kappa*s), cos(phi)*(1-cos(kappa*s))/kappa],
                       [sin(phi)*cos(phi)*(cos(kappa*s)-1), cos(phi)**2*(1-cos(kappa*s))+cos(kappa*s), sin(phi)*sin(kappa*s), sin(phi)*(1-cos(kappa*s))/kappa],
                       [-cos(phi)*sin(kappa*s), -sin(phi)*sin(kappa*s), cos(kappa*s), sin(kappa*s)/kappa],
                       [0, 0, 0, 1]])
      return T

   def get_diff_angle(self, degree=False):
      alpha = acos(np.dot(self.tangent_vec, self.tangent_vec_goal)) / \
                  (norm(self.tangent_vec) * norm(self.tangent_vec_goal))
      return alpha*180/np.pi if degree else alpha

   def render(self, mode="human", pause=0.0000001, save_frames=False):
      """ renders the 3d plot of the robot's arc, pause (float) determines how long each frame is shown
          when save frames is set to True each frame of the plot is saved in an png file"""
      if self.steps % 5 == 0 and mode=="string":
         print("STEP {:3d}\tDISTANCE: {:5.2f}mm".format(self.steps, 1000*norm(self.goal-self.tip_vec)))
      elif mode=="human":
         if self.fig == None:
            self.init_render()

         points = self.points_on_arc(self.kappa, 100) # points to be plotted from base to robot's tip

         while self.ax.lines:
             self.ax.lines.pop() # delete plots of previous frame
         self.ax.plot(points[:,0], points[:,1], points[:,2], label="Segment 1", c="black", linewidth=4)
         self.ax.plot([self.goal[0]], [self.goal[1]], [self.goal[2]], linestyle=None, label="Ziel", c="magenta", marker="x", markersize=8)
         self.ax.legend()

         # delete arrows of previous frame, except base frame
         while len(self.ax.artists) > 3:
             self.ax.artists.pop()
         # add current frenet or bishop coordinate frame in plot
         normal_vec = self.normal_vec
         binormal_vec = self.binormal_vec
         tangent_vec = self.tangent_vec

         anormal = Arrow3D([self.tip_vec[0], self.tip_vec[0]+self.arrow_len*normal_vec[0]],
                           [self.tip_vec[1], self.tip_vec[1]+self.arrow_len*normal_vec[1]],
                           [self.tip_vec[2], self.tip_vec[2]+self.arrow_len*normal_vec[2]],
                           arrowstyle="-|>", lw=self.arrow_lw, mutation_scale=self.arrow_ms, color="r")
         abinormal = Arrow3D([self.tip_vec[0], self.tip_vec[0]+self.arrow_len*binormal_vec[0]],
                             [self.tip_vec[1], self.tip_vec[1]+self.arrow_len*binormal_vec[1]],
                             [self.tip_vec[2], self.tip_vec[2]+self.arrow_len*binormal_vec[2]],
                             arrowstyle="-|>", lw=self.arrow_lw, mutation_scale=self.arrow_ms, color="g")
         atangent = Arrow3D([self.tip_vec[0], self.tip_vec[0]+self.arrow_len*tangent_vec[0]],
                            [self.tip_vec[1], self.tip_vec[1]+self.arrow_len*tangent_vec[1]],
                            [self.tip_vec[2], self.tip_vec[2]+self.arrow_len*tangent_vec[2]],
                            arrowstyle="-|>", lw=self.arrow_lw, mutation_scale=self.arrow_ms, color="b")
         self.ax.add_artist(anormal)
         self.ax.add_artist(abinormal)
         self.ax.add_artist(atangent)
         # tangent vector indicating orientation of goal point
         atangent_goal = Arrow3D([self.goal[0], self.goal[0]+self.arrow_len*self.tangent_vec_goal[0]],
                                 [self.goal[1], self.goal[1]+self.arrow_len*self.tangent_vec_goal[1]],
                                 [self.goal[2], self.goal[2]+self.arrow_len*self.tangent_vec_goal[2]],
                                 arrowstyle="fancy", lw=self.arrow_lw, mutation_scale=0.75*self.arrow_ms, color="cyan")
         self.ax.add_artist(atangent_goal)
         plot_loop_pause(pause) # updates plot without losing focus
         if save_frames == True:
             self.fig.savefig("./figures/frame"+str(self.frame)[1:]+".png")
             self.frame += 1

   def points_on_arc(self, kappa, num_points):
      """ returns np.array([num_points, 3]) of arc points [x(s), y(s), z(s)] for plot [m] """
      points = np.zeros((num_points, 3))
      s = np.linspace(0, self.seg_len, num_points)
      for i in range(num_points):
         points[i] = np.matmul(self.transformation_matrix(self.kappa, self.phi, s[i]),
                               np.array([0.0, 0.0, 0.0, 1]))[0:3]
      return points

   def init_render(self):
      """ sets up 3d plot """
      plt.ion() # interactive plot mode, panning, zooming enabled
      self.fig = plt.figure(figsize=(9.5,7.2))
      self.ax = self.fig.add_subplot(111, projection="3d") # attach z-axis to plot
      # set axe limits and labels
      self.ax.set_xlim([-0.5*self.lmax, 0.5*self.lmax])
      self.ax.set_ylim([-0.5*self.lmax, 0.5*self.lmax])
      self.ax.set_zlim([0.0, self.lmax])
      self.ax.set_xlabel("x in [m]")
      self.ax.set_ylabel("y in [m]")
      self.ax.set_zlabel("z in [m]")
      # add coordinate 3 arrows of base frame, have to be defined once!
      self.arrow_len = 0.02
      self.arrow_lw = 1
      self.arrow_ms = 10
      ax_base = Arrow3D([0.0, self.arrow_len], [0.0, 0.0], [0.0, 0.0],
                        arrowstyle="-|>", lw=self.arrow_lw, mutation_scale=self.arrow_ms, color="r")
      ay_base = Arrow3D([0.0, 0.0], [0.0, self.arrow_len], [0.0, 0.0],
                        arrowstyle="-|>", lw=self.arrow_lw, mutation_scale=self.arrow_ms, color="g")
      az_base = Arrow3D([0.0, 0.0], [0.0, 0.0], [0.0, self.arrow_len],
                        arrowstyle="-|>", lw=self.arrow_lw, mutation_scale=self.arrow_ms, color="b")
      self.ax.add_artist(ax_base)
      self.ax.add_artist(ay_base)
      self.ax.add_artist(az_base)
      plt.show() # display figure and bring focus (once) to plotting window
      self.fig.tight_layout() # fits the plot to window size

#diffs10_5 = []
#env = TendonOneSegmentEnv()
#env.reset()
#episodes = 10000
#for episode in range(episodes):
##   state, reward, done, info = env.step(np.random.uniform(-env.delta_l, env.delta_l, 3))
##   state, reward, done, info = env.step(env.delta_l*np.ones(3))
#   state, reward, done, info = env.step(np.random.uniform(-env.delta_l, env.delta_l, 3))
##   env.render(mode="human")
#   if done:
#      episode += 1
#      env.reset()
#
#print("MEAN", np.mean(np.array(diffs10_5)))
#print("MAX", np.max(np.array(diffs10_5)))
#print("MIN", np.min(np.array(diffs10_5)))