from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
from rllab.utils.plot_utils import Arrow3D, plot_loop_pause

import numpy as np
from numpy.linalg import norm
from math import sqrt, asin, atan2, cos, sin, acos
import matplotlib.pyplot as plt

class TendonTwoSegmentEnv(Env):
   """
   class handling the forward kinematics of a two segment tendon driven continuum robot
   l1min, l1max: min max tendon lengths of first segment [m]
   l2min, l2max: min max tendon lenghts of second segment [m]
   d: pitch distance to cable guides [m]:
   n: number of units (spacer discs) within one segment
   """

   precision_digits = 16 # rounding precision needed for handling the singularity at l1=l2=l3
   total_episodes = 0
   total_goals_reached = 0

   def __init__(self):
      self.l1min = 0.085
      self.l1max = 0.115
      self.l2min = 0.185
      self.l2max = 0.215
      self.d = 0.01
      self.n = 10
      self.dependent_actuation = True # I like dependent! indicates the way change of tendon lenghts interact with robot
      self.rewardfn_num = 2 # number of chosen reward fn

      self.base = np.array([0.0, 0.0, 0.0, 1.0]) # base vector used for transformations
      self.l11 = None; self.l12 = None; self.l13 = None; # tendon lengths
      self.l21 = None; self.l22 = None; self.l23 = None; # absolute segment 2 tendon lengths
      self.dl21 = None; self.dl22 = None; self.dl23 = None; # effective  segment 2 tendon lengths (needed for kappa2, phi2, seg_len2)
      self.lengths1 = None # [l11, l12, l13]
      self.lengths2 = None # [l11, l12, l13]

      self.kappa1 = None # curvature kappa [m^(-1)]
      self.kappa2 = None # curvature kappa [m^(-1)]
      self.phi1 = None # angle rotating arc out of x-z plane [rad]
      self.phi2 = None # angle rotating arc out of x-z plane [rad]
      self.seg_len1 = None # total arc length [m]
      self.seg_len2 = None # total arc length [m]

      self.T01 = None # bishop transformation matrix from base to segment 1 tip
      self.T12 = None # bishop transformation matrix from segment 1 tip to segment 2 tip
      self.T02 = None # bishop transformation matrix from base to segment 2 tip
      self.normal_vec1 = None # Frenet: pointing towards center point of arc radius # Bishop: aligned with the base frame
      self.normal_vec2 = None # Frenet: pointing towards center point of arc radius # Bishop: aligned with the base frame
      self.binormal_vec1 = None # segment1 binormal vector
      self.binormal_vec2 = None # robot's tip binormal vector
      self.tangent_vec1 = None # segment1 tangent vector
      self.tangent_vec2 = None # tangent vector robot's tip
      self.tip_vec1 = None # segment1 tip vector [m] [x, y, z]
      self.tip_vec2 = None # robot's tip vector [m] [x, y, z]
      self.old_dist_vec = None # goal-tip_vec in vector form, at time step t
      self.new_dist_vec = None # goal-tip_vec in vector form, at time step t+1 after taking action a_t
      self.old_dist_euclid = None # euclid dist to goal at time step t
      self.new_dist_euclid = None # euclid dist to goal at time step t+1 after taking action a_t

      self.fig = None # fig variable used for plotting
      self.frame = 10000 # used to name frames when save_frames is active in render()

      # variables needed in episodic reinforcement learning
      self._state = None # state vector containing l1, l2, l3, tip position, and goal position
      self.info = {} # used to store adaditional environment data
      self.steps = None # current step the episode is in
      self.goal = None # goal to be reached by the robot's tip [x, y, z] [m]
      self.goal_lengths = None # tendon lengths of generated goal
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
      return Box(low=-np.inf, high=np.inf, shape=(13,))

   @property
   def action_space(self):
      """allowed value range for actions"""
      return Box(low=-self.delta_l, high=self.delta_l, shape=(6,))

   def reset(self, lengths=None, goal=None, tangent_vec_goal=None):
      """ Resets the environment and updates other variables accordingly. Returns state of new episode. """
      self.l11 = np.random.uniform(self.l1min, self.l1max) if lengths is None else lengths[0]
      self.l12 = np.random.uniform(self.l1min, self.l1max) if lengths is None else lengths[1]
      self.l13 = np.random.uniform(self.l1min, self.l1max) if lengths is None else lengths[2]
      self.l21 = np.random.uniform(self.l2min, self.l2max) if lengths is None else lengths[3]
      self.l22 = np.random.uniform(self.l2min, self.l2max) if lengths is None else lengths[4]
      self.l23 = np.random.uniform(self.l2min, self.l2max) if lengths is None else lengths[5]
      self.update_workspace()
      # create goal with a little distance away from tip-vetor
      if goal is None and tangent_vec_goal is None: # create random goal here
         self.goal = self.tip_vec2
         while norm(self.goal-self.tip_vec2) < 2*self.eps:
            self.set_goal() # set a new goal for the episode
      elif goal is not None and tangent_vec_goal is not None: # create goal given by fn params
         assert len(goal) == 3 and len(tangent_vec_goal) == 3
         self.goal = goal
         self.tangent_vec_goal = tangent_vec_goal
      elif goal is not None and tangent_vec_goal is None: # this is the case when goal lenghts are given instead of point in workspace
         assert len(goal) == 6
         self.set_goal(goallengths=goal)
      else:
         raise NotImplementedError

      self._state = self.get_state()
      self.dist_start = norm(self.goal-self.tip_vec2)
      self.dist_min = self.dist_start
      self.anglet_min = self.get_diff_angle()
      self.steps = 0
      self.info["str"] = "Reset the environment."
      self.info["goal"] = False
      return self._state

   def set_goal(self, goallengths=None):
      """ Sets the goal to a random point of the robot's workspace [x, y, z] in [m]
      and sets the tangent vector accordingly."""
      if goallengths is None:
         l11goal = np.random.uniform(self.l1min, self.l1max)
         l12goal = np.random.uniform(self.l1min, self.l1max)
         l13goal = np.random.uniform(self.l1min, self.l1max)
         l21goal = np.random.uniform(self.l2min, self.l2max)
         l22goal = np.random.uniform(self.l2min, self.l2max)
         l23goal = np.random.uniform(self.l2min, self.l2max)
      else:
         l11goal = goallengths[0]; l12goal = goallengths[1]; l13goal = goallengths[2]
         l21goal = goallengths[3]; l22goal = goallengths[4]; l23goal = goallengths[5]
      self.goal_lengths = np.array([l11goal, l12goal, l13goal, l21goal, l22goal, l23goal])
      dl21goal = l21goal-l11goal; dl22goal = l22goal-l12goal; dl23goal = l23goal-l13goal
      kappa1, phi1, seg_len1 = self.arc_params(l11goal, l12goal, l13goal)
      kappa2, phi2, seg_len2 = self.arc_params(dl21goal, dl22goal, dl23goal)
      T01 = self.transformation_matrix(kappa1, phi1, seg_len1)
      T12 = self.transformation_matrix(kappa2, phi2, seg_len2)
      T02 = np.matmul(T01, T12)
      self.goal = np.matmul(T02, self.base)[0:3]
      self.tangent_vec_goal =  T02[0:3, 2]

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
      return np.array([self.l11, self.l12, self.l13, self.l21, self.l22, self.l23,
                       self.tip_vec2[0], self.tip_vec2[1], self.tip_vec2[2],
                       self.goal[0], self.goal[1], self.goal[2], norm(self.goal-self.tip_vec2)])

   def update_workspace(self):
      """updates configuration and work space variables after changing tendon lengths"""
      self.lengths1 = np.array([self.l11, self.l12, self.l13])
      self.lengths2 = np.array([self.l21, self.l22, self.l23])
      self.dl21 = self.l21-self.l11; self.dl22 = self.l22-self.l12; self.dl23 = self.l23-self.l13;
      self.kappa1, self.phi1, self.seg_len1 = self.arc_params(self.l11, self.l12, self.l13)
      self.kappa2, self.phi2, self.seg_len2 = self.arc_params(self.dl21, self.dl22, self.dl23)
      self.T01 = self.transformation_matrix(self.kappa1, self.phi1, self.seg_len1)
      self.T12 = self.transformation_matrix(self.kappa2, self.phi2, self.seg_len2)
      self.T02 = np.matmul(self.T01, self.T12)
      self.normal_vec1 = self.T01[0:3, 0]
      self.normal_vec2 = self.T02[0:3, 0]
      self.binormal_vec1 = self.T01[0:3, 1]
      self.binormal_vec2 = self.T02[0:3, 1]
      self.tangent_vec1 = self.T01[0:3, 2]
      self.tangent_vec2 = self.T02[0:3, 2]
      self.tip_vec1 = np.matmul(self.T01, self.base)[0:3]
      self.tip_vec2 = np.matmul(self.T02, self.base)[0:3]

   def take_action(self, action):
      """executes action at timestep t, and updates configuration/work space
      as well as distance variables"""
      if self.dependent_actuation: # this is what i like
         self.l11 += action[0]; self.l12 += action[1]; self.l13 += action[2]
         self.l21 += action[3]; self.l22 += action[4]; self.l23 += action[5]
         # make sure tendon lengths are within min, max
         lengths1 = [self.l11, self.l12, self.l13]
         lengths1 = [self.l1min if l < self.l1min else l for l in lengths1]
         lengths1 = [self.l1max if l > self.l1max else l for l in lengths1]
         self.l11 = lengths1[0]; self.l12 = lengths1[1]; self.l13 = lengths1[2]
         lengths2 = [self.l21, self.l22, self.l23]
         lengths2 = [self.l2min if l < self.l2min else l for l in lengths2]
         lengths2 = [self.l2max if l > self.l2max else l for l in lengths2]
         self.l21 = lengths2[0]; self.l22 = lengths2[1]; self.l23 = lengths2[2]
      else: # independent acutation
         l11 = self.l11; l12 = self.l12; l13 = self.l13
         l11 += action[0]; l12 += action[1]; l13 += action[2]
         lengths1 = [l11, l12, l13] # make sure actuator limits are not exceeded
         lengths1 = [self.l1min if l < self.l1min else l for l in lengths1]
         lengths1 = [self.l1max if l > self.l1max else l for l in lengths1]
         # calculate the actual change of tendon lenghts for first segment
         a0_actual = lengths1[0]-self.l11; a1_actual = lengths1[1]-self.l12; a2_actual = lengths1[2]-self.l13
         self.l11 = lengths1[0]; self.l12 = lengths1[1]; self.l13 = lengths1[2] # update tendon lenghts of segment 1
         self.l21 += a0_actual; self.l22 += a1_actual; self.l23 += a2_actual # apply actual tendon change of segment 1 to segment 2
         self.l21 += action[3]; self.l22 += action[4]; self.l23 += action[5] # apply action to segment 2
         # make sure tendon lengths for segment 2 are within min, max
         lengths2 = [self.l21, self.l22, self.l23]
         lengths2 = [self.l2min if l < self.l2min else l for l in lengths2]
         lengths2 = [self.l2max if l > self.l2max else l for l in lengths2]
         self.l21 = lengths2[0]; self.l22 = lengths2[1]; self.l23 = lengths2[2]

      self.old_dist_vec = self.goal-self.tip_vec2
      self.old_dist_euclid = norm(self.old_dist_vec)
      self.update_workspace()
      self.new_dist_vec = self.goal-self.tip_vec2
      self.new_dist_euclid = norm(self.new_dist_vec)
      if self.new_dist_euclid < self.dist_min: # update min distance within one episode
         self.dist_min = self.new_dist_euclid
         self.anglet_min = self.get_diff_angle()

   def get_reward_done_info(self, new_dist_euclid, old_dist_euclid, steps):
      """returns reward, done, info dict after taking action"""
      done = False
      alpha = 0.4; c1 = 1; c2 = 100; gamma=0.99; cpot = 50 # reward function params
      # regular step without terminating episode
      if self.rewardfn_num == 1:
         reward = -1+cpot*(-gamma*(new_dist_euclid/self.dist_start)**alpha \
                           + (old_dist_euclid/self.dist_start)**alpha)
      elif self.rewardfn_num == 2:
         reward = cpot*(-gamma*(new_dist_euclid/self.dist_start)**alpha \
                        +(old_dist_euclid/self.dist_start)**alpha)
      elif self.rewardfn_num == 3:
         reward = -c1*((new_dist_euclid/self.dist_start)**alpha)
      elif self.rewardfn_num == 4:
         reward = -c2*(new_dist_euclid-old_dist_euclid)
      elif self.rewardfn_num == 5:
         reward = -c1*(new_dist_euclid/self.dist_start)**alpha \
                  -c2*(new_dist_euclid-old_dist_euclid)
      elif self.rewardfn_num == 6:
         reward = cpot*(-gamma*(new_dist_euclid/self.dist_start)**alpha \
                        +(old_dist_euclid/self.dist_start)**alpha) \
                  -c2*(new_dist_euclid-old_dist_euclid)
      elif self.rewardfn_num == 7:
         reward = gamma*(1-(new_dist_euclid/self.dist_start)**alpha) - \
                        (1-(old_dist_euclid/self.dist_start)**alpha)
      elif self.rewardfn_num == 8:
         reward = 1-((new_dist_euclid/self.dist_start)**alpha) \
                  + cpot*(-gamma*((new_dist_euclid/self.dist_start)**alpha) \
                          +((old_dist_euclid/self.dist_start)**alpha))
      elif self.rewardfn_num == 9:
         reward = 1-((new_dist_euclid/self.eps)**alpha)

      """R7 like R5 but leave out gamma_t at the bottom"""

      self.info["str"] = "Regular step @ {:3d}, dist covered: {:5.2f}" \
                         .format(self.steps, 1000*(new_dist_euclid-old_dist_euclid))

      #terminate episode, when
      # 1. moving too far from goal
      # 2. reaching goal
      # 3. exceeding max steps of environment
      if (new_dist_euclid > self.dist_start + 15*self.eps) or \
         (new_dist_euclid < self.eps) or \
         self.steps >= self.max_steps:

         done = True
         self.dist_end = new_dist_euclid

         if new_dist_euclid > self.dist_start+15*self.eps:
            reward = -1000
            self.info["str"] = "Moving too far away from finish @step {:3d}, start_dist: {:5.2f}mm, end_dist: {:5.2f}mm." \
                               .format(self.steps, 1000*self.dist_start, 1000*norm(self.goal-self.tip_vec2))
         elif new_dist_euclid < self.eps:
            reward = 100
            self.info["goal"] = True
            self.info["str"] = "GOAL! Distance {:.2f}mm @step {:3d}, total distance covered {:.2f}mm." \
                               .format(1000*norm(self.goal-self.tip_vec2), self.steps, 1000*norm(self.dist_start-self.dist_end))
         elif self.steps >= self.max_steps:
            self.info["str"] = "Max steps {}, distance to finish {:5.2f}mm, total distance covered {:5.2f}mm." \
                               .format(self.max_steps, 1000*self.dist_end, 1000*(self.dist_start-self.dist_end))

#      gamma_t = 1-(self.steps/(self.max_steps+1))
#      reward = gamma_t*reward
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
      alpha = acos(np.dot(self.tangent_vec2, self.tangent_vec_goal)) / \
                  (norm(self.tangent_vec2) * norm(self.tangent_vec_goal))
      return alpha*180/np.pi if degree else alpha

   def render(self, mode="human", pause=0.0000001, save_frames=False):
      """ renders the 3d plot of the robot's arc, pause (float) determines how long each frame is shown
          when save frames is set to True each frame of the plot is saved in an png file"""
      if self.steps % 5 == 0 and mode=="string":
         print("STEP {:3d}\tDISTANCE: {:5.2f}mm".format(self.steps, 1000*norm(self.goal-self.tip_vec2)))
      elif mode=="human":
         if self.fig == None:
            self.init_render()

         points1, points2 = self.points_on_arc(100) # points to be plotted from base to robot's tip

         while self.ax.lines:
             self.ax.lines.pop() # delete plots of previous frame
         self.ax.plot(points1[:,0], points1[:,1], points1[:,2], label="Segment 1", c="black", linewidth=3)
         self.ax.plot(points2[:,0], points2[:,1], points2[:,2], label="Segment 2", c="grey", linewidth=2)
         self.ax.plot([self.goal[0]], [self.goal[1]], [self.goal[2]], linestyle=None, label="Ziel", c="magenta", marker="x", markersize=11)
         self.ax.legend()

         # delete arrows of previous frame, except base frame
         while len(self.ax.artists) > 3:
             self.ax.artists.pop()
         # add coordinate frames to plot
         anormal1 = self.create_arrow(self.tip_vec1, self.normal_vec1, self.arrow_len,
                                      "-|>", self.arrow_lw, self.arrow_ms, c="r")
         abinormal1 = self.create_arrow(self.tip_vec1, self.binormal_vec1, self.arrow_len,
                                        "-|>", self.arrow_lw, self.arrow_ms, c="g")
         atangent1 = self.create_arrow(self.tip_vec1, self.tangent_vec1, self.arrow_len,
                                       "-|>", self.arrow_lw, self.arrow_ms, c="b")
         anormal2 = self.create_arrow(self.tip_vec2, self.normal_vec2, self.arrow_len,
                                      "-|>", self.arrow_lw, self.arrow_ms, c="r")
         abinormal2 = self.create_arrow(self.tip_vec2, self.binormal_vec2, self.arrow_len,
                                        "-|>", self.arrow_lw, self.arrow_ms, c="g")
         atangent2 = self.create_arrow(self.tip_vec2, self.tangent_vec2, self.arrow_len,
                                       "-|>", self.arrow_lw, self.arrow_ms, c="b")
         self.ax.add_artist(anormal1)
         self.ax.add_artist(abinormal1)
         self.ax.add_artist(atangent1)
         self.ax.add_artist(anormal2)
         self.ax.add_artist(abinormal2)
         self.ax.add_artist(atangent2)
         atangent_goal = self.create_arrow(self.goal, self.tangent_vec_goal, self.arrow_len,
                                           "-|>", self.arrow_lw, self.arrow_ms, c="b")
         self.ax.add_artist(atangent_goal)
         plot_loop_pause(pause) # updates plot without losing focus
         if save_frames == True:
            self.fig.savefig("./figures/frame"+str(self.frame)[1:]+".png")
            self.frame += 1

   def points_on_arc(self, num_points):
      """ returns np.arrays [num_points, 3] arc points [x(s), y(s), z(s)] [m] """
      points1 = np.zeros((num_points, 3)) # points placeholder for segment 1
      points2 = np.zeros((num_points, 3)) # points placeholder for segment 2
      s1 = np.linspace(0.0, self.seg_len1, num_points) # variable arc length 1
      s2 = np.linspace(0.0, self.seg_len2, num_points) # variable arc length 2
      for i in range(num_points):
         points1[i] = np.matmul(self.transformation_matrix(self.kappa1, self.phi1, s1[i]), self.base)[0:3]
      for i in range(num_points):
         T02_s = np.matmul(self.T01, self.transformation_matrix(self.kappa2, self.phi2, s2[i]))
         points2[i] = np.matmul(T02_s, self.base)[0:3]
      return points1, points2

   def create_arrow(self, start_vec, dir_vec, alen, astyle, alw, ams, c):
      """Returns a 3D arrow pointing to dir_vec from start_vec. dir_vec should be a unit vector"""
      a = Arrow3D([start_vec[0], start_vec[0]+alen*dir_vec[0]],
                  [start_vec[1], start_vec[1]+alen*dir_vec[1]],
                  [start_vec[2], start_vec[2]+alen*dir_vec[2]],
                  arrowstyle=astyle, lw=alw, mutation_scale=ams, color=c)
      return a

   def init_render(self):
      """ sets up 3d plot """
      plt.ion() # interactive plot mode, panning, zooming enabled
      self.fig = plt.figure(figsize=(9.5,7.2))
      self.ax = self.fig.add_subplot(111, projection="3d") # attach z-axis to plot
      # set axe limits and labels
      self.ax.set_xlim([-0.5*self.l2max, 0.5*self.l2max])
      self.ax.set_ylim([-0.5*self.l2max, 0.5*self.l2max])
      self.ax.set_zlim([0.0, self.l2max])
      self.ax.set_xlabel("x in [m]")
      self.ax.set_ylabel("y in [m]")
      self.ax.set_zlabel("z in [m]")
      # add coordinate 3 arrows of base frame, have to be defined once!
      self.arrow_len = 0.03
      self.arrow_lw = 1.5
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

