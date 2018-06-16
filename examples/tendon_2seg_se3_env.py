from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
from rllab.utils.plot_utils import Arrow3D, plot_loop_pause

import numpy as np
from numpy.linalg import norm
from math import sqrt, asin, atan2, cos, sin, acos
import matplotlib.pyplot as plt

class TendonTwoSegmentSE3Env(Env):
   """
   class handling the forward kinematics also taking orientation into account
   of a two segment tendon driven continuum robot
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
      self.dependent_actuation = True
      self.rewardfn_num = 8 # choose which reward fn should be used

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
      self.dist_vec_new = None # goal-tip_vec in vector form, at time step t+1 after taking action a_t
      self.dist_vec_old = None # goal-tip_vec in vector form, at time step t
      self.dist_euclid_new = None # euclid dist to goal at time step t+1 after taking action a_t
      self.dist_euclid_old = None # euclid dist to goal at time step t

      self.Rnew = None; self.Pnew = None; self.Ynew = None # RPY angles at timestep t+1
      self.Rold = None; self.Pold = None; self.Yold = None # RPY angles at timestep t
      self.Rgoal = None; self.Pgoal = None; self.Ygoal = None # RPY goal angles
      self.Rmin = None; self.Pmin = None; self.Ymin = None # min RPY angles difference between goal's and robot's RPY

      self.anglen_new = None; self.angleb_new = None; self.anglet_new = None # angles between normal, binormal, tangent vector of goal and tip vec at timestep t+1
      self.anglen_old = None; self.angleb_old = None; self.anglet_old = None # angles between normal, binormal, tangent vector of goal and tip vec at timestep t
      self.anglen_min = None; self.angleb_min = None; self.anglet_min = None # min angles between normal, binormal and tangent vectors

      self.qnew = None; self.qold = None # robot's tip quaternion at timestep t+1 and t
      self.qmin = None # min difference between goal's and robot's quaternion
      self.qgoal = None # goal quaternion

      # variables needed in episodic reinforcement learning
      self._state = None # state vector containing l1, l2, l3, tip position, and goal position
      self.info = {} # used to store adaditional environment data
      self.steps = None # current step the episode is in
      self.goal = None # goal to be reached by the robot's tip [x, y, z] [m]
      self.normal_vec_goal = None # normal vector of goal position
      self.binormal_vec_goal = None # binormal vector of goal position
      self.tangent_vec_goal = None # tangent vector of goal position
      self.goal_lengths = None # tendon lengths of generated goal
      self.delta_l = 0.001 # max change of tendon length per timestep
      self.max_steps = 75 # max steps per episode
      self.eps_dist = 1e-3 # distance tolerance to reach goal
      self.eps_angle = 5 * np.pi / 180 # angle tolerance between tangent_vec_goal and tangent_vec2 in rad
      self.eps_q = None # quaternion tolerance ...
      self.dist_start = None # euclidean start distance to goal
      self.dist_end = None # euclidean end distance to goal of episode
      self.dist_min = None # euclidean mmin distance to goal reached within each episode
      self.fig = None # fig variable used for plotting
      self.frame = 10000

   @property
   def observation_space(self):
      """allowed value range for states"""
      return Box(low=-np.inf, high=np.inf, shape=(20,))

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
         while norm(self.goal-self.tip_vec2) < 2*self.eps_dist:
            self.set_goal() # set a new goal for the episode
      elif goal is not None and tangent_vec_goal is not None: # create goal given by fn params
         self.goal = goal
         self.tangent_vec_goal = tangent_vec_goal
      else:
         raise NotImplementedError

      self.dist_start = norm(self.goal-self.tip_vec2)
      self.dist_min = self.dist_start

      self.anglen_new, self.angleb_new, self.anglet_new = self.get_diff_angles() # angles between vectors
      self.anglen_min, self.angleb_min, self.anglet_min = self.anglen_new, self.angleb_new, self.anglet_new

      self._state = self.get_state()

      self.steps = 0
      self.info["str"] = "Reset the environment."
      self.info["goal"] = False
      return self._state

   def set_goal(self):
      """ Sets the goal to a random point of the robot's workspace [x, y, z] in [m]
      and sets the tangent vector accordingly."""
      l11goal = np.random.uniform(self.l1min, self.l1max)
      l12goal = np.random.uniform(self.l1min, self.l1max)
      l13goal = np.random.uniform(self.l1min, self.l1max)
      l21goal = np.random.uniform(self.l2min, self.l2max)
      l22goal = np.random.uniform(self.l2min, self.l2max)
      l23goal = np.random.uniform(self.l2min, self.l2max)
      dl21goal = l21goal-l11goal; dl22goal = l22goal-l12goal; dl23goal = l23goal-l13goal
      kappa1, phi1, seg_len1 = self.arc_params(l11goal, l12goal, l13goal)
      kappa2, phi2, seg_len2 = self.arc_params(dl21goal, dl22goal, dl23goal)
      T01 = self.transformation_matrix(kappa1, phi1, seg_len1)
      T12 = self.transformation_matrix(kappa2, phi2, seg_len2)
      T02 = np.matmul(T01, T12)
      self.goal_lengths = np.array([l11goal, l12goal, l13goal, l21goal, l22goal, l23goal])
      self.goal = np.matmul(T02, self.base)[0:3]
      self.normal_vec_goal = T02[0:3, 0]
      self.binormal_vec_goal = T02[0:3, 1]
      self.tangent_vec_goal =  T02[0:3, 2]
      # calculate RPY angles of goal here or quaternion

   def step(self, action):
      """Steps the environment and returns new state, reward, done, info."""
      self.steps += 1
      self.take_action(action) # changes tendon lengths and updates configuration/work space
      self._state = self.get_state()
      reward, done, self.info = self.get_reward_done_info()

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
                       self.goal[0], self.goal[1], self.goal[2], norm(self.goal-self.tip_vec2),
                       self.tangent_vec2[0], self.tangent_vec2[1], self.tangent_vec2[2],
                       self.tangent_vec_goal[0], self.tangent_vec_goal[1], self.tangent_vec_goal[2],
                       self.anglet_new])

   def update_workspace(self):
      """Updates configuration and work space variables after changing tendon lengths"""
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
      if self.dependent_actuation: # actuator decoupled actuation
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
      else: # actuator coupled actuation
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

      self.dist_vec_old = self.goal-self.tip_vec2
      self.dist_euclid_old = norm(self.dist_vec_old)
      self.anglen_old, self.angleb_old, self.anglet_old = self.get_diff_angles()
      self.update_workspace() # timestep t to t+1 for workspace variables
      self.dist_vec_new = self.goal-self.tip_vec2
      self.dist_euclid_new = norm(self.dist_vec_new)
      self.anglen_new, self.angleb_new, self.anglet_new = self.get_diff_angles()
      if self.dist_euclid_new < self.dist_min: # update min distance within one episode
         self.dist_min = self.dist_euclid_new

   def get_reward_done_info(self):
      """returns reward, done, info dict after taking action"""
      done = False
      # get values needed to calculate rewards
      dnew = self.dist_euclid_new; dold = self.dist_euclid_old; dstart = self.dist_start # new and old euclidean distances
      atnew = self.anglet_new; atold = self.anglet_old # new and old angle between tangent vectors
      # reward fn params
      alpha = 0.4; beta = 0.4; alphap = 0.4; alphao = 0.1;
      wp = 1; # 'w'eight of total 'p'osition reward
      wo = (1-(dnew/dstart))**beta if dnew < dstart else 0 # 'w'eight of total 'o'rientation reward
      wsp = 50 # 'w'eight of 's'haping reward for 'p'osition
      wso = 50 # 'w'eight of 's'haping reward for 'o'rientation
      gamma = 0.99 # discount for potential based shaping
      wpp = 1; # 'w'eight of 'p'otential reward for 'p'osition
      wpo = 1; # 'w'eight of 'p'otential reward for 'o'rientation
      wlp = 100; # 'w'eight of 'l'inear reward for 'p'osition
      wlo = 100; # 'w'eight of 'l'inear reward for 'o'rientation

      # regular step without terminating episode, rp = position reward, ro = orientation reward
      if self.rewardfn_num == 1:
         rp = -1+wsp*(-gamma*(dnew/dstart)**alpha + (dold/dstart)**alpha)
         ro =    wso*(-gamma*(atnew/np.pi)**alpha + (atold/np.pi)**alpha)
      elif self.rewardfn_num == 2:
         rp = wsp*(-gamma*(dnew/dstart)**alpha + (dold/dstart)**alpha)
         ro = wso*(-gamma*(atnew/np.pi)**alpha + (atold/np.pi)**alpha)
      elif self.rewardfn_num == 3:
         rp = -wpp*((dnew/dstart)**alpha)
         ro = -wpo*((atnew/np.pi)**alpha)
      elif self.rewardfn_num == 4:
         rp = -wlp*(dnew-dold)
         ro = -wlo*(atnew-atold)
      elif self.rewardfn_num == 5:
         rp = -wpp*(dnew/dstart)**alpha - wlp*(dnew-dold)
         ro = -wpo*(atnew/np.pi)**alpha - wlo*(atnew-atold)
      elif self.rewardfn_num == 6:
         rp = 1-((dnew/dstart)**alpha)
         ro = 1-((atnew/np.pi)**alpha)
      elif self.rewardfn_num == 7:
         pass
      elif self.rewardfn_num == 8:
         rp = 1-((dnew/dstart)**alpha) + wsp*(-gamma*((dnew/dstart)**alpha) + ((dold/dstart)**alpha))
         ro = 1-((atnew/np.pi)**alpha) + wso*(-gamma*((atnew/np.pi)**alpha) + ((atnew/np.pi)**alpha))
      elif self.rewardfn_num == 9:
         rp = 1-((dnew/self.eps_dist)**alphap)
         ro = 1-((atnew/self.eps_angle)**alphao)

      reward = wp*rp + wo*ro

      self.info["str"] = "Regular step @ {:3d}, dist covered: {:5.2f}" \
                         .format(self.steps, 1000*(dnew-dold))

      #terminate episode, when
      # 1. reaching goal and angle tolerance
      # 2. exceeding max steps of environment
      if (dnew < self.eps_dist and atnew < self.eps_angle) or self.steps >= self.max_steps:

         done = True
         self.dist_end = dnew
#         self.diff_angle_end = atnew

         if dnew < self.eps_dist and atnew < self.eps_angle:
            reward = 150
            self.info["goal"] = True
            self.info["str"] = "GOAL! Distance {:.2f}mm @step {:3d}, total distance covered {:.2f}mm." \
                               .format(1000*norm(self.goal-self.tip_vec2), self.steps, 1000*norm(dstart-self.dist_end))
         elif self.steps >= self.max_steps:
            self.info["str"] = "Max steps {}, distance to finish {:5.2f}mm, total distance covered {:5.2f}mm." \
                               .format(self.max_steps, 1000*self.dist_end, 1000*(dstart-self.dist_end))

#      gamma_t = 1-(self.steps/(self.max_steps+1))
#      reward = gamma_t*reward
      return reward, done, self.info

   def arc_params(self, l1, l2, l3):
      """Returns arc parameters kappa, phi, s of continuum robot."""
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

   def get_diff_angles(self, degree=False):
      """Returns the angles between normal, binormal, and tangent vector of goal and robot orientation"""
      anglen = acos(np.dot(self.normal_vec2, self.normal_vec_goal)) / (norm(self.normal_vec2)*norm(self.normal_vec_goal))
      angleb = acos(np.dot(self.binormal_vec2, self.binormal_vec_goal)) / (norm(self.binormal_vec2)*norm(self.binormal_vec_goal))
      anglet = acos(np.dot(self.tangent_vec2, self.tangent_vec_goal)) / (norm(self.tangent_vec2)*norm(self.tangent_vec_goal))
      if degree:
         return anglen*180/np.pi, angleb*180/np.pi, anglet*180/np.pi
      else:
         return anglen, angleb, anglet

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
#         self.ax.plot([self.goal[0]], [self.goal[1]], [self.goal[2]], linestyle=None, label="Ziel", c="magenta", marker="x", markersize=11)
         self.ax.plot([self.goal[0]], [self.goal[1]], [self.goal[2]], label="Ziel", c="magenta", marker="x", markersize=11)
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
         # tangent vector indicating orientation of goal point
#         anormal_goal = self.create_arrow(self.goal, self.normal_vec_goal, self.arrow_len,
#                                          "-|>", 0.75*self.arrow_lw, self.arrow_ms, c="r")
#         abinormal_goal = self.create_arrow(self.goal, self.binormal_vec_goal, self.arrow_len,
#                                          "-|>", 0.75*self.arrow_lw, self.arrow_ms, c="g")
         atangent_goal = self.create_arrow(self.goal, self.tangent_vec_goal, self.arrow_len,
                                           "-|>", 0.75*self.arrow_lw, self.arrow_ms, c="b")
#         self.ax.add_artist(anormal_goal)
#         self.ax.add_artist(abinormal_goal)
         self.ax.add_artist(atangent_goal)

         plot_loop_pause(pause) # updates plot without losing window focus
         if save_frames == True:
            self.fig.savefig("./figures/frame"+str(self.frame)[1:]+".png")
            self.frame += 1

   def points_on_arc(self, num_points):
      """Returns np.arrays of arc points with shape [num_points, 3] [x(s), y(s), z(s)] [m]"""
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
      """Returns a 3D arrow pointing from start_vec to dir_vec. dir_vec should be a unit vector"""
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
      # set axis limits and labels
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
