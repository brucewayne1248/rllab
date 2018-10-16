from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
from rllab.utils.plot_utils import Arrow3D, plot_loop_pause

import numpy as np
import quaternion
from numpy.linalg import norm
from math import sqrt, asin, atan2, cos, sin, acos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class TendonTwoSegmentEnv(Env):
   """
   class handling the forward kinematics of a two segment tendon driven continuum robot
   l1min, l1max: min max tendon lengths of first segment [m]
   l2min, l2max: min max tendon lenghts of second segment [m]
   d: pitch distance to cable guides [m]:
   n: number of units (spacer discs) within one segment
   """

   precision_digits = 16 # rounding precision needed for handling the singularity at l1=l2=l3

   def __init__(self):
      self.l1min = 0.085
      self.l1max = 0.115
      self.l2min = 0.185
      self.l2max = 0.215
      self.d = 0.01
      self.n = 10
      self.dependent_actuation = True # I like dependent! indicates the way change of tendon lenghts interact with robot
      self.rewardfn_num = 10 # number of chosen reward fn

      self.base = np.array([0.0, 0.0, 0.0, 1.0]) # base vector used for transformations
      self.l11 = None; self.l12 = None; self.l13 = None; # tendon lengths
      self.l21 = None; self.l22 = None; self.l23 = None; # absolute segment 2 tendon lengths
      self.dl21 = None; self.dl22 = None; self.dl23 = None; # effective  segment 2 tendon lengths (needed for kappa2, phi2, seg_len2)
      self.lengths1 = None # [l11, l12, l13]
      self.lengths2 = None # [l11, l12, l13]
      self.closest_lengths = None # tendon lengths at closest position within one episode

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
      self.tip_vec2min = None # closest coordinates [x, y, z] within one episode
      self.old_dist_vec = None # goal-tip_vec in vector form, at time step t
      self.new_dist_vec = None # goal-tip_vec in vector form, at time step t+1 after taking action a_t
      self.old_dist_euclid = None # euclid dist to goal at time step t
      self.new_dist_euclid = None # euclid dist to goal at time step t+1 after taking action a_t

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

      self.fig = None # fig variable used for plotting

      # variables needed in episodic reinforcement learning
      self._state = None # state vector containing l1, l2, l3, tip position, and goal position
      self.info = {} # used to store adaditional environment data
      self.steps = None # current step the episode is in
      self.goal = None # goal to be reached by the robot's tip [x, y, z] [m]
      self.goal_lengths = None # tendon lengths of generated goal
      self.normal_vec_goal = None # normal vector of goal position
      self.binormal_vec_goal = None # binormal vector of goal position
      self.tangent_vec_goal = None # tangent vector of goal position
      self.delta_l = 0.001 # max tendon length change per timestep
      self.max_steps = 100 # max steps per episode
      self.eps = 1e-3 # distance tolerance to reach goal
      self.dist_start = None # start distance to goal
      self.dist_end = None # end distance to goal of episode
      self.dist_min = None # min distance to goal reached within each episode
      self.dist = None # current distance to goal
      self.total_episodes = 0
      self.total_goals_reached = 0
      self.ep = 0 # current episode of simulating a policy

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

      self.tip_vec2min = self.tip_vec2
      self.dist_start = norm(self.goal-self.tip_vec2)
      self.dist_min = self.dist_start
      self.closest_lengths = np.array([self.l11, self.l12, self.l13, self.l21, self.l22, self.l23])

      self.anglen_new, self.angleb_new, self.anglet_new = self.get_diff_angles() # angles between vectors
      self.anglen_min, self.angleb_min, self.anglet_min = self.anglen_new, self.angleb_new, self.anglet_new

      self.Rmin, self.Pmin, self.Ymin = self.get_orientation(self.T02)

      self.qmin = quaternion.as_float_array(quaternion.from_rotation_matrix(self.T02))
      self.qmin = np.sign(self.qmin[0]) * self.qmin

      self._state = self.get_state()
      self.steps = 0
      self.info["goal"] = False
      self.ep += 1
      self.frame = 0
      self.max_steps = 75
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
      self.normal_vec_goal = T02[0:3, 0]
      self.binormal_vec_goal = T02[0:3, 1]
      self.tangent_vec_goal =  T02[0:3, 2]
      # quaternions
      self.qgoal = quaternion.as_float_array(quaternion.from_rotation_matrix(T02[:3, :3]))
      if np.sign(self.qgoal[0]) != 0.0:
         self.qgoal = np.sign(self.qgoal[0]) * self.qgoal # make sure that quaternions all have the same sign for w
      # calculate RPY angles of goal here or quaternion
      self.Rgoal, self.Pgoal, self.Ygoal = self.get_orientation(T02)

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
      self.anglen_old, self.angleb_old, self.anglet_old = self.get_diff_angles()
      self.update_workspace()
      self.new_dist_vec = self.goal-self.tip_vec2
      self.new_dist_euclid = norm(self.new_dist_vec)
      self.anglen_new, self.angleb_new, self.anglet_new = self.get_diff_angles()
      if self.new_dist_euclid < self.dist_min: # update min distance within one episode
         self.tip_vec2min = self.tip_vec2
         self.closest_lengths = np.array([self.l11, self.l12, self.l13, self.l21, self.l22, self.l23])
         self.dist_min = self.new_dist_euclid
         self.anglen_min, self.angleb_min, self.anglet_min = self.anglen_new, self.angleb_new, self.anglet_new
         self.Rmin, self.Pmin, self.Ymin = self.get_orientation(self.T02)
         self.qmin = quaternion.as_float_array(quaternion.from_rotation_matrix(self.T02[:3, :3]))
         if np.sign(self.qmin[0]) != 0.0:
            self.qmin *= np.sign(self.qmin[0])

   def get_reward_done_info(self, new_dist_euclid, old_dist_euclid, steps):
      """returns reward, done, info dict after taking action"""
      done = False
      xi = 0.4;
      # regular step without terminating episode
      reward = 1-((new_dist_euclid/self.eps)**xi)

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
         if new_dist_euclid < self.eps:
            reward = 100
            self.info["goal"] = True
            self.info["str"] = "GOAL! Distance {:.2f}mm @step {:3d}, total distance covered {:.2f}mm." \
                               .format(1000*norm(self.goal-self.tip_vec2), self.steps, 1000*norm(self.dist_start-self.dist_end))
         elif self.steps >= self.max_steps:
            self.info["str"] = "Max steps {}, distance to finish {:5.2f}mm, total distance covered {:5.2f}mm." \
                               .format(self.max_steps, 1000*self.dist_end, 1000*(self.dist_start-self.dist_end))

      return reward, done, self.info

   def arc_params(self, l1, l2, l3):
      """returns kappa, phi, l of configuration space"""
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
      anglen = acos(np.dot(self.normal_vec2, self.normal_vec_goal)) / ((norm(self.normal_vec2)*norm(self.normal_vec_goal)) + 1e-8)
      angleb = acos(np.dot(self.binormal_vec2, self.binormal_vec_goal)) / ((norm(self.binormal_vec2)*norm(self.binormal_vec_goal)) +1e-8)
      anglet = acos(np.dot(self.tangent_vec2, self.tangent_vec_goal)) / ((norm(self.tangent_vec2)*norm(self.tangent_vec_goal)) + 1e-8)
      return (anglen*180/np.pi, angleb*180/np.pi, anglet*180/np.pi) if degree else (anglen, angleb, anglet)

   def get_orientation(self, T, representation="eulerZYX", degree=False):
      T = np.array(T)
      if representation == "eulerZYX":
      # R = R_x(theta1) R_y(theta2) R_z(theta3) corresponds to RPY - theta1 = PHI/R, theta2 = Theta/P, theta3 = Psi/Y
      # See https://pdfs.semanticscholar.org/6681/37fa4b875d890f446e689eea1e334bcf6bf6.pdf
      # handles the singularity at theta2 +- pi/2 by setting theta1 = 0 and theta3 accounts for whole rotation
         theta1 = atan2(T[1,2], T[2,2])
         c2 = sqrt(T[0,0]**2+T[0,1]**2)
         theta2 = atan2(-T[0,2], c2)
         s1 = sin(theta1); c1 = cos(theta1)
         theta3 = atan2(s1*T[2,0]-c1*T[1,0], c1*T[1,1]-s1*T[2,1])
#         theta3 = atan2(T[0,1], T[00])
      elif representation == "eulerZYZ":
      # R = R_z(theta1) R_y(theta2) R_z(theta3)
      # NEEDS HANDLING OF SINGULARITY FOR THETA2 ~ 0
         theta2 = atan2(sqrt(T[2,0]**2+T[2,1]**2), T[2,2])
         theta1 = atan2(T[1,2]/sin(theta2), T[0,2]/sin(theta2))
         theta3 = atan2(T[2,1]/sin(theta2), -T[2,0]/sin(theta2))
      return (theta1*180/np.pi, theta2*180/np.pi, theta3*180/np.pi) if degree else (theta1, theta2, theta3)



   def render(self, mode="human", pause=0.0000000001, save_frames=False, render_coordinate_frames=True,
               render_spacer_discs=True, render_goal=True, render_legend=False,
               render_segment1=True, render_segment2=True,
               render_axes=False, xlim=None, ylim=None, zlim=None, view=None,
               num_points=25):
      """ renders the 3d plot of the robot's arc, pause (float) determines how long each frame is shown
          when save frames is set to True each frame of the plot is saved in an png file"""
      lw = 4.5

#      self.iter = 0

      if self.steps % 5 == 0 and mode=="string":
         print("STEP {:3d}\tDISTANCE: {:5.2f}mm".format(self.steps, 1000*norm(self.goal-self.tip_vec2)))
      elif mode=="human":
         if self.fig == None:
            self.init_render()

         if render_axes is False:
            self.ax._axis3don = False

         if xlim is not None:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.ax.set_zlim(zlim)

         if view is not None:
            self.ax.view_init(elev=view[0], azim=view[1])

#         self.azim += 1
         self.ax.view_init(elev=self.elev, azim=self.azim)

         points1, points2 = self.points_on_arc(100) # points to be plotted from base to robot's tip

         while self.ax.lines:
             self.ax.lines.pop() # delete plots of previous frame
         if render_segment2:
            self.ax.plot(points2[:,0], points2[:,1], points2[:,2], label="Segment 2", c="k", linewidth=lw)
         if render_segment1:
            self.ax.plot(points1[:,0], points1[:,1], points1[:,2], label="Segment 1", c="black", linewidth=lw)
         if render_goal:
            self.ax.plot([self.goal[0]], [self.goal[1]], [self.goal[2]], linestyle=None, label="Ziel", c="magenta", marker="o", markersize=9)
#            self.ax.plot([self.goal[0]], [self.goal[1]], [self.goal[2]], label="Ziel", c="magenta", marker="x", markersize=11)
         if render_legend: self.ax.legend()

         # clear spacer discs
         self.ax.collections.clear() # clear previous spacer disc polygons

         if render_spacer_discs: self.render_spacer_discs(render_segment2, num_points)
         # delete arrows of previous frame, except base frame
         while len(self.ax.artists) > 3:
             self.ax.artists.pop()
         # add coordinate frames to plot
         if render_segment1:
            anormal1 = self.create_arrow(self.tip_vec1, self.normal_vec1, self.arrow_len,
                                         "-|>", self.arrow_lw, self.arrow_ms, c="r")
            abinormal1 = self.create_arrow(self.tip_vec1, self.binormal_vec1, self.arrow_len,
                                           "-|>", self.arrow_lw, self.arrow_ms, c="g")
            atangent1 = self.create_arrow(self.tip_vec1, self.tangent_vec1, self.arrow_len,
                                          "-|>", self.arrow_lw, self.arrow_ms, c="b")
            self.ax.add_artist(anormal1)
            self.ax.add_artist(abinormal1)
            self.ax.add_artist(atangent1)
         if render_segment2:
            anormal2 = self.create_arrow(self.tip_vec2, self.normal_vec2, self.arrow_len,
                                         "-|>", self.arrow_lw, self.arrow_ms, c="r")
            abinormal2 = self.create_arrow(self.tip_vec2, self.binormal_vec2, self.arrow_len,
                                           "-|>", self.arrow_lw, self.arrow_ms, c="g")
            atangent2 = self.create_arrow(self.tip_vec2, self.tangent_vec2, self.arrow_len,
                                          "-|>", self.arrow_lw, self.arrow_ms, c="b")
            self.ax.add_artist(anormal2)
            self.ax.add_artist(abinormal2)
            self.ax.add_artist(atangent2)
         # tangent vector indicating orientation of goal point
         if render_goal:
            anormal_goal = self.create_arrow(self.goal, self.normal_vec_goal, self.arrow_len,
                                             "-|>", 0.75*self.arrow_lw, self.arrow_ms, c="r")
            abinormal_goal = self.create_arrow(self.goal, self.binormal_vec_goal, self.arrow_len,
                                             "-|>", 0.75*self.arrow_lw, self.arrow_ms, c="g")
            atangent_goal = self.create_arrow(self.goal, self.tangent_vec_goal, self.arrow_len,
                                              "-|>", self.arrow_lw, self.arrow_ms, c=[0.2, 0.2, 1])
            self.ax.add_artist(anormal_goal)
            self.ax.add_artist(abinormal_goal)
            self.ax.add_artist(atangent_goal)

         self.ax.text(-0.1,0.0,-0.05, "Iteration " + str(self.iter), fontname="Carlito", size="xx-large")

         plot_loop_pause(pause) # updates plot without losing window focus
         if save_frames == True:
            filename = "./figures/itr" + str(self.iter) + "_ep" + str(self.ep).rjust(3, '0') +"_frame" + str(self.frame).rjust(3, '0')+".png"
            self.fig.savefig(filename, format='png')#, dpi=200)
            self.frame += 1

   def render_spacer_discs(self, render_segment2, num_points=25):
      if num_points < 13 or (num_points-1) % 12 is not 0:
         num_points = 25 # minimum amount of points has to be num_points = k*12 + 1
      lw = 1
      enlarge_factor = 1.5
      r = enlarge_factor * self.d
      alpha_face = 0.35 # opacity for spacer discs
      alpha_edge = 0.35
      n = self.n # number of spacer discs
      thickness = 0.002 # of spacer discs

      color_segment1 = [0.00000,0.313725,0.607843, alpha_face] # imes blue
      color_segment2 = [0.90588235,0.48235,0.160784, alpha_face] # imes orange
      color_edge = [0,0,0, alpha_edge]
      # indexes used to plot tendons between spacer discs
      degree_step = 2*np.pi / (num_points-1)
      idx1 = round(1/2*np.pi / degree_step)
      idx2 = round(7/6*np.pi / degree_step)
      idx3 = round(11/6*np.pi / degree_step)

      phi_values = np.linspace(0, 2*np.pi, num_points)
      base_points = np.zeros((num_points, 3))
      s_values = np.linspace(0.0, self.seg_len1, n)
      for i in range(num_points):
         x = r * cos(phi_values[i])
         y = r * sin(phi_values[i])
         base_points[i] = [x, y, 0]
      base_points_transform = np.hstack((base_points, np.ones((num_points, 1), dtype=np.float32)))
      ########## segment 1 spacer discs ##########
      rect = np.zeros((4,3)) # used to create polygons of cylinder's shell
      centers = np.zeros((num_points, 3))
      tops = np.zeros((num_points, 3))
      bottoms = np.zeros((num_points, 3))
      tops_old = np.zeros((num_points, 3))
      for i in range(n):
         T = self.transformation_matrix(self.kappa1, self.phi1, s_values[i])
         tan_vec = T[:3, 2]
         for j in range(num_points):
            centers[j] = np.matmul(T, base_points_transform[j])[:3]
            tops[j] = centers[j] + 0.5*thickness*tan_vec
            bottoms[j] = centers[j] - 0.5*thickness*tan_vec
         for k in range(num_points-1):
            # render shell of cylinder/spacer discs
            rect[0] = bottoms[k]
            rect[1] = bottoms[k+1]
            rect[2] = tops[k+1]
            rect[3] = tops[k]
            verts = [list(zip(rect[:,0], rect[:,1], rect[:,2]))]
            collection = Poly3DCollection(verts, facecolors=color_segment1, alpha=alpha_face)
            collection.set_facecolor(color_segment1)
            self.ax.add_collection3d(collection)
         # bottom and top polygons of spacer discs
         verts = [list(zip(bottoms[:,0], bottoms[:,1], bottoms[:,2]))]
         collection = Poly3DCollection(verts)
         collection.set_edgecolor(color_edge)
         collection.set_facecolor(color_segment1)
         self.ax.add_collection3d(collection)
         verts = [list(zip(tops[:,0], tops[:,1], tops[:,2]))]
         collection = Poly3DCollection(verts)
         collection.set_edgecolor(color_edge)
         collection.set_facecolor(color_segment1)
         self.ax.add_collection3d(collection)
         ########## segment 1 tendons ##########
         if i > 0:
            self.ax.plot([tops_old[idx1, 0], bottoms[idx1, 0]], [tops_old[idx1, 1], bottoms[idx1, 1]], [tops_old[idx1, 2], bottoms[idx1, 2]], color="k", linewidth=lw)
            self.ax.plot([tops_old[idx2, 0], bottoms[idx2, 0]], [tops_old[idx2, 1], bottoms[idx2, 1]], [tops_old[idx2, 2], bottoms[idx2, 2]], color="k", linewidth=lw)
            self.ax.plot([tops_old[idx3, 0], bottoms[idx3, 0]], [tops_old[idx3, 1], bottoms[idx3, 1]], [tops_old[idx3, 2], bottoms[idx3, 2]], color="k", linewidth=lw)
         tops_old = tops.copy()
      ########## segment 2 spacer discs ##########
      if render_segment2:
         s_values = np.linspace(0, self.seg_len2, n+1)
         for i in range(1, n+1):
            T = np.matmul(self.T01, self.transformation_matrix(self.kappa2, self.phi2, s_values[i]))
            tan_vec = T[:3, 2]
            for j in range(num_points):
               centers[j] = np.matmul(T, base_points_transform[j])[:3]
               tops[j] = centers[j] + 0.5*thickness*tan_vec
               bottoms[j] = centers[j] - 0.5*thickness*tan_vec
            for k in range(num_points-1):
               # render shell of cylinder/spacer discs
               rect[0] = bottoms[k]
               rect[1] = bottoms[k+1]
               rect[2] = tops[k+1]
               rect[3] = tops[k]
               verts = [list(zip(rect[:,0], rect[:,1], rect[:,2]))]
               collection = Poly3DCollection(verts, facecolors=color_segment2, alpha=alpha_face)
               collection.set_facecolor(color_segment2)
               self.ax.add_collection3d(collection)
            # bottom and top polygons of spacer discs
            verts = [list(zip(bottoms[:,0], bottoms[:,1], bottoms[:,2]))]
            collection = Poly3DCollection(verts)
            collection.set_edgecolor(color_edge)
            collection.set_facecolor(color_segment2)
            self.ax.add_collection3d(collection)
            verts = [list(zip(tops[:,0], tops[:,1], tops[:,2]))]
            collection = Poly3DCollection(verts)
            collection.set_edgecolor(color_edge)
            collection.set_facecolor(color_segment2)
            self.ax.add_collection3d(collection)

            ########## segment 2 tendons ##########
            if i > 0:
               self.ax.plot([tops_old[idx1, 0], bottoms[idx1, 0]], [tops_old[idx1, 1], bottoms[idx1, 1]], [tops_old[idx1, 2], bottoms[idx1, 2]], color="k", linewidth=lw)
               self.ax.plot([tops_old[idx2, 0], bottoms[idx2, 0]], [tops_old[idx2, 1], bottoms[idx2, 1]], [tops_old[idx2, 2], bottoms[idx2, 2]], color="k", linewidth=lw)
               self.ax.plot([tops_old[idx3, 0], bottoms[idx3, 0]], [tops_old[idx3, 1], bottoms[idx3, 1]], [tops_old[idx3, 2], bottoms[idx3, 2]], color="k", linewidth=lw)
            tops_old = tops.copy()

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
      self.elev = 25
      self. azim = 45
      plt.ion() # interactive plot mode, panning, zooming enabled
      window_scale = 1.25
      self.fig = plt.figure(figsize=(window_scale*8,window_scale*6))
      self.ax = self.fig.add_subplot(111, projection="3d") # attach z-axis to plot
      # set axis limits and labels
      shrink = 0.85
      self.ax.set_xlim([-0.5*shrink*self.l2max, 0.5*shrink*self.l2max])
      self.ax.set_ylim([-0.5*shrink*self.l2max, 0.5*shrink*self.l2max])
      self.ax.set_zlim([0.0, shrink*self.l2max])
      self.ax.set_xlabel("x [m]")
      self.ax.set_ylabel("y [m]")
      self.ax.set_zlabel("z [m]")
      # add coordinate 3 arrows of base frame, have to be defined once!
      self.arrow_len = 0.045
      self.arrow_lw = 2.7*1.5
      self.arrow_ms = 17
      ax_base = Arrow3D([0.0, self.arrow_len], [0.0, 0.0], [0.0, 0.0],
                        arrowstyle="-|>", lw=self.arrow_lw, mutation_scale=self.arrow_ms, color="r")
      ay_base = Arrow3D([0.0, 0.0], [0.0, self.arrow_len], [0.0, 0.0],
                        arrowstyle="-|>", lw=self.arrow_lw, mutation_scale=self.arrow_ms, color="g")
      az_base = Arrow3D([0.0, 0.0], [0.0, 0.0], [0.0, self.arrow_len],
                        arrowstyle="-|>", lw=self.arrow_lw, mutation_scale=self.arrow_ms, color="b")
      self.ax.add_artist(ax_base)
      self.ax.add_artist(ay_base)
      self.ax.add_artist(az_base)
      self.fig.tight_layout() # fits the plot to window size
      plt.show() # display figure and bring focus (once) to plotting window

plt.close("all")