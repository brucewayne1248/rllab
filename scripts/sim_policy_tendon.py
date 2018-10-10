import argparse
import os
import joblib
import tensorflow as tf
import pickle, scipy.io
import logging
import numpy as np
import quaternion
from rllab.sampler.utils import rollout_tendon
from datetime import datetime
import csv

# used to name benchmark files and log files
timestr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def create_logger(filename):
    directory = os.path.dirname(os.path.abspath(args.file))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(message)s")
    logfilename = directory+"/benchmarklog" + str(args.episodes) + "_" + timestr + ".log"
    log_hdlr = logging.FileHandler(logfilename, mode="w")
    log_hdlr.setFormatter(formatter)
    logger.handlers.clear() # for usage in ipython console of spyder
    logger.addHandler(log_hdlr)
    return logger

def long_env_info(env, env_wrapped):
#   print(vars(env))
#   print(env_wrapped.__dict__)
   logger.info(env._Serializable__args)
   if hasattr(env_wrapped, 'lmin'):
      logger.info("lmin: {}".format(env_wrapped.lmin))
      logger.info("lmax: {}".format(env_wrapped.lmax))
   else:
     logger.info("l1min: {}".format(env_wrapped.l1min))
     logger.info("l1max: {}".format(env_wrapped.l1max))
     logger.info("l2min: {}".format(env_wrapped.l2min))
     logger.info("l2max: {}".format(env_wrapped.l2max))
   logger.info("d: {}".format(env_wrapped.d))
   logger.info("n: {}".format(env_wrapped.n))
   logger.info("state: {}".format(env_wrapped._state))
   logger.info("delta_l: {}".format(env_wrapped.delta_l))
   logger.info("max_steps: {}".format(env_wrapped.max_steps))
   if hasattr(env_wrapped, "eps"):
      logger.info("eps: {}".format(env_wrapped.eps))
   else:
      logger.info("eps dist: {}".format(env_wrapped.eps_dist))
      logger.info("eps_angle: {}".format(env_wrapped.eps_angle*180/np.pi))
   if hasattr(env_wrapped, "dependent_actuation"):
      logger.info("dependent_actuation: {}".format(env_wrapped.dependent_actuation))
   logger.info("goals: {}/{}".format(env_wrapped.total_goals_reached, env_wrapped.total_episodes))

def retry_ep(goal, tangent_vec_goal, max_retries=1, verbose=False):
    lengths = None
    for retry in range(max_retries):
        path = rollout_tendon(env, policy, always_return_paths=True,
                              render_mode="human",
                              save_frames=1,
                              lengths=lengths, goal=goal, tangent_vec_goal=tangent_vec_goal)

        if env._wrapped_env.info["goal"] == True: break

    if env._wrapped_env.info["goal"] == True: # goal reached
        if verbose: print("Goal reached after {} retries.".format(retry+1))
        return True
    else:
        if verbose: print("Goal still not reached after {} retries.".format(retry+1))
        return False

def save_benchmark(filename):
    global dist_mins, arc_lens, dist_relmins, anglediffs_tangent, RPYgoals, RPYmins, Rdiffs, Pdiffs, Ydiffs, goal_coordinates, goal_quaternions, closest_coordinates, closest_quaternions
    directory = os.path.dirname(os.path.abspath(filename))
    result_dict = {"dist_mins": dist_mins, "arc_lens":arc_lens, "dist_relmins":dist_relmins, "anglediffs_tangent":anglediffs_tangent,
                   "RPYgoals": RPYgoals, "RPYmins": RPYmins, "Rdiffs": Rdiffs, "Pdiffs": Pdiffs, "Ydiffs": Ydiffs, "goal_coordinates": goal_coordinates,
                   "goal_quaternions": goal_quaternions, "closest_coordinates": closest_coordinates, "closest_quaternions": closest_quaternions}
    abspath_pkl = directory + "/benchmark" + str(args.episodes) + "_" + timestr +".pkl"
    abspath_mat = directory + "/benchmark" + str(args.episodes) + "_" + timestr +".mat"
    with open(abspath_pkl, "wb") as f:
        pickle.dump(result_dict, f)
    scipy.io.savemat(abspath_mat, mdict=result_dict)
    print("saved {} and {} to {}".format(os.path.basename(abspath_pkl), os.path.basename(abspath_mat), directory))

def log_results():
    global steps_goal, dist_mins, arc_lens, dist_relmins, anglediffs_tangent, RPYgoals, RPYmins, Rdiffs, Pdiffs, Ydiffs, goal_coordinates, goal_quaternions, closest_coordinates, closest_quaternions
    logger.info("Goals reached: {:3d}/{:3d}".format(goals_reached, total_episodes))
    print("Goals reached: {:3d}/{:3d}".format(goals_reached, total_episodes))
    logger.info("Goals reached in %: {:.2f}".format((goals_reached/total_episodes)*100))
    if args.retry: print("Goals reached after retries {:3d}/{:3d}".format(goals_reached+goals_reached_after_retries, total_episodes))
    logger.info("mean steps to goal: {:.2f}".format(steps_goal.mean()))
    logger.info("mean min dist in mm: {:.2f}".format(1000*dist_mins.mean()))
    logger.info("std min dist in mm: {:.2f}".format(1000*dist_mins.std()))
    logger.info("mean total arc len in mm {:.2f}".format(1000*arc_lens.mean()))
    logger.info("rel mean error based on arc len in %: {:.2f}".format(100*dist_relmins.mean()))
    logger.info("max of min dist in mm: {:.2f}".format(1000*np.max(dist_mins)))
    logger.info("abs mean tangent angle diff goal in deg: {:.2f}".format(180/np.pi*anglediffs_tangent.mean()))
    logger.info("max tangent angle diff goal in deg: {:.2f}".format(180/np.pi*np.max(anglediffs_tangent)))
    if RPYgoals.size: # use this statement when RPY angles are calculated while benchmarking
       logger.info("mean RPY diffs in deg: {:.2f} {:.2f} {:.2f}".format(180/np.pi*Rdiffs.mean(), 180/np.pi*Pdiffs.mean(), 180/np.pi*Ydiffs.mean()))
       logger.info("convenient info for copy/paste into overview spreadsheet\n{}/{} {} {} {} {} {} {} {} {} {} {} {}".format(
                 goals_reached+goals_reached_after_retries, total_episodes, steps_goal.mean(),
                 1000*dist_mins.mean(), 1000*dist_mins.std(), 1000*arc_lens.mean(), 100*dist_relmins.mean(),
                 1000*np.max(dist_mins), 180/np.pi*anglediffs_tangent.mean(), 180/np.pi*np.max(anglediffs_tangent),
                 180/np.pi*Rdiffs.mean(), 180/np.pi*Pdiffs.mean(), 180/np.pi*Ydiffs.mean()
             ))
       print("convenient info for copy/paste into overview spreadsheet\n{}/{} {} {} {} {} {} {} {} {} {} {} {}".format(
              goals_reached+goals_reached_after_retries, total_episodes, steps_goal.mean(),
              1000*dist_mins.mean(), 1000*dist_mins.std(), 1000*arc_lens.mean(), 100*dist_relmins.mean(),
              1000*np.max(dist_mins), 180/np.pi*anglediffs_tangent.mean(), 180/np.pi*np.max(anglediffs_tangent),
              180/np.pi*Rdiffs.mean(), 180/np.pi*Pdiffs.mean(), 180/np.pi*Ydiffs.mean()
              ))
    else:
       logger.info("{}/{} {} {} {} {} {} {} {} {}".format(
                 goals_reached+goals_reached_after_retries, total_episodes, steps_goal.mean(),
                 1000*dist_mins.mean(), 1000*dist_mins.std(), 1000*arc_lens.mean(), 100*dist_relmins.mean(),
                 1000*np.max(dist_mins), 180/np.pi*anglediffs_tangent.mean(), 180/np.pi*np.max(anglediffs_tangent)
             ))
       print("{}/{} {} {} {} {} {} {} {} {}".format(
                 goals_reached+goals_reached_after_retries, total_episodes, steps_goal.mean(),
                 1000*dist_mins.mean(), 1000*dist_mins.std(), 1000*arc_lens.mean(), 100*dist_relmins.mean(),
                 1000*np.max(dist_mins), 180/np.pi*anglediffs_tangent.mean(), 180/np.pi*np.max(anglediffs_tangent)
             ))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--test_batch_file', type=str, default='',
                        help='Path to test_batch file for benchmarking learned policies')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Total simulation episodes')
    parser.add_argument('--render_mode', type=str, default="",
                        help='render_mode: "human" - 3D plot, "string" - print distances within episode, "" - no rendering')
    parser.add_argument('--save_frames', type=int, default=0,
                        help='Saves frames of human rendered environment if evaluated to True')
    parser.add_argument('--save_benchmark', type=int, default=0,
                        help='Saves .pickle and .mat file in directory of snapshot file if int(save_benchmark) evaluates to True')
    parser.add_argument('--retry', type=int, default=0,
                        help='Integer indicating how many retries one episode should have \
                        to same goal with different starting position, in case goal is not reached:\
                        0 - no retries; n - max of n retries')
    parser.add_argument('--dependent_actuation', type=int, default=1,
                        help='Indicating the robot is actuated, 0 or 1 accepted values')
    parser.add_argument('--learning_curve', type=int, default=0,
                        help='Indicating whether learning curve statistics are saved when evaluated to true.')
    args = parser.parse_args()


    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]

    goals_reached = 0
    goals_reached_after_retries = 0
    cur_goal_reached = True
    cur_goal = None
    cur_tangent_vec_goal = None
    # lists for analyzing performance
    goal_coordinates = [] # goal coordinates x, y, z
    goal_quaternions = [] # goal quaternions
    goal_lengths = [] # goal tendon lengths
    closest_coordinates = [] # closest coordinates x, y, z during episode
    closest_quaternions = [] # quaternion at closest position
    closest_lengths = [] # tendon lengths of closest position to goal
    steps_goal = [] # steps needed to reach goal
    dist_mins = [] # minimal distance to goal within episode
    dists_start = [] # start distances not really necessary
    arc_lens = [] # total arc lengths at every episode at closest point to goal
    dist_relmins = [] # minimal distances to goal within episode divided by total arc length
    anglediffs_tangent = []
    RPYgoals = []
    RPYmins = []
    Rdiffs, Pdiffs, Ydiffs = [], [], []
    paths = []

    test_data = None # test data meaning test batch of positions and goals
    if args.test_batch_file:
        test_data = pickle.load(open(args.test_batch_file, "rb")) # load benchmark data for testing trained environments

    episode = 0
    total_episodes = args.episodes if test_data==None else len(test_data["lengths"])


    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env'] # wrapped env, access TendonOneSegmentEnv with env._wrapped_env
        lengths=None; goal=None; tangent_vec_goal=None


        if hasattr(env._wrapped_env, "dependent_actuation"):
           pass
        else:
           env._wrapped_env.dependent_actuation = args.dependent_actuation
           print("Created depenpent_actuation attribute for Tendon Env.")

        if hasattr(env._wrapped_env, "rewardfn_num"):
           pass
        else:
           env._wrapped_env.rewardfn_num = None
           print("Created rewardfn_num attribute for tendon env: value {}".format(env._wrapped_env.rewardfn_num))

        if hasattr(env._wrapped_env, "ep"): pass
        else: env._wrapped_env.ep = 1

        while episode < total_episodes:
            if test_data is not None: # set starting lengths and goal according to test batch
                lengths=test_data["lengths"][episode]
                goal=test_data["goal"][episode]
                tangent_vec_goal=test_data["tangent_vec_goal"][episode]

            path = rollout_tendon(
                    env, policy, always_return_paths=True,
                    render_mode=args.render_mode, save_frames=args.save_frames,
                    lengths=lengths, goal=goal, tangent_vec_goal=tangent_vec_goal)
            episode += 1

            # saving goal variables
            goal_coordinates.append(env._wrapped_env.goal)
            goal_lengths.append(env._wrapped_env.goal_lengths)
            goal_quaternions.append(env._wrapped_env.qgoal)
            # saving closest point to goal
            closest_coordinates.append(env._wrapped_env.tip_vec2min)
            closest_lengths.append(env._wrapped_env.closest_lengths)
            closest_quaternions.append(env._wrapped_env.qmin)
            # remaining statistics
            dist_mins.append(env._wrapped_env.dist_min)
            arc_lens.append(env._wrapped_env.seg_len1+env._wrapped_env.seg_len2)
            dist_relmins.append(env._wrapped_env.dist_min/(env._wrapped_env.seg_len1+env._wrapped_env.seg_len2))
            anglediffs_tangent.append(env._wrapped_env.anglet_min)
            dists_start.append(env._wrapped_env.dist_start)
            try: # RPY angles are not recorded for 2seg env i suppose
               RPYgoals.append((env._wrapped_env.Rgoal, env._wrapped_env.Pgoal, env._wrapped_env.Ygoal))
               RPYmins.append((env._wrapped_env.Rmin, env._wrapped_env.Pmin, env._wrapped_env.Ymin))
               Rdiffs.append(np.sqrt((env._wrapped_env.Rgoal-env._wrapped_env.Rmin)**2))
               Pdiffs.append(np.sqrt((env._wrapped_env.Pgoal-env._wrapped_env.Pmin)**2))
               Ydiffs.append(np.sqrt((env._wrapped_env.Ygoal-env._wrapped_env.Ymin)**2))
            except:
               pass
            if path["env_infos"]["info"]["goal"] == True: # goal reached
                steps_goal.append(env._wrapped_env.steps)
                goals_reached += 1
                cur_goal_reached = True
            else: # goal not reached in this episode
                cur_goal_reached = False

        # convert lists to numpy arrays
        goal_coordinates = np.array(goal_coordinates)
        closest_coordinates = np.array(closest_coordinates)
        if type(goal_quaternions[0]) == np.quaternion:
            goal_quaternions = quaternion.as_float_array(goal_quaternions)
        if type(closest_quaternions[0]) == quaternion.quaternion:
            closest_quaternions = quaternion.as_float_array(closest_quaternions)
        goal_quaternions = np.array(goal_quaternions)
        closest_quaternions = np.array(closest_quaternions)
        # convert to numpy arrays
        steps_goal = np.array(steps_goal)
        dist_mins = np.array(dist_mins)
        arc_lens = np.array(arc_lens)
        dist_relmins = np.array(dist_relmins)
        anglediffs_tangent = np.array(anglediffs_tangent)
        if RPYgoals:
            RPYgoals = np.array(RPYgoals)
            RPYmins = np.array(RPYmins)
            Rdiffs = np.array(Rdiffs)
            Pdiffs = np.array(Pdiffs)
            Ydiffs = np.array(Ydiffs)

        if args.save_benchmark:
            logger = create_logger(args.file)
            save_benchmark(args.file)
            logger.info("ANN shapes: {}".format(policy._cached_param_shapes))
            long_env_info(env, env._wrapped_env)
            log_results()

        if args.learning_curve:
               # write to csv file
            directory = os.path.dirname(os.path.abspath(args.file))
            filepath = directory + "/evaluation.csv"
            create_header = False
            try:
                file = open(filepath, "r")
            except:
                file = open(filepath, "a")
                create_header = True
            file = open(filepath, "a")
            writer = csv.writer(file)

            if create_header is True:
                writer.writerow(["itr", "episodes", "goals", "dist_mean", "dist_std", "dist_rel", "steps_to_goal"])
            itr = int("".join(filter(str.isdigit, os.path.basename(args.file))))
            writer.writerow([itr, args.episodes, goals_reached/args.episodes*100,
                             dist_mins.mean()*1000, dist_mins.std()*1000, dist_relmins.mean()*100, steps_goal.mean()])
