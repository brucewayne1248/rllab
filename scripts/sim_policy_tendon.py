import argparse
import os
import joblib
import tensorflow as tf
import pickle, scipy.io

import numpy as np
from rllab.sampler.utils import rollout_tendon

def print_env_info(env, env_wrapped):
#   print(vars(env))
#   print(env_wrapped.__dict__)
   print(env._Serializable__args)
   if hasattr(env_wrapped, 'lmin'):
      print("lmin:", env_wrapped.lmin)
      print("lmax:", env_wrapped.lmax)
   else:
     print("l1min:", env_wrapped.l1min)
     print("l1max:", env_wrapped.l1max)
     print("l2min:", env_wrapped.l2min)
     print("l2max:", env_wrapped.l2max)
   print("d:", env_wrapped.d)
   print("n:", env_wrapped.n)
   print("state:", env_wrapped._state)
   print("delta_l:", env_wrapped.delta_l)
   print("max_steps:", env_wrapped.max_steps)
   if hasattr(env_wrapped, "eps"):
      print("eps:", env_wrapped.eps)
   else:
      print("eps dist:", env_wrapped.eps_dist)
      print("eps_angle:", env_wrapped.eps_angle*180/np.pi)
   if hasattr(env_wrapped, "dependent_actuation"):
      print("dependent_actuation:", env_wrapped.dependent_actuation)
   print("goals: {}/{}".format(env_wrapped.total_goals_reached, env_wrapped.total_episodes))

def retry_ep(goal, tangent_vec_goal, max_retries, verbose=False):
    lengths = None
    for retry in range(max_retries):
        path = rollout_tendon(env, policy, always_return_paths=True,
                              render_mode=args.render_mode,
                              save_frames=args.save_frames,
                              lengths=lengths, goal=goal, tangent_vec_goal=tangent_vec_goal)

        if env._wrapped_env.dist_end < env._wrapped_env.eps: break

    if env._wrapped_env.dist_end < env._wrapped_env.eps: # goal reached
        if verbose: print("Goal reached after {} retries.".format(retry+1))
        return True
    else:
        if verbose: print("Goal still not reached after {} retries.".format(retry+1))
        return False

def save_results(filename):
    global dist_mins, arc_lens, dist_relmins, anglediffs_tangent, RPYgoals, RPYmins, Rdiffs, Pdiffs, Ydiffs
    directory = os.path.dirname(os.path.abspath(filename))
    result_dict = {"dist_mins": dist_mins, "arc_lens":arc_lens, "dist_relmins":dist_relmins, "anglediffs_tangent":anglediffs_tangent,
                   "RPYgoals":RPYgoals, "RPYmins":RPYmins, "Rdiffs":Rdiffs, "Pdiffs":Pdiffs, "Ydiffs":Ydiffs}
    with open(directory+"/benchmark.pkl", "wb") as f:
        pickle.dump(result_dict, f)
    scipy.io.savemat(directory+"/benchmark.mat", mdict=result_dict)
    print("saved results to {}".format(directory))
    pickle.dump

def print_results():
    global steps_goal, dist_mins, arc_lens, dist_relmins, anglediffs_tangent, RPYgoals, RPYmins, Rdiffs, Pdiffs, Ydiffs
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
    print("Goals reached {:3d}/{:3d}".format(goals_reached, total_episodes))
    if args.retry: print("Goals reached after retries {:3d}/{:3d}".format(goals_reached+goals_reached_after_retries, total_episodes))
    print("mean steps to goal: {:.2f}".format(steps_goal.mean()))
    print("mean min dist in mm: {:.2f}".format(1000*dist_mins.mean()))
    print("std min dist in mm: {:.2f}".format(1000*dist_mins.std()))
    print("mean total arc len in mm {:.2f}".format(1000*arc_lens.mean()))
    print("rel mean error based on arc len in %: {:.2f}".format(100*dist_relmins.mean()))
    print("max of min dist in mm: {:.2f}".format(1000*np.max(dist_mins)))
    print("abs mean tangent angle diff goal in deg: {:.2f}".format(180/np.pi*anglediffs_tangent.mean()))
    print("max tangent angle diff goal in deg: {:.2f}".format(180/np.pi*np.max(anglediffs_tangent)))
    if RPYgoals.all():
       print("mean RPY diffs in deg: {:.2f} {:.2f} {:.2f}".format(180/np.pi*Rdiffs.mean(), 180/np.pi*Pdiffs.mean(), 180/np.pi*Ydiffs.mean()))
       print("{}/{} {} {} {} {} {} {} {} {} {} {} {}".format(
             goals_reached+goals_reached_after_retries, total_episodes, steps_goal.mean(),
             1000*dist_mins.mean(), 1000*dist_mins.std(), 1000*arc_lens.mean(), 100*dist_relmins.mean(),
             1000*np.max(dist_mins), 180/np.pi*anglediffs_tangent.mean(), 180/np.pi*np.max(anglediffs_tangent),
             180/np.pi*Rdiffs.mean(), 180/np.pi*Pdiffs.mean(), 180/np.pi*Ydiffs.mean()
             ))
    else:
       print("{}/{} {} {} {} {} {} {} {} {}".format(
             goals_reached+goals_reached_after_retries, total_episodes, steps_goal.mean(),
             1000*dist_mins.mean(), 1000*dist_mins.std(), 1000*arc_lens.mean(), dist_relmins.mean(),
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
    parser.add_argument('--save_results', type=int, default=0,
                        help='Saves .pickle and .mat file in directory of snapshot file if int(save_results) evaluates to True')
    parser.add_argument('--retry', type=int, default=0,
                        help='Integer indicating how many retries one episode should have \
                        to same goal with different starting position, in case goal is not reached:\
                        0 - no retries; n - max of n retries')
    parser.add_argument('--dependent_actuation', type=int, default=1,
                        help='Indicating the robot is actuated, 0 or 1 accepted values')
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
    steps_goal = [] # steps needed to reach goal
    dist_mins = [] # minimal distance to goal within episode
    arc_lens = [] # total arc lengths at every episode at closest point to goal
    dist_relmins = [] # minimal distances to goal within episode divided by total arc length
    anglediffs_tangent = []
    RPYgoals = []
    RPYmins = []
    Rdiffs, Pdiffs, Ydiffs = [], [], []

    test_data = None
    if args.test_batch_file:
        test_data = pickle.load(open(args.test_batch_file, "rb")) # load benchmark data for testing trained environments

    episode = 0
    total_episodes = args.episodes if test_data==None else len(test_data["lengths"])

    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env'] # wrapped env, access TendonOneSegmentEnv with env._wrapped_env
        lengths=None; goal=None; tangent_vec_goal=None

        print("ANN shapes",policy._cached_param_shapes)
        print_env_info(env, env._wrapped_env)

        if hasattr(env._wrapped_env, "dependent_actuation"):
           pass
        else:
           env._wrapped_env.dependent_actuation = args.dependent_actuation
           print("Created depenpent_actuation attribute for Tendon Env.")

        if hasattr(env._wrapped_env, "rewardfn_num"):
           pass
        else:
           env._wrapped_env.rewardfn_num = 1
           print("Created rewardfn_num attribute for tendon env")

        # set max steps of environment to high value to test max performance of agent
#        env._wrapped_env.max_steps = 70
#        print("NEW MAX STEPS", env._wrapped_env.max_steps )

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

            # Analyze episode and gather statistics
            dist_mins.append(env._wrapped_env.dist_min)
            arc_lens.append(env._wrapped_env.seg_len1+env._wrapped_env.seg_len2)
            dist_relmins.append(env._wrapped_env.dist_min/(env._wrapped_env.seg_len1+env._wrapped_env.seg_len2))
            anglediffs_tangent.append(env._wrapped_env.anglet_min)
            try:
               RPYgoals.append((env._wrapped_env.Rgoal, env._wrapped_env.Pgoal, env._wrapped_env.Ygoal))
               RPYmins.append((env._wrapped_env.Rmin, env._wrapped_env.Pmin, env._wrapped_env.Ymin))
               Rdiffs.append(np.sqrt((env._wrapped_env.Rgoal-env._wrapped_env.Rmin)**2))
               Pdiffs.append(np.sqrt((env._wrapped_env.Pgoal-env._wrapped_env.Pmin)**2))
               Ydiffs.append(np.sqrt((env._wrapped_env.Ygoal-env._wrapped_env.Ymin)**2))
            except:
               pass
            if path["env_infos"]["info"]["goal"] == True: # goal reached
                steps_goal.append(env._wrapped_env.steps)
#                try:
#                   anglediffs_tangent.append(env._wrapped_env.get_diff_angle(degree=True))
#                except:
#                   anglediffs_tangent.append(env._wrapped_env.get_diff_angles(degree=True)[2])
                goals_reached += 1
                cur_goal_reached = True
            else: # goal not reached in this episode
                cur_goal_reached = False

        if args.save_results:
            save_results(args.file)

        print_results()


