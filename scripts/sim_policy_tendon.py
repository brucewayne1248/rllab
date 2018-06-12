import argparse
import sys
import joblib
import tensorflow as tf
import pickle

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
   print("eps:", env_wrapped.eps)
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
    parser.add_argument('--analyze', type=int, default=0,
                        help='True value indicates that rollouts should be analyzed')
    parser.add_argument('--retry', type=int, default=0,
                        help='Integer indicating how many retries one episode should have \
                        to same goal with different starting position, in case goal is not reached:\
                        0 - no retries; n+ - max of n retries')
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
    if args.analyze:
        ep_lens_goal_reached = []
        dist_mins = []
        diff_angles = []

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

        print(policy._cached_param_shapes)
        print_env_info(env, env._wrapped_env)

        if hasattr(env._wrapped_env, "dependent_actuation"):
           pass
        else:
           env._wrapped_env.dependent_actuation = args.dependent_actuation
           print("Created depenpent_actuation attribute for Tendon Env.")

        if hasattr(env._wrapped_env, "rewardfn_num"):
           pass
        else: env._wrapped_env.rewardfn_num = 1; print("Created rewardfn_num attribute for tendon env")

        # set max steps of environment to high value to test max performance of agent
#        env._wrapped_env.max_steps = 200
#        print("NEW MAX STEPS", env._wrapped_env.max_steps )

        while episode < total_episodes:
            if test_data is not None: # set starting lengths and goal according to test batch
                lengths=test_data["lengths"][episode]
                goal=test_data["goal"][episode]
                tangent_vec_goal=test_data["tangent_vec_goal"][episode]

            path = rollout_tendon(env, policy, always_return_paths=True,
                                  render_mode=args.render_mode,
                                  save_frames=args.save_frames,
                                  lengths=lengths, goal=goal, tangent_vec_goal=tangent_vec_goal)

            if path["env_infos"]["info"]["goal"] == True: # goal reached
#                print(env._wrapped_env.goal, env._wrapped_env.tip_vec2)
#                print(norm(env._wrapped_env.goal-env._wrapped_env.tip_vec2))
                if args.analyze:
                    ep_lens_goal_reached.append(env._wrapped_env.steps)
                    diff_angles.append(env._wrapped_env.get_diff_angle(degree=True))
                    goals_reached += 1
                cur_goal_reached = True
            else: # goal not reached
                if args.analyze:
                    dist_mins.append(env._wrapped_env.dist_min)
                cur_goal_reached = False

            if args.retry > 0 and cur_goal_reached == False:
                goal_reached_retry= retry_ep(env._wrapped_env.goal,
                                             env._wrapped_env.tangent_vec_goal,
                                             args.retry, verbose=True)
                goals_reached_after_retries += goal_reached_retry

            episode += 1

        if args.analyze:
            eps = 1e-10
            print("Goals reached {:3d}/{:3d}".format(goals_reached, total_episodes))
            print("Goals reached after retries {:3d}/{:3d}".format(goals_reached+goals_reached_after_retries, total_episodes))
            print("Average steps needed to reach goal: {:5.1f}.".format(sum(ep_lens_goal_reached)/(len(ep_lens_goal_reached)+eps)))
            print("Average min distance from goal within one episode, when goal not reached: {:5.2f}mm.".format(1000*sum(dist_mins)/(len(dist_mins)+eps)))
            print("Average angle difference when goal reached: {:5.2f}°.".format(sum(diff_angles)/(len(diff_angles)+eps)))
