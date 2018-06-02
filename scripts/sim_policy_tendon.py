import argparse

import joblib
import tensorflow as tf
import pickle

#from examples.tendon_1seg_env import TendonOneSegmentEnv
#from examples.tendon_2seg_env import TendonTwoSegmentEnv
#from math import acos
import numpy as np
from rllab.sampler.utils import rollout_tendon


def retry_ep(goal, tangent_vec_goal, max_retries, verbose=False):
    lengths = None
    for retry in range(max_retries):
#        print(vars(env._wrapped_env))
        path = rollout_tendon(env, policy, always_return_paths=True,
                              #render_mode=args.render_mode,
                              render_mode="human",
                              save_frames=1,
                              lengths=lengths, goal=goal, tangent_vec_goal=tangent_vec_goal)

        if env._wrapped_env.dist_end < env._wrapped_env.eps: break

#    print("lenghts1", env._wrapped_env.lengths1)
#    print("lenghts2", env._wrapped_env.lengths2)
#    print("tip1", env._wrapped_env.tip_vec1)
#    print("tip2", env._wrapped_env.tip_vec2)
#    print("phi1", env._wrapped_env.phi1)
    print(vars(env._wrapped_env))
    if env._wrapped_env.dist_end < env._wrapped_env.eps: # goal reached
        if verbose: print("Goal reached after {} retries.".format(retry+1))
#        print(vars(env._wrapped_env), sep='\n')

        return True
    else:
        if verbose: print("Goal still not reached after {} retries.".format(retry+1))
#        print(vars(env._wrapped_env), sep='\n')
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

        printenv = env
        printenv_wrapped = env._wrapped_env

        print(vars(policy))
        print(vars(env))
        print(env._wrapped_env.__dict__)
        print(policy._cached_param_shapes)
        print(env._Serializable__args)
        print("lmin:", printenv_wrapped.lmin)
        print("lmax:", printenv_wrapped.lmax)
#        print("l1min:", printenv_wrapped.l1min)
#        print("l1max:", printenv_wrapped.l1max)
#        print("l2min:", printenv_wrapped.l2min)
#        print("l2max:", printenv_wrapped.l2max)
        print("d:", printenv_wrapped.d)
        print("n:", printenv_wrapped.n)
        print("state:", printenv_wrapped._state)
#        print("n_states:", len(printenv_wrapped._state))
        print("delta_l:", printenv_wrapped.delta_l)
        print("max_steps:", printenv_wrapped.max_steps)
        print("eps:", printenv_wrapped.eps)


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
#            if env._wrapped_env.dist_end < env._wrapped_env.eps: # goal reached
                if args.analyze:
                    ep_lens_goal_reached.append(env._wrapped_env.steps)
                    diff_angles.append(env._wrapped_env.get_diff_angle(degree=True))
                    goals_reached += 1
                cur_goal_reached = True
            else: # goal not reached
                if args.analyze:
                    dist_mins.append(env._wrapped_env.dist_end)
                cur_goal_reached = False

            if args.retry > 0 and cur_goal_reached == False:
                goal_reached_retry= retry_ep(env._wrapped_env.goal,
                                             env._wrapped_env.tangent_vec_goal,
                                             args.retry, verbose=True)
                goals_reached_after_retries += goal_reached_retry
#                break

            episode += 1

        if args.analyze:
            eps = 1e-10
            print("Goals reached {:3d}/{:3d}".format(goals_reached, total_episodes))
            print("Goals reached after retries {:3d}/{:3d}".format(goals_reached+goals_reached_after_retries, total_episodes))
            print("Average steps needed to reach goal: {:5.1f}.".format(sum(ep_lens_goal_reached)/(len(ep_lens_goal_reached)+eps)))
            print("Average min distance from goal within one episode, when goal not reached: {:5.2f}mm.".format(1000*sum(dist_mins)/(len(dist_mins)+eps)))
            print("Average angle difference when goal reached: {:5.2f}°.".format(sum(diff_angles)/(len(diff_angles)+eps)))