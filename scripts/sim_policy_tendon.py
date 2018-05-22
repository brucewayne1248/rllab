import argparse

import joblib
import tensorflow as tf

from examples.tendon_1seg_env import TendonOneSegmentEnv
from examples.tendon_2seg_env import TendonTwoSegmentEnv
from math import acos
import numpy as np
#from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout_tendon

# environment used for rendering
render_env_1seg = TendonOneSegmentEnv()
render_env_2seg = TendonTwoSegmentEnv()

def render_episode(env, tendon_lengths, goal, tangent_vec_goal):
   env.reset()
   env.goal = goal
   env.tangent_vec_goal = tangent_vec_goal

   n_seg = int(len(tendon_lengths[0]) / 3)

   for i in range(len(tendon_lengths)):
      if n_seg == 1:
         env.l1 = tendon_lengths[i, 0]; env.l2 = tendon_lengths[i, 1]; env.l3 = tendon_lengths[i, 2]
      elif n_seg == 2:
         env.l11 = tendon_lengths[i, 0]; env.l12 = tendon_lengths[i, 1]; env.l13 = tendon_lengths[i, 2]
         env.l21 = tendon_lengths[i, 3]; env.l22 = tendon_lengths[i, 4]; env.l23 = tendon_lengths[i, 5]
      env.update_workspace()
      env.render(mode="human", save_frames=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Total simulation episodes')
    parser.add_argument('--animated', type=int, default=1,
                        help='Boolean determining if environment is animated')
    parser.add_argument('--render', type=int, default=1,
                        help='0 no render, else render 3d of continuum robot')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    print("RENDER: ", args.render)
    goals_reached = 0
    total_episodes = args.episodes
    episode = 0
    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env'] # wrapped env, access TendonOneSegmentEnv with env._wrapped_env
        while episode < total_episodes:
            path = rollout_tendon(env, policy, max_path_length=args.max_path_length,
                                  animated=args.animated, speedup=args.speedup, always_return_paths=True)

#            print("VARS: ", vars(env))
#            print("VARS [_wrapped_env]", vars(env._wrapped_env))
            tangent_vec_goal = env._wrapped_env.tangent_vec_goal
#            print("TANGENT VEC GOAL", tangent_vec_goal)
#            tangent_vec_goal = path["env_infos"]["info"]["tangent_vec_goal"][0]
#            print("TANGENT VEC GOAL", tangent_vec_goal)
            if len(path["observations"][0, :]) == 10:
               tendon_lengths = path["observations"][:, 0:3]
               goal = path["observations"][0, 6:9]
               if args.render:
                  render_episode(render_env_1seg, tendon_lengths, goal, tangent_vec_goal)
            elif len(path["observations"][0, :]) == 13:
               tendon_lengths = path["observations"][:, 0:6]
               goal = path["observations"][0, 9:12]
               if args.render:
                  render_episode(render_env_2seg, tendon_lengths, goal, tangent_vec_goal)

            if "GOAL" in str(path["env_infos"]["info"]["str"]):
               goals_reached += 1
            episode += 1

        print("Goals reached {:3d}/{:3d}".format(goals_reached, total_episodes))