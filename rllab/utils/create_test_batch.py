from examples.tendon_1seg_env import TendonOneSegmentEnv
from examples.tendon_2seg_env import TendonTwoSegmentEnv
import pickle
import numpy as np
import argparse

def create_test_batch(segments, batch_size):
   data = {}
   lenghts = []
   goals = []
   tangent_vec_goals = []

   assert segments == 1 or segments == 2, "Segments must be 1 or 2 baby"
   env = TendonOneSegmentEnv() if segments == 1 else TendonTwoSegmentEnv()

   for i in range(args.batch_size):
      env.reset()
      # only save necessary variables defining starting position of continuum robot
      # and the goal position plus it's tangent vector
      lenghts.append(env.lengths) if segments == 1 else \
      lenghts.append(np.concatenate((env.lengths1, env.lengths2)))
      goals.append(env.goal)
      tangent_vec_goals.append(env.tangent_vec_goal)

   data["lengths"] = np.array(lenghts)
   data["goal"] = np.array(goals)
   data["tangent_vec_goal"] = np.array(tangent_vec_goals)

   filename = "test_batch_{}seg_{}episodes.pkl".format(segments, batch_size)

   pickle.dump(data, open(filename, "wb"))

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--batch_size', type=int, default=20000,
                       help='Amount of test episodes to create variables for')
   parser.add_argument('--segments', type=int, default=2,
                       help='Segments of continuum robot, possible values [1, 2]')
   args = parser.parse_args()

   create_test_batch(args.segments, args.batch_size)


