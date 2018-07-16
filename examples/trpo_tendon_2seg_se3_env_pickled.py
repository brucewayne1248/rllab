from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from examples.tendon_2seg_se3_env import TendonTwoSegmentSE3Env
#import lasagne.nonlinearities as NL
import numpy as np

def run_task(*_):
   env = normalize(TendonTwoSegmentSE3Env())

   policy = GaussianMLPPolicy(
         env_spec = env.spec,
         hidden_sizes=(64, 64)
#         output_nonlinearity=NL.tanh
         #  std_hidden_nonlinearity=NL.rectify,
         #  hidden_nonlinearity=NL.rectify,
   )

   baseline = LinearFeatureBaseline(env_spec=env.spec)

   algo = TRPO(
         env=env,
         policy=policy,
         baseline=baseline,
         batch_size = 4000,
         max_path_length=np.inf,
         n_itr=5,
         discount=0.99,
         step_size=0.01,
         gae_lambda=0.95,
   )
   algo.train()

run_experiment_lite(
      run_task,
      n_parallel=2,
      exp_name="GAEtest",
      snapshot_mode="gap",
      snapshot_gap=1,
      seed=1,
 )