from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from examples.tendon_2seg_env import TendonTwoSegmentEnv
#import lasagne.nonlinearities as NL

def run_task(*_):
   env = normalize(TendonTwoSegmentEnv())

   policy = GaussianMLPPolicy(
         env_spec = env.spec,
         hidden_sizes=(64, 64)
#         output_nonlinearity=NL.tanh
   )

   baseline = LinearFeatureBaseline(env_spec=env.spec)

   algo = TRPO(
         env=env,
         policy=policy,
         baseline=baseline,
         batch_size = 4000,
         max_path_length=150,
         n_itr=20001,
         discount=0.99,
         step_size=0.01,
   )
   algo.train()

run_experiment_lite(
      run_task,
      n_parallel=2,
      exp_name="s2_r7_a04_h64_T100_dl1_itrmax20000",
      snapshot_mode="gap",
      snapshot_gap=100,
      seed=1,
 )