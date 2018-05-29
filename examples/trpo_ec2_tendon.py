from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from examples.tendon_1seg_env import TendonOneSegmentEnv
from examples.tendon_2seg_env import TendonTwoSegmentEnv
import sys


def run_task(v):
    env = normalize(TendonOneSegmentEnv())

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(64, 64)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=100,
        discount=0.99,
        step_size=0.01,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()



run_experiment_lite(
    run_task,
    exp_name="test_tendon_ec2",
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="gap",
    snapshot_gap=10,
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # mode="local",
    mode="ec2",
    # variant=dict(step_size=step_size, seed=seed)
    # plot=True,
    # terminate_machine=False,
)
# sys.exit()
