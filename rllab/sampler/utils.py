import numpy as np
from rllab.misc import tensor_utils
import time


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
#            print(env_infos)
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.0005
            time.sleep(timestep / speedup)

    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )

def rollout_tendon(env, agent, always_return_paths=True,
                   render_mode="", save_frames=False,
                   lengths=None, goal=None, tangent_vec_goal=None):
    """adjusted rollout method
    agent=policy"""
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    # reset environment according to given starting point, and goal, or randomly
    if lengths is not None and goal is not None and tangent_vec_goal is not None:
        o = env._wrapped_env.reset(lengths, goal, tangent_vec_goal)
    elif lengths is None and goal is not None and tangent_vec_goal is not None:
        o = env._wrapped_env.reset(lengths, goal, tangent_vec_goal)
    elif lengths is None and goal is not None and tangent_vec_goal is None:
        o = env._wrapped_env.reset(lengths, goal, tangent_vec_goal)
    else:
        o = env.reset()
    agent.reset()
    if render_mode:
        env.render(mode=render_mode, save_frames=save_frames)
    while True:
        a, agent_info = agent.get_action(o) # agent = policy
#        print(agent_info)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        if render_mode:
            env.render(mode=render_mode, save_frames=save_frames)
        if d:
            env.render(mode=render_mode, save_frames=save_frames)
            observations.append(env.observation_space.flatten(next_o)) # also append terminal observation
            env_infos.append(env_info) # only append terminal info
            break
        o = next_o

    if not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )