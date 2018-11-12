import numpy as np


def extract_trajectory(states, actions):

    # Dimensions
    [sample_count, horizon, state_dimension] = states.shape
    [_, _, action_dimension] = actions.shape

    # Reshape Inputs and Targets
    inputs = np.reshape(states, (sample_count*horizon, state_dimension))
    targets = np.reshape(actions, (sample_count*horizon, action_dimension))

    return inputs, targets


def rollout_trajectories(env, policy, horizon, sample_count=1):

    # States and Actions
    states = np.zeros((sample_count, horizon, env.stateDimension))
    actions = np.zeros((sample_count, horizon, env.actionDimension))
    rewards = np.zeros((sample_count, horizon))

    # Sample Trajectories
    for s in range(sample_count):
        for t in range(horizon):

            # Initialization
            if t == 0:
                states[s,t,:] = env.reset()
            # Transition and Reward
            else:
                states[s, t, :], rewards[s,t-1], _1, _2 = env.step(states[s, t-1, :], actions[s, t-1, :])

            # Action Selection
            actions[s, t, :] = policy.query(states[s, t, :])

        _, rewards[s, horizon-1], _1, _2 = env.step(states[s, horizon-1, :], actions[s, horizon-1, :])

    return states, actions, rewards
