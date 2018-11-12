import numpy as np


# Importance Sampling
def importance_sampling(states, actions, rewards, policy_sample_count=0):

    # Dimensions
    [sample_count, horizon, state_dimension] = states.shape
    [_, _, action_dimension] = actions.shape

    if policy_sample_count <= 0:
        policy_sample_count = sample_count

    # Weighting
    if sum(rewards) == 0:
        weights = np.ones(sample_count) / sample_count
    else:
        weights = rewards / sum(rewards)

    # Resampling
    index = np.random.choice(range(sample_count), size=policy_sample_count, p=weights, replace=True)

    # New Trajectories
    states_new = np.zeros((policy_sample_count, horizon, state_dimension))
    actions_new = np.zeros((policy_sample_count, horizon, action_dimension))
    for s in range(policy_sample_count):
        states_new[s] = states[index[s], :, :]
        actions_new[s] = actions[index[s], :, :]

    return [states_new, actions_new]