import numpy as np
def discount_and_normalize_reward(rewards, gamma):
    # discount
    discounted_rewards = np.zeros_like(rewards)
    cum_reward = 0.0
    for i in reversed(range(len(rewards))):
        if rewards[i] == 0:
            cum_reward = 0
        else:
            cum_reward = cum_reward * gamma + rewards[i]
            discounted_rewards[i] = cum_reward
    # for i, reward in enumerate(reversed(rewards)):
    #     cum_reward = cum_reward * gamma + reward
    #     discounted_rewards[i] = cum_reward

    # normalize
    r_mean = np.mean(discounted_rewards)
    r_std = np.std(discounted_rewards)
    for i in range(len(discounted_rewards)):
        discounted_rewards[i] = (discounted_rewards[i] - r_mean) / r_std

    return discounted_rewards
    

