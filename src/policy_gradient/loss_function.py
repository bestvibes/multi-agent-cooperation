
def policy_gradient_loss_function(action_distribution, action, reward):
    return -action_distribution.log_prob(action) * reward