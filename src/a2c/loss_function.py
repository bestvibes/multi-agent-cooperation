def policy_gradient_loss_function(action_distribution,
                                  action,
                                  v_state,
                                  v_next_state,
                                  reward,
                                  gamma):
    q_sa = reward + gamma * v_next_state
    return -action_distribution.log_prob(action) * (q_sa - v_state)