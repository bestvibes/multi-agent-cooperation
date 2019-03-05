def evaluate(q_net, select_action, env, max_trial_number, max_episode_steps):
    scores = []
    trial_number = 0
    while (trial_number < max_trial_number):
        state = list(env.reset())
        episode_step = 0
        while (episode_step < max_episode_steps):
            #env.render()
            action = select_action(state, q_net, 0)
            next_state, reward, done, _ = env.step(action.item())
            if done: break
            state = list(next_state)
            episode_step += 1
        scores.append(episode_step)
        trial_number += 1
    return scores