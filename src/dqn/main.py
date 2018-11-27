import copy

import pytorch

from src.replay_memory import ReplayMemoryPusher
from src.dqn import DQN
from src.dqn.optimize_model import ComputeLoss
from src.dqn.select_action_dqn import select_action_dqn

def dqn_trainer(initial_env,
				start_state: tuple,
				save_path: str,
				learning_rate: float=1e-4,
				batch_size: int=128,
				gamma: float=0.999,
				memory_capacity: int=1000,
				max_training_steps: int=1000,
				max_episode_steps: int=25,
				epsilon: int=0.1,
				target_network_update_interval_steps: int=10):

	# Initialization
	memory = []
	replay_memory_pusher = ReplayMemoryPusher(memory_capacity)

	policy_net = DQN()
	target_net = DQN()
	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters())
    loss_function = ComputeLoss(batch_size, gamma)

    training_step = 0
    while(training_step <= max_training_steps):
    	# reset env
    	env = copy.deepcopy(initial_env)

    	state = start_state
    	episode_step = 0
        while(episode_step <= max_episode_steps):
    		action = select_action_dqn(state, policy_net, epsilon)

    		next_state, reward, done = env(action)

    		# store the transition in memory
    		memory = replay_memory_pusher(memory, state, action, next_state, reward)

			# optimize model
			loss = loss_function(memory, policy_net, target_net)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if done: break

			episode_step += 1

		# update target network
		if (training_step % target_network_update_interval_steps == 0):
			target_net.load_state_dict(policy_net.state_dict())

		training_step += 1

	torch.save(policy_net.state_dict(), save_path)

    