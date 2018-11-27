from dqn import DQN
from optimize_model import ComputeLoss
import pytorch


def main():
	# Parameters
	learning_rate = 1e-4

	# Initilization
	policy_net = DQN()
	target_net = DQN()
    optimizer = torch.optim.Adam(policy_net.parameters())

    while True:
    	# reset env

    	# get state
    	while True:
    		# select action

    		# env.step --> new state; done; reward

    		# Transform into tensor

    		# store the transition in memory

			# optimize model:
				loss = ComputeLoss()
				optimizer.zero_grad()
				loss.backward()

				optimizer.step()

			# check terminal condition

			# update target network

	# save model


    