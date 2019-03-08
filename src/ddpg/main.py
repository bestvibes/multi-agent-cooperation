import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.replay_memory import ReplayMemoryPusher
from src.dqn.dqn import DQN
from src.policy_gradient.policy_net import PolicyNet
from src.dqn.optimize_model import ComputeLoss as CriticLoss

from src.ddpg.actor_loss import ActorLoss

def ddpg_trainer(initial_env,
                start_state: tuple,
                save_path: str,
                learning_rate: float=1e-4,
                batch_size: int=128,
                gamma: float=0.999,
                memory_capacity: int=1000,
                max_training_steps: int=1000,
                max_episode_steps: int=25,
                epsilon: int=0.1,
                target_network_update_interval_steps: int=10,
                plot_interval: float=0.01,
                tau: float=0.001):
    # Initilization
    memory = []
    replay_memory_pusher = ReplayMemoryPusher

    critic = DQN()
    critic_target = DQN()
    critic_target.load_state_dict(critic.state_dict())
    critic_target.eval()

    actor = PolicyNet()
    actor_target = PolicyNet()
    actor_target.load_state_dict(actor.state_dict())
    actor_target.eval()

    critic_optimizer = torch.optim.Adam(critic.parameters())
    critic_loss = CriticLoss(batch_size, gamma)

    actor_optimizer = torch.optim.Adam(actor.parameters())
    actor_loss = ActorLoss()

    training_step = 0
    while(training_step <= max_training_steps):
        # reset env
        env = copy.deepcopy(initial_env)

        state = start_state
        episode_step = 0
        while(episode_step <= max_episode_steps):
            # select action

            

    






