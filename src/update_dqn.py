import torch
import torch.nn.functional as F

import src.util_types as UT
import src.util as U

def update_dqn(Q_net, target_net, memory, learning_rate=1, batch_size=1, gamma=0.99):
    old_param = Q_net.parameters()
    print("old", list(old_param))
    optimizer = torch.optim.Adam(Q_net.parameters(), lr=learning_rate)
    state_batch, loss = compute_loss(Q_net, target_net, memory, gamma=gamma, batch_size=batch_size)
    optimizer.zero_grad()
    out=loss.backward()
    optimizer.step()
    # print("old", list(old_param))
    print("new", list(Q_net.parameters()))

    # print("state", state_batch.grad)

    return Q_net


def compute_loss(Q_net, target_net, memory, gamma, batch_size):
    if len(memory) >= batch_size:
        transitions = U.list_batch_random_sample(memory, batch_size)
    else:
        transitions  = U.list_batch_random_sample(memory, len(memory)) 

    batch = UT.Transition(*zip(*transitions))

    state_batch = torch.Tensor(batch.state)
    print(state_batch.grad)
    next_state_batch = torch.Tensor(batch.next_state)
    reward_batch = torch.Tensor(batch.reward)
    action_batch = torch.LongTensor([[action] for action in batch.action])

    state_action_values = Q_net(state_batch).gather(1, action_batch)

    expected_state_action_values = target_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = torch.add((expected_state_action_values * gamma), reward_batch)
    expected_state_action_values = torch.Tensor([[value] for value in expected_state_action_values])

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    # print("loss", loss+1000)

    return state_batch, loss



