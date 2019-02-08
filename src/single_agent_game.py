import copy
import time

from src.trainer import Trainer

class SingleAgentGame(object):
    def __init__(self, f_transition: callable, f_reward: callable, trainer: Trainer):
        self.f_transition = f_transition
        self.f_reward = f_reward
        self.trainer = trainer

    def __call__(self,
                    start_state,
                    f_done: callable,
                    max_training_steps: int=1000,
                    max_episode_steps: int=25):
        training_step = 0
        while(training_step <= max_training_steps):
            state = copy.deepcopy(start_state)

            episode_step = 0
            while(episode_step <= max_episode_steps):
                action = self.trainer.select_action(state)
                #print(action)
                next_state = self.f_transition(state, action)
                reward = self.f_reward(state, action, next_state)

                self.trainer.train_episode_step(state, action, next_state, reward)

                if (f_done(next_state)): break

                state = next_state
                episode_step += 1

            self.trainer.train_training_step()
            training_step += 1

        return self.trainer.get_policy()

class SingleAgentRunner(object):
    def __init__(self, policy, renderer, f_transition: callable, f_done: callable):
        self.policy = policy
        self.renderer = renderer if renderer else lambda state: None
        self.f_transition = f_transition
        self.f_done = f_done

    def __call__(self, start_state, max_running_steps=100, render_interval: float=0.5):
        state = copy.deepcopy(start_state)

        actions = []

        self.renderer(state)
        for i in range(0, max_running_steps):
            action = self.policy(state)
            next_state = self.f_transition(state, action)

            state = next_state
            actions.append(action)

            print(i)
            self.renderer(state)
            if self.f_done(state): break
            time.sleep(render_interval)

        return actions
