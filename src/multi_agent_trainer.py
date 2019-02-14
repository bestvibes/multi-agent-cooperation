import copy
import time
from itertools import starmap

from src.algorithm import Algorithm

class TrainMultiAgentPolicies(object):
    def __init__(self, f_transition: callable, f_rewards: list, algorithms: list):
        assert(len(f_rewards) == len(algorithms))
        self.f_transition = f_transition
        self.f_rewards = f_rewards
        self.algorithms = algorithms

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
                actions = list(map(lambda t: t.select_action(state), self.algorithms))
                next_state = self.f_transition(state, actions)
                rewards = list(starmap(lambda rf, a: rf(state, a, next_state), zip(self.f_rewards, actions)))

                list(starmap(lambda t, a, r: t.train_episode_step(state, a, next_state, r), zip(self.algorithms, actions, rewards)))

                if (f_done(next_state)): break

                state = next_state
                episode_step += 1

            list(map(lambda t: t.train_training_step(), self.algorithms))
            training_step += 1
            if (training_step % 50 == 0):
                print(f"training step: {training_step}")

        return list(map(lambda t: t.get_policy(), self.algorithms))

class RenderMultiAgentPolicies(object):
    def __init__(self, renderer, policies: list, f_transition: callable, f_done: callable):
        self.policies = policies
        self.renderer = renderer
        self.f_transition = f_transition
        self.f_done = f_done

    def __call__(self, start_state, max_running_steps=100, render_interval: float=0.5):
        state = copy.deepcopy(start_state)

        self.renderer(state)
        for i in range(0, max_running_steps):
            actions = list(map(lambda p: p(state), self.policies))
            next_state = self.f_transition(state, actions)

            state = next_state

            print(i)
            self.renderer(state)
            if self.f_done(state): break
            time.sleep(render_interval)
