import numbers
from typing import NamedTuple, Sequence

import numpy as np

from src.env import ShadedPVEnv
from src.policy import Policy


class Experience(NamedTuple):
    obs: np.ndarray
    action: np.ndarray
    reward: numbers.Real
    done: bool
    next_obs: np.ndarray


class ExperienceSource:
    def __init__(self, policy: Policy):
        self.policy = policy
        self.reset()

    def reset(self) -> None:
        self.obs = self.env.reset()
        self.done = False
        self.info = {}

    def play_step(self) -> Experience:
        if self.done:
            self.reset()

        action = self.policy(self.obs, self.info)
        obs_, reward, done, self.info = self.env.step(action)

        if done:
            self.done = True
            return Experience(self.obs, action, reward, done, None)

        obs, self.obs = self.obs, obs_
        return Experience(obs, action, reward, done, self.obs)

    def play_episode(self) -> Sequence[Experience]:
        ep_history = []

        while True:
            experience = self.play_step()
            ep_history.append(experience)

            if experience.next_obs is None:
                return ep_history

    @property
    def env(self) -> ShadedPVEnv:
        return self.policy.env


if __name__ == "__main__":
    from src.model import SimpleModel

    env = ShadedPVEnv.get_envs(num_envs=1, env_names=["po_train"])
    model = SimpleModel(env, policy_name="po")

    exp_source = ExperienceSource(model.policy)
    exp_source.play_episode()
    # ex

    model.quit()
