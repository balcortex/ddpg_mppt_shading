import numbers
import collections
from typing import List, NamedTuple, Sequence, Deque, Dict, Tuple
import gym

import numpy as np
import torch as th

from src.env import ShadedPVEnv
from src.policy import Policy

Tensor = th.Tensor


class Experience(NamedTuple):
    obs: np.ndarray
    action: np.ndarray
    reward: numbers.Real
    done: bool
    next_obs: np.ndarray


class ExperienceBatch(NamedTuple):
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    next_obs: np.ndarray


class ExperienceTensorBatch(NamedTuple):
    obs: Tensor
    action: Tensor
    reward: Tensor
    done: Tensor
    next_obs: Tensor


class PrioritizedExperienceBatch(NamedTuple):
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    next_obs: np.ndarray
    weights: np.ndarray
    indices: np.ndarray


class PrioritizedExperienceTensorBatch(NamedTuple):
    obs: Tensor
    action: Tensor
    reward: Tensor
    done: Tensor
    next_obs: Tensor
    weights: Tensor
    indices: np.ndarray


class ExperienceSource:
    def __init__(
        self, policy: Policy, env: gym.Env, gamma: float = 1.0, n_steps: int = 1
    ):
        self.policy = policy
        self.env = env
        self.gamma = gamma
        self.n_steps = n_steps
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
            # Need to mask the last_state
            return Experience(self.obs, action, reward, done, self.obs)

        obs, self.obs = self.obs, obs_
        return Experience(obs, action, reward, done, self.obs)

    def play_episode(self) -> Sequence[Experience]:
        ep_history = []

        while True:
            experience = self.play_step()
            ep_history.append(experience)

            # if experience.next_obs is experience.obs:
            if self.done:
                self.reset()
                return ep_history

    def play_n_steps(self) -> Experience:
        history: List[Experience] = []
        discounted_reward = 0.0

        for step_idx in range(self.n_steps):
            exp = self.play_step()
            discounted_reward += exp.reward * self.gamma ** step_idx
            history.append(exp)

            if exp.done:
                break

        return Experience(
            obs=history[0].obs,
            action=history[0].action,
            reward=discounted_reward,
            done=history[-1].done,
            next_obs=history[-1].next_obs,
        )

    # @property
    # def env(self) -> ShadedPVEnv:
    #     return self.policy.env

    @property
    def config_dic(self) -> Dict[str, str]:
        ignore = ["obs", "done", "info", "env"]
        return {k: str(v) for k, v in self.__dict__.items() if k not in ignore}


class ReplayBuffer:
    """
    Buffer to save the interactions of the agent with the environment

    Parameters:
        capacity: buffers' capacity to store a experience tuple

    Returns:
    Numpy arrays
    """

    def __init__(self, capacity: int, name: str = ""):
        assert isinstance(capacity, int)
        self.name = name
        self.buffer: Deque[Experience] = collections.deque(maxlen=capacity)

        self._cum_rew = 0
        self._n = 0
        self._min_rew = 0
        self._max_rew = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer {self.buffer.maxlen}"

    def append(self, experience) -> None:
        self._cum_rew += experience.reward
        self._n += 1
        self._max_rew = max(self._max_rew, experience.reward)
        self._min_rew = min(self._min_rew, experience.reward)

        self.buffer.append(experience)

    def sample(self, batch_size: int) -> ExperienceBatch:
        assert (
            len(self.buffer) >= batch_size
        ), f"Cannot sample {batch_size} elements from buffer of length {len(self.buffer)}"
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs, actions, rewards, dones, next_obs = zip(
            *[self.buffer[idx] for idx in indices]
        )

        return ExperienceBatch(
            np.array(obs),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_obs),
        )

    def reward_mean_std(self) -> Tuple[float, float]:
        _, _, rew, *_ = zip(*self.buffer)
        return np.mean(rew), np.std(rew)

    @property
    def config_dic(self) -> Dict[str, str]:
        ignore = ["buffer", "name"]
        return {k: str(v) for k, v in self.__dict__.items() if k not in ignore}

    @property
    def mean_rew(self) -> float:
        return self._cum_rew / self._n

    @property
    def min_rew(self) -> float:
        return self._min_rew

    @property
    def max_rew(self) -> float:
        return self._max_rew


class PrioritizedReplayBuffer(object):
    def __init__(
        self, capacity: int, alpha: float = 0.6, beta: float = 0.4, sort: bool = True
    ):
        self.alpha = alpha
        self.beta = beta
        self.sort = sort
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        self._cum_rew = 0
        self._n = 0
        self._min_rew = 0
        self._max_rew = 0

    def __len__(self):
        return len(self.buffer)

    def __repr__(self) -> str:
        return f"PrioritizedReplayBuffer {self.capacity}"

    def append(self, experience) -> None:
        self._cum_rew += experience.reward
        self._n += 1
        self._max_rew = max(self._max_rew, experience.reward)
        self._min_rew = min(self._min_rew, experience.reward)

        max_prio = self.priorities.max() if self.buffer else 1.0
        self.priorities[self.pos] = max_prio

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.pos = (self.pos + 1) % self.capacity

        if self.sort:
            ind = np.argsort(self.priorities)[::-1]
            self.priorities = self.priorities[ind]

            buf = []
            for i, _ in zip(ind, self.buffer):
                buf.append(self.buffer[i])
            self.buffer = buf

            # self.priorities = self.priorities[::-1]
            # self.buffer = self.buffer[::-1]

    def sample(self, batch_size: int) -> PrioritizedExperienceBatch:
        assert (
            len(self.buffer) >= batch_size
        ), f"Cannot sample {batch_size} elements from buffer of length {len(self.buffer)}"

        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        # print(indices)

        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        obs, actions, rewards, dones, next_obs = zip(
            *[self.buffer[idx] for idx in indices]
        )

        return PrioritizedExperienceBatch(
            np.array(obs),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_obs),
            np.array(weights, dtype=np.float32),
            np.array(indices, dtype=np.int32),
        )

    def update_priorities(
        self,
        batch_indices: Sequence[int],
        batch_priorities: Sequence[numbers.Real],
        tau: float = 0.1,
    ):
        for idx, prio in zip(batch_indices, batch_priorities):
            old_prio = self.priorities[idx]
            new_prio = (1 - tau) * old_prio + tau * prio
            self.priorities[idx] = new_prio

    def reward_mean_std(self) -> Tuple[float, float]:
        _, _, rew, *_ = zip(*self.buffer)
        return np.mean(rew), np.std(rew)

    @property
    def config_dic(self) -> Dict[str, str]:
        ignore = ["buffer", "pos", "priorities"]
        return {k: str(v) for k, v in self.__dict__.items() if k not in ignore}

    @property
    def mean_rew(self) -> float:
        return self._cum_rew / self._n

    @property
    def min_rew(self) -> float:
        return self._min_rew

    @property
    def max_rew(self) -> float:
        return self._max_rew


if __name__ == "__main__":
    from src.env import DummyEnv
    from src.policy import RandomPolicy

    prio_buffer = PrioritizedReplayBuffer(100)
    env = DummyEnv()
    policy = RandomPolicy(env.action_space)
    exp_source = ExperienceSource(policy, env, gamma=1.0, n_steps=1)

    for _ in range(3):
        exp = exp_source.play_step()
        prio_buffer.append(exp)

    # from src.model import PerturbObserveModel

    # env = ShadedPVEnv.get_envs(num_envs=1, env_names=["po_train"])
    # model = SimpleModel(env, policy_name="po")
    # buffer = ReplayBuffer(10)

    # for i in range(5):
    #     buffer.append(model.exp_source.play_step())

    # model.quit()

    # model = PerturbObserveModel(exp_source_kwargs={"gamma": 0.1, "n_steps": 2})

    # done = False
    # steps = 0
    # while True:
    #     exp = model.exp_source.play_step()
    #     steps += 1
    #     if exp.done:
    #         print(steps)
    #         break

    # model.env.reset()
    # done = False
    # steps = 0
    # while True:
    #     exp = model.exp_source.play_n_steps()
    #     steps += 1
    #     if exp.done:
    #         print(steps)
    #         break
