import numbers
from collections import deque
from typing import List, NamedTuple, Optional, Sequence, Deque, Dict, Tuple
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

    def __init__(self, capacity: int, **kwargs):
        assert isinstance(capacity, int)
        self.capacity = capacity

        self._obs_deq = deque(maxlen=capacity)
        self._act_deq = deque(maxlen=capacity)
        self._rew_deq = deque(maxlen=capacity)
        self._done_deq = deque(maxlen=capacity)
        self._next_obs_deq = deque(maxlen=capacity)

        self._cum_rew = 0.0
        self._n = 0
        self._min_rew = 0.0
        self._max_rew = 0.0

    def __len__(self) -> int:
        return len(self._obs_deq)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def append(self, experience: Experience) -> None:
        self._cum_rew += experience.reward
        self._n += 1
        self._max_rew = max(self._max_rew, experience.reward)
        self._min_rew = min(self._min_rew, experience.reward)

        self._obs_deq.append(experience.obs)
        self._act_deq.append(experience.action)
        self._rew_deq.append(experience.reward)
        self._done_deq.append(experience.done)
        self._next_obs_deq.append(experience.next_obs)

    def sample(self, batch_size: int) -> ExperienceBatch:
        assert (
            len(self) >= batch_size
        ), f"Cannot sample {batch_size} elements from buffer of length {len(self)}"

        self._indices = self._get_indices(batch_size)

        obs = [self._obs_deq[idx] for idx in self._indices]
        actions = [self._act_deq[idx] for idx in self._indices]
        rewards = [self._rew_deq[idx] for idx in self._indices]
        dones = [self._done_deq[idx] for idx in self._indices]
        next_obs = [self._next_obs_deq[idx] for idx in self._indices]

        return ExperienceBatch(
            np.array(obs),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_obs),
        )

    def _get_indices(self, batch_size: int) -> np.ndarray:
        return np.random.choice(len(self), batch_size, replace=False)

    def reward_mean_std(self) -> Tuple[float, float]:
        rew_ = np.array(self._rew_deq)
        return rew_.mean(), rew_.std()

    @property
    def total_mean_rew(self) -> float:
        return self._cum_rew / self._n

    @property
    def min_rew(self) -> float:
        return self._min_rew

    @property
    def max_rew(self) -> float:
        return self._max_rew

    @property
    def config_dic(self) -> Dict[str, str]:
        return {k: str(v) for k, v in self.__dict__.items() if not k.startswith("_")}


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        tau: float = 1.0,
        sort: bool = False,
    ):
        super().__init__(capacity)

        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.sort = sort
        self._prios_deq = deque(maxlen=capacity)

    def append(self, experience: Experience) -> None:
        super().append(experience)

        max_prio = max(self._prios_deq) if self._prios_deq else 1.0
        self._prios_deq.append(max_prio)

        if self.sort:
            self._sort_by_priority()

    def _sort_by_priority(self) -> None:
        idx = np.argsort(self._prios_deq)

        obs = deque([self._obs_deq[i] for i in idx], maxlen=self.capacity)
        act = deque([self._act_deq[i] for i in idx], maxlen=self.capacity)
        rew = deque([self._rew_deq[i] for i in idx], maxlen=self.capacity)
        done = deque([self._done_deq[i] for i in idx], maxlen=self.capacity)
        next_obs = deque([self._next_obs_deq[i] for i in idx], maxlen=self.capacity)
        prios = deque([self._prios_deq[i] for i in idx], maxlen=self.capacity)

        self._obs_deq = obs
        self._act_deq = act
        self._rew_deq = rew
        self._done_deq = done
        self._next_obs_deq = next_obs
        self._prios_deq = prios

    def sample(self, batch_size: int) -> PrioritizedExperienceBatch:
        exp_batch = super().sample(batch_size)

        weights = (len(self) * self._probs[self._indices]) ** (-self.beta)
        weights /= weights.max()

        return PrioritizedExperienceBatch(
            *exp_batch,
            np.array(weights, dtype=np.float32),
            np.array(self._indices, dtype=np.int32),
        )

    def _get_indices(self, batch_size: int) -> np.ndarray:
        prios = np.array(self._prios_deq)
        probs = prios ** self.alpha
        probs /= probs.sum()

        self._probs = probs

        return np.random.choice(len(self), batch_size, replace=False, p=probs)

    def update_priorities(
        self,
        batch_indices: np.ndarray,
        batch_priorities: np.ndarray,
    ):
        for idx, prio in zip(batch_indices, batch_priorities):
            old_prio = self._prios_deq[idx]
            new_prio = (1 - self.tau) * old_prio + self.tau * float(prio)
            self._prios_deq[idx] = new_prio


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
