import numbers
from collections import deque
from typing import List, NamedTuple, Sequence, Dict, Tuple, Optional, Any, Union
import gym

import numpy as np
import torch as th

from src.policy import Policy
from src.env import EnvironmentTracker

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
        self,
        policy: Policy,
        env: gym.Env,
        gamma: float = 1.0,
        n_steps: int = 1,
        env_tracker: Optional[EnvironmentTracker] = None,
    ):
        """
        Class that automates the interaction with the environment, making the reset calls transparent when the episode is finished.
        The methods return trajectories,

        Parameters:
            - policy: the policy that decides the actions taken
            - env: the environment on which the policy performs
            - gamma: the discount factor (when n_steps > 1)
            - n_steps: perform the number of steps in the environment and discount the reward
        """
        self.policy = policy
        self.env = env
        self.gamma = gamma
        self.n_steps = n_steps

        self.env_tracker = env_tracker or EnvironmentTracker(env)
        self.obs = None

        if not self.env_tracker.is_ready:
            self.env_tracker.reset()

    def play_step(self) -> Experience:
        """Play one step in the environent and return the trajectory"""
        if self.env_tracker.done:
            self.env_tracker.reset()

        self.obs = self.env_tracker.obs
        action = self.policy(self.env_tracker.obs, self.env_tracker.info)
        step = self.env_tracker.step(action)

        return Experience(self.obs, action, step.reward, step.done, step.obs)

    def play_episode(self) -> Sequence[Experience]:
        """
        Play one step in the environment until the end of the episode and return the complete trajectory
        """
        ep_history = []

        while True:
            experience = self.play_step()
            ep_history.append(experience)

            if self.env_tracker.done:
                self.env_tracker.reset()
                return ep_history

    def play_n_steps(self) -> Experience:
        """Play `n_steps` in the environment and return the condensed trajectory"""
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

    @property
    def config_dic(self) -> Dict[str, str]:
        """Returns the parameters of the class as a dictionary"""
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
        """Add a experience to the buffer"""
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
        """Sample a random batch of experiences from the buffer"""
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
        """Perform uniform sampling"""
        return np.random.choice(len(self), batch_size, replace=False)

    def reward_mean_std(self) -> Tuple[float, float]:
        """Return the mean and standard deviation of all rewards in the buffer"""
        rew_ = np.array(self._rew_deq)
        return rew_.mean(), rew_.std()

    @property
    def total_mean_rew(self) -> float:
        """Return the total mean reward of all the rewards seen by the buffer"""
        return self._cum_rew / self._n

    @property
    def min_rew(self) -> float:
        """Return the minumum reward seen by the buffer"""
        return self._min_rew

    @property
    def max_rew(self) -> float:
        """Return the maximum reward seen by the buffer"""
        return self._max_rew

    @property
    def config_dic(self) -> Dict[str, str]:
        """Returns the parameters of the class as a dictionary"""
        return {k: str(v) for k, v in self.__dict__.items() if not k.startswith("_")}


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Buffer to save the interactions of the agent with the environment. The sampling is prioritized based on each element priority.

    Parameters:
        capacity: buffers' capacity to store a experience tuple
        alpha: how much prioritization is made (0 -> uniform, 1 -> full prioritization)
        beta: importance sampling (IS) correction for non uniform probabilities
            (0 -> no correction, 1 -> full correction)
        tau: soft update between old priorities and new priorities
        sort: keep the greatest priorities (greatest error) on the right side of the buffer
            (the left side is overwritten with new experiences)

    Returns:
    Numpy arrays
    """

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
        """Keep the experiencies with the greatest priority on the right of the buffer"""
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
        """Performs prioritized sampling"""
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
        """Update the priorities of the given indices"""
        for idx, prio in zip(batch_indices, batch_priorities):
            old_prio = self._prios_deq[idx]
            new_prio = (1 - self.tau) * old_prio + self.tau * float(prio)
            self._prios_deq[idx] = new_prio


if __name__ == "__main__":
    from src.env import DummyEnv
    from src.policy import RandomPolicy

    env = DummyEnv(max_steps=5)
    es = ExperienceSource(RandomPolicy(env.action_space), env)

    es.env_tracker.counter_total_steps
    es.env_tracker.counter_episode_steps
    es.env_tracker.counter_total_episodes
    es.env_tracker._cum_reward
    es.env_tracker.history["ep_reward"]

    es.play_step()
    # es.env_tracker.reset()

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
