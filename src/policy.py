from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import gym
import numpy as np
import torch as th

from src.noise import Noise
from src.schedule import Schedule


class Policy(ABC):
    """Policy base class"""

    def __init__(
        self,
        env: gym.Env,
        noise: Optional[Noise],
        schedule: Optional[Schedule],
        decrease_noise: bool,
    ):
        self.env = env
        self.noise = noise
        self.schedule = schedule
        self.decrease_noise = decrease_noise

        self.low = env.action_space.low[0]
        self.high = env.action_space.high[0]

    @abstractmethod
    def __call__(
        self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Select an action based on the observation
        The info dictionary may content useful information
        """

    def reset(self) -> None:
        """Reset the epsilon schedule and the noise process"""
        if self.noise is not None:
            self.noise.reset()
        if self.schedule is not None:
            self.schedule.reset()

    def unscale_actions(
        self,
        scled_actions: Union[np.ndarray, th.Tensor],
    ) -> Union[np.ndarray, th.Tensor]:
        """Unscale the actions to match the environment limits"""
        return scled_actions

    def add_noise(
        self,
        actions: Union[np.ndarray, th.Tensor],
    ) -> Union[np.ndarray, th.Tensor]:
        """Add noise to the actions"""
        if self.noise is None:
            return actions

        if self.schedule is not None:
            epsilon = self.schedule()
            self.schedule.step()
        else:
            epsilon = 0.0

        if epsilon > np.random.rand():
            noise = self.noise.sample()
            if self.decrease_noise:
                noise *= epsilon
            actions += noise

        return actions

    def clamp_actions(
        self,
        actions: Union[np.ndarray, th.Tensor],
    ) -> Union[np.ndarray, th.Tensor]:
        """Clamp the actions between environment limits"""
        if isinstance(actions, np.ndarray):
            return actions.clip(self.low, self.high)
        return actions.clamp(self.low, self.high)

    def process_actions(
        self,
        scled_actions: Union[np.ndarray, th.Tensor],
    ) -> Union[np.ndarray, th.Tensor]:
        """Unscale actions, add noise and clamp them"""
        actions = self.unscale_actions(scled_actions)
        actions = self.add_noise(actions)
        actions = self.clamp_actions(actions)
        return actions

    @property
    def config_dic(self) -> Dict[str, str]:
        return {k: str(v) for k, v in self.__dict__.items()}


class RandomPolicy(Policy):
    """
    Policy that returns a random action depending on the environment

    Parameters:
        env: gym environment
    """

    def __init__(self, env: gym.Env):
        self.env = env

    def __call__(
        self,
        obs: np.ndarray,
        info: Optional[Dict[Any, Any]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Get an action according to the observation

        Paramters:
            obs: observations from the environment
            info: additional info passed to the policy (not used)
        """
        return self.env.action_space.sample()

    def __str__(self):
        return "RandomPolicy"


class PerturbObservePolicy(Policy):
    """
    Perturb & Observe algorithm
    """

    def __init__(
        self,
        env: gym.Env,
        dc_step: float = 0.01,
        dv_key: str = "delta_voltage",
        dp_key: str = "delta_power",
        noise: Optional[Noise] = None,
        schedule: Optional[Schedule] = None,
        decrease_noise: bool = False,
    ):
        super().__init__(
            env=env, noise=noise, schedule=schedule, decrease_noise=decrease_noise
        )

        self.dc_step = dc_step
        self.dv_key = dv_key
        self.dp_key = dp_key

    def __call__(self, obs: np.ndarray, info: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Get an action according to the observation

        Parameters:
            obs: observations from the environment
            info: additional info passed to the policy
        """
        delta_p = info.get(self.dp_key, 0.0)
        delta_v = info.get(self.dv_key, 0.0)

        if delta_p >= 0:
            if delta_v > 0:
                action = -self.dc_step
            else:
                action = self.dc_step
        else:
            if delta_v >= 0:
                action = self.dc_step
            else:
                action = -self.dc_step

        action = np.array([action])
        action = self.process_actions(action)

        return action

    def __str__(self):
        return "PerturbObservePolicy"


class MLPPolicy(Policy):
    def __init__(self):
        pass


if __name__ == "__main__":
    from src.env import ShadedPVEnv

    env = ShadedPVEnv.get_envs(num_envs=1)
    policy = PerturbObservePolicy(env=env, dc_step=0.01)
