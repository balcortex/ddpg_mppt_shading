from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from copy import deepcopy

from gym import spaces
import numpy as np
import torch as th
import torch.nn as nn

from src.env import ShadedPVEnv
from src.noise import Noise
from src.schedule import Schedule


class Policy(ABC):
    """
    Policy base class
    Transform observations into actions

    Parameters:
        - action_space: the action space of the environment
        - noise: class that adds noise to the selected actions
        - schedule: class that keeps track of the steps taken in the environment
            and returns a float to decide whether to add noise or not (comparing it to
            a uniform random float)
        - decrese_noise: whether to decrease the noise based on the value of the schudule
            or keeping it constant.
    """

    def __init__(
        self,
        action_space: spaces.Box,
        noise: Optional[Noise],
        schedule: Optional[Schedule],
        decrease_noise: bool,
    ):
        self.action_space = action_space
        self.noise = noise
        self.schedule = schedule
        self.decrease_noise = decrease_noise

        self.low = action_space.low[0]
        self.high = action_space.high[0]

        self.reset()

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
        """"Return the parameters of the class as a dictionary"""
        return {k: str(v) for k, v in self.__dict__.items()}


class RandomPolicy(Policy):
    """
    Policy that returns a random action depending on the action space

    Parameters:
        action_space: the action space of the environment
    """

    def __init__(self, action_space: spaces.Box):
        self.action_space = action_space

    def __call__(
        self,
        obs: np.ndarray,
        info: Optional[Dict[Any, Any]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Get an action according to the observation

        Paramters:
            obs: not used
            info: not used
        """
        return self.action_space.sample()

    def __str__(self):
        return "RandomPolicy"


class PerturbObservePolicy(Policy):
    """
    Perturb & Observe algorithm

    Parameters:
        - action_space: the action space of the environment
        - dc_step: the size perturbation on the duty cycle
        - dv_key: the key of the `delta_voltage` variable pased on the `info` parameter in
            the `call` method
        - dp_key: the key of the `delta_power` variable pased on the `info` parameter in
            the `call` method
        - noise: class that adds noise to the selected actions
        - schedule: class that keeps track of the steps taken in the environment
            and returns a float to decide whether to add noise or not (comparing it to
            a uniform random float)
        - decrese_noise: whether to decrease the noise based on the value of the schudule
            or keeping it constant.
    """

    def __init__(
        self,
        action_space: spaces.Box,
        dc_step: float = 0.01,
        dv_key: str = "delta_voltage",
        dp_key: str = "delta_power",
        noise: Optional[Noise] = None,
        schedule: Optional[Schedule] = None,
        decrease_noise: bool = False,
    ):
        super().__init__(
            action_space=action_space,
            noise=noise,
            schedule=schedule,
            decrease_noise=decrease_noise,
        )

        self.dc_step = dc_step
        self.dv_key = dv_key
        self.dp_key = dp_key

    def __call__(self, obs: np.ndarray, info: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Get an action according to the observation

        Parameters:
            obs: not used
            info: dictionary containing the `dv_key` and `dp_key` variables
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
    """
    Multilayer Perceptron policy.
    Transform observation into actions

    Parameters:
        - action_space: the action space of the environment
        - net: the network that maps the observations to actions. The network outputs must
            be between [0, 1] (tanh activation).
        - noise: class that adds noise to the selected actions
        - schedule: class that keeps track of the steps taken in the environment
            and returns a float to decide whether to add noise or not (comparing it to
            a uniform random float)
        - decrese_noise: whether to decrease the noise based on the value of the schudule
            or keeping it constant.
        - device: the device on which the net runs (`cpu` or `cuda`)
    """

    def __init__(
        self,
        action_space: spaces.Box,
        net: nn.Module,
        noise: Optional[Noise] = None,
        schedule: Optional[Schedule] = None,
        decrease_noise: bool = False,
        device: str = "cpu",
    ):
        super().__init__(
            action_space=action_space,
            noise=noise,
            schedule=schedule,
            decrease_noise=decrease_noise,
        )
        self.net = net
        self.device = device

    @th.no_grad()
    def __call__(self, obs: np.ndarray, info: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Get an action according to the observation

        Parameters:
            obs: observations from the environment
            info: not used
        """
        obs_v = th.tensor(obs, dtype=th.float32)
        actions = self.net(obs_v)
        actions = self.process_actions(actions)
        return actions.cpu().numpy()

    def unscale_actions(self, scled_actions: th.Tensor) -> th.Tensor:
        "Unscale the actions to match the environment limits"
        actions = self.low + (scled_actions + 1) * (self.high - self.low) / 2
        return actions

    def __str__(self) -> str:
        return "MLPPolicy"


if __name__ == "__main__":
    from src.env import ShadedPVEnv

    env = ShadedPVEnv.get_envs(
        num_envs=1, env_names=["po_train"], weather_paths=["train_1_4_0.5"]
    )
    policy = PerturbObservePolicy(action_space=env.action_space, dc_step=0.01)

    obs = env.reset()
    info = {}

    action = policy(obs, info)
    obs, _, _, info = env.step(action)