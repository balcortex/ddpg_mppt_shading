from __future__ import annotations
import numbers
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple, Union
from collections import defaultdict, namedtuple
import itertools

import gym
from gym import spaces
import abc
import numpy as np
import pandas as pd

from src.pvsys import ShadedArray, SimulinkModelOutput
from src import utils

DEFAULT_WEATHER_PATH = Path("data/synthetic_weather.csv")
DEFAULT_STATES = (
    # "voltage",
    "norm_voltage",
    # "delta_voltage",
    "norm_delta_voltage",
    # "power",
    "norm_power",
    # "delta_power",
    # "norm_delta_power",
)
DEFAULT_LOG_STATES = ("power", "date", "duty_cycle", "delta_duty_cycle", "delta_power")
DEFAULT_NORM_DIC = {"power": 200, "voltage": 36}


class CustomEnv(gym.Env, abc.ABC):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self._called_reset = False
        self._done = True

    def step(self, action) -> Tuple[np.ndarray, numbers.Real, bool, Dict[Any, Any]]:
        assert (
            self._called_reset == True
        ), f"Cannot call env.step() before calling reset()"
        assert self._done == False, f"The episode ended, must call reset() first"
        return self._step(action=action)

    def reset(self) -> np.ndarray:
        self._called_reset = True
        self._done = False
        return self._reset()

    def render(self, mode="human"):
        return self._render(mode=mode)

    def close(self):
        return self._close()

    @abc.abstractmethod
    def _step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, numbers.Real, bool, Dict[Any, Any]]:
        """Play a step in the environment and return the observation"""

    @abc.abstractmethod
    def _reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation"""

    @abc.abstractmethod
    def _get_action_space(self) -> spaces.Space:
        """Return the action space of the environment"""

    @abc.abstractmethod
    def _get_observation_space(self) -> spaces.Space:
        """Return the observation space of the environment"""

    def _render(self, mode):
        raise NotImplementedError

    def _close(self):
        raise NotImplementedError


class ShadedPVEnv(CustomEnv):
    """PV system under partial shading that implements the openAI gym interface

    Parameters
        - state: List of strings. Add `norm_` to `state` to get a normalized state.
                Add `delta_` to get the difference of the actual state wrt the last.
            - Possible states: "voltage", "power", "date", "duty_cycle", "duty_cycle"
    """

    def __init__(
        self,
        pvarray: ShadedArray,
        weather_df: pd.DataFrame,
        states: Sequence[str],
        log_states: Sequence[str] = [],
        dic_normalizer: Optional[Dict[str, numbers.Real]] = None,
    ):
        self.pvarray = pvarray
        self.weather_df = weather_df
        self._name_states = states
        self._name_log_states = log_states
        # like set, but ordered
        self._name_all_states = {
            k: None for k in itertools.chain(states, log_states)
        }.keys()
        self._history = defaultdict(list)

        # List of available columns for irradiance and ambient temperature
        self._key_cols = {
            "irradiance": [col for col in self.weather_df.columns if "Irr" in col],
            "amb_temperature": [col for col in self.weather_df.columns if "Amb" in col],
        }
        # Make a namedtuple to output states
        self._StatesTuple = namedtuple("States", self._name_states)
        self._LogStatesTuple = namedtuple("LogStates", self._name_log_states)

        self._norm_dic = {} or dic_normalizer

        super().__init__()

    def _reset(self) -> np.ndarray:
        """Reset the environment"""
        self._history.clear()
        self._action = self.action_space.sample() * 0
        self._row_idx = -1

        return self.step(action=np.array([0.0]))[0]

    def _step(
        self, action: Union[numbers.Real, np.ndarray]
    ) -> Tuple[np.ndarray, numbers.Real, bool, Dict[Any, Any]]:
        """Play a step in the environment"""
        self._row_idx += 1
        self._action += action
        res = self.pvarray.simulate(
            duty_cycle=self._action[0],
            irradiance=self._current_g,
            ambient_temperature=self._current_amb_t,
        )
        self._add_history(res)

        self._done = self._row_idx == len(self.weather_df) - 1
        info = self.log_state_as_tuple._asdict()

        return self.state, self.reward, self.done, info

    def _add_history(self, result: SimulinkModelOutput) -> None:
        """Save the step result in the history"""
        for name in self._name_all_states:
            res = self._resolve_state(result, name)
            self._history[name].append(res)

    def _resolve_state(
        self, result: SimulinkModelOutput, state_name: str
    ) -> Union[numbers.Real, str]:
        """Logic to get the states from the enviroment"""
        # Check if the state is in the keys of the namedtuple and return it
        state = getattr(result, state_name, None)
        if state is not None:
            return state

        # If the state contain the word `delta` return (actual state - last state)
        if "delta" in state_name:
            name = state_name.replace("delta_", "")
            return (
                0
                if self._row_idx == 0
                else (
                    self._resolve_state(result, name)
                    - self._history[name][self._row_idx - 1]
                )
            )
        # If the state is to be normalized
        elif "norm" in state_name:
            name = state_name.replace("norm_", "")
            assert (
                name in self._norm_dic.keys()
            ), f"Must provide a normalizer for {name} in the norm_dic"
            return self._resolve_state(result, name) / self._norm_dic.get(name, 1.0)
        # `power` is not available directly in the simulation results
        elif state_name == "power":
            return result.voltage * result.current
        elif state_name == "date":
            return str(self.weather_df.index[self._row_idx])
        # elif state_name == "duty_cycle":
        #     return result.

        # No key found
        else:
            raise KeyError(f"The state `{state_name}` is not recognized")

    def _get_action_space(self) -> spaces.Space:
        """Return the action space"""
        # The action space is the perturbation applied to the DC-DC converter
        return spaces.Box(low=-1, high=1, shape=(1,))

    def _get_observation_space(self) -> spaces.Space:
        """Return the observation space"""
        # The shape of the observation space match the list of states provided by
        # the user
        len_ = len(self._name_states)
        limit = np.array([np.inf] * len_)
        return spaces.Box(low=-limit, high=limit, shape=limit.shape)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def reward(self) -> numbers.Real:
        return self._history["delta_power"][-1]

    @property
    def state_as_tuple(self) -> NamedTuple[numbers.Real]:
        """Return the states as a namedtuple"""
        return self._StatesTuple(
            *(self._history[state][-1] for state in self._name_states)
        )

    @property
    def state(self) -> np.ndarray:
        """Return the states as a numpy array"""
        return np.array(self.state_as_tuple)

    @property
    def log_state_as_tuple(self) -> NamedTuple[numbers.Real]:
        """Return the states as a namedtuple"""
        return self._LogStatesTuple(
            *(self._history[state][-1] for state in self._name_log_states)
        )

    @property
    def history_all(self) -> Dict[str, Sequence[numbers.Real]]:
        return self._history

    @property
    def history(self) -> Dict[str, Sequence[numbers.Real]]:
        return {
            key: val for key, val in self._history.items() if key in self._name_states
        }

    @property
    def history_log(self) -> Dict[str, Sequence[numbers.Real]]:
        return {
            key: val
            for key, val in self._history.items()
            if key in self._name_log_states
        }

    @property
    def _current_g(self) -> Sequence[numbers.Number]:
        """Values of irradiance for the taken step"""
        val = self.weather_df.iloc[self._row_idx][self._key_cols["irradiance"]].values
        return list(val)

    @property
    def _current_amb_t(self) -> Sequence[numbers.Number]:
        """Values of ambient temperature for the taken step"""
        val = self.weather_df.iloc[self._row_idx][
            self._key_cols["amb_temperature"]
        ].values
        return list(val)

    @classmethod
    def get_default_env(cls) -> ShadedPVEnv:
        """Get a default env"""
        pvarray = ShadedArray.get_default_array()
        weather_df = utils.csv_to_dataframe(DEFAULT_WEATHER_PATH)
        return cls(
            pvarray=pvarray,
            weather_df=weather_df,
            states=DEFAULT_STATES,
            log_states=DEFAULT_LOG_STATES,
            dic_normalizer=DEFAULT_NORM_DIC,
        )

    @staticmethod
    def to_dataframe(dic: Dict[Any, Any]) -> pd.DataFrame:
        df = pd.DataFrame(dic)

        # Rename `delta_voltage` -> `Delta Voltage`
        df.columns = [
            " ".join(c.capitalize() for c in col.split("_")) for col in df.columns
        ]

        # Set `Date` as index
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", drop=True, inplace=True)

        return df


if __name__ == "__main__":
    # env = ShadedPVEnv.get_default_env()
    # env.reset()
    # env.step([0.1])

    # df = ShadedPVEnv.to_dataframe(env.history_all)

    # df.head()

    import gym
    from stable_baselines3 import PPO

    env = ShadedPVEnv.get_default_env()

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # env.render()

    # env.close()
    env.to_dataframe(env.history_all)