from __future__ import annotations

import abc
import datetime
import itertools
import numbers
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces

from src import utils
from src.pvsys import ShadedArray, SimulinkModelOutput

DEFAULT_WEATHER_PATH = Path("data/synthetic_weather.csv")
DEFAULT_STATES = (
    "norm_voltage",
    "norm_delta_voltage",
    "norm_power",
    "norm_delta_power",
    "duty_cycle",
    "delta_duty_cycle",
)
DEFAULT_LOG_STATES = (
    "norm_voltage",
    "norm_delta_voltage",
    "power",
    "optimum_power",
    "date",
    "optimum_duty_cycle",
    "voltage",
    "delta_voltage",
    "delta_power",
)
DEFAULT_PLOT_STATES = {
    # key -> filename, val -> plot from df
    "power": ("power", "optimum_power"),
    "duty_cycle": ("duty_cycle", "optimum_duty_cycle"),
}
DEFAULT_NORM_DIC = {"power": 200, "voltage": 36}
DEFAULT_LOG_PATH = Path("default")


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
        - pvarray: The pvarray
        - weather_df: DataFrame containg the weather to perform the simulations
        - states: List of strings. Return these states as the observations
            Add `norm_` to `state` to get a normalized state.
            Add `delta_` to get the difference of the actual state wrt the last.
            Possible states: "voltage", "power", "date", "duty_cycle", "duty_cycle"
        - log_states: List of strings. Return these states in the info dictionary.
        - dic_normalizer: Provide a dic with the maximum values of the states to perform
            normalization.
        - log_path: Path to save the csv files and png images
        - plot_states: Plot the states in the dictionary. The key is used as a filename
            and the values are the states to be plotted.
    """

    def __init__(
        self,
        pvarray: ShadedArray,
        weather_df: pd.DataFrame,
        states: Sequence[str],
        log_states: Sequence[str] = [],
        dic_normalizer: Optional[Dict[str, numbers.Real]] = None,
        log_path: Optional[Path] = None,
        plot_states: Optional[Dict[str, Union[str, Sequence[str]]]] = None,
    ):
        self.pvarray = pvarray
        self.weather_df = weather_df
        self._name_states = states
        self._name_log_states = log_states
        self._dic_plot_states = plot_states
        self._history = defaultdict(list)
        self._row_idx = -1

        # like set, but ordered  ->  avoid repetition on states and log states
        self._name_all_states = {
            k: None for k in itertools.chain(states, log_states)
        }.keys()

        if log_path:
            self.path = log_path

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
        self._save_history()
        self._history.clear()
        self._weather_comb = {}  # save all the unique weather combinations (ordered)
        self._action: np.ndarray = self.action_space.sample() * 0
        self._row_idx = -1

        return self.step(action=np.array([0.0]))[0]

    def _step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, numbers.Real, bool, Dict[Any, Any]]:
        """Play a step in the environment"""
        self._row_idx += 1
        self._action += action
        self._action = np.clip(self._action, 0.0, 1.0)
        res = self.pvarray.simulate(
            duty_cycle=self._action[0],
            irradiance=self._current_g,
            ambient_temperature=self._current_amb_t,
        )
        self._add_history(res)
        self._weather_comb[(self._current_g, self._current_amb_t)] = None

        self._done = self._row_idx == len(self.weather_df) - 1
        info = self.log_state_as_tuple._asdict()

        return self.state, self.reward, self.done, info

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
                if self._row_idx <= 0
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
        elif "optimum" in state_name:
            name = state_name.replace("optimum_", "")
            result_ = self.pvarray.get_mpp(verbose=False)
            return self._resolve_state(result_, name)
        # `power` is not available directly in the simulation results
        elif state_name == "power":
            return result.voltage * result.current
        elif state_name == "date":
            return str(self.weather_df.index[self._row_idx])

        # No key found
        else:
            raise KeyError(f"The state `{state_name}` is not recognized")

    def _save_history(self, minimim_length: int = 100) -> None:
        # Save the history if any
        if self._row_idx > minimim_length:
            self._now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Save the csv
            path = self.path.joinpath(self.time + ".csv")
            self.to_dataframe(self.history_all).to_csv(path)

            # Save figures
            for k, v in self._dic_plot_states.items():
                self._plot_state_df(k, v)

            # Efficiency
            df = self.to_dataframe(self.history_all, rename_cols=False)
            max_p = np.array(df["optimum_power"])
            p = np.array(df["power"])
            eff = np.mean(p / max_p) * 100
            with open(self.path.joinpath("efficiency.txt"), "a") as f:
                f.write(f"{self.time}:{eff:.2f}\n")

            print(f"Saved to {path}")

    def _plot_state_df(self, name: str, states: Union[str, Sequence[str]]) -> None:
        """Plot states the states and save to a file"""
        df = self.to_dataframe(self.history_all, rename_cols=False)
        ax = df.plot(y=list(states))
        fig = ax.get_figure()
        fig.savefig(self.path.joinpath(f"{self.time}_{name}.png"))
        plt.close()

    def quit(self) -> None:
        self.pvarray.quit()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def reward(self) -> numbers.Real:
        return self._history["norm_delta_power"][-1]

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
    def path(self) -> Path:
        return self._log_path

    @path.setter
    def path(self, path: Path) -> None:
        path.mkdir(exist_ok=True)
        self._log_path = path

    @property
    def _current_g(self) -> Sequence[numbers.Number]:
        """Values of irradiance for the taken step"""
        val = self.weather_df.iloc[self._row_idx][self._key_cols["irradiance"]].values
        return tuple(val)

    @property
    def _current_amb_t(self) -> Sequence[numbers.Number]:
        """Values of ambient temperature for the taken step"""
        val = self.weather_df.iloc[self._row_idx][
            self._key_cols["amb_temperature"]
        ].values
        return tuple(val)

    @property
    def time(self) -> str:
        """Return the time when reset() was called last"""
        return self._now

    @property
    def unique_weathers(self) -> Tuple[Tuple[Tuple[numbers.Real, ...]]]:
        """
        Return all the unique combinations of irradiance and temperature in the
        weather dataframe
        """
        return tuple(self._weather_comb.keys())

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
            log_path=DEFAULT_LOG_PATH,
            plot_states=DEFAULT_PLOT_STATES,
        )

    @classmethod
    def get_envs(
        cls,
        num_envs,
        log_path: Optional[Path] = None,
        env_names: Optional[Sequence[str]] = None,
    ) -> Union[ShadedPVEnv, Sequence[ShadedPVEnv]]:
        """Get a sequence of ShadedPVEnvs, each one logged to a different folder"""
        pvarray = ShadedArray.get_default_array()
        weather_df = utils.csv_to_dataframe(DEFAULT_WEATHER_PATH)
        now_ = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        path = log_path or DEFAULT_LOG_PATH
        path.mkdir(exist_ok=True)

        if env_names is not None:
            paths = [path.joinpath(now_ + "_" + env_name) for env_name in env_names]
        else:
            paths = [
                path.joinpath(f"{now_}_env{str(i).zfill(3)}") for i in range(num_envs)
            ]

        envs = [
            cls(
                pvarray=pvarray,
                weather_df=weather_df,
                states=DEFAULT_STATES,
                log_states=DEFAULT_LOG_STATES,
                dic_normalizer=DEFAULT_NORM_DIC,
                log_path=path,
                plot_states=DEFAULT_PLOT_STATES,
            )
            for path in paths
        ]

        if len(envs) == 1:
            return envs[0]

        return envs

    @staticmethod
    def to_dataframe(dic: Dict[Any, Any], rename_cols: bool = True) -> pd.DataFrame:
        """Convert a dictionary with PV states to a dataframe"""
        df = pd.DataFrame(dic)

        # Rename `delta_voltage` -> `Delta Voltage`
        if rename_cols:
            df.columns = [
                " ".join(c.capitalize() for c in col.split("_")) for col in df.columns
            ]

        # Set `Date` as index
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", drop=True, inplace=True)

        return df
