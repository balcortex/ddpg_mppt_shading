from __future__ import annotations

import abc
import datetime
import itertools
import numbers
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Any, DefaultDict, Dict, NamedTuple, Optional, Sequence, Tuple, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces

from src import utils
from src.pvsys import ShadedArray, SimulinkModelOutput

DEFAULT_WEATHER_PATH = Path("data/synthetic_weather_test.csv")
DEFAULT_STATES = (
    # "norm_delta_mod1_voltage",
    # "norm_delta_mod2_voltage",
    # "norm_delta_mod3_voltage",
    # "norm_delta_mod4_voltage",
    "norm_mod1_voltage",
    "norm_mod2_voltage",
    "norm_mod3_voltage",
    "norm_mod4_voltage",
    # "norm_voltage",
    # "norm_delta_voltage",
    "duty_cycle",
    "delta_duty_cycle",
    "norm_power",
    # "norm_delta_power",
)
DEFAULT_LOG_STATES = (
    "date",
    "power",
    "optimum_power",
    "duty_cycle",
    "optimum_duty_cycle",
    "norm_voltage",
    "norm_delta_voltage",
    "norm_power",
    "norm_delta_power",
    "delta_duty_cycle",
    "voltage",
    "delta_voltage",
    "delta_power",
    "reward",
)
DEFAULT_PLOT_STATES = {
    # key -> filename, val -> plot from df
    # "power": ("power", "optimum_power"),
    "duty_cycle": ("duty_cycle", "optimum_duty_cycle"),
}
DEFAULT_NORM_DIC = {
    "power": 200,
    "voltage": 36,
    "mod1_voltage": 9,
    "mod2_voltage": 9,
    "mod3_voltage": 9,
    "mod4_voltage": 9,
}
DEFAULT_LOG_PATH = Path("default")
DEFAULT_REWARD = 0
DEFAULT_WEATHER_PATH_NAMES = ["train_1_4_0.5", "test_1_4_0.5"]


class EnvStep(NamedTuple):
    obs: np.ndarray
    reward: numbers.Real
    done: bool
    info: Dict[Any, Any]


class EnvironmentTracker:
    def __init__(self, env: gym.Env):
        """
        Keep track of the state of an Environment

        Parameters:
            - env: the environment
        """

        self.env = env

        self.counter_episode_steps = 0
        self.counter_total_steps = 0
        self.counter_total_episodes = -1
        self._logged_at_episode = 0
        self._logged_at_step = 0

        self._cum_reward = np.nan
        self.history = DefaultDict(list)

        self.obs = None

    def reset(self) -> np.ndarray:
        """Reset the environment and return its first observation"""
        if not np.isnan(self._cum_reward):
            self.track("ep_reward", self._cum_reward)
        self._cum_reward = 0.0

        self.counter_total_episodes += 1
        self.counter_episode_steps = 0

        return self(self.env.reset())

    def step(self, action: np.ndarray) -> EnvStep:
        """Take a step in the environment and return the new observation"""
        return self(self.env.step(action))

    def track(self, name: str, value: numbers.Real) -> None:
        """
        Append the `value` to a list in the dictionary `user` whose key is `name`.
        Two lists are created, one that keeps track of all values and the other to
        give the user the ability to modify the list on demand.

        """
        self.history[name].append(value)
        # self.history[name + "_all"].append(value)

    def __call__(self, step: Union[np.ndarray, EnvStep]) -> Union[np.ndarray, EnvStep]:
        """Store the information of `step` and return it without changes"""
        if isinstance(step, np.ndarray):
            self.obs = step
            self.reward = 0.0
            self.done = False
            self.info = {}
        else:
            self._cum_reward += step.reward

            self.counter_total_steps += 1
            self.counter_episode_steps += 1

            self.obs = step.obs
            self.reward = step.reward
            self.done = step.done
            self.info = step.info

        return step

    @property
    def is_ready(self) -> bool:
        return not self.obs is None

    @property
    def new_episode_available(self) -> bool:
        return self.counter_total_episodes > self._logged_at_episode

    @property
    def steps_elapsed(self) -> int:
        return self.counter_total_steps - self._logged_at_step

    @new_episode_available.setter
    def new_episode_available(self, value: bool):
        if value == False:
            self._logged_at_episode = self.counter_total_episodes
            self._logged_at_step = self.counter_total_steps

    def print_tracking(
        self, avg: int = 0, ignore: Optional[Sequence[str]] = None
    ) -> None:
        """
        Print the tracked values

        Parameters:
            - avg: the num of values to average. If `0` average all values

        """
        ignore = ignore or []

        for key, lst in self.history.items():
            if any(ign in key for ign in ignore):
                continue

            mean = np.mean(lst[-avg:])
            std = np.std(lst[-avg:])
            print(f"{key}={mean:0.2f} +/- {std:0.2f}")

    def save_as_csv(self, hist_key: str, parent_path: Path) -> None:
        """
        Save the list contained in the history dict

        Parameters:
            - hist_key: the key
            - parent_path: the path
        """
        arr = np.array(self.history[hist_key])
        arr.tofile(parent_path.joinpath(f"{hist_key}.csv"), sep="\n")

    def save_all_as_csv(
        self, parent_path: Path, ignore: Optional[Sequence[str]] = None
    ) -> None:
        """
        Save all lists in the history dict

        Parameters:
            - parent_path: the path
            - ignore: ignore the keys that contains the strings in this sequence
        """

        ignore = ignore or []

        for key in self.history.keys():
            if any(ign in key for ign in ignore):
                continue
            self.save_as_csv(key, parent_path)

    def save_plot_metrics(self, parent_path: Path) -> None:
        for k, seq in self.history.items():
            self._fig = utils.plot_seq(
                seq,
                ylabel=k,
                allow_ylog=not (
                    "eff" in k or "reward" in k or "q_filter" in k or "actor" in k
                ),
            )
            self._fig.savefig(parent_path.joinpath(f"{k}.png"))
            self._fig.clf()
            plt.close(self._fig)


class CustomEnv(gym.Env, abc.ABC):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self._called_reset = False
        self._done = True

    def step(self, action) -> EnvStep:
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
    def _step(self, action: np.ndarray) -> EnvStep:
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


class DummyEnv(CustomEnv):
    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps
        super().__init__()

    def _reset(self) -> np.ndarray:
        """Reset the environment"""
        self.cur_step = 0
        return np.array([self.cur_step])

    def _step(self, action: np.ndarray) -> EnvStep:
        """Play a step in the environment"""
        self.cur_step += 1

        return EnvStep(
            np.array([self.cur_step]), 1.0, self.cur_step == self.max_steps, {}
        )

    def _get_action_space(self) -> spaces.Space:
        """Return the action space"""
        # The action space is the perturbation applied to the DC-DC converter
        return spaces.Box(low=-1, high=1, shape=(1,))

    def _get_observation_space(self) -> spaces.Space:
        """Return the observation space"""
        return spaces.Box(low=np.array([0]), high=np.array([self.max_steps]))


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
        - reward:   0 -> norm_delta_power
                    1 -> max(norm_delta_power, 0)
                    2 -> -1 if norm_delta_power < 0, else norm_delta_power
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
        reward: int = 0,
    ):
        self.pvarray = pvarray
        self.weather_dfs = [df for _, df in weather_df.groupby(weather_df.index.date)]
        # self.weather_df = weather_df
        self._name_states = states
        self._name_log_states = log_states
        self._dic_plot_states = plot_states
        self._history = defaultdict(list)
        self._row_idx = -1
        self._day_idx = -1

        # like set, but ordered  ->  avoid repetition on states and log states
        self._name_all_states = {
            k: None for k in itertools.chain(states, log_states)
        }.keys()

        if log_path:
            self.path = log_path

        # List of available columns for irradiance and ambient temperature
        self._key_cols = {
            "irradiance": sorted(
                [col for col in self.weather_df.columns if "Irr" in col]
            ),
            "amb_temperature": sorted(
                [col for col in self.weather_df.columns if "Amb" in col]
            ),
        }
        # Make a namedtuple to output states
        self._StatesTuple = namedtuple("States", self._name_states)
        self._LogStatesTuple = namedtuple("LogStates", self._name_log_states)

        self._norm_dic = {} or dic_normalizer
        self._reward = reward

        self.env_tracker = EnvironmentTracker(self)

        super().__init__()

    def __str__(self) -> str:
        return f"ShadedPVEnv"

    def _reset(self) -> np.ndarray:
        """Reset the environment"""
        self._day_idx += 1
        if self._day_idx == self.available_weather_days:
            self._day_idx = 0

        self._save_history()
        self._history.clear()
        self._weather_comb = {}  # save all the unique weather combinations (ordered)
        self._action = np.array([0.25])
        self._row_idx = -1

        return self.step(action=self._action)[0]

    def _step(self, action: np.ndarray) -> EnvStep:
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
        # Add weather to a set to save unique combinations
        self._weather_comb[(self._current_g, self._current_amb_t)] = None

        self._done = self._row_idx == len(self.weather_df) - 1

        info = self.log_state_as_tuple._asdict()

        return EnvStep(self.state, self.reward, self.done, info)

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
        elif state_name == "reward":
            return self.reward

        # No key found
        else:
            raise KeyError(f"The state `{state_name}` is not recognized")

    def _save_history(self, minimim_length: int = 10) -> None:
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

            self.env_tracker.track("ep_efficiency", eff)

            # Rewards
            rew = self.env_tracker.history["ep_reward"][-1]
            with open(self.path.joinpath("ep_reward.txt"), "a") as f:
                f.write(f"{self.time}:{rew:.4f}\n")

    def _plot_state_df(self, name: str, states: Union[str, Sequence[str]]) -> None:
        """Plot states the states and save to a file"""
        df = self.to_dataframe(self.history_all, rename_cols=False)
        self._ax = df.plot(y=list(states))
        self._fig = self._ax.get_figure()
        self._fig.savefig(self.path.joinpath(f"{self.time}_{name}.png"))
        self._fig.clf()
        plt.close(self._fig)

    def quit(self) -> None:
        """Save the dataframe and exit the MATLAB engine"""
        self.reset()  # Save if anything must be saved

        mean_eff = np.mean(self.env_tracker.history["ep_efficiency"])
        with open(self.path.joinpath("efficiency.txt"), "a") as f:
            f.write(f"mean:{mean_eff:.2f}\n")

        # Rewards
        mean_rew = np.mean(self.env_tracker.history["ep_reward"])
        with open(self.path.joinpath("ep_reward.txt"), "a") as f:
            f.write(f"mean:{mean_rew:.4f}\n")

        self.pvarray.quit()

    @property
    def done(self) -> bool:
        """Return `True` if the episode is done"""
        return self._done

    @property
    def reward(self) -> numbers.Real:
        """Return the reward at each step"""
        rew = self._history["norm_delta_power"][-1]
        # rew = self._history["norm_power"][-1]
        # rew = self._history["power"][-1] / self._history["optimum_power"][-1]

        if self._reward == 0:
            return rew
        elif self._reward == 1:
            return max(0, rew)
        elif self._reward == 2:
            return rew if rew > 0 else -1
        else:
            raise NotImplementedError

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
        """Return all the states as a dictionary"""
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
    def weather_df(self) -> pd.DataFrame:
        return self.weather_dfs[self._day_idx]

    @property
    def available_weather_days(self) -> int:
        return len(self.weather_dfs)

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

    @property
    def config_dic(self) -> Dict[str, str]:
        return {
            "states": self._name_states,
            "dic_normalizer": self._norm_dic,
            "reward": self._reward,
        }

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
            reward=DEFAULT_REWARD,
        )

    @classmethod
    def get_envs(
        cls,
        num_envs: Optional[int] = None,
        log_path: Optional[Path] = None,
        env_names: Optional[Sequence[str]] = None,
        weather_paths: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> Union[ShadedPVEnv, Sequence[ShadedPVEnv]]:
        """Get a sequence of ShadedPVEnvs, each one logged to a different folder"""
        weather_paths = weather_paths or DEFAULT_WEATHER_PATH_NAMES

        if weather_paths is not None and env_names is not None:
            assert len(weather_paths) == len(env_names)
            num_envs = len(weather_paths)

        pvarray = ShadedArray.get_default_array()
        now_ = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        path = log_path or DEFAULT_LOG_PATH
        path.mkdir(exist_ok=True)

        if env_names is not None:
            paths = [path.joinpath(now_ + "_" + env_name) for env_name in env_names]
        else:
            num_envs = len(weather_paths)
            paths = [
                path.joinpath(f"{now_}_env{str(i).zfill(3)}") for i in range(num_envs)
            ]

        dic = {
            "pvarray": pvarray,
            "states": DEFAULT_STATES,
            "log_states": DEFAULT_LOG_STATES,
            "dic_normalizer": DEFAULT_NORM_DIC,
            "plot_states": DEFAULT_PLOT_STATES,
            "reward": DEFAULT_REWARD,
        }
        if kwargs is not None:
            dic.update(kwargs)

        # Add Weather dataframes
        if weather_paths is None:
            if num_envs == 1:
                weather_paths = ["test"]
            elif num_envs == 2:
                weather_paths = ["train", "test"]
            else:
                raise NotImplementedError(
                    "Must provide the paths for the weather dataframes"
                )
        # Dic lookup
        weather_dic_path = {
            "test": Path("data/synthetic_weather_test.csv"),
            "test_uniform": Path("data/synthetic_weather_test_uniform.csv"),
            "train": Path("data/synthetic_weather_train.csv"),
            "train_1_4_0.5": Path("data/synthetic_weather_train_1_4_0.5.csv"),
            "test_1_4_0.5": Path("data/synthetic_weather_test_1_4_0.5.csv"),
            "train_0_4_0.5": Path("data/synthetic_weather_train_0_4_0.5.csv"),
            "test_0_4_0.5": Path("data/synthetic_weather_test_0_4_0.5.csv"),
        }
        df_paths = [weather_dic_path[p] for p in weather_paths]
        weather_dfs = [utils.csv_to_dataframe(df_path) for df_path in df_paths]

        envs = [
            cls(weather_df=df, log_path=path, **dic)
            for df, path in zip(weather_dfs, paths)
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
