from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence

import numpy as np

import src.policy
from src.env import ShadedPVEnv
from src.experience import Experience, ExperienceSource
from src.policy import Policy


class Model(ABC):
    def __init__(
        self,
        env: ShadedPVEnv,
        policy_name: str,
        policy_kwargs: Optional[Dict[Any, Any]],
    ):

        self.env = env
        self.policy = self.get_policy(policy_name, env, policy_kwargs)
        self.exp_source = ExperienceSource(self.policy)

        self._save_config_dic()

    @abstractmethod
    def learn(self) -> None:
        """Run the learning process"""

    def predict(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        return self.policy(obs=obs, info=info)

    def _save_config_dic(self) -> None:
        config_path = self.path.joinpath("config.json")
        with open(config_path, "w") as f:
            f.write(json.dumps(self.config_dic))

    def quit(self) -> None:
        self.env.quit()

    def play_episode(self) -> Sequence[Experience]:
        return self.exp_source.play_episode()

    @property
    def config_dic(self) -> Dict[str, Dict[str, str]]:
        dic = {
            "policy": self.policy.config_dic,
        }
        return dic

    @property
    def path(self) -> str:
        return self.env.path

    @staticmethod
    def get_policy(
        policy_name: str,
        env: ShadedPVEnv,
        policy_kwargs: Optional[Dict[Any, Any]] = None,
    ) -> Policy:
        """Return a policy by name"""
        dic = {
            "po": "PerturbObservePolicy",
            "random": "RandomPolicy",
        }
        if policy_name not in dic.keys():
            raise NotImplementedError(f"The policy {policy_name} is not implemented")

        if policy_kwargs is None:
            policy_kwargs = {}

        policy_kwargs_ = policy_kwargs.copy()
        policy_kwargs_["env"] = env

        return getattr(src.policy, dic[policy_name])(**policy_kwargs_)


class SimpleModel(Model):
    """This model does not need training"""

    def __init__(
        self,
        env: ShadedPVEnv,
        policy_name: str,
        policy_kwargs: Optional[Dict[Any, Any]] = None,
    ):
        super().__init__(env, policy_name, policy_kwargs=policy_kwargs)

    def learn(self) -> None:
        return self


if __name__ == "__main__":
    env = ShadedPVEnv.get_envs(num_envs=1, env_names=["po_train"])
    model = SimpleModel(env, policy_name="po")
    model.play_episode()
    model.play_episode()
    # run_model(model)
    model.quit()

    env = ShadedPVEnv.get_envs(num_envs=1, env_names=["random_train"])
    model = SimpleModel(env, policy_name="random")
    model.play_episode()
    # run_model(model)
    model.quit()
