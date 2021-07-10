from typing import Optional, Dict, Any

import stable_baselines3 as sb3

from src.model import Model
from src.env import ShadedPVEnv
from src.experience import ExperienceSource
from src.policy import SB3Policy
import src.utils


class SB3Model(Model):
    """
    Use Stable Baselines 3 to solve the MPPT problem

    Parameters:
        - model_name: 'ddpg', 'td3', sac, 'a2c' or 'ppo'
        - model_kwargs: keyword arguments for the selected model
            (see the Stable Baseline documentation)
        - env_kwargs: keyword arguments for the enviroment
            (see the ShadedPVEnv documentation)
    """

    def __init__(
        self,
        model_name: str,
        model_kwargs: Optional[Dict[Any, Any]],
        env_kwargs: Optional[Dict[Any, Any]] = None,
        **kwargs,
    ):
        self._model_name = model_name.lower()

        model_kwargs = model_kwargs or {}

        env_kwargs = src.utils.new_dic(env_kwargs)
        env_kwargs.setdefault(
            "env_names",
            [f"sb3_{self._model_name}_train", f"sb3_{self._model_name}_test"],
        )
        self.env, self.test_env = ShadedPVEnv.get_envs(**env_kwargs)

        model = getattr(sb3, self._model_name.upper())
        self.model = model(
            policy="MlpPolicy",
            env=self.env,
            device="cpu",
            **model_kwargs,
            **kwargs,
        )

        self.policy = SB3Policy(self.model)
        self.exp_source = ExperienceSource(
            self.policy, self.env, gamma=model_kwargs["gamma"]
        )

        self.test_policy = SB3Policy(self.model, deterministic=True)
        self.agent_exp_source = ExperienceSource(
            self.policy, self.test_env, gamma=model_kwargs["gamma"]
        )

        self.save_config_dic()

    def learn(
        self,
        timesteps: int,
        val_every_timesteps: int = -1,
        n_eval_episodes: int = 1,
    ):
        self.model = self.model.learn(
            total_timesteps=timesteps,
            log_interval=1,
            reset_num_timesteps=False,
            eval_env=self.test_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=val_every_timesteps,
        )

    @classmethod
    def run_from_grid(
        cls,
        dic: Dict[Any, Any],
        total_timesteps: int,
        val_every_timesteps: int = 0,
        repeat_run: int = 1,
        **kwargs,
    ) -> None:
        # Get a permutation of the dic if needed
        gg = src.utils.grid_product(dic)
        for dic in gg:
            for _ in range(repeat_run):
                model = cls(**dic, **kwargs)
                model.learn(total_timesteps, val_every_timesteps)
                model.quit()


if __name__ == "__main__":
    dic = {
        "model_name": "ppo",
        "model_kwargs": {"gamma": 0.01},
        "env_kwargs": {
            "weather_paths": [["train_1_4_0.5", "test_1_4_0.5"]],
        },
    }
    model = SB3Model.run_from_grid(
        dic, total_timesteps=30_000, val_every_timesteps=1_000
    )
