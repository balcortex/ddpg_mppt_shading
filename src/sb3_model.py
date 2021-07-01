from typing import Optional, Dict, Any

import stable_baselines3 as sb3

from src.model import TrainableModel
from src.env import ShadedPVEnv
from src.experience import ExperienceSource
from src.policy import SB3Policy


class SB3Model(TrainableModel):
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

    def run(
        self,
        total_timesteps: int,
        val_every_timesteps: int = -1,
        n_eval_episodes: int = 1,
    ):
        self._env_steps += total_timesteps
        self.learn(total_timesteps, val_every_timesteps, n_eval_episodes)

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

    def save_plot_losses(self) -> None:
        pass


class SB3DDPG(SB3Model):
    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.1,
        verbose: int = 1,
        env_kwargs: Optional[Dict[Any, Any]] = None,
        **kwargs,
    ):

        model_kwargs = {
            "learning_rate": lr,
            "gamma": gamma,
            "verbose": verbose,
        }

        super().__init__(
            "ddpg",
            model_kwargs=model_kwargs,
            env_kwargs=env_kwargs,
            **kwargs,
        )


class SB3TD3(SB3Model):
    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.1,
        verbose: int = 1,
        env_kwargs: Optional[Dict[Any, Any]] = None,
        **kwargs,
    ):
        model_kwargs = {
            "learning_rate": lr,
            "gamma": gamma,
            "verbose": verbose,
        }

        super().__init__(
            "td3",
            model_kwargs,
            env_kwargs=env_kwargs,
            **kwargs,
        )


class SB3A2C(SB3Model):
    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.1,
        verbose: int = 1,
        env_kwargs: Optional[Dict[Any, Any]] = None,
        **kwargs,
    ):
        model_kwargs = {
            "learning_rate": lr,
            "gamma": gamma,
            "verbose": verbose,
        }

        super().__init__(
            "a2c",
            model_kwargs,
            env_kwargs=env_kwargs,
            **kwargs,
        )


class SB3SAC(SB3Model):
    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.1,
        verbose: int = 1,
        env_kwargs: Optional[Dict[Any, Any]] = None,
        **kwargs,
    ):
        model_kwargs = {
            "learning_rate": lr,
            "gamma": gamma,
            "verbose": verbose,
        }

        super().__init__(
            "sac",
            model_kwargs,
            env_kwargs=env_kwargs,
            **kwargs,
        )


class SB3PPO(SB3Model):
    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.1,
        verbose: int = 1,
        env_kwargs: Optional[Dict[Any, Any]] = None,
        **kwargs,
    ):
        model_kwargs = {
            "learning_rate": lr,
            "gamma": gamma,
            "verbose": verbose,
        }

        super().__init__(
            "ppo",
            model_kwargs,
            env_kwargs=env_kwargs,
            **kwargs,
        )
