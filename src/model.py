from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence
import gym

import numpy as np
import torch as th
from torch.optim import Adam

from tqdm import tqdm
import src.policy
from src.env import ShadedPVEnv
from src.experience import Experience, ExperienceSource, ReplayBuffer
from src.policy import Policy
from src import rl
from src.noise import GaussianNoise
from src.schedule import ConstantSchedule, LinearSchedule
import src.utils


class Model(ABC):
    def __init__(
        self,
        env: gym.Env,
        policy_name: str,
        policy_kwargs: Dict[Any, Any],
        exp_source_kwargs: Dict[Any, Any],
    ):
        policy_kwargs = policy_kwargs or {}
        exp_source_kwargs = exp_source_kwargs or {}

        self.env = env
        self.policy = self.get_policy(policy_name, self.env, policy_kwargs)
        self.exp_source = ExperienceSource(self.policy, self.env, **exp_source_kwargs)

        self._save_config_dic()

    @abstractmethod
    def learn(self) -> None:
        """Run the learning process"""

    def predict(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        return self.policy(obs=obs, info=info)

    def _save_config_dic(self) -> None:
        config_path = self.path.joinpath("config.json")
        with open(config_path, "w") as f:
            f.write(json.dumps(self.all_config_dics()))

    def quit(self) -> None:
        self.env.quit()

    def play_episode(self) -> Sequence[Experience]:
        return self.exp_source.play_episode()

    def all_config_dics(self) -> Dict[str, Dict[str, str]]:
        dic = {
            "self": self.config_dic,
            "policy": self.policy.config_dic,
            "env": self.env.config_dic,
            "exp_sorce": self.exp_source.config_dic,
        }
        return dic

    @property
    def config_dic(self) -> Dict[str, Dict[str, str]]:
        dic = {k: str(v) for k, v in self.__dict__.items()}
        return dic

    @property
    def path(self) -> str:
        return self.env.path

    @staticmethod
    def get_policy(
        policy_name: str,
        env: ShadedPVEnv,
        policy_kwargs: Dict[Any, Any],
    ) -> Policy:
        """Return a policy by name"""
        dic = {
            "po": "PerturbObservePolicy",
            "random": "RandomPolicy",
            "mlp": "MLPPolicy",
        }
        if policy_name not in dic.keys():
            raise NotImplementedError(f"The policy {policy_name} is not implemented")

        policy_kwargs_ = policy_kwargs.copy()
        policy_kwargs_["action_space"] = env.action_space

        return getattr(src.policy, dic[policy_name])(**policy_kwargs_)


class DummyModel(Model):
    """Dummy model for testing. This model does not learn"""

    def __init__(
        self,
        env: gym.Env,
        policy_name: str,
        policy_kwargs: Dict[Any, Any],
        exp_source_kwargs: Dict[Any, Any],
    ):
        super().__init__(env, policy_name, policy_kwargs, exp_source_kwargs)

    def _save_config_dic(self) -> None:
        pass

    def learn(self) -> None:
        pass


class PerturbObserveModel(Model):
    """Perturb & Obserb model. This model does not need training"""

    def __init__(
        self,
        env_kwargs: Optional[Dict[Any, Any]] = None,
        policy_kwargs: Optional[Dict[Any, Any]] = None,
        exp_source_kwargs: Dict[Any, Any] = None,
    ):
        env_kwargs = env_kwargs or {}
        env_kwargs.update({"num_envs": 1, "env_names": ["po"]})
        env = ShadedPVEnv.get_envs(**env_kwargs)

        super().__init__(
            env,
            policy_name="po",
            policy_kwargs=policy_kwargs,
            exp_source_kwargs=exp_source_kwargs,
        )

    def learn(self) -> None:
        return None


class RandomModel(Model):
    """Random model. This model does not need training"""

    def __init__(
        self,
        env_kwargs: Optional[Dict[Any, Any]] = None,
        policy_kwargs: Optional[Dict[Any, Any]] = None,
        exp_source_kwargs: Dict[Any, Any] = None,
    ):
        env_kwargs = env_kwargs or {}
        env_kwargs.update({"num_envs": 1, "env_names": ["random"]})
        env = ShadedPVEnv.get_envs(**env_kwargs)

        super().__init__(
            env,
            policy_name="random",
            policy_kwargs=policy_kwargs,
            exp_source_kwargs=exp_source_kwargs,
        )

    def learn(self) -> None:
        return None


class DDPG(Model):
    """DDPG Model"""

    def __init__(
        self,
        env_kwargs: Optional[Dict[Any, Any]] = None,
        policy_kwargs: Optional[Dict[Any, Any]] = None,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        actor_l2: float = 0.0,
        critic_l2: float = 0.0,
        tau: float = 1e-3,
        gamma: float = 0.99,
        n_steps: int = 1,
        norm_rewards: bool = False,
        train_steps: int = 1,
        collect_steps: int = 1,
        prefill_buffer: Optional[int] = None,
    ):
        if env_kwargs is None:
            env_kwargs_ = {}
        else:
            env_kwargs_ = env_kwargs.copy()
        env_kwargs_.update({"num_envs": 2, "env_names": ["ddpg_train", "ddpg_test"]})
        env, self.env_test = ShadedPVEnv.get_envs(**env_kwargs_)

        self.actor, self.critic = rl.create_mlp_actor_critic(env=env)

        test_policy = self.get_policy(
            "mlp", self.env_test, policy_kwargs={"net": self.actor}
        )
        self.gamma = gamma
        self.n_steps = n_steps
        exp_source_kwargs = {"gamma": gamma, "n_steps": n_steps}
        self.agent_test_source = ExperienceSource(
            test_policy, self.env_test, exp_source_kwargs
        )

        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.tau = tau
        self.norm_rewards = norm_rewards
        self.train_steps = train_steps
        self.collect_steps = collect_steps

        self.actor_optim = Adam(
            self.actor.parameters(), lr=actor_lr, weight_decay=actor_l2
        )
        self.critic_optim = Adam(
            self.critic.parameters(), lr=critic_lr, weight_decay=critic_l2
        )

        self.actor_target = rl.TargetNet(self.actor)
        self.critic_target = rl.TargetNet(self.critic)
        self.actor_target.sync()
        self.critic_target.sync()

        if policy_kwargs is None:
            policy_kwargs_ = {}
        else:
            policy_kwargs_ = policy_kwargs.copy()
        policy_kwargs_.update({"net": self.actor})
        super().__init__(
            env,
            policy_name="mlp",
            policy_kwargs=policy_kwargs_,
            exp_source_kwargs=exp_source_kwargs,
        )

        # Fill replay buffer
        fill_steps = prefill_buffer or batch_size
        for _ in tqdm(range(fill_steps), desc="Pre-filling replay buffer"):
            self.buffer.append(self.collect_exp_source.play_n_steps())

    def learn(
        self,
        epochs: int = 1000,
    ) -> None:
        for i in tqdm(range(1, epochs + 1)):

            self._train_net(self.train_steps)
            self._collect_steps(self.collect_steps)

    @property
    def collect_exp_source(self) -> ExperienceSource:
        return self.exp_source

    def _train_net(self, train_steps: int) -> None:
        for _ in range(train_steps):
            batch = self._prepare_batch()

            # Critic training
            pred_last_action = self.actor_target(batch.next_obs)
            q_last = self.critic_target(batch.next_obs, pred_last_action).squeeze(-1)
            q_last[batch.done] = 0.0  # Mask the value of terminal states
            q_ref = batch.reward + q_last * self.gamma ** self.n_steps
            q_pred = self.critic(batch.obs, batch.action).squeeze(-1)
            # .detach() to stop gradient propogation for q_ref
            critic_loss = th.nn.functional.mse_loss(q_ref.detach(), q_pred)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # Actor trainig
            act_pred = self.actor(batch.obs)
            actor_loss = -self.critic(batch.obs, act_pred).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.critic_target.alpha_sync(self.tau)
            self.actor_target.alpha_sync(self.tau)

    def _collect_steps(self, collect_steps: int) -> None:
        for _ in range(collect_steps):
            self.buffer.append(self.collect_exp_source.play_n_steps())

    def _prepare_batch(self) -> rl.ExperienceTensorBatch:
        "Get a training batch for network's weights update"
        batch = self.buffer.sample(self.batch_size)
        obs = th.tensor(batch.obs, dtype=th.float32)
        action = th.tensor(batch.action, dtype=th.float32)
        reward = th.tensor(batch.reward, dtype=th.float32)
        done = th.tensor(batch.done, dtype=th.bool)
        next_obs = th.tensor(batch.next_obs, dtype=th.float32)

        if self.norm_rewards:
            reward = reward - reward.mean()
            # reward = (reward - reward.mean()) / (reward.std() + 1e-6)

        return rl.ExperienceTensorBatch(obs, action, reward, done, next_obs)

    @classmethod
    def from_grid(
        cls,
        dic: Dict[Any, Any],
        repeat_run: int = 1,
        training_iter: int = 10,
        epochs: int = 1000,
        **kwargs,
    ) -> Sequence[DDPG]:
        # Get a permutation of the dic if needed
        gg = src.utils.grid_generator_nested(dic)
        for dic in gg:
            for _ in range(repeat_run):
                model = cls(**dic, **kwargs)
                for _ in range(training_iter):
                    model.learn(epochs=epochs)
                    model.agent_test_source.play_episode()
                model.quit()

    @property
    def config_dic(self) -> Dict[str, Dict[str, str]]:
        dic = {k: str(v) for k, v in self.__dict__.items()}
        return dic


if __name__ == "__main__":
    # model = PerturbObserveModel(
    #     env_kwargs={"env_names": ["po_testt"], "weather_paths": ["test_1_4_0.5"]}
    # )
    # model.play_episode()
    # model.quit()

    # model = RandomModel()
    # model.play_episode()
    # model.quit()

    # # Direct run of DDPG
    # model = DDPG(
    #     buffer_size=10_000,
    #     batch_size=256,
    #     gamma=0.1,
    #     norm_rewards=False,
    #     policy_kwargs={
    #         "noise": GaussianNoise(mean=0.0, std=0.2),
    #         "schedule": LinearSchedule(max_steps=5000),
    #         "decrease_noise": True,
    #     },
    # )

    # for _ in range(10):
    #     model.learn(epochs=1_000)
    #     model.agent_test_source.play_episode()
    # model.quit()

    # Grid run of DDPG
    dic = {
        "buffer_size": 100_000,
        "batch_size": 64,
        "actor_lr": 1e-3,
        "critic_lr": 1e-2,
        "actor_l2": 1e-2,
        "critic_l2": 1e-4,
        "gamma": 0.9,
        "n_steps": 1,
        "tau": 1e-3,
        "norm_rewards": False,
        "train_steps": 1,
        "collect_steps": 1,
        "prefill_buffer": 3_000,
        "policy_kwargs": {
            "noise": [GaussianNoise(mean=0.0, std=0.3)],
            "schedule": [LinearSchedule(max_steps=10_000)],
            "decrease_noise": True,
        },
        "env_kwargs": {
            "reward": [2],
            "states": [
                ["norm_voltage", "norm_power", "norm_delta_power"],
            ],
            # "weather_paths": [["test", "test"]],
            "weather_paths": [["test_1_4_0.5", "test_1_4_0.5"]],
            # "weather_paths": [["test_0_4_0.5", "test_0_4_0.5"]],
            # "weather_paths": [["test_uniform", "test_uniform"]],
        },
    }
    models = DDPG.from_grid(
        dic,
        repeat_run=1,
        training_iter=100,
    )
