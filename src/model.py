from __future__ import annotations
import sys

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Union
import gym
import numbers
import matplotlib

import numpy as np
import torch as th
from torch.optim import Adam

from tqdm import tqdm
import src.policy
from src.env import ShadedPVEnv
from src.experience import (
    Experience,
    ExperienceSource,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    ExperienceTensorBatch,
    PrioritizedExperienceTensorBatch,
)
from src.policy import Policy, RandomPolicy
from src import rl
from src.noise import GaussianNoise
from src.schedule import ConstantSchedule, LinearSchedule
import src.utils
import matplotlib.pyplot as plt


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
        buffer_kwargs: Optional[Dict[Any, Any]] = None,
        test_policy_kwargs: Optional[Dict[Any, Any]] = None,
        batch_size: int = 64,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        actor_l2: float = 0.0,
        critic_l2: float = 0.0,
        tau_critic: float = 1e-3,
        tau_actor: float = 1e-3,
        gamma: float = 0.99,
        n_steps: int = 1,
        norm_rewards: bool = False,
        train_steps: int = 1,
        collect_steps: int = 1,
        prefill_buffer: Optional[int] = None,
        use_per: bool = False,
        warmup: int = 0,
    ):
        """
        per: use the Prioritized Experience Buffer
        """
        if env_kwargs is None:
            env_kwargs_ = {}
        else:
            env_kwargs_ = env_kwargs.copy()
        env_kwargs_.update(self._get_env_names())
        env, self.env_test = ShadedPVEnv.get_envs(**env_kwargs_)

        self.actor, self.critic = rl.create_mlp_actor_critic(
            env=env, weight_init_fn=None
        )

        if test_policy_kwargs is None:
            test_policy_kwargs = {}
        else:
            test_policy_kwargs = test_policy_kwargs.copy()
        test_policy_kwargs.update({"net": self.actor})
        test_policy = self.get_policy(
            "mlp",
            self.env_test,
            policy_kwargs=test_policy_kwargs,
        )
        self.gamma = gamma
        self.n_steps = n_steps
        exp_source_kwargs = {"gamma": gamma, "n_steps": n_steps}
        self.agent_test_source = ExperienceSource(
            test_policy, self.env_test, exp_source_kwargs
        )

        self.use_per = use_per
        self.buffer = (
            ReplayBuffer(**buffer_kwargs)
            if not use_per
            else PrioritizedReplayBuffer(**buffer_kwargs)
        )
        self.batch_size = batch_size
        self.tau_critic = tau_critic
        self.tau_actor = tau_actor
        self.norm_rewards = norm_rewards
        self.train_steps = train_steps
        self.collect_steps = collect_steps
        self.warmup = warmup

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

        # Explore policy
        random_policy = RandomPolicy(env.action_space)
        self.warmup_exp_source = ExperienceSource(random_policy, env, gamma, n_steps)

        # After calling super the class members are not saved

        # Fill replay buffer
        fill_steps = prefill_buffer or batch_size
        for _ in tqdm(range(fill_steps), desc="Pre-filling replay buffer"):
            # self.buffer.append(self.collect_exp_source.play_n_steps())
            self.buffer.append(self.warmup_exp_source.play_n_steps())

        self.step_counter = 0
        self.actor_loss = []
        self.critic_loss = []

        self._pre_warmup_init()

        # Warm-up training
        for i in tqdm(range(1, self.warmup + 1), desc="Warm-up training"):
            self._train_net(train_steps=1)

    def _pre_warmup_init(self) -> None:
        pass

    def learn(
        self,
        epochs: int = 1000,
    ) -> None:
        for i in tqdm(range(1, epochs + 1), desc="Training"):

            self._train_net(self.train_steps)
            self._collect_steps(self.collect_steps)

    @property
    def collect_exp_source(self) -> ExperienceSource:
        return self.exp_source

    def _get_env_names(self) -> Dict[str, Any]:
        return {"num_envs": 2, "env_names": ["ddpg_train", "ddpg_test"]}

    def _train_net(self, train_steps: int) -> None:
        for _ in range(train_steps):
            self.step_counter += 1
            batch = self._prepare_batch()

            # We do this since each weight will squared in MSE loss
            weights = np.sqrt(batch.weights)
            # weights = batch.weights

            # Critic training
            q_target = self.critic_target(
                batch.next_obs, self.actor_target(batch.next_obs)
            )
            q_target[batch.done] = 0.0
            y = (batch.reward + q_target * self.gamma ** self.n_steps).detach()
            q = self.critic(batch.obs, batch.action)
            td_error = y - q
            weighted_td_error = th.mul(td_error, weights)

            # Create a zero tensor to compare against
            zero_tensor = th.zeros_like(weighted_td_error)
            critic_loss = th.nn.functional.mse_loss(weighted_td_error, zero_tensor)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # Actor trainig
            actor_loss = -self.critic(batch.obs, self.actor(batch.obs)).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.critic_target.alpha_sync(self.tau_critic)
            self.actor_target.alpha_sync(self.tau_actor)

            # For prioritized exprience replay
            # Update priorities of experiences with TD errors
            if self.use_per:
                error_np = weighted_td_error.detach().cpu().numpy()
                new_priorities = np.abs(error_np)  # + 1e-6
                self.buffer.update_priorities(batch.indices, new_priorities)

            # Keep track of losses
            self.critic_loss.append(critic_loss.detach().numpy())
            self.actor_loss.append(actor_loss.detach().numpy())

    def _collect_steps(self, collect_steps: int) -> None:
        for _ in range(collect_steps):
            self.buffer.append(self.collect_exp_source.play_n_steps())

    def _prepare_batch(self) -> PrioritizedExperienceTensorBatch:
        "Get a training batch for network's weights update"
        batch = self.buffer.sample(self.batch_size)
        obs = th.tensor(batch.obs, dtype=th.float32)
        action = th.tensor(batch.action, dtype=th.float32)
        reward = th.tensor(batch.reward, dtype=th.float32)
        done = th.tensor(batch.done, dtype=th.bool)
        next_obs = th.tensor(batch.next_obs, dtype=th.float32)

        if self.norm_rewards == 1:
            reward = reward - self.buffer.total_mean_rew
        elif self.norm_rewards == 2:
            mean, std = self.buffer.reward_mean_std()
            reward = (reward - mean) / (std + 1e-6)
        elif self.norm_rewards == 3:
            reward = (reward - reward.mean()) / (reward.std() + 1e-6)
        elif self.norm_rewards == 4:
            reward = (reward - self.buffer.min_rew) / (
                self.buffer.max_rew - self.buffer.min_rew
            )
        else:
            pass

        if self.use_per:
            weights = th.tensor(batch.weights, dtype=th.float32)
            indices = batch.indices
        else:
            weights = th.ones_like(reward)
            indices = np.array([1])  # dummy data

        reward = reward.reshape((self.batch_size, 1))
        weights = weights.reshape((self.batch_size, 1))

        return PrioritizedExperienceTensorBatch(
            obs, action, reward, done, next_obs, weights, indices
        )

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
                for i in range(training_iter):
                    print(f"{i+1}/{training_iter}")
                    model.learn(epochs=epochs)
                    model.agent_test_source.play_episode()
                    model.save_plot_losses()
                model.quit()

        return model

    @property
    def config_dic(self) -> Dict[str, Dict[str, str]]:
        dic = {k: str(v) for k, v in self.__dict__.items()}
        return dic

    def all_config_dics(self) -> Dict[str, Dict[str, str]]:
        dic = super().all_config_dics()
        dic.update({"buffer": self.buffer.config_dic})
        return dic

    def plot_loss(self, loss: str) -> matplotlib.figure.Figure:
        """
        loss: 'critic_loss' or 'actor_loss'
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(range(self.step_counter), getattr(self, loss))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(loss)

        if "critic" in loss:
            ax.set_yscale("log")

        return fig

    def save_plot_losses(self) -> None:
        fig_actor_loss = self.plot_loss("actor_loss")
        fig_critic_loss = self.plot_loss("critic_loss")

        fig_actor_loss.savefig(self.path.joinpath("actor_loss.png"))
        fig_critic_loss.savefig(self.path.joinpath("critic_loss.png"))

        plt.close(fig_actor_loss)
        plt.close(fig_critic_loss)


class TD3(DDPG):
    def __init__(
        self,
        env_kwargs: Optional[Dict[Any, Any]] = None,
        policy_kwargs: Optional[Dict[Any, Any]] = None,
        buffer_kwargs: Optional[Dict[Any, Any]] = None,
        test_policy_kwargs: Optional[Dict[Any, Any]] = None,
        batch_size: int = 64,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        actor_l2: float = 0.0,
        critic_l2: float = 0.0,
        tau_critic: float = 1e-3,
        tau_actor: float = 1e-3,
        gamma: float = 0.99,
        n_steps: int = 1,
        norm_rewards: bool = False,
        train_steps: int = 1,
        collect_steps: int = 1,
        prefill_buffer: Optional[int] = None,
        use_per: bool = False,
        warmup: int = 0,
        policy_delay: int = 2,
        target_action_epsilon_noise: float = 0.001,
    ):

        self.policy_delay = policy_delay
        self.target_epsilon_noise = target_action_epsilon_noise

        super().__init__(
            env_kwargs=env_kwargs,
            policy_kwargs=policy_kwargs,
            buffer_kwargs=buffer_kwargs,
            test_policy_kwargs=test_policy_kwargs,
            batch_size=batch_size,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            actor_l2=actor_l2,
            critic_l2=critic_l2,
            tau_critic=tau_critic,
            tau_actor=tau_actor,
            gamma=gamma,
            n_steps=n_steps,
            norm_rewards=norm_rewards,
            train_steps=train_steps,
            collect_steps=collect_steps,
            prefill_buffer=prefill_buffer,
            use_per=use_per,
            warmup=warmup,
        )

    def _pre_warmup_init(self) -> None:
        self.critic2_loss = []
        self.critic2 = rl.create_mlp_critic(self.env, weight_init_fn=None)

        params = self.critic_optim.param_groups[0]
        self.critic2_optim = Adam(
            self.critic2.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )

        self.critic2_target = rl.TargetNet(self.critic2)
        self.critic2_target.sync()

    def _train_net(self, train_steps: int) -> None:
        for _ in range(train_steps):
            self.step_counter += 1
            batch = self._prepare_batch()

            # We do this since each weight will squared in MSE loss
            weights = np.sqrt(batch.weights)

            # Add noise to the actions
            act_target = self.actor_target(batch.next_obs)
            noise = th.rand_like(act_target) * 2 - 1  # Normal noise between [-1, 1]
            noise *= self.target_epsilon_noise
            act_target += noise
            act_target = act_target.clamp(-1, 1)

            # Critics training
            q_critic1 = self.critic_target(batch.next_obs, act_target)
            q_critic2 = self.critic2_target(batch.next_obs, act_target)
            q_critic1[batch.done] = 0.0
            q_critic2[batch.done] = 0.0

            # Compute target with the minumim of both critics
            y = (
                batch.reward + th.min(q_critic1, q_critic2) * self.gamma ** self.n_steps
            ).detach()

            # Critic 1 training
            q_target_1 = self.critic(batch.obs, batch.action)
            td_error_1 = y - q_target_1
            weighted_td_error_1 = th.mul(td_error_1, weights)
            zero_tensor_1 = th.zeros_like(weighted_td_error_1)
            critic_loss_1 = th.nn.functional.mse_loss(
                weighted_td_error_1, zero_tensor_1
            )
            self.critic_optim.zero_grad()
            critic_loss_1.backward()
            self.critic_optim.step()

            # Critic 2 training
            q_target_2 = self.critic2(batch.obs, batch.action)
            td_error_2 = y - q_target_2
            weighted_td_error_2 = th.mul(td_error_2, weights)
            zero_tensor_2 = th.zeros_like(weighted_td_error_2)
            critic_loss_2 = th.nn.functional.mse_loss(
                weighted_td_error_2, zero_tensor_2
            )
            self.critic2_optim.zero_grad()
            critic_loss_2.backward()
            self.critic2_optim.step()

            self.critic_target.alpha_sync(self.tau_critic)
            self.critic2_target.alpha_sync(self.tau_critic)

            # Actor trainig
            if self.step_counter % self.policy_delay == 0 or self.step_counter == 1:
                actor_loss = -self.critic(batch.obs, self.actor(batch.obs)).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                self.actor_target.alpha_sync(self.tau_actor)
            else:
                actor_loss = th.tensor(self.actor_loss[-1])

            # For prioritized exprience replay
            # Update priorities of experiences with TD errors
            if self.use_per:
                error_np = weighted_td_error_1.detach().cpu().numpy()
                new_priorities = np.abs(error_np) + 1e-6
                self.buffer.update_priorities(batch.indices, new_priorities)

            # Keep track of losses
            self.critic_loss.append(critic_loss_1.detach().numpy())
            self.critic2_loss.append(critic_loss_2.detach().numpy())
            self.actor_loss.append(actor_loss.detach().numpy())

    # def _train_net(self, train_steps: int) -> None:
    #     for _ in range(train_steps):
    #         self.step_counter += 1
    #         batch = self._prepare_batch()

    #         # We do this since each weight will squared in MSE loss
    #         weights = np.sqrt(batch.weights)

    #         # Add noise to the actions
    #         act_target = self.actor_target(batch.next_obs)
    #         noise = th.rand_like(act_target) * 2 - 1  # Normal noise between [-1, 1]
    #         act_target += noise * self.target_epsilon_noise
    #         act_target = act_target.clamp(-1, 1)

    #         # Critics training
    #         q_critic1 = self.critic_target(batch.next_obs, act_target)
    #         q_critic2 = self.critic2_target(batch.next_obs, act_target)
    #         q_critic1[batch.done] = 0.0
    #         q_critic2[batch.done] = 0.0

    #         q_cat = th.cat((q_critic1, q_critic2), dim=1)
    #         min_ = th.min(q_cat, dim=1)
    #         indices = min_.indices.unsqueeze(-1)
    #         q_min = q_cat.gather(1, indices)

    #         # Compute target with the minumim of both critics
    #         # y = (
    #         #     batch.reward + th.min(q_critic1, q_critic2) * self.gamma ** self.n_steps
    #         # ).detach()
    #         y = (batch.reward + q_min * self.gamma ** self.n_steps).detach()

    #         # Critic 1 training
    #         q_target_1 = self.critic(batch.obs, batch.action)
    #         td_error_1 = y - q_target_1
    #         weighted_td_error_1 = th.mul(td_error_1, weights)
    #         zero_tensor_1 = th.zeros_like(weighted_td_error_1)
    #         critic_loss_1 = th.nn.functional.mse_loss(
    #             weighted_td_error_1, zero_tensor_1
    #         )
    #         self.critic_optim.zero_grad()
    #         critic_loss_1.backward()
    #         self.critic_optim.step()

    #         # Critic 1 training
    #         q_target_2 = self.critic2(batch.obs, batch.action)
    #         td_error_2 = y - q_target_2
    #         weighted_td_error_2 = th.mul(td_error_2, weights)
    #         zero_tensor_2 = th.zeros_like(weighted_td_error_2)
    #         critic_loss_2 = th.nn.functional.mse_loss(
    #             weighted_td_error_2, zero_tensor_2
    #         )
    #         self.critic2_optim.zero_grad()
    #         critic_loss_2.backward()
    #         self.critic2_optim.step()

    #         self.critic_target.alpha_sync(self.tau_critic)
    #         self.critic2_target.alpha_sync(self.tau_critic)

    #         # Actor trainig
    #         if self.step_counter % self.policy_delay == 0 or self.step_counter == 1:
    #             actor_loss_1 = -self.critic(batch.obs, self.actor(batch.obs))
    #             actor_loss_2 = -self.critic2(batch.obs, self.actor(batch.obs))
    #             actor_loss = (
    #                 th.cat((actor_loss_1, actor_loss_2), dim=1)
    #                 .gather(1, indices)
    #                 .mean()
    #             )
    #             self.actor_optim.zero_grad()
    #             actor_loss.backward()
    #             self.actor_optim.step()

    #             self.actor_target.alpha_sync(self.tau_actor)
    #         else:
    #             actor_loss = th.tensor(self.actor_loss[-1])

    #         # For prioritized exprience replay
    #         # Update priorities of experiences with TD errors
    #         if self.use_per:
    #             error_np = (
    #                 th.cat((weighted_td_error_1, weighted_td_error_2), dim=1)
    #                 .gather(1, indices)
    #                 .detach()
    #                 .cpu()
    #                 .numpy()
    #             )
    #             new_priorities = np.abs(error_np) + 1e-6
    #             self.buffer.update_priorities(batch.indices, new_priorities)

    #         # Keep track of losses
    #         self.critic_loss.append(critic_loss_1.detach().numpy())
    #         self.critic2_loss.append(critic_loss_2.detach().numpy())
    #         self.actor_loss.append(actor_loss.detach().numpy())

    def _get_env_names(self) -> Dict[str, Any]:
        return {"num_envs": 2, "env_names": ["td3_train", "td3_test"]}

    def save_plot_losses(self) -> None:
        super().save_plot_losses()
        fig_critic_loss = self.plot_loss("critic2_loss")
        fig_critic_loss.savefig(self.path.joinpath("critic2_loss.png"))
        plt.close(fig_critic_loss)


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
        "batch_size": 64,  # 64
        "actor_lr": 1e-4,  # 1e-3
        "critic_lr": 1e-3,
        "tau_critic": 1e-4,  # 1e-3
        "tau_actor": 1e-4,  # 1e-4
        "actor_l2": 0,
        "critic_l2": 0,
        "gamma": 0.01,  # 0.6
        "n_steps": 1,
        "norm_rewards": [0, 1, 2, 3, 4],
        "train_steps": 1,  # 5
        "collect_steps": 1,
        "prefill_buffer": 100,
        "use_per": True,  # True,
        # "policy_delay": 2,
        # "target_action_epsilon_noise": 0.001,
        "warmup": 1000,
        "policy_kwargs": {
            "noise": [GaussianNoise(mean=0.0, std=0.3)],
            "schedule": [LinearSchedule(max_steps=10_000)],
            "decrease_noise": True,
        },
        # "test_policy_kwargs": {
        #     "noise": GaussianNoise(0.0, 0.01),
        #     "schedule": ConstantSchedule(1.0),
        # },
        "buffer_kwargs": {
            "capacity": 50_000,
            # "alpha": 0.9,  # 0.9
            # "beta": 0.2,  # 0.2
            # "tau": 1.0,  # 0.9
            # "sort": True,  # False
        },
        "env_kwargs": {
            "reward": [0],
            "states": [
                [
                    "mod1_voltage",
                    "mod2_voltage",
                    "mod3_voltage",
                    "mod4_voltage",
                    "duty_cycle",
                    "delta_duty_cycle",
                    "norm_power",
                ]
            ],
            # "weather_paths": [["test", "test"]],
            "weather_paths": [["train_1_4_0.5", "test_1_4_0.5"]],
            # "weather_paths": [["train_0_4_0.5", "test_0_4_0.5"]],
            # "weather_paths": [["test_uniform", "test_uniform"]],
        },
    }
    # models = DDPG.from_grid(
    #     dic,
    #     repeat_run=10,
    #     epochs=1000,
    #     training_iter=30,
    # )
    models = TD3.from_grid(
        dic,
        repeat_run=10,
        epochs=1000,
        training_iter=50,
    )
