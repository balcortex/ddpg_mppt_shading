from __future__ import annotations

import json
import numbers
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Union

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch.optim import Adam
from tqdm import tqdm
import stable_baselines3 as sb3

import src.policy
import src.utils
from src import rl
from src.env import DummyEnv, ShadedPVEnv
from src.experience import (
    Experience,
    ExperienceSource,
    ExperienceTensorBatch,
    PrioritizedExperienceBatch,
    PrioritizedExperienceTensorBatch,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from src.noise import GaussianNoise
from src.policy import PerturbObservePolicy, Policy, RandomPolicy, SB3Policy
from src.schedule import ConstantSchedule, LinearSchedule


class Model(ABC):
    def __init__(
        self,
        env: gym.Env,
        policy_name: str,
        policy_kwargs: Optional[Dict[Any, Any]] = None,
        exp_source_kwargs: Optional[Dict[Any, Any]] = None,
    ):
        policy_kwargs = policy_kwargs or {}
        exp_source_kwargs = exp_source_kwargs or {}

        self.env = env
        self.policy = self.get_policy(policy_name, self.env, policy_kwargs)
        self.exp_source = ExperienceSource(self.policy, self.env, **exp_source_kwargs)

    @abstractmethod
    def learn(self) -> None:
        """Run the learning process"""

    def predict(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        return self.policy(obs=obs, info=info)

    def quit(self) -> None:
        self.env.quit()

    def play_episode(self) -> Sequence[Experience]:
        return self.exp_source.play_episode()

    def save_config_dic(self) -> None:
        config_path = self.path.joinpath("config.json")
        with open(config_path, "w") as f:
            f.write(json.dumps(self.all_config_dics))

    @property
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

    @classmethod
    def run_from_grid(
        cls,
        dic: Dict[Any, Any],
        episodes: int = 1,
        **kwargs,
    ) -> None:
        # Get a permutation of the dic if needed
        gg = src.utils.grid_product(dic)
        for dic in gg:
            model = cls(**dic, **kwargs)
            for _ in tqdm(range(episodes), desc="Playing episodes"):
                model.play_episode()
            model.quit()

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

        return getattr(src.policy, dic[policy_name])(
            action_space=env.action_space, **policy_kwargs
        )


class DummyModel(Model):
    """Dummy model for testing. This model does not learn"""

    def __init__(
        self,
        env: Optional[gym.Env] = None,
        policy_name: Optional[Dict[Any, Any]] = None,
        policy_kwargs: Optional[Dict[Any, Any]] = None,
        exp_source_kwargs: Optional[Dict[Any, Any]] = None,
    ):
        if not env:
            env = DummyEnv()

        super().__init__(env, policy_name, policy_kwargs, exp_source_kwargs)

    def learn(self) -> None:
        pass


class PerturbObserveModel(Model):
    """Perturb & Obserb model. This model does not need training"""

    def __init__(
        self,
        dc_step: float = 0.01,
        env_kwargs: Optional[Dict[Any, Any]] = None,
        policy_kwargs: Optional[Dict[Any, Any]] = None,
        exp_source_kwargs: Optional[Dict[Any, Any]] = None,
    ):
        env_kwargs = src.utils.new_dic(env_kwargs)
        env_kwargs.setdefault("env_names", ["po"])
        env = ShadedPVEnv.get_envs(**env_kwargs)

        policy_kwargs = src.utils.new_dic(policy_kwargs)
        policy_kwargs.setdefault("dc_step", dc_step)

        super().__init__(
            env,
            policy_name="po",
            policy_kwargs=policy_kwargs,
            exp_source_kwargs=exp_source_kwargs,
        )

        self.save_config_dic()

    def learn(self) -> None:
        return None


class RandomModel(Model):
    """Random model. This model does not need training"""

    def __init__(
        self,
        env_kwargs: Dict[Any, Any] = {},
        policy_kwargs: Dict[Any, Any] = {},
        exp_source_kwargs: Dict[Any, Any] = {},
    ):
        env_kwargs = env_kwargs or {}
        env_kwargs.setdefault("env_names", ["random"])
        env = ShadedPVEnv.get_envs(**env_kwargs)

        super().__init__(
            env,
            policy_name="random",
            policy_kwargs=policy_kwargs,
            exp_source_kwargs=exp_source_kwargs,
        )

    def learn(self) -> None:
        return None


class TrainableModel(Model):
    def __init__(
        self,
        env: gym.Env,
        policy_name: str,
        policy_kwargs: Optional[Dict[Any, Any]],
        exp_source_kwargs: Optional[Dict[Any, Any]],
    ):
        super().__init__(
            env,
            policy_name,
            policy_kwargs=policy_kwargs,
            exp_source_kwargs=exp_source_kwargs,
        )

    @abstractmethod
    def learn(self) -> None:
        """Run the learning process"""

    @abstractmethod
    def save_plot_losses(self) -> None:
        """Save the losses during the training process"""

    def run(
        self,
        total_timesteps: int,
        val_every_timesteps: int = -1,
        n_eval_episodes: int = 1,
    ):
        val_every_timesteps = val_every_timesteps or total_timesteps
        iters = total_timesteps // val_every_timesteps

        for it in range(iters):
            self.learn(val_every_timesteps)
            for ep in range(n_eval_episodes):
                self.play_test_episode()
            self.save_plot_losses()

    def play_test_episode(self) -> None:
        self.agent_exp_source.play_episode()

    @property
    def collect_exp_source(self) -> ExperienceSource:
        return self.exp_source

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
                model.run(total_timesteps, val_every_timesteps)
                model.quit()


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


class DDPG(TrainableModel):
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
        norm_rewards: int = 0,
        train_steps: int = 1,
        collect_steps: int = 1,
        prefill_buffer: Optional[int] = None,
        use_per: bool = False,
        warmup: int = 0,
    ):
        """
        per: use the Prioritized Experience Buffer
        """
        env_kwargs = env_kwargs or {}
        env_kwargs.setdefault("env_names", ["ddpg_train", "ddpg_test"])
        env, self.env_test = ShadedPVEnv.get_envs(**env_kwargs)
        self.actor, self.critic = rl.create_mlp_actor_critic(env=env)

        test_policy_kwargs = test_policy_kwargs or {}
        test_policy_kwargs.update({"net": self.actor})
        test_policy = self.get_policy("mlp", self.env_test, test_policy_kwargs)

        self.gamma = gamma
        self.n_steps = n_steps
        self.use_per = use_per
        self.batch_size = batch_size
        self.tau_critic = tau_critic
        self.tau_actor = tau_actor
        self.norm_rewards = norm_rewards
        self.train_steps = train_steps
        self.collect_steps = collect_steps
        self.prefill_buffer = prefill_buffer
        self.warmup = warmup

        self.agent_exp_source = ExperienceSource(
            test_policy, self.env_test, gamma=gamma, n_steps=n_steps
        )
        self.buffer = (
            ReplayBuffer(**buffer_kwargs)
            if not use_per
            else PrioritizedReplayBuffer(**buffer_kwargs)
        )
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

        policy_kwargs = policy_kwargs or {}
        policy_kwargs.update({"net": self.actor})
        super().__init__(
            env,
            policy_name="mlp",
            policy_kwargs=policy_kwargs,
            exp_source_kwargs={"gamma": gamma, "n_steps": n_steps},
        )

        self.step_counter = 0
        self.actor_loss = []
        self.critic_loss = []

        self._prefill_buffer()
        self._pre_trained = False

        self.save_config_dic()

    def learn(
        self,
        epochs: int = 1000,
    ) -> None:
        if not self._pre_trained:
            for i in tqdm(range(1, self.warmup + 1), desc="Warm-up training"):
                self._train_net(train_steps=1)
            self._pre_trained = True

        for i in tqdm(range(1, epochs + 1), desc="Training"):
            self._train_net(self.train_steps)
            self._collect_steps(self.collect_steps)

    def plot_loss(self, loss: str, log: bool = False) -> matplotlib.figure.Figure:
        """
        loss: 'critic_loss' or 'actor_loss'
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(range(self.step_counter), getattr(self, loss))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(loss)

        if log:
            ax.set_yscale("log")

        return fig

    def save_plot_losses(self) -> None:
        fig_actor_loss = self.plot_loss("actor_loss")
        fig_critic_loss = self.plot_loss("critic_loss", log=True)

        fig_actor_loss.savefig(self.path.joinpath("actor_loss.png"))
        fig_critic_loss.savefig(self.path.joinpath("critic_loss.png"))

        plt.close(fig_actor_loss)
        plt.close(fig_critic_loss)

    def _train_net(self, train_steps: int) -> None:
        for _ in range(train_steps):
            self.step_counter += 1
            batch = self._prepare_batch()

            # We do this since each weight will squared in MSE loss
            weights = np.sqrt(batch.weights)

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
                new_priorities = np.abs(error_np) + 1e-6
                self.buffer.update_priorities(batch.indices, new_priorities)

            # Keep track of losses
            self.critic_loss.append(critic_loss.detach().numpy())
            self.actor_loss.append(actor_loss.detach().numpy())

    def _prefill_buffer(self) -> None:
        # Explore policy
        random_policy = RandomPolicy(self.env.action_space)
        self.warmup_exp_source = ExperienceSource(
            random_policy, self.env, self.gamma, self.n_steps
        )

        # Fill replay buffer
        fill_steps = self.prefill_buffer or self.batch_size
        for _ in tqdm(range(fill_steps), desc="Pre-filling replay buffer"):
            self.buffer.append(self.warmup_exp_source.play_n_steps())

    def _collect_steps(self, collect_steps: int) -> None:
        for _ in range(collect_steps):
            self.buffer.append(self.collect_exp_source.play_n_steps())

    def _prepare_batch(
        self, buffer: Optional[ReplayBuffer] = None
    ) -> PrioritizedExperienceTensorBatch:
        "Get a training batch for network's weights update"

        buffer = buffer or self.buffer

        batch = buffer.sample(self.batch_size)
        obs = th.tensor(batch.obs, dtype=th.float32)
        action = th.tensor(batch.action, dtype=th.float32)
        reward = th.tensor(batch.reward, dtype=th.float32)
        done = th.tensor(batch.done, dtype=th.bool)
        next_obs = th.tensor(batch.next_obs, dtype=th.float32)

        if self.norm_rewards == 1:
            reward = reward - buffer.total_mean_rew
        elif self.norm_rewards == 2:
            mean, std = buffer.reward_mean_std()
            reward = (reward - mean) / (std + 1e-6)
        elif self.norm_rewards == 3:
            reward = (reward - reward.mean()) / (reward.std() + 1e-6)
        elif self.norm_rewards == 4:
            reward = (reward - buffer.min_rew) / (buffer.max_rew - buffer.min_rew)
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

    @property
    def all_config_dics(self) -> Dict[str, Dict[str, str]]:
        dic = super().all_config_dics
        dic.update({"buffer": self.buffer.config_dic})
        return dic


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
        norm_rewards: int = 0,
        train_steps: int = 1,
        collect_steps: int = 1,
        prefill_buffer: Optional[int] = None,
        use_per: bool = False,
        warmup: int = 0,
        policy_delay: int = 2,
        target_action_epsilon_noise: float = 0.001,
        training_type: int = 0,
    ):
        env_kwargs = env_kwargs or {}
        env_kwargs.setdefault("env_names", ["td3_train", "td3_test"])

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

        self.policy_delay = policy_delay
        self.target_epsilon_noise = target_action_epsilon_noise
        self.training_type = training_type

        self.critic2_loss = []
        self.critic2 = rl.create_mlp_critic(self.env, weight_init_fn=None)
        self.critic2_optim = Adam(
            self.critic2.parameters(),
            lr=critic_lr,
            weight_decay=critic_l2,
        )
        self.critic2_target = rl.TargetNet(self.critic2)
        self.critic2_target.sync()

        self.save_config_dic()

    def _train_net(self, train_steps: int) -> None:
        for _ in range(train_steps):
            self.step_counter += 1
            batch = self._prepare_batch()

            # We do this since each weight will squared in MSE loss
            weights = np.sqrt(batch.weights)

            # Add noise to the actions
            act_target = self.actor_target(batch.next_obs)
            noise = th.rand_like(act_target) * 2 - 1  # Normal noise between [-1, 1]
            act_target += noise * self.target_epsilon_noise
            act_target = act_target.clamp(-1, 1)

            # Critics training
            q_critic1 = self.critic_target(batch.next_obs, act_target)
            q_critic2 = self.critic2_target(batch.next_obs, act_target)
            q_critic1[batch.done] = 0.0
            q_critic2[batch.done] = 0.0

            q_cat = th.cat((q_critic1, q_critic2), dim=1)
            min_ = th.min(q_cat, dim=1)
            indices = min_.indices.unsqueeze(-1)
            q_min = q_cat.gather(1, indices)

            # Compute target with the minumim of both critics
            y = (batch.reward + q_min * self.gamma ** self.n_steps).detach()

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

            # Training using TD3 original algorithm (use the critic 1 to
            # calculate the loss of the actor) or use the indices of the
            # minimum q values (new algorithm).
            if self.training_type == 0:
                # Use the q values of the critic 1
                # indices = (indices * 0).detach()
                indices = indices * 0
            else:
                # Use the indices of the min(critic1, critic2)
                # indices = indices.detach()
                pass

            # Actor trainig
            if self.step_counter % self.policy_delay == 0 or self.step_counter == 1:
                actor_loss_1 = -self.critic(batch.obs, self.actor(batch.obs))
                actor_loss_2 = -self.critic2(batch.obs, self.actor(batch.obs))
                actor_loss = (
                    th.cat((actor_loss_1, actor_loss_2), dim=1)
                    .gather(1, indices)
                    .mean()
                )
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                self.actor_target.alpha_sync(self.tau_actor)
            else:
                actor_loss = th.tensor(self.actor_loss[-1])

            # For prioritized exprience replay
            # Update priorities of experiences with TD errors
            if self.use_per:
                error_np = (
                    th.cat((weighted_td_error_1, weighted_td_error_2), dim=1)
                    .gather(1, indices)
                    .detach()
                    .cpu()
                    .numpy()
                )
                new_priorities = np.abs(error_np) + 1e-6
                self.buffer.update_priorities(batch.indices, new_priorities)

            # Keep track of losses
            self.critic_loss.append(critic_loss_1.detach().numpy())
            self.critic2_loss.append(critic_loss_2.detach().numpy())
            self.actor_loss.append(actor_loss.detach().numpy())

    def save_plot_losses(self) -> None:
        super().save_plot_losses()
        fig_critic_loss = self.plot_loss("critic2_loss", log=True)
        fig_critic_loss.savefig(self.path.joinpath("critic2_loss.png"))
        plt.close(fig_critic_loss)


class TD3Exp(TD3):
    def __init__(
        self,
        env_kwargs: Optional[Dict[Any, Any]] = None,
        policy_kwargs: Optional[Dict[Any, Any]] = None,
        buffer_kwargs: Optional[Dict[Any, Any]] = None,
        demo_buffer_kwargs: Optional[Dict[Any, Any]] = None,
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
        norm_rewards: int = 0,
        train_steps: int = 1,
        collect_steps: int = 1,
        prefill_buffer: Optional[int] = None,
        use_per: bool = False,
        warmup: int = 0,
        policy_delay: int = 2,
        target_action_epsilon_noise: float = 0.001,
        training_type: int = 0,
        lambda_bc: float = 1.0,
        q_filter: bool = False,
    ):
        env_kwargs = src.utils.new_dic(env_kwargs)
        env_kwargs.setdefault("env_names", ["td3exp_train", "td3exp_test"])

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
            policy_delay=policy_delay,
            target_action_epsilon_noise=target_action_epsilon_noise,
            training_type=training_type,
        )

        self.lambda_bc = lambda_bc
        self.q_filter = q_filter

        self.bc_loss = []
        self.q_filter_num = []

        self.demo_buffer = (
            ReplayBuffer(**demo_buffer_kwargs)
            if not use_per
            else PrioritizedReplayBuffer(**demo_buffer_kwargs)
        )
        self._fill_demo_buffer()

        self.save_config_dic()

    def _fill_demo_buffer(self) -> None:
        # PO policy
        po_policy = PerturbObservePolicy(self.env.action_space)
        self.expert_exp_source = ExperienceSource(
            po_policy, self.env, self.gamma, self.n_steps
        )

        # Fill demo replay buffer
        fill_steps = self.demo_buffer.capacity
        for _ in tqdm(range(fill_steps), desc="Filling demo buffer"):
            self.demo_buffer.append(self.expert_exp_source.play_n_steps())

    def save_plot_losses(self) -> None:
        super().save_plot_losses()
        bc_loss = self.plot_loss("bc_loss", log=True)
        bc_loss.savefig(self.path.joinpath("bc_loss.png"))
        plt.close(bc_loss)

        q_f = self.plot_loss("q_filter_num")
        q_f.savefig(self.path.joinpath("q_filter_num.png"))
        plt.close(q_f)

    def _train_net(self, train_steps: int) -> None:
        for _ in range(train_steps):
            self.step_counter += 1

            batch = self._prepare_batch(self.buffer)

            # We do this since each weight will squared in MSE loss
            weights = np.sqrt(batch.weights)

            # Add noise to the actions
            act_target = self.actor_target(batch.next_obs)
            noise = th.rand_like(act_target) * 2 - 1  # Normal noise between [-1, 1]
            act_target += noise * self.target_epsilon_noise
            act_target = act_target.clamp(-1, 1)

            # Critics training
            q_critic1 = self.critic_target(batch.next_obs, act_target)
            q_critic2 = self.critic2_target(batch.next_obs, act_target)
            q_critic1[batch.done] = 0.0
            q_critic2[batch.done] = 0.0

            q_cat = th.cat((q_critic1, q_critic2), dim=1)
            min_ = th.min(q_cat, dim=1)
            indices = min_.indices.unsqueeze(-1)
            q_min = q_cat.gather(1, indices)

            # Compute target with the minumim of both critics
            y = (batch.reward + q_min * self.gamma ** self.n_steps).detach()

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

            # Training using TD3 original algorithm (use the critic 1 to
            # calculate the loss of the actor) or use the indices of the
            # minimum q values (new algorithm).
            if self.training_type == 0:
                # Use the q values of the critic 1
                indices = (indices * 0).detach()
                # indices = indices * 0
            else:
                # Use the indices of the min(critic1, critic2)
                indices = indices.detach()
                # pass

            # Actor trainig
            if self.step_counter % self.policy_delay == 0 or self.step_counter == 1:
                # BC Loss
                demo_batch = self._prepare_batch(buffer=self.demo_buffer)
                demo_weights = np.sqrt(demo_batch.weights)
                agent_action = self.actor_target(demo_batch.obs)

                if self.q_filter:
                    q1_expert = self.critic_target(demo_batch.obs, demo_batch.action)
                    q2_expert = self.critic2_target(demo_batch.obs, demo_batch.action)
                    q_expert = th.min(q1_expert, q2_expert)

                    q1_agent = self.critic_target(demo_batch.obs, agent_action)
                    q2_agent = self.critic2_target(demo_batch.obs, agent_action)
                    q_agent = th.min(q1_agent, q2_agent)

                    ind = (q_expert > q_agent).detach()
                    q_filter_num = sum(ind)

                else:
                    ind = th.ones_like(agent_action, dtype=th.bool)
                    q_filter_num = 0

                error = demo_batch.action[ind] - agent_action[ind]
                print(demo_batch.action[ind])
                print(agent_action[ind])
                weighted_error = th.mul(error, demo_weights)
                zero_tensor = th.zeros_like(weighted_error)
                bc_loss = th.nn.functional.mse_loss(weighted_error, zero_tensor)
                bc_loss *= self.lambda_bc
                if th.isnan(bc_loss) or not self._pre_trained:
                    bc_loss = th.tensor(0.0)
                else:
                    if self.use_per:
                        error_demo = weighted_error.detach().cpu().numpy()
                        new_priorities = np.abs(error_demo) + 1e-6
                        print(demo_batch.indices[ind.squeeze(-1)])
                        print(new_priorities)
                        self.demo_buffer.update_priorities(
                            demo_batch.indices[ind.squeeze(-1)], new_priorities
                        )

                # RL Loss
                actor_loss_1 = -self.critic(batch.obs, self.actor(batch.obs))
                actor_loss_2 = -self.critic2(batch.obs, self.actor(batch.obs))
                actor_loss_rl = (
                    th.cat((actor_loss_1, actor_loss_2), dim=1)
                    .gather(1, indices)
                    .mean()
                )

                actor_loss = actor_loss_rl + bc_loss

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                self.actor_target.alpha_sync(self.tau_actor)
            else:
                actor_loss = th.tensor(self.actor_loss[-1])
                q_filter_num = self.q_filter_num[-1]
                bc_loss = th.tensor(self.bc_loss[-1])

            # For prioritized exprience replay
            # Update priorities of experiences with TD errors
            if self.use_per:
                error_np = (
                    th.cat((weighted_td_error_1, weighted_td_error_2), dim=1)
                    .gather(1, indices)
                    .detach()
                    .cpu()
                    .numpy()
                )
                new_priorities = np.abs(error_np) + 1e-6
                self.buffer.update_priorities(batch.indices, new_priorities)

            # Keep track of losses
            self.critic_loss.append(critic_loss_1.detach().numpy())
            self.critic2_loss.append(critic_loss_2.detach().numpy())
            self.actor_loss.append(actor_loss.detach().numpy())
            self.bc_loss.append(bc_loss.detach().numpy())
            self.q_filter_num.append(q_filter_num)


if __name__ == "__main__":
    pass

    # td3_dic = {
    #     "batch_size": 64,  # 64
    #     "actor_lr": 1e-4,  # 1e-3
    #     "critic_lr": 1e-3,
    #     "tau_critic": 1e-4,  # 1e-3
    #     "tau_actor": 1e-4,  # 1e-4
    #     "actor_l2": 0,
    #     "critic_l2": 0,
    #     "gamma": 0.01,  # 0.6
    #     "n_steps": 1,
    #     "norm_rewards": 0,
    #     "train_steps": 1,  # 5
    #     "collect_steps": 1,
    #     "prefill_buffer": 1000,
    #     "use_per": True,  # True,
    #     "warmup": 1000,
    #     "policy_delay": 2,
    #     "target_action_epsilon_noise": 0.001,
    #     "training_type": 0,
    #     "policy_kwargs": {
    #         "noise": [GaussianNoise(mean=0.0, std=0.3)],
    #         "schedule": [LinearSchedule(max_steps=10_000)],
    #         "decrease_noise": True,
    #     },
    #     "buffer_kwargs": {
    #         "capacity": 50_000,
    #     },
    #     "env_kwargs": {
    #         "weather_paths": [["train_1_4_0.5", "test_1_4_0.5"]],
    #     },
    # }
    # TD3.run_from_grid(td3_dic, total_timesteps=30_000, val_every_timesteps=1_000)

    dic = {
        "batch_size": 64,  # 64
        "actor_lr": 1e-4,  # 1e-3
        "critic_lr": 1e-3,
        "tau_critic": 1e-4,  # 1e-3
        "tau_actor": 1e-4,  # 1e-4
        "actor_l2": 0,
        "critic_l2": 0,
        "gamma": 0.1,  # 0.6
        "n_steps": 1,
        "norm_rewards": 0,
        "train_steps": 1,  # 5
        "collect_steps": 1,
        "prefill_buffer": 0,
        # "use_per": [False, True],  # True,
        "use_per": True,  # True,
        "warmup": 0,
        "policy_delay": 1,
        "target_action_epsilon_noise": 0.001,
        "training_type": 0,
        # "q_filter": [False, True],
        "q_filter": True,
        "lambda_bc": 0.1,
        # "lambda_bc": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        # "q_filter": True,
        # "lambda_bc": 0.1,
        # "policy_kwargs": {
        #     "noise": [GaussianNoise(mean=0.0, std=0.3)],
        #     "schedule": [LinearSchedule(max_steps=10_000)],
        #     "decrease_noise": True,
        # },
        "buffer_kwargs": {
            "capacity": 50_000,
            # "alpha": 0.1,  # 0.9
            # "beta": 1.0,  # 0.2
        },
        "demo_buffer_kwargs": {
            "capacity": 1_000,
            # "alpha": 0.1,  # 0.9
            # "beta": 1.0,  # 0.2
        },
        "env_kwargs": {
            "weather_paths": [["train_1_4_0.5", "test_1_4_0.5"]],
        },
    }
    TD3Exp.run_from_grid(
        dic, total_timesteps=30_000, val_every_timesteps=1_000, repeat_run=1
    )
