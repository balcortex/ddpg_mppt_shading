from __future__ import annotations

import json
import numbers
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch.optim import Adam
from tqdm import tqdm

import src.policy
import src.utils
from src import rl
from src.env import DummyEnv, ShadedPVEnv
from src.experience import (
    Experience,
    ExperienceSource,
    ExperienceTensorBatch,
    PrioritizedExperienceTensorBatch,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from src.policy import PerturbObservePolicy, Policy, RandomPolicy


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

        self._env_steps = 0

    @abstractmethod
    def learn(self) -> None:
        """Run the learning process"""

    def predict(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        return self.policy(obs=obs, info=info)

    def quit(self) -> None:
        self.env.quit()

    def play_episode(self) -> Sequence[Experience]:
        episode = self.exp_source.play_episode()
        self._env_steps += len(episode) * self.exp_source.n_steps
        return episode

    def collect_step(self) -> Experience:
        self._env_steps += self.exp_source.n_steps
        return self.exp_source.play_n_steps()

    def save_log(self) -> Path:
        path = self.path.joinpath("log.txt")
        with open(path, "w") as f:
            f.write(f"Environment steps: {self._env_steps}\n")

        return path

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
    def path(self) -> Path:
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
            model.save_log()
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

        self._train_steps = 0

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
            self.save_log()
            self.save_plot_losses()

    def play_test_episode(self) -> None:
        self.agent_exp_source.play_episode()

    def save_log(self) -> None:
        path = super().save_log()
        with open(path, "a") as f:
            f.write(f"Train steps: {self._train_steps}\n")

    @property
    def collect_exp_source(self) -> ExperienceSource:
        return self.exp_source

    @staticmethod
    def apply_losses(losses: Sequence[th.Tensor], optim: Adam) -> None:

        losses = th.cat(tuple(l.unsqueeze(-1) for l in losses))

        if all(th.isnan(losses)):
            return th.tensor(np.nan)

        mask = th.isnan(losses)
        total_loss = th.sum(losses[~mask])

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        return total_loss

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
        prefill_buffer: int = 0,
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
        self.batch_size = batch_size
        self.tau_critic = tau_critic
        self.tau_actor = tau_actor
        self.norm_rewards = norm_rewards
        self.train_steps = train_steps
        self.collect_steps = collect_steps
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

        # self.step_counter = 0
        self.actor_loss = []
        self.critic_loss = []

        # Fill replay buffer with random policy (explore policy)
        random_policy = RandomPolicy(self.env.action_space)
        explore_exp_source = ExperienceSource(
            random_policy, self.env, self.gamma, self.n_steps
        )
        # if warmup:
        #     prefill_buffer = prefill_buffer or batch_size
        self._env_steps += self.fill_buffer(
            self.buffer,
            explore_exp_source,
            num_experiences=prefill_buffer,
            description="Filling buffer with random experiences",
        )
        self._update_losses(losses={}, repeat=prefill_buffer)

        self._pre_trained = False

        self.save_config_dic()

    @property
    def _loss_names(self) -> Sequence[str]:
        return ["actor_loss", "critic_loss"]

    def learn(
        self,
        epochs: int = 1000,
    ) -> None:
        if not self._pre_trained:
            for i in tqdm(range(1, self.warmup + 1), desc="Warm-up training"):
                self._train_nets(train_steps=1)
            self._pre_trained = True

        for i in tqdm(range(1, epochs + 1), desc="Training"):
            losses = self._train_nets(self.train_steps)

            # Collect steps using the trained policy
            self._env_steps += self.fill_buffer(
                self.buffer, self.collect_exp_source, num_experiences=1
            )

            # Update losses tracking
            self._update_losses(losses)

    def _update_losses(
        self, losses: Optional[Dict[str, numbers.Real]] = None, repeat: int = 1
    ) -> None:
        if not losses:
            for _ in range(repeat):
                for l in self._loss_names:
                    getattr(self, l).append(np.nan)
            return None

        for loss, value in losses.items():
            for _ in range(repeat):
                getattr(self, loss).append(value)

    def plot_loss(self, loss: str, log_y: bool = False) -> matplotlib.figure.Figure:
        """
        loss: 'critic_loss' or 'actor_loss'
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)

        x = np.arange(self._env_steps)
        y = np.array(getattr(self, loss))
        mask = np.isfinite(y)

        ax.plot(x[mask], y[mask])
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(loss)

        if log_y:
            ax.set_yscale("log")

        return fig

    # def save_loss(self, loss: str) -> None:
    #     path = self.path.joinpath(f'{loss}.csv')

    def save_plot_losses(self) -> None:
        fig_actor_loss = self.plot_loss("actor_loss")
        fig_critic_loss = self.plot_loss("critic_loss", log_y=True)

        fig_actor_loss.savefig(self.path.joinpath("actor_loss.png"))
        fig_critic_loss.savefig(self.path.joinpath("critic_loss.png"))

        plt.close(fig_actor_loss)
        plt.close(fig_critic_loss)

    def _train_nets(self, train_steps: int) -> Dict[str, numbers.Real]:
        for _ in range(train_steps):
            self._train_steps += 1

            batch = self.prepare_batch(self.buffer, self.batch_size, self.norm_rewards)

            critic_loss = self._get_critic_loss(batch, self.buffer)
            critic_loss = self.apply_losses([critic_loss], self.critic_optim)

            actor_loss = self._get_actor_loss(batch)
            actor_loss = self.apply_losses([actor_loss], self.actor_optim)

            # Update target networks
            self.critic_target.alpha_sync(self.tau_critic)
            self.actor_target.alpha_sync(self.tau_actor)

            # Keep track of losses
            # self.critic_loss.append(critic_loss.detach().numpy())
            # self.actor_loss.append(actor_loss.detach().numpy())
            return {
                "critic_loss": critic_loss.detach().numpy(),
                "actor_loss": actor_loss.detach().numpy(),
            }

    def _get_critic_loss(
        self, batch: ExperienceTensorBatch, buffer: ReplayBuffer
    ) -> th.Tensor:
        if batch is None:
            return th.tensor(np.nan)

        weights = np.sqrt(batch.weights)

        q_target = self.critic_target(batch.next_obs, self.actor_target(batch.next_obs))
        q_target[batch.done] = 0.0
        y = (batch.reward + q_target * self.gamma ** self.n_steps).detach()
        q = self.critic(batch.obs, batch.action)

        td_error = y - q
        weighted_td_error = th.mul(td_error, weights)

        # Update PER priorities if needed
        self.update_priorities(buffer, batch.indices, weighted_td_error)

        # Create a zero tensor to compare against
        zero_tensor = th.zeros_like(weighted_td_error)

        return th.nn.functional.mse_loss(weighted_td_error, zero_tensor)

    def _get_actor_loss(self, batch: PrioritizedExperienceTensorBatch) -> None:
        if batch is None:
            return th.tensor(np.nan)

        loss = -self.critic(batch.obs, self.actor(batch.obs))
        w_loss = th.mul(loss, batch.weights)
        mean_ = w_loss.mean()
        return mean_

        # return -self.critic(batch.obs, self.actor(batch.obs)).mean()

    @property
    def all_config_dics(self) -> Dict[str, Dict[str, str]]:
        dic = super().all_config_dics
        dic.update({"buffer": self.buffer.config_dic})
        return dic

    @staticmethod
    def prepare_batch(
        buffer: ReplayBuffer,
        batch_size: int,
        norm_rewards: int = 0,
    ) -> PrioritizedExperienceTensorBatch:
        "Get a training batch for network's weights update"

        if len(buffer) < batch_size:
            return None

        batch = buffer.sample(batch_size)
        obs = th.tensor(batch.obs, dtype=th.float32)
        action = th.tensor(batch.action, dtype=th.float32)
        reward = th.tensor(batch.reward, dtype=th.float32)
        done = th.tensor(batch.done, dtype=th.bool)
        next_obs = th.tensor(batch.next_obs, dtype=th.float32)

        if norm_rewards == 0:
            pass
        elif norm_rewards == 1:
            reward = reward - buffer.total_mean_rew
        elif norm_rewards == 2:
            mean, std = buffer.reward_mean_std()
            reward = (reward - mean) / (std + 1e-6)
        elif norm_rewards == 3:
            reward = (reward - reward.mean()) / (reward.std() + 1e-6)
        elif norm_rewards == 4:
            reward = (reward - buffer.min_rew) / (buffer.max_rew - buffer.min_rew)
        else:
            raise NotImplementedError()

        if isinstance(buffer, PrioritizedReplayBuffer):
            weights = th.tensor(batch.weights, dtype=th.float32)
            indices = batch.indices
        else:
            weights = th.ones_like(reward)
            indices = np.array([1])  # dummy data

        reward = reward.reshape((batch_size, 1))
        weights = weights.reshape((batch_size, 1))

        return PrioritizedExperienceTensorBatch(
            obs, action, reward, done, next_obs, weights, indices
        )

    @staticmethod
    def fill_buffer(
        buffer: ReplayBuffer,
        exp_source: ExperienceSource,
        num_experiences: Optional[int] = None,
        description: Optional[str] = None,
    ) -> int:
        if num_experiences == -1:
            num_experiences = buffer.capacity

        for _ in tqdm(
            range(num_experiences), desc=description, disable=description is None
        ):
            exp = exp_source.play_n_steps()
            buffer.append(exp)

        return num_experiences

    @staticmethod
    def update_priorities(
        buffer: PrioritizedReplayBuffer,
        indices: np.ndarray,
        error: th.Tensor,
    ) -> None:
        if not isinstance(buffer, PrioritizedReplayBuffer):
            return None

        prio_np = th.abs(error).detach().cpu().numpy() + 1e-6
        buffer.update_priorities(indices, prio_np)


class DDPGExp(DDPG):
    def __init__(
        self,
        env_kwargs: Optional[Dict[Any, Any]] = None,
        policy_kwargs: Optional[Dict[Any, Any]] = None,
        test_policy_kwargs: Optional[Dict[Any, Any]] = None,
        buffer_kwargs: Optional[Dict[Any, Any]] = None,
        demo_buffer_kwargs: Optional[Dict[Any, Any]] = None,
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
        prefill_buffer: int = 0,
        use_per: bool = False,
        warmup: int = 0,
        lambda_bc: float = 0.1,
        use_q_filter: bool = False,
    ):

        env_kwargs = src.utils.new_dic(env_kwargs)
        env_kwargs.setdefault("env_names", ["ddpgexp_train", "ddpgexp_test"])

        self.lambda_bc = lambda_bc
        self.use_q_filter = use_q_filter

        self.bc_loss = []
        self.q_filter_num = []

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

        self.demo_buffer = (
            ReplayBuffer(**demo_buffer_kwargs)
            if not use_per
            else PrioritizedReplayBuffer(**demo_buffer_kwargs)
        )

        # Fill demo buffer
        po_policy = PerturbObservePolicy(self.env.action_space)
        expert_exp_source = ExperienceSource(
            po_policy, self.env, self.gamma, self.n_steps
        )
        demo_experiences = self.fill_buffer(
            self.demo_buffer,
            expert_exp_source,
            num_experiences=-1,
            description="Filling demo buffer",
        )
        self._env_steps += demo_experiences
        self._update_losses(losses={}, repeat=demo_experiences)

        self.save_config_dic()

    @property
    def _loss_names(self) -> Sequence[str]:
        return super()._loss_names + ["bc_loss", "q_filter_num"]

    def _train_nets(self, train_steps: int) -> Dict[str, numbers.Real]:
        for _ in range(train_steps):
            self._train_steps += 1

            batch = self.prepare_batch(self.buffer, self.batch_size, self.norm_rewards)
            demo_batch = self.prepare_batch(
                self.demo_buffer, self.batch_size, self.norm_rewards
            )

            critic_loss_rl = self._get_critic_loss(batch, self.buffer)
            critic_loss_demo = self._get_critic_loss(demo_batch, self.demo_buffer)
            critic_loss = self.apply_losses(
                [critic_loss_rl, critic_loss_demo], self.critic_optim
            )

            actor_loss_rl = self._get_actor_loss(batch)
            actor_loss_demo = self._get_actor_loss(demo_batch)
            actor_loss_bc, q_filter_num = self._get_bc_loss(demo_batch)
            actor_loss = self.apply_losses(
                [actor_loss_rl, actor_loss_demo, actor_loss_bc], self.actor_optim
            )

            # Update target networks
            self.critic_target.alpha_sync(self.tau_critic)
            self.actor_target.alpha_sync(self.tau_actor)

            return {
                "critic_loss": critic_loss.detach().numpy(),
                "actor_loss": actor_loss.detach().numpy(),
                "bc_loss": actor_loss_bc.detach().numpy(),
                "q_filter_num": q_filter_num,
            }

    def _get_bc_loss(
        self, demo_batch: ExperienceTensorBatch
    ) -> Tuple[th.Tensor, numbers.Real]:
        agent_action = self.actor(demo_batch.obs)

        if self.use_q_filter:
            q_expert = self.critic(demo_batch.obs, demo_batch.action)
            q_agent = self.critic(demo_batch.obs, agent_action)
            ind = (q_expert > q_agent).detach()
            q_filter_num = sum(ind)
        else:
            ind = th.ones_like(agent_action, dtype=th.bool)
            q_filter_num = 0

        error = demo_batch.action[ind] - agent_action[ind]

        demo_weights = np.sqrt(demo_batch.weights)
        weighted_error = th.mul(error, demo_weights[ind])
        zero_tensor = th.zeros_like(weighted_error)
        bc_loss = th.nn.functional.mse_loss(weighted_error, zero_tensor)
        bc_loss *= self.lambda_bc

        return bc_loss, q_filter_num

    def save_plot_losses(self) -> None:
        super().save_plot_losses()
        bc_loss = self.plot_loss("bc_loss", log_y=True)
        bc_loss.savefig(self.path.joinpath("bc_loss.png"))
        plt.close(bc_loss)

        q_f = self.plot_loss("q_filter_num")
        q_f.savefig(self.path.joinpath("q_filter_num.png"))
        plt.close(q_f)


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
    ):
        env_kwargs = env_kwargs or {}
        env_kwargs.setdefault("env_names", ["td3_train", "td3_test"])

        self.policy_delay = policy_delay
        self.target_epsilon_noise = target_action_epsilon_noise

        self.critic2_loss = []

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

        self.critic2 = rl.create_mlp_critic(self.env, weight_init_fn=None)
        self.critic2_optim = Adam(
            self.critic2.parameters(),
            lr=critic_lr,
            weight_decay=critic_l2,
        )
        self.critic2_target = rl.TargetNet(self.critic2)
        self.critic2_target.sync()

        self.save_config_dic()

    @property
    def _loss_names(self) -> Sequence[str]:
        return super()._loss_names + ["critic2_loss"]

    def _get_critic_loss(
        self, batch: ExperienceTensorBatch, buffer: ReplayBuffer
    ) -> Tuple[th.Tensor, th.Tensor]:
        if batch is None:
            return th.tensor(np.nan), th.Tensor(np.nan)

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

        # Compute target with the minumim of both critics
        y = (
            batch.reward + th.min(q_critic1, q_critic2) * self.gamma ** self.n_steps
        ).detach()

        # Critic 1 training
        q_target_1 = self.critic(batch.obs, batch.action)
        td_error_1 = y - q_target_1
        weighted_td_error_1 = th.mul(td_error_1, weights)
        zero_tensor_1 = th.zeros_like(weighted_td_error_1)
        critic_loss_1 = th.nn.functional.mse_loss(weighted_td_error_1, zero_tensor_1)

        # Critic 2 training
        q_target_2 = self.critic2(batch.obs, batch.action)
        td_error_2 = y - q_target_2
        weighted_td_error_2 = th.mul(td_error_2, weights)
        zero_tensor_2 = th.zeros_like(weighted_td_error_2)
        critic_loss_2 = th.nn.functional.mse_loss(weighted_td_error_2, zero_tensor_2)

        # Update PER priorities if needed
        self.update_priorities(buffer, batch.indices, weighted_td_error_1)

        return critic_loss_1, critic_loss_2

    def _train_nets(self, train_steps: int) -> Dict[str, numbers.Real]:
        for _ in range(train_steps):
            self._train_steps += 1

            batch = self.prepare_batch(self.buffer, self.batch_size, self.norm_rewards)

            critic_loss, critic2_loss = self._get_critic_loss(batch, self.buffer)
            critic_loss = self.apply_losses([critic_loss], self.critic_optim)
            critic2_loss = self.apply_losses([critic2_loss], self.critic2_optim)

            if self._train_steps % self.policy_delay == 0:
                actor_loss = self._get_actor_loss(batch)
                actor_loss = self.apply_losses([actor_loss], self.actor_optim)
                self.actor_target.alpha_sync(self.tau_actor)
            else:
                actor_loss = th.tensor(np.nan)

            # Update target networks
            self.critic_target.alpha_sync(self.tau_critic)
            self.critic2_target.alpha_sync(self.tau_critic)

            return {
                "critic_loss": critic_loss.detach().numpy(),
                "critic2_loss": critic2_loss.detach().numpy(),
                "actor_loss": actor_loss.detach().numpy(),
            }

    def save_plot_losses(self) -> None:
        super().save_plot_losses()
        fig_critic_loss = self.plot_loss("critic2_loss", log_y=True)
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
        lambda_bc: float = 1.0,
        use_q_filter: bool = False,
    ):
        env_kwargs = src.utils.new_dic(env_kwargs)
        env_kwargs.setdefault("env_names", ["td3exp_train", "td3exp_test"])

        self.lambda_bc = lambda_bc
        self.use_q_filter = use_q_filter

        self.bc_loss = []
        self.q_filter_num = []

        self.demo_buffer = (
            ReplayBuffer(**demo_buffer_kwargs)
            if not use_per
            else PrioritizedReplayBuffer(**demo_buffer_kwargs)
        )

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
        )

        # Fill demo buffer
        po_policy = PerturbObservePolicy(self.env.action_space)
        expert_exp_source = ExperienceSource(
            po_policy, self.env, self.gamma, self.n_steps
        )
        demo_experiences = self.fill_buffer(
            self.demo_buffer,
            expert_exp_source,
            num_experiences=-1,
            description="Filling demo buffer",
        )
        self._env_steps += demo_experiences
        self._update_losses(losses={}, repeat=demo_experiences)

        self.save_config_dic()

    @property
    def _loss_names(self) -> Sequence[str]:
        return super()._loss_names + ["bc_loss", "q_filter_num"]

    def _train_nets(self, train_steps: int) -> Dict[str, numbers.Real]:
        for _ in range(train_steps):
            self._train_steps += 1

            batch = self.prepare_batch(self.buffer, self.batch_size, self.norm_rewards)
            demo_batch = self.prepare_batch(
                self.demo_buffer, self.batch_size, self.norm_rewards
            )

            critic_loss, critic2_loss = self._get_critic_loss(batch, self.buffer)
            critic_loss_demo, critic2_loss_demo = self._get_critic_loss(
                demo_batch, self.demo_buffer
            )
            critic_loss = self.apply_losses(
                [critic_loss, critic_loss_demo], self.critic_optim
            )
            critic2_loss = self.apply_losses(
                [critic2_loss, critic2_loss_demo], self.critic2_optim
            )

            actor_loss_bc, q_filter_num = self._get_bc_loss(demo_batch)
            if self._train_steps % self.policy_delay == 0:
                actor_loss_rl = self._get_actor_loss(batch)
                actor_loss_demo = self._get_actor_loss(demo_batch)
                actor_loss = self.apply_losses(
                    [actor_loss_rl, actor_loss_demo, actor_loss_bc], self.actor_optim
                )

                self.actor_target.alpha_sync(self.tau_actor)
            else:
                actor_loss = th.tensor(np.nan)

            # Update target networks
            self.critic_target.alpha_sync(self.tau_critic)
            self.critic2_target.alpha_sync(self.tau_critic)

            return {
                "critic_loss": critic_loss.detach().numpy(),
                "critic2_loss": critic2_loss.detach().numpy(),
                "actor_loss": actor_loss.detach().numpy(),
                "bc_loss": actor_loss_bc.detach().numpy(),
                "q_filter_num": q_filter_num,
            }

    def _get_bc_loss(
        self, demo_batch: ExperienceTensorBatch
    ) -> Tuple[th.Tensor, numbers.Real]:
        agent_action = self.actor(demo_batch.obs)

        if self.use_q_filter:
            q1_expert = self.critic(demo_batch.obs, demo_batch.action)
            q2_expert = self.critic2(demo_batch.obs, demo_batch.action)
            q_expert = th.min(q1_expert, q2_expert)

            q1_agent = self.critic(demo_batch.obs, agent_action)
            q2_agent = self.critic2(demo_batch.obs, agent_action)
            q_agent = th.min(q1_agent, q2_agent)
            ind = (q_expert > q_agent).detach()
            q_filter_num = sum(ind)
        else:
            ind = th.ones_like(agent_action, dtype=th.bool)
            q_filter_num = 0

        error = demo_batch.action[ind] - agent_action[ind]

        demo_weights = np.sqrt(demo_batch.weights)
        weighted_error = th.mul(error, demo_weights[ind])
        zero_tensor = th.zeros_like(weighted_error)
        bc_loss = th.nn.functional.mse_loss(weighted_error, zero_tensor)
        bc_loss *= self.lambda_bc

        return bc_loss, q_filter_num

    def save_plot_losses(self) -> None:
        super().save_plot_losses()
        bc_loss = self.plot_loss("bc_loss", log_y=True)
        bc_loss.savefig(self.path.joinpath("bc_loss.png"))
        plt.close(bc_loss)

        q_f = self.plot_loss("q_filter_num")
        q_f.savefig(self.path.joinpath("q_filter_num.png"))
        plt.close(q_f)


if __name__ == "__main__":
    pass

    # dic_ddpgexp = {
    #     "batch_size": 64,  # 64
    #     "actor_lr": 1e-4,  # 1e-3
    #     "critic_lr": 1e-3,
    #     "tau_critic": 1e-3,  # 1e-3
    #     "tau_actor": 1e-3,  # 1e-4
    #     "actor_l2": 0,
    #     "critic_l2": 0,
    #     "gamma": 0.01,  # 0.6
    #     "n_steps": 1,
    #     "norm_rewards": 0,
    #     "train_steps": 1,  # 5
    #     "collect_steps": 1,
    #     "prefill_buffer": 600,
    #     "use_per": [False, True],  # True,
    #     "warmup": 10_000,
    #     "lambda_bc": [0.01, 0.2, 0.5, 0.7, 0.99],
    #     "use_q_filter": True,
    #     "buffer_kwargs": {
    #         "capacity": 50_000,
    #     },
    #     "demo_buffer_kwargs": {
    #         "capacity": 10_000,
    #     },
    #     "env_kwargs": {
    #         "weather_paths": [["train_1_4_0.5", "test_1_4_0.5"]],
    #     },
    # }
    # DDPGExp.run_from_grid(
    #     dic_ddpgexp, total_timesteps=30_000, val_every_timesteps=1_000, repeat_run=1
    # )

    # dic_ddpg = {
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
    #     "prefill_buffer": 0,
    #     "use_per": False,  # True,
    #     "warmup": 1000,
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
    # DDPG.run_from_grid(
    #     dic_ddpg, total_timesteps=30_000, val_every_timesteps=1_000, repeat_run=1
    # )

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
    #     "use_per": False,  # True,
    #     "warmup": 1000,
    #     "policy_delay": 2,
    #     "target_action_epsilon_noise": 0.001,
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
        "prefill_buffer": 600,
        "use_per": False,  # True,
        "warmup": 10_000,
        "policy_delay": 2,
        "target_action_epsilon_noise": 0.001,
        "use_q_filter": True,
        "lambda_bc": 0.1,
        "buffer_kwargs": {
            "capacity": 50_000,
        },
        "demo_buffer_kwargs": {
            "capacity": 10_000,
        },
        "env_kwargs": {
            "weather_paths": [["train_1_4_0.5", "test_1_4_0.5"]],
        },
    }
    TD3Exp.run_from_grid(
        dic, total_timesteps=30_000, val_every_timesteps=1_000, repeat_run=1
    )
