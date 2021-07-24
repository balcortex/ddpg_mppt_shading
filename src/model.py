from __future__ import annotations

import json
import numbers
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import gym
import numpy as np
import torch as th
from torch.optim import Adam, Optimizer
from tqdm import tqdm

import src.policy
import src.utils
from src import rl
from src.env import DummyEnv, EnvironmentTracker, ShadedPVEnv
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

    # def predict(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
    #     return self.policy(obs=obs, info=info)

    def quit(self) -> None:
        self.env.quit()

    def play_episode(self) -> Sequence[Experience]:
        episode = self.exp_source.play_episode()
        self._env_steps += len(episode) * self.exp_source.n_steps
        return episode

    # def collect_step(self) -> Experience:
    #     self._env_steps += self.exp_source.n_steps
    #     return self.exp_source.play_n_steps()

    # def save_log(self) -> Path:
    #     path = self.path.joinpath("log.txt")
    #     with open(path, "w") as f:
    #         f.write(f"Environment steps: {self._env_steps}\n")

    #     return path

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
            "exp_source": self.exp_source.config_dic,
        }
        return dic

    @property
    def config_dic(self) -> Dict[str, Dict[str, str]]:
        dic = {k: str(v) for k, v in self.__dict__.items()}
        return dic

    @property
    def path(self) -> Path:
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


class StaticModel(Model):
    """This model class does not learn"""

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

    def learn(self) -> None:
        pass

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


class PerturbObserveModel(StaticModel):
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


class RandomModel(StaticModel):
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


class TD3Experience(Model):
    def __init__(
        self,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        demo_buffer_size: int = 1200,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        actor_l2: float = 0.0,
        critic_l2: float = 0.0,
        tau_critic: float = 1e-4,
        tau_actor: float = 1e-4,
        gamma: float = 0.01,
        n_steps: int = 1,
        norm_rewards: int = 0,
        train_steps: int = 1,
        collect_steps: int = 1,
        prefill_buffer: int = 600,
        use_per: bool = False,
        warmup_train_steps: int = 1000,
        lambda_demo_critic: float = 1.0,
        lambda_demo_actor: float = 1.0,
        lambda_bc: float = 1.0,
        use_q_filter: bool = False,
        policy_delay: int = 1,
        target_action_epsilon_noise: float = 0.0,
        env_kwargs: Dict[str, Any] = None,
        policy_kwargs: Optional[Dict[Any, Any]] = None,
        test_policy_kwargs: Optional[Dict[Any, Any]] = None,
        buffer_kwargs: Optional[Dict[Any, Any]] = None,
        buffer_demo_kwargs: Optional[Dict[Any, Any]] = None,
        include_demo_metrics: bool = True,
    ):

        # - - - Class members
        self.batch_size = batch_size
        self.buffer_size = buffer_size  # not used
        self.buffer_demo_size = demo_buffer_size  # not used
        self.actor_lr = actor_lr  # not used
        self.critic_lr = critic_lr  # not used
        self.actor_l2 = actor_l2  # not used
        self.critic_l2 = critic_l2  # not used
        self.gamma = gamma
        self.n_steps = n_steps
        self.norm_rewards = norm_rewards
        self.train_steps_per_timestep = train_steps
        self.collect_steps_per_timestep = collect_steps
        self.prefill_buffer = prefill_buffer  # not used
        self.use_per = use_per
        self.warmup_train_steps = warmup_train_steps  # not used
        self.lambda_demo_critic = lambda_demo_critic
        self.lambda_demo_actor = lambda_demo_actor
        self.lambda_bc = lambda_bc
        self.use_q_filter = use_q_filter
        self.policy_delay = policy_delay
        self.target_action_noise = target_action_epsilon_noise
        self.include_demo_metrics = include_demo_metrics

        # - - - Dict handling
        env_kwargs = src.utils.new_dic(env_kwargs)
        env_kwargs.setdefault("env_names", ["td3exp_train", "td3exp_test"])

        policy_kwargs = src.utils.new_dic(policy_kwargs)
        test_policy_kwargs = src.utils.new_dic(test_policy_kwargs)

        buffer_kwargs = src.utils.new_dic(buffer_kwargs)
        buffer_kwargs.setdefault("capacity", buffer_size)

        buffer_demo_kwargs = src.utils.new_dic(buffer_demo_kwargs)
        buffer_demo_kwargs.setdefault("capacity", demo_buffer_size)

        # - - - Environments
        envs: Sequence[ShadedPVEnv] = ShadedPVEnv.get_envs(**env_kwargs)
        self.env = envs[0]
        self.env_test = envs[1]

        # - - - Folders
        self.path_plots = self.env.path.joinpath("plots")
        self.path_csvs = self.env.path.joinpath("csvs")
        self.path_plots_test = self.env_test.path.joinpath("plots")
        self.path_csvs_test = self.env_test.path.joinpath("csvs")
        Path.mkdir(self.path_plots)
        Path.mkdir(self.path_csvs)
        Path.mkdir(self.path_plots_test)
        Path.mkdir(self.path_csvs_test)

        # - - - Networks
        self.actor = rl.create_mlp_actor(self.env)
        self.critic = rl.create_mlp_critic(self.env)
        self.critic2 = rl.create_mlp_critic(self.env)
        self.actor_target = rl.TargetNet(self.actor, alpha=tau_actor)
        self.critic_target = rl.TargetNet(self.critic, alpha=tau_critic)
        self.critic2_target = rl.TargetNet(self.critic2, alpha=tau_critic)

        # - - - Optimizers
        self.actor_optim = Adam(
            self.actor.parameters(), lr=actor_lr, weight_decay=actor_l2
        )
        self.critic_optim = Adam(
            self.critic.parameters(), lr=critic_lr, weight_decay=critic_l2
        )
        self.critic2_optim = Adam(
            self.critic2.parameters(), lr=critic_lr, weight_decay=critic_l2
        )

        # - - - Loss tracking
        self._counter_train_steps = 0
        # self.losses_env_steps = defaultdict(list)
        # self.losses_train_steps = defaultdict(list)

        # - - - Experience sources
        self.env_tracker = self.env.env_tracker
        self.exp_source = self._get_experience_source(
            env=self.env, env_tracker=self.env_tracker, **policy_kwargs
        )
        self.exp_source_explore = ExperienceSource(
            RandomPolicy(self.env.action_space),
            self.env,
            gamma,
            n_steps,
            env_tracker=self.env_tracker,
        )
        self.exp_source_demo = ExperienceSource(
            PerturbObservePolicy(self.env.action_space),
            self.env,
            self.gamma,
            self.n_steps,
            env_tracker=self.env_tracker,
        )
        self.env_tracker_test = self.env_test.env_tracker
        self.exp_source_test = self._get_experience_source(
            env=self.env_test,
            env_tracker=self.env_tracker_test,
            **test_policy_kwargs,
        )

        # - - - Buffers
        self.buffer = self._get_buffer(use_per=use_per, **buffer_kwargs)
        self.buffer_demo = self._get_buffer(use_per=use_per, **buffer_demo_kwargs)
        if not include_demo_metrics:
            self.env._day_idx = 80  # Use the last days to obtain expert samples
            self.env.allow_tracking = False  # Do not count the expert episodes
            demo_steps = 0
        else:
            demo_steps = self.buffer_demo.capacity
        self._fill_buffer(
            self.buffer_demo,
            self.exp_source_demo,
            num_experiences=-1,
            description="Filling replay demo buffer",
        )
        if not self.env.allow_tracking:
            self.env_tracker.reset()
            self.env._day_idx = -1
            self.env_tracker.reset_all()
            self.env_tracker.reset()  # Bug: must call reset twice
        self.env.allow_tracking = True
        self._fill_buffer(
            self.buffer,
            self.exp_source_explore,
            num_experiences=prefill_buffer,
            description="Filling replay buffer with random experiences",
        )

        # Add losses (np.nan) for each step taken in the prefilling for each loss
        dic = {k: np.nan for k in self.available_losses}
        for _ in range(self.prefill_buffer + demo_steps):
            self._track(self.env_tracker, dic)
        # And for the ep metrics
        dic = {f"ep_{k}": np.nan for k in self.available_losses}
        for _ in range(self.env_tracker.counter_total_episodes):
            self._track(self.env_tracker, dic)

        self.env_tracker.new_episode_available = False

        self.save_config_dic()

    def quit(self) -> None:
        super().quit()
        self.env_test.quit()

        path = Path("default/log.txt")
        et = self.env_tracker
        ett = self.env_tracker_test
        with open(path, "a") as f:
            f.write(f"{self.__class__.__name__}\n")
            f.write(f"Train episodes: {et.counter_total_episodes}\n")
            f.write(f"Train mean ep eff: {np.mean(et.history['ep_efficiency'])}\n")
            f.write(f"Train mean ep rew: {np.mean(et.history['ep_reward'])}\n")
            f.write(f"Test episodes: {ett.counter_total_episodes}\n")
            f.write(f"Test mean ep eff: {np.mean(ett.history['ep_efficiency'])}\n")
            f.write(f"Test mean ep rew: {np.mean(ett.history['ep_reward'])}\n\n")

    def learn(
        self,
        timesteps: int,
        val_every_timesteps: int = 0,
        n_eval_episodes: int = -1,
        log_interval: int = 1,
        plot_every_timesteps: int = 0,
    ):

        timesteps = timesteps + self.warmup_train_steps

        if n_eval_episodes == -1:
            n_eval_episodes = self.env_test.available_weather_days

        val_every_timesteps = val_every_timesteps or timesteps
        plot_every_timesteps = plot_every_timesteps or timesteps

        for ts_counter in tqdm(range(1, timesteps + 1), desc="Training"):
            # ts_counter = self.env_tracker.counter_total_steps

            # - - - Testing
            if (
                ts_counter % val_every_timesteps == 0
                and ts_counter > self.warmup_train_steps
            ):
                for _ in range(n_eval_episodes):
                    self.exp_source_test.play_episode()

                print("\n * Test episode")
                self.env_tracker_test.print_tracking(avg=n_eval_episodes)
                print()

            # - - - Logging
            if self.env_tracker.new_episode_available:
                print(f"\nEval num_timesteps={ts_counter}")
                print(f"training_steps={self._counter_train_steps}")
                self.env_tracker.print_tracking(
                    avg=log_interval, ignore=["loss", "filter"]
                )
                print()

                # Loss per episode
                dic = {}
                for k, lst in self.env_tracker.history.items():
                    if k.startswith("ep"):
                        continue
                    arr = np.array(lst[-self.env_tracker.steps_elapsed :])
                    mask = np.isfinite(arr)
                    if all(~mask):
                        dic[f"ep_{k}"] = np.nan
                    else:
                        dic[f"ep_{k}"] = arr[mask].mean()
                self._track(self.env_tracker, dic)

                self.env_tracker.new_episode_available = False

            # - - - Training
            for _ in range(self.train_steps_per_timestep):
                self._counter_train_steps += 1
                losses = self._train_nets()
            self._track(self.env_tracker, losses)  # Keep only the last set of losses

            # - - - Collect
            if ts_counter > self.warmup_train_steps:
                self._fill_buffer(
                    self.buffer,
                    self.exp_source,
                    num_experiences=self.collect_steps_per_timestep,
                )

            # - - - Plotting
            if ts_counter % plot_every_timesteps == 0 or ts_counter == timesteps:
                self.env_tracker.save_plot_metrics(self.path_plots)
                self.env_tracker.save_all_as_csv(self.path_csvs)
                self.env_tracker_test.save_plot_metrics(self.path_plots_test)
                self.env_tracker_test.save_all_as_csv(self.path_csvs_test)

    @property
    def available_losses(self) -> Sequence[str]:
        return [
            "critic_loss_rl",
            "critic_loss_demo",
            "critic_loss",
            "critic2_loss_rl",
            "critic2_loss_demo",
            "critic2_loss",
            "actor_loss_rl",
            "actor_loss",
            "actor_loss_demo",
            "bc_loss",
            "q_filter",
        ]

    @property
    def available_metrics(self) -> Sequence[str]:
        return ["efficiency", "ep_reward"]

    @property
    def policy(self) -> Policy:
        return getattr(self.exp_source, "policy", None)

    @staticmethod
    def _track(
        tracker: EnvironmentTracker,
        dic: Dict[str, Union[numbers.Real], Sequence[numbers.Real]],
    ):
        """Interface method to track using a dictionary"""
        for k, v in dic.items():
            tracker.track(k, float(v))

    def _train_nets(self) -> Dict[str, numbers.Real]:
        batch = self._prepare_batch(
            self.buffer, self.batch_size, norm_rewards=self.norm_rewards
        )
        batch_demo = self._prepare_batch(
            self.buffer_demo, self.batch_size, norm_rewards=self.norm_rewards
        )

        critics_loss = self._get_critic_loss(batch, self.buffer)
        critics_loss_demo = self._get_critic_loss(batch_demo, self.buffer_demo)
        critic_loss_rl = critics_loss[0]
        critic_loss_demo = critics_loss_demo[0] * self.lambda_demo_critic
        critic_loss = self._apply_losses(
            [critic_loss_rl, critic_loss_demo], self.critic_optim
        )
        critic2_loss_rl = critics_loss[1]
        critic2_loss_demo = critics_loss_demo[1] * self.lambda_demo_critic
        critic2_loss = self._apply_losses(
            [critic2_loss_rl, critic2_loss_demo], self.critic2_optim
        )

        actor_loss_rl = self._get_actor_loss(batch)
        actor_loss_demo = self._get_actor_loss(batch_demo) * self.lambda_demo_actor
        bc_losses = self._get_bc_loss(batch_demo)
        bc_loss = bc_losses[0] * self.lambda_bc
        q_filter = bc_losses[1]

        apply = self._counter_train_steps % self.policy_delay == 0
        actor_loss = self._apply_losses(
            [actor_loss_rl, actor_loss_demo, bc_loss], self.actor_optim, apply=apply
        )

        losses_dic = {
            "critic_loss_rl": critic_loss_rl.detach().cpu().numpy(),
            "critic_loss_demo": critic_loss_demo.detach().cpu().numpy(),
            "critic_loss": critic_loss.detach().cpu().numpy(),
            "critic2_loss_rl": critic2_loss_rl.detach().cpu().numpy(),
            "critic2_loss_demo": critic2_loss_demo.detach().cpu().numpy(),
            "critic2_loss": critic2_loss.detach().cpu().numpy(),
            "actor_loss_rl": actor_loss_rl.detach().cpu().numpy(),
            "actor_loss_demo": actor_loss_demo.detach().cpu().numpy(),
            "actor_loss": actor_loss.detach().cpu().numpy(),
            "bc_loss": bc_loss.detach().cpu().numpy(),
            "q_filter": q_filter,
        }

        return {k: v for k, v in losses_dic.items() if k in self.available_losses}

    def _get_bc_loss(
        self, demo_batch: ExperienceTensorBatch
    ) -> Tuple[th.Tensor, numbers.Real]:
        if demo_batch is None:
            return th.tensor(np.nan), np.nan

        agent_action = self.actor(demo_batch.obs)

        if self.use_q_filter:
            q_expert = self.critic(demo_batch.obs, demo_batch.action)
            q_agent = self.critic(demo_batch.obs, agent_action)
            ind = (q_expert > q_agent).detach()
        else:
            ind = th.ones_like(agent_action, dtype=th.bool)
        q_filter_num = sum(ind)

        error = demo_batch.action[ind] - agent_action[ind]

        # demo_weights = np.sqrt(demo_batch.weights)
        # weighted_error = th.mul(error, demo_weights[ind])
        zero_tensor = th.zeros_like(error)
        bc_loss = th.nn.functional.mse_loss(error, zero_tensor)
        # bc_loss *= self.lambda_bc

        return bc_loss, q_filter_num

    def _get_actor_loss(self, batch: PrioritizedExperienceTensorBatch) -> None:
        if batch is None or self.critic is None:
            return th.tensor(np.nan)

        # w_loss = th.mul(loss, batch.weights)
        # mean_ = w_loss.mean()
        # return mean_

        loss = -self.critic(batch.obs, self.actor(batch.obs))
        return loss.mean()

    def _get_critic_loss(
        self, batch: ExperienceTensorBatch, buffer: ReplayBuffer
    ) -> Tuple[th.Tensor, th.Tensor]:
        if batch is None or (self.critic is None and self.critic2 is None):
            return th.tensor(np.nan), th.tensor(np.nan)

        weights = np.sqrt(batch.weights)

        # Add noise to the actions
        act_target = self.actor_target(batch.next_obs)
        noise = th.rand_like(act_target) * 2 - 1  # Normal noise between [-1, 1]
        act_target += noise * self.target_action_noise
        act_target = act_target.clamp(-1, 1)

        # Critics training
        q_critic1 = self.critic_target(batch.next_obs, act_target)
        q_critic1[batch.done] = 0.0
        if not self.critic2 is None:
            q_critic2 = self.critic2_target(batch.next_obs, act_target)
            q_critic2[batch.done] = 0.0
        else:
            q_critic2 = q_critic1

        # Compute target with the minumim of both critics
        y = (
            batch.reward + th.min(q_critic1, q_critic2) * self.gamma ** self.n_steps
        ).detach()

        # Critic 1 training
        q_target_1 = self.critic(batch.obs, batch.action)
        td_error_1 = y - q_target_1
        weighted_td_error_1 = th.mul(td_error_1, weights)
        zero_t1 = th.zeros_like(weighted_td_error_1)
        critic_loss_1 = th.nn.functional.mse_loss(weighted_td_error_1, zero_t1)

        # Critic 2 training
        if not self.critic2 is None:
            q_target_2 = self.critic2(batch.obs, batch.action)
            td_error_2 = y - q_target_2
            weighted_td_error_2 = th.mul(td_error_2, weights)
            zero_t2 = th.zeros_like(weighted_td_error_2)
            critic_loss_2 = th.nn.functional.mse_loss(weighted_td_error_2, zero_t2)
        else:
            critic_loss_2 = th.tensor(np.nan)

        # Update PER priorities if needed
        self.update_priorities(buffer, batch.indices, weighted_td_error_1)

        return critic_loss_1, critic_loss_2

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

    @staticmethod
    def _prepare_batch(
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

    def _get_experience_source(self, env: ShadedPVEnv, **kwargs) -> ExperienceSource:
        """
        Return a ExperienceSource with a mlp policy driven by the actor network of the class

        Parameters:
            - env: the environment to interact with
            - kwargs: policy kwargs (see Policy class)
        """
        kwargs["net"] = self.actor
        env_tracker = kwargs.pop("env_tracker", None)
        policy = self.get_policy(policy_name="mlp", env=env, policy_kwargs=kwargs)
        return ExperienceSource(
            policy, env, self.gamma, self.n_steps, env_tracker=env_tracker
        )

    @staticmethod
    def _get_buffer(
        use_per: bool, **kwargs
    ) -> Union[ReplayBuffer, PrioritizedReplayBuffer]:
        """
        Return a ReplayBuffer or a PrioritizedReplayBuffer with the specified kwargs

        Parameters:
            - use_per: use a prioritized replay buffer
            - kwargs: buffer kwargs (see ReplayBuffer class)

        """
        if use_per:
            return PrioritizedReplayBuffer(**kwargs)
        return ReplayBuffer(**kwargs)

    @staticmethod
    def _apply_losses(
        losses: Union[th.Tensor, Sequence[th.Tensor]],
        optim: Optimizer,
        apply: bool = True,
    ) -> th.Tensor:
        """ "
        Computes the gradient of the sum of the provided loss tensors and performs a single optimization step.
        If one tensor is NaN, its contribution is not taken into account.
        If all tensors are NaN, the function returns a NaN tensor as the loss,
        else returns the sum of the provided losses.

        Parameters:
            - losses: sequence of torch.Tensors
            - optim: optimizer that performs the weights optimization
            - apply: take the optimization step or just sum the losses
        """

        if not isinstance(losses, Sequence):
            losses = tuple([losses])

        losses = th.cat(tuple(l.unsqueeze(-1) for l in losses))

        if all(th.isnan(losses)):
            return th.tensor(np.nan)

        mask = th.isnan(losses)
        total_loss = th.sum(losses[~mask])

        if apply and not optim is None:
            optim.zero_grad()
            total_loss.backward()
            optim.step()

        return total_loss

    @staticmethod
    def _fill_buffer(
        buffer: ReplayBuffer,
        exp_source: ExperienceSource,
        num_experiences: Optional[int] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Fill the buffer using the experience source specified.

        Parameters:
            - buffer: the buffer to store the experiences
            - exp_source: the experience source that provides the experiences
            - num_experiences: the number of experiences to add to the buffer.
                Use `-1` to fill the buffer to its full capacity.
        """
        if num_experiences == 0:
            return None

        if num_experiences == -1:
            num_experiences = buffer.capacity

        for _ in tqdm(
            range(num_experiences), desc=description, disable=description is None
        ):
            # exp = exp_source.play_n_steps()
            exp = exp_source.play_step()
            buffer.append(exp)

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


class BC(TD3Experience):
    def __init__(
        self,
        batch_size: int = 64,
        demo_buffer_size: int = 10_000,
        actor_lr: float = 1e-3,
        actor_l2: float = 0.0,
        n_steps: int = 1,
        norm_rewards: int = 0,
        train_steps: int = 1,
        use_per: bool = False,
        warmup_train_steps: int = 100,
        lambda_bc: float = 1.0,
        target_action_epsilon_noise: float = 0,
        env_kwargs: Optional[Dict[Any, Any]] = None,
        policy_kwargs: Optional[Dict[Any, Any]] = None,
        test_policy_kwargs: Optional[Dict[Any, Any]] = None,
        buffer_kwargs: Optional[Dict[Any, Any]] = None,
        buffer_demo_kwargs: Optional[Dict[Any, Any]] = None,
    ):

        env_kwargs = src.utils.new_dic(env_kwargs)
        env_kwargs.setdefault("env_names", ["bc_train", "bc_test"])

        super().__init__(
            batch_size=batch_size,
            buffer_size=0,
            demo_buffer_size=demo_buffer_size,
            actor_lr=actor_lr,
            critic_lr=1e-3,
            actor_l2=actor_l2,
            critic_l2=0.0,
            tau_critic=1e-3,
            tau_actor=1e-3,
            gamma=1.0,
            n_steps=n_steps,
            norm_rewards=norm_rewards,
            train_steps=train_steps,
            collect_steps=0,
            prefill_buffer=0,
            use_per=use_per,
            warmup_train_steps=warmup_train_steps,
            lambda_demo_critic=0,
            lambda_demo_actor=0,
            lambda_bc=lambda_bc,
            use_q_filter=False,
            policy_delay=1,
            target_action_epsilon_noise=target_action_epsilon_noise,
            env_kwargs=env_kwargs,
            policy_kwargs=policy_kwargs,
            test_policy_kwargs=test_policy_kwargs,
            buffer_kwargs=buffer_kwargs,
            buffer_demo_kwargs=buffer_demo_kwargs,
            include_demo_metrics=True,
        )

        self.critic = None
        self.critic2_optim = None
        self.critic2 = None
        self.critic2_optim = None

    @property
    def available_losses(self) -> Sequence[str]:
        return ["bc_loss"]


class TD3(TD3Experience):
    def __init__(
        self,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        actor_l2: float = 0.0,
        critic_l2: float = 0.0,
        tau_critic: float = 1e-4,
        tau_actor: float = 1e-4,
        gamma: float = 0.1,
        n_steps: int = 1,
        norm_rewards: int = 0,
        train_steps: int = 1,
        collect_steps: int = 1,
        prefill_buffer: int = 2000,
        use_per: bool = False,
        warmup_train_steps: int = 1000,
        policy_delay: int = 2,
        target_action_epsilon_noise: float = 0.001,
        env_kwargs: Dict[str, Any] = None,
        policy_kwargs: Optional[Dict[Any, Any]] = None,
        test_policy_kwargs: Optional[Dict[Any, Any]] = None,
        buffer_kwargs: Optional[Dict[Any, Any]] = None,
        buffer_demo_kwargs: Optional[Dict[Any, Any]] = None,
    ):

        env_kwargs = src.utils.new_dic(env_kwargs)
        env_kwargs.setdefault("env_names", ["td3_train", "td3_test"])

        super().__init__(
            batch_size=batch_size,
            buffer_size=buffer_size,
            demo_buffer_size=0,
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
            warmup_train_steps=warmup_train_steps,
            lambda_demo_critic=0.0,
            lambda_demo_actor=0.0,
            lambda_bc=0.0,
            use_q_filter=False,
            policy_delay=policy_delay,
            target_action_epsilon_noise=target_action_epsilon_noise,
            env_kwargs=env_kwargs,
            policy_kwargs=policy_kwargs,
            test_policy_kwargs=test_policy_kwargs,
            buffer_kwargs=buffer_kwargs,
            buffer_demo_kwargs=buffer_demo_kwargs,
            include_demo_metrics=True,
        )

    @property
    def available_losses(self) -> Sequence[str]:
        return ["critic_loss", "critic2_loss", "actor_loss"]


class DDPGExperience(TD3Experience):
    def __init__(
        self,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        demo_buffer_size: int = 1200,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        actor_l2: float = 0.0,
        critic_l2: float = 0.0,
        tau_critic: float = 1e-4,
        tau_actor: float = 1e-4,
        gamma: float = 0.1,
        n_steps: int = 1,
        norm_rewards: int = 0,
        train_steps: int = 1,
        collect_steps: int = 1,
        prefill_buffer: int = 2000,
        use_per: bool = False,
        warmup_train_steps: int = 1000,
        lambda_demo_critic: float = 0.1,
        lambda_demo_actor: float = 0.1,
        lambda_bc: float = 0.1,
        use_q_filter: bool = False,
        policy_delay: int = 2,
        target_action_epsilon_noise: float = 0.001,
        env_kwargs: Dict[str, Any] = None,
        policy_kwargs: Optional[Dict[Any, Any]] = None,
        test_policy_kwargs: Optional[Dict[Any, Any]] = None,
        buffer_kwargs: Optional[Dict[Any, Any]] = None,
        buffer_demo_kwargs: Optional[Dict[Any, Any]] = None,
        include_demo_metrics: bool = True,
    ):
        env_kwargs = src.utils.new_dic(env_kwargs)
        env_kwargs.setdefault("env_names", ["ddpgexp_train", "ddpgexp_test"])

        super().__init__(
            batch_size=batch_size,
            buffer_size=buffer_size,
            demo_buffer_size=demo_buffer_size,
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
            warmup_train_steps=warmup_train_steps,
            lambda_demo_critic=lambda_demo_critic,
            lambda_demo_actor=lambda_demo_actor,
            lambda_bc=lambda_bc,
            use_q_filter=use_q_filter,
            policy_delay=policy_delay,
            target_action_epsilon_noise=target_action_epsilon_noise,
            env_kwargs=env_kwargs,
            policy_kwargs=policy_kwargs,
            test_policy_kwargs=test_policy_kwargs,
            buffer_kwargs=buffer_kwargs,
            buffer_demo_kwargs=buffer_demo_kwargs,
            include_demo_metrics=include_demo_metrics,
        )

        self.critic2 = None
        self.critic2_optim = None

    @property
    def available_losses(self) -> Sequence[str]:
        return [
            "critic_loss_rl",
            "critic_loss_demo",
            "critic_loss",
            "actor_loss_rl",
            "actor_loss",
            "actor_loss_demo",
            "bc_loss",
            "q_filter",
        ]


class DDPG(DDPGExperience):
    def __init__(
        self,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        actor_l2: float = 0.0,
        critic_l2: float = 0.0,
        tau_critic: float = 1e-4,
        tau_actor: float = 1e-4,
        gamma: float = 0.1,
        n_steps: int = 1,
        norm_rewards: int = 0,
        train_steps: int = 1,
        collect_steps: int = 1,
        prefill_buffer: int = 2000,
        use_per: bool = False,
        warmup_train_steps: int = 1000,
        policy_delay: int = 2,
        target_action_epsilon_noise: float = 0.001,
        env_kwargs: Dict[str, Any] = None,
        policy_kwargs: Optional[Dict[Any, Any]] = None,
        test_policy_kwargs: Optional[Dict[Any, Any]] = None,
        buffer_kwargs: Optional[Dict[Any, Any]] = None,
        buffer_demo_kwargs: Optional[Dict[Any, Any]] = None,
    ):

        env_kwargs = src.utils.new_dic(env_kwargs)
        env_kwargs.setdefault("env_names", ["ddpg_train", "ddpg_test"])

        super().__init__(
            batch_size=batch_size,
            buffer_size=buffer_size,
            demo_buffer_size=0,
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
            warmup_train_steps=warmup_train_steps,
            lambda_demo_critic=0,
            lambda_demo_actor=0,
            lambda_bc=0,
            use_q_filter=False,
            policy_delay=policy_delay,
            target_action_epsilon_noise=target_action_epsilon_noise,
            env_kwargs=env_kwargs,
            policy_kwargs=policy_kwargs,
            test_policy_kwargs=test_policy_kwargs,
            buffer_kwargs=buffer_kwargs,
            buffer_demo_kwargs=buffer_demo_kwargs,
            include_demo_metrics=True,
        )

    @property
    def available_losses(self) -> Sequence[str]:
        return ["critic_loss", "actor_loss"]


class PO(DDPGExperience):
    def __init__(
        self,
        demo_buffer_size: int = 30_000,
        gamma: float = 0.01,
        n_steps: int = 1,
        norm_rewards: int = 0,
        env_kwargs: Optional[Dict[str, Any]] = None,
    ):
        env_kwargs = src.utils.new_dic(env_kwargs)
        env_kwargs.setdefault("env_names", ["po_train", "po_test"])
        super().__init__(
            batch_size=64,
            buffer_size=0,
            demo_buffer_size=demo_buffer_size,
            actor_lr=1e-3,
            critic_lr=1e-3,
            actor_l2=0.0,
            critic_l2=0.0,
            tau_critic=1e-4,
            tau_actor=1e-4,
            gamma=gamma,
            n_steps=n_steps,
            norm_rewards=norm_rewards,
            train_steps=1,
            collect_steps=1,
            prefill_buffer=0,
            use_per=False,
            warmup_train_steps=0,
            lambda_demo_critic=1.0,
            lambda_demo_actor=1.0,
            lambda_bc=1.0,
            use_q_filter=False,
            policy_delay=1,
            target_action_epsilon_noise=0.0,
            env_kwargs=env_kwargs,
            policy_kwargs=None,
            test_policy_kwargs=None,
            buffer_kwargs=None,
            buffer_demo_kwargs=None,
            include_demo_metrics=True,
        )

        self.exp_source_test = ExperienceSource(
            PerturbObservePolicy(self.env_test.action_space),
            self.env_test,
            self.gamma,
            self.n_steps,
            env_tracker=self.env_tracker_test,
        )

    @property
    def available_losses(self) -> Sequence[str]:
        return []


def run_ddpgexp():
    model = DDPGExperience(
        demo_buffer_size=20000,
        use_q_filter=True,
        warmup_train_steps=5000,
        prefill_buffer=1000,
        train_steps=TRAIN_STEPS,
        lambda_bc=1.0,
        policy_kwargs=explore_policy_kwargs2,
        include_demo_metrics=False,
    )
    model.learn(timesteps=59_000)
    model.quit()

    return model


def run_td3exp():
    model = TD3Experience(
        demo_buffer_size=20000,
        use_q_filter=True,
        warmup_train_steps=5000,
        prefill_buffer=1000,
        train_steps=TRAIN_STEPS,
        lambda_bc=1.0,
        policy_kwargs=explore_policy_kwargs2,
        include_demo_metrics=False,
    )
    model.learn(timesteps=59_000)
    model.quit()

    return model


def run_ddpg():
    model = DDPG(
        warmup_train_steps=0,
        prefill_buffer=1000,
        train_steps=TRAIN_STEPS,
        policy_kwargs=explore_policy_kwargs2,
    )
    model.learn(timesteps=59_000)
    model.quit()

    return model


def run_td3():
    model = TD3(
        warmup_train_steps=0,
        prefill_buffer=1000,
        train_steps=TRAIN_STEPS,
        policy_kwargs=explore_policy_kwargs2,
    )
    model.learn(timesteps=59_000)
    model.quit()

    return model


def run_po():
    model = PO(demo_buffer_size=60_000)
    model.learn(timesteps=1)
    model.quit()

    return model


if __name__ == "__main__":
    from src.noise import GaussianNoise
    from src.schedule import LinearSchedule
    import gc

    explore_policy_kwargs = {
        "noise": GaussianNoise(0, 0.05),
        "schedule": LinearSchedule(max_steps=30_000),
        "decrease_noise": True,
    }

    explore_policy_kwargs2 = {
        "noise": GaussianNoise(0, 0.3),
        "schedule": LinearSchedule(max_steps=9_000),
        "decrease_noise": True,
    }

    # model = BC(demo_buffer_size=5000)
    # model.learn(timesteps=30_000, val_every_timesteps=1_000, plot_every_timesteps=1000)
    # model.quit()

    NUM_EXPS = 10
    TRAIN_STEPS = 1

    for _ in range(NUM_EXPS):
        run_ddpgexp()
        run_td3exp()
        run_ddpg()
        run_td3()
        run_po()
