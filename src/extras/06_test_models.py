from src.model import (
    SB3DDPG,
    SB3TD3,
    SB3A2C,
    SB3SAC,
    SB3PPO,
    PerturbObserveModel,
    RandomModel,
    DDPG,
    TD3,
)
from src.noise import GaussianNoise
from src.schedule import LinearSchedule

# Test SB3 Models
models = [
    SB3DDPG,
    SB3TD3,
    # SB3A2C,
    # SB3SAC,
    # SB3PPO,
]
dic_sb3 = {
    "env_kwargs": {
        "weather_paths": [["train_1_4_0.5", "test_1_4_0.5"]],
    },
}
for m in models:
    m.run_from_grid(
        dic=dic_sb3,
        total_timesteps=1_000,
        val_every_timesteps=1_000,
    )


# Test PO model
dic_po = {
    "dc_step": [0.01, 0.02],
    "env_kwargs": {
        "weather_paths": [["test_1_4_0.5"]],
    },
}
PerturbObserveModel.run_from_grid(dic_po)

# Test Random Model
dic_rand = {
    "env_kwargs": {
        "weather_paths": [["test_1_4_0.5"]] * 2,
    },
}
RandomModel.run_from_grid(dic_rand)

# Test DDPG Model
dic_ddpg = {
    "batch_size": 64,  # 64
    "actor_lr": 1e-4,  # 1e-3
    "critic_lr": 1e-3,
    "tau_critic": 1e-4,  # 1e-3
    "tau_actor": 1e-4,  # 1e-4
    "actor_l2": 0,
    "critic_l2": 0,
    "gamma": 0.01,  # 0.6
    "n_steps": 1,
    "norm_rewards": 0,
    "train_steps": 1,  # 5
    "collect_steps": 1,
    "prefill_buffer": 100,
    "use_per": True,  # True,
    "warmup": 1000,
    "policy_kwargs": {
        "noise": [GaussianNoise(mean=0.0, std=0.3)],
        "schedule": [LinearSchedule(max_steps=10_000)],
        "decrease_noise": True,
    },
    "buffer_kwargs": {
        "capacity": 50_000,
        # "alpha": 0.9,  # 0.9
        # "beta": 0.2,  # 0.2
        # "tau": 1.0,  # 0.9
        # "sort": True,  # False
    },
    "env_kwargs": {
        "weather_paths": [["train_1_4_0.5", "test_1_4_0.5"]],
    },
}
DDPG.run_from_grid(dic_ddpg, total_timesteps=1_000, val_every_timesteps=1_000)

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
    "norm_rewards": 0,
    "train_steps": 1,  # 5
    "collect_steps": 1,
    "prefill_buffer": 100,
    "use_per": True,  # True,
    "warmup": 1000,
    "policy_delay": 2,
    "target_action_epsilon_noise": 0.001,
    "training_type": 0,
    "policy_kwargs": {
        "noise": [GaussianNoise(mean=0.0, std=0.3)],
        "schedule": [LinearSchedule(max_steps=10_000)],
        "decrease_noise": True,
    },
    "buffer_kwargs": {
        "capacity": 50_000,
        # "alpha": 0.9,  # 0.9
        # "beta": 0.2,  # 0.2
        # "tau": 1.0,  # 0.9
        # "sort": True,  # False
    },
    "env_kwargs": {
        "weather_paths": [["train_1_4_0.5", "test_1_4_0.5"]],
    },
}
TD3.run_from_grid(dic, total_timesteps=1_000, val_every_timesteps=1_000)