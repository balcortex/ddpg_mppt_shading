from src.model import DDPG, TD3, DDPGExp, TD3Exp
from src.noise import GaussianNoise
from src.schedule import LinearSchedule

# TOTAL_TIMESTEPS = 30_000
TOTAL_TIMESTEPS = 30_000
VAL_EVERY_TIMESTEPS = 1_000
REPEAT_RUN = 1

COMMON_KWARGS = {
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
    "train_steps": 5,  # 5
    "collect_steps": 0,
    "prefill_buffer": 600,
    "use_per": False,
    "warmup": 10_000,
    "policy_delay": 2,
    "target_action_epsilon_noise": 0.001,
    "use_q_filter": False,
    "lambda_bc": 0.1,
    "policy_kwargs": {
        "noise": [GaussianNoise(mean=0.0, std=0.3)],
        "schedule": [LinearSchedule(max_steps=10_000)],
        "decrease_noise": True,
    },
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

DDPG_DEL_KEYS = [
    "policy_delay",
    "target_action_epsilon_noise",
    "use_q_filter",
    "lambda_bc",
    "demo_buffer_kwargs",
]
DPPGEXP_DEL_KEYS = [
    "policy_delay",
    "target_action_epsilon_noise",
    "policy_kwargs",
]
TD3_DEL_KEYS = [
    "use_q_filter",
    "lambda_bc",
    "demo_buffer_kwargs",
]
TD3EXP_DEL_KEYS = [
    "policy_kwargs",
]


def run_ddpg():
    dic = {k: v for k, v in COMMON_KWARGS.items() if k not in DDPG_DEL_KEYS}
    DDPG.run_from_grid(dic, TOTAL_TIMESTEPS, VAL_EVERY_TIMESTEPS, REPEAT_RUN)


def run_ddpgexp():
    dic = {k: v for k, v in COMMON_KWARGS.items() if k not in DPPGEXP_DEL_KEYS}
    DDPGExp.run_from_grid(dic, TOTAL_TIMESTEPS, VAL_EVERY_TIMESTEPS, REPEAT_RUN)


def run_td3():
    dic = {k: v for k, v in COMMON_KWARGS.items() if k not in TD3_DEL_KEYS}
    TD3.run_from_grid(dic, TOTAL_TIMESTEPS, VAL_EVERY_TIMESTEPS, REPEAT_RUN)


def run_td3exp():
    dic = {k: v for k, v in COMMON_KWARGS.items() if k not in TD3EXP_DEL_KEYS}
    TD3Exp.run_from_grid(dic, TOTAL_TIMESTEPS, VAL_EVERY_TIMESTEPS, REPEAT_RUN)


if __name__ == "__main__":
    run_ddpg()
    run_ddpgexp()
    run_td3()
    run_td3exp()
