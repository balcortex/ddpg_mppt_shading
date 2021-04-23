from pathlib import Path
from typing import Any, Dict, Generator, Optional

from src import utils
from src.env import ShadedPVEnv
import src.policy
from src.schedule import ConstantSchedule
from src.noise import GaussianNoise

POLICIES = {"po": src.policy.PerturbObservePolicy, "random": src.policy.RandomPolicy}


def policy_run(
    policy_name: str, options: Dict[str, Any], dic_generator: Optional[Generator] = None
) -> None:
    path = Path(f"default/{policy_name}")

    if dic_generator:
        opt_gen = dic_generator(options)
    else:
        opt_gen = (options,)  #  Make an iterable

    for opt in opt_gen:
        env = ShadedPVEnv.get_envs(num_envs=1, log_path=path, env_names=["default_env"])

        utils.save_dic_txt(
            {k: str(v) for k, v in opt.items()},
            env.path.joinpath("config.txt"),
            overwrite=True,
        )

        policy = POLICIES[policy_name](env=env, **opt)

        obs, done, info = env.reset(), False, {}
        while not done:
            action = policy(obs, info)
            obs, _, done, info = env.step(action)

        env.quit()  #  Quit the matlab engines


if __name__ == "__main__":
    policies = {
        "po": {
            "dc_step": [0.01] * 10,
            "noise": [GaussianNoise(mean=0.0, std=0.05)],
            "schedule": [ConstantSchedule(0.1)],
        },
        # "random": {},
    }

    for policy_name, options in policies.items():
        policy_run(policy_name, options, dic_generator=utils.grid_generator)
