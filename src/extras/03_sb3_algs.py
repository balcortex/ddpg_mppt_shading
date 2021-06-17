from pathlib import Path
from typing import Any, Dict

import numpy as np
import src.alg
from src import utils
from src.env import ShadedPVEnv
from src.pvsys import ShadedArray

SB3 = ["DDPG", "A2C", "TD3", "SAC", "PPO"]


def exp(model_name: str, kwargs: Dict[Any, Any], iters: int = 300) -> None:
    path = Path(f"default/{model_name}")
    train_env, test_env = ShadedPVEnv.get_envs(
        num_envs=2,
        log_path=path,
        env_names=["train", "test"],
        weather_paths=["test_1_4_0.5", "test_1_4_0.5"],
    )

    utils.save_dic_txt(
        {k: str(v) for k, v in kwargs.items()},
        train_env.path.joinpath("config.txt"),
        overwrite=True,
    )

    if model_name in SB3:
        model = getattr(stable_baselines3, model_name)(env=train_env, **kwargs)
    else:
        model = getattr(src.alg, model_name)(**kwargs)

    for _ in range(iters):
        model.learn(total_timesteps=1000)

        obs = test_env.reset()
        done = False
        info = {}
        while not done:
            if model_name in SB3:
                action, _states = model.predict(obs, deterministic=True)
            else:
                action, _states = model.predict(obs, info)
            obs, reward, done, info = test_env.step(action)

        test_env.reset()

    train_env.quit()

    return model


def test_po() -> None:
    env = ShadedPVEnv.get_envs(num_envs=1, env_names=["PO"])
    model = src.alg.PO(v_step=0.01)

    obs = env.reset()
    done = False
    info = {}
    while not done:
        action, _states = model.predict(obs, info)
        obs, reward, done, info = env.step(action)

    wc = env.unique_weathers
    obs = env.reset()  # Save the csv files

    for g, t in wc:
        curve = env.pvarray.get_shaded_iv_curve(irradiance=g, ambient_temperature=t)
        fig1 = ShadedArray.plot_mpp_curve(curve, type_="iv")
        fig2 = ShadedArray.plot_mpp_curve(curve, type_="pv")
        fig3 = ShadedArray.plot_mpp_curve(curve, type_="pd")


def test_alg(model_name: str) -> None:
    common_kwargs = {
        "policy": "MlpPolicy",
        "verbose": 0,
        "device": "cpu",
        "learning_rate": [1e-3] * 1,
        "gamma": [0.1],
    }
    kwargs = {
        "DDPG": {
            "action_noise": [
                NormalActionNoise(np.array([0.0]), np.array([0.2])),
            ],
        },
        "TD3": {
            "action_noise": [
                NormalActionNoise(np.array([0.0]), np.array([0.2])),
            ],
        },
        "SAC": {
            "action_noise": [
                NormalActionNoise(np.array([0.0]), np.array([0.2])),
            ],
            # "use_sde": [True, False],
        },
        "A2C": {
            # "gae_lambda": [0.1, 0.9],
            # "ent_coef": [0.1, 0.9],
            # "normalize_advantage": [True, False],
            # "use_sde": [True, False],
        },
        "PPO": {
            # "gae_lambda": [0.1, 0.9],
            # "ent_coef": [0.1, 0.9],
            # "use_sde": [True, False],
        },
        "PO": {"dc_step": 0.01},
    }

    for kw in utils.grid_generator(kwargs.get(model_name, {})):
        print("model_name", model_name)
        print("kw", kw)
        for c_kw in utils.grid_generator(common_kwargs):
            print("c_kw", c_kw)
            if model_name in SB3:
                model = exp(model_name, {**c_kw, **kw})
            else:
                model = exp(model_name, {**kw}, iters=1)


if __name__ == "__main__":
    import stable_baselines3
    from src import utils
    from stable_baselines3.common.noise import NormalActionNoise

    # test_po()

    algs = [
        "PO",
        "DDPG",
        "TD3",
        "SAC",
        "A2C",
        "PPO",
    ]
    for alg in algs:
        test_alg(alg)
