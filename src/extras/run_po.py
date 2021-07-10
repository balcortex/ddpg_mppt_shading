from src.model import PerturbObserveModel

if __name__ == "__main__":
    model = PerturbObserveModel(
        dc_step=0.01,
        env_kwargs={"weather_paths": ["train_1_4_0.5"]},
    )

    model.play_episode()
    model.play_episode()
    model.save_log()

    dic = {
        "dc_step": [0.01, 0.02],
        "env_kwargs": {"weather_paths": [["train_1_4_0.5"]]},
    }
    PerturbObserveModel.run_from_grid(dic, episodes=5)
