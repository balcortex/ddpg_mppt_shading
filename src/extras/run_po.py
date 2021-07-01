from src.model import PerturbObserveModel

if __name__ == "__main__":
    model = PerturbObserveModel(
        dc_step=0.01,
        env_kwargs={"weather_paths": ["train_1_4_0.5"]},
    )

    model.collect_step()
    model.collect_step()
    model.collect_step()
    model.collect_step()
    model.collect_step()
    model.play_episode()
    model.save_log()
