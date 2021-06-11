from src.model import PerturbObserveModel, RandomModel

model = PerturbObserveModel(
    env_kwargs={"env_names": ["po_test"], "weather_paths": ["test_1_4_0.5"]}
)
model.play_episode()
model.quit()

# model = RandomModel(
#     env_kwargs={"env_names": ["random_test"], "weather_paths": ["test_1_4_0.5"]}
# )
# model.play_episode()
# model.quit()
