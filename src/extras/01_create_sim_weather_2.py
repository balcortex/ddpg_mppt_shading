import datetime
import numbers
import random
from typing import Any, DefaultDict, Generator, Iterable, Optional, Sequence

import pandas as pd
from pathlib import Path

from src import utils


def choice_repeat(
    iterable: Iterable, num_choices: int, size: int
) -> Generator[Any, None, None]:
    choices = random.choices(iterable, k=num_choices)
    for choice in choices:
        for _ in range(size):
            yield choice


def make_random_synthetic_weather(
    modules: int,
    max_shaded_modules: int,
    irradiance: Sequence[numbers.Real],
    amb_temperature: Sequence[numbers.Real],
    days: int,
    steps_per_day: int,
    shaded_mod_prob: float = 0.5,
    initial_date: datetime.datetime = datetime.datetime(2020, 1, 1, 8),
    final_date: datetime.datetime = datetime.datetime(2020, 1, 1, 18),
    time_delta: datetime.timedelta = datetime.timedelta(minutes=1),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Make a dataframe with random weather measurements.

    Parameters:
        - modules: number of independent modulues
        - max_shaded_modules: maximum number of shaded modules per step
        - irradiance: list of possible values of irradiance
        - amb_temperature: list of possible values of ambient temperature
        - days: number of days to generate
        - steps_per_day: the number of weather steps per day
        - shaded_mod_prob: probability of shading at each step
        - initial_date: initial date of the dataframe
        - final_date: this date is used to separate the dataframe into different days.
            The day must be the same as `initial_date` and only differ in the hour.
        - time_delta: time elapsed between two adjacent samples
        - seed: seed for reproducibility

    """
    dic = DefaultDict(list)
    g_key = "Irradiance {}"
    t_key = "Ambient Temperature {}"
    day_size = (final_date - initial_date).seconds // time_delta.seconds
    step_size = day_size // steps_per_day

    print("Day size", day_size)
    print("Step size", step_size)

    total_shaded = 0
    total_non_shaded = 0
    random.seed(seed)

    for day in range(days):
        for step in range(steps_per_day):
            # produce a set of shaded modules if the probability is greater than a random num
            if shaded_mod_prob > random.random():
                shaded_mods = {
                    random.randint(0, modules - 1) for _ in range(max_shaded_modules)
                }
            else:
                shaded_mods = {}

            if len(shaded_mods) > 0:
                total_shaded += 1
            else:
                total_non_shaded += 1

            # fill the shaded mods
            for s_m in shaded_mods:
                dic[g_key.format(s_m)].extend(choice_repeat(irradiance, 1, step_size))
                dic[t_key.format(s_m)].extend(
                    choice_repeat(amb_temperature, 1, step_size)
                )

            # fill the non-shaded mods
            non_shaded_g = list(choice_repeat(irradiance, 1, step_size))
            non_shaded_t = list(choice_repeat(amb_temperature, 1, step_size))
            for m in range(modules):
                if m in shaded_mods:
                    continue
                dic[g_key.format(m)].extend(non_shaded_g)
                dic[t_key.format(m)].extend(non_shaded_t)

    print("Steps shaded", total_shaded)
    print("Steps non-shaded", total_non_shaded)

    # add the date
    if initial_date:
        dic["Date"].extend(
            utils.date_steps(
                initial_date,
                time_delta,
                steps=step_size * steps_per_day * days,
                stop=final_date,
                as_str=True,
            )
        )

    df = pd.DataFrame(dic)
    if dic.get("Date", None):
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", drop=True, inplace=True)

    return df


def create_train_test_set(
    modules: int,
    max_shaded_modules: int,
    irradiance: Sequence[numbers.Real],
    amb_temperature: Sequence[numbers.Real],
    days_train: int,
    days_test: int,
    steps_per_day: int = 6,
    shaded_mod_prob: float = 0.5,
    seed: int = 42,
) -> None:

    df_train = make_random_synthetic_weather(
        modules=modules,
        max_shaded_modules=max_shaded_modules,
        irradiance=irradiance,
        amb_temperature=amb_temperature,
        days=days_train,
        steps_per_day=steps_per_day,
        shaded_mod_prob=shaded_mod_prob,
        seed=seed,
    )

    df_test = make_random_synthetic_weather(
        modules=modules,
        max_shaded_modules=max_shaded_modules,
        irradiance=irradiance,
        amb_temperature=amb_temperature,
        days=days_test,
        steps_per_day=steps_per_day,
        shaded_mod_prob=shaded_mod_prob,
        seed=seed + 2,
    )

    basepath = Path("data")
    train_path = basepath.joinpath(
        f"synthetic_weather_train_{max_shaded_modules}_{modules}_{shaded_mod_prob}.csv"
    )
    test_path = basepath.joinpath(
        f"synthetic_weather_test_{max_shaded_modules}_{modules}_{shaded_mod_prob}.csv"
    )

    df_train.to_csv(train_path)
    df_test.to_csv(test_path)


if __name__ == "__main__":
    create_train_test_set(
        modules=4,
        max_shaded_modules=4,
        irradiance=[200, 400, 600, 800, 1000],
        amb_temperature=[25],
        days_train=300,
        days_test=60,
        shaded_mod_prob=0.9,
    )
