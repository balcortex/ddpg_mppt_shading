import datetime
import numbers
import random
from typing import Any, DefaultDict, Generator, Iterable, Optional, Sequence

import pandas as pd

from src import utils


def choice_repeat(
    iterable: Iterable, num_choices: int, size: int
) -> Generator[Any, None, None]:
    choices = random.choices(iterable, k=num_choices)
    for choice in choices:
        for _ in range(size):
            yield choice


def repeat_each(iterable: Iterable, size: int) -> Generator[Any, None, None]:
    for it in iterable:
        for _ in range(size):
            yield it


def make_random_synthetic_weather(
    modules: int,
    irradiance: Sequence[numbers.Real],
    amb_temperature: Sequence[numbers.Real],
    days: int,
    steps_per_day: int,
    initial_date: datetime.datetime = datetime.datetime(2020, 1, 1, 8),
    final_date: datetime.datetime = datetime.datetime(2020, 1, 1, 18),
    time_delta: datetime.timedelta = datetime.timedelta(minutes=1),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Make a dataframe with random weather measurements.

    Parameters:
        - irradiance: list of possible values of irradiance
        - amb_temperature: list of possible values of ambient temperature
        - days: number of days to generate
        - steps_per_day: the number of weather steps per day
        - initial_date: initial date of the dataframe
        - final_date: this date is used to separate the dataframe into different days.
            The day must be the same as `initial_date` and only differ in the hour.
        - time_delta: time elapsed between two adjacent samples

    """
    dic = DefaultDict(list)

    g_key = "Irradiance {}"
    t_key = "Ambient Temperature {}"
    day_size = (final_date - initial_date).seconds // time_delta.seconds
    step_size = day_size // steps_per_day

    random.seed(seed)
    for _ in range(days):
        for m in range(1, modules + 1):
            dic[g_key.format(m)].extend(
                choice_repeat(irradiance, steps_per_day, step_size)
            )
        for m in range(1, modules + 1):
            dic[t_key.format(m)].extend(
                choice_repeat(amb_temperature, steps_per_day, step_size)
            )

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


def make_synthetic_weather(
    irradiance: Sequence[Sequence[numbers.Real]],
    amb_temperature: Sequence[Sequence[numbers.Real]],
    size: int,
    initial_date: Optional[datetime.datetime] = None,
    final_date: Optional[datetime.datetime] = None,
    time_delta: Optional[datetime.timedelta] = datetime.timedelta(minutes=1),
) -> pd.DataFrame:
    dic = DefaultDict(list)

    if initial_date:
        dic["Date"].extend(
            utils.date_steps(
                initial_date,
                time_delta,
                steps=size * len(irradiance[0]),
                stop=final_date,
                as_str=True,
            )
        )

    g_key = "Irradiance {}"
    t_key = "Ambient Temperature {}"
    for i, (g, t) in enumerate(zip(irradiance, amb_temperature), start=1):
        dic[g_key.format(i)].extend(repeat_each(g, size))
        dic[t_key.format(i)].extend(repeat_each(t, size))

    df = pd.DataFrame(dic)
    if dic.get("Date", None):
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", drop=True, inplace=True)

    return df


if __name__ == "__main__":
    # df = make_synthetic_weather(
    #     irradiance=[
    #         [1000, 200, 800, 1000, 200],
    #         [400, 800, 800, 400, 800],
    #         [200, 200, 200, 200, 200],
    #         [600, 200, 20, 600, 200],
    #     ],
    #     amb_temperature=[
    #         [20, 20, 20, 20, 20],
    #         [25, 25, 25, 25, 25],
    #         [25, 25, 25, 25, 25],
    #         [20, 20, 20, 20, 20],
    #     ],
    #     size=200,
    #     initial_date=datetime.datetime(2020, 1, 1, 1),
    #     final_date=datetime.datetime(2020, 1, 1, 23),
    # )

    # df.to_csv(r"data\synthetic_weather_test.csv")

    df = make_random_synthetic_weather(
        modules=4,
        irradiance=[200, 400, 600, 800, 1000],
        amb_temperature=[20, 25],
        days=365,
        steps_per_day=6,
    )

    df.to_csv(r"data\synthetic_weather_train.csv")

    # df = make_synthetic_weather(
    #     irradiance=[
    #         [1000, 400, 800, 1000, 200],
    #         [1000, 400, 800, 1000, 200],
    #         [1000, 400, 800, 1000, 200],
    #         [1000, 400, 800, 1000, 200],
    #     ],
    #     amb_temperature=[
    #         [20, 20, 20, 20, 20],
    #         [20, 20, 20, 20, 20],
    #         [20, 20, 20, 20, 20],
    #         [20, 20, 20, 20, 20],
    #     ],
    #     size=200,
    #     initial_date=datetime.datetime(2020, 1, 1, 1),
    #     final_date=datetime.datetime(2020, 1, 1, 23),
    # )

    # df.to_csv(r"data\synthetic_weather_test_uniform.csv")