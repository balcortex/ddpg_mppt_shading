import datetime
import numbers
import random
from sys import modules
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
            # print(it)
            yield it


def make_random_synthetic_weather(
    modules: int,
    irradiance: Sequence[numbers.Real],
    amb_temperature: Sequence[numbers.Real],
    num_choices: int,
    size: int,
    initial_date: Optional[datetime.datetime] = None,
    time_delta: Optional[datetime.timedelta] = datetime.timedelta(minutes=10),
) -> pd.DataFrame:
    dic = DefaultDict(list)

    if initial_date:
        dic["Date"].extend(
            utils.date_steps(
                initial_date, time_delta, steps=size * num_choices, as_str=True
            )
        )

    g_key = "Irradiance {}"
    t_key = "Ambient Temperature {}"
    for m in range(1, modules + 1):
        dic[g_key.format(m)].extend(choice_repeat(irradiance, num_choices, size))
    for m in range(1, modules + 1):
        dic[t_key.format(m)].extend(choice_repeat(amb_temperature, num_choices, size))

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
    time_delta: Optional[datetime.timedelta] = datetime.timedelta(minutes=1),
) -> pd.DataFrame:
    dic = DefaultDict(list)

    if initial_date:
        dic["Date"].extend(
            utils.date_steps(
                initial_date, time_delta, steps=size * len(irradiance[0]), as_str=True
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
    # df = make_random_synthetic_weather(
    #     modules=4,
    #     irradiance=[200, 400, 600, 800, 1000],
    #     amb_temperature=[20, 25],
    #     num_choices=3,
    #     size=100,
    #     initial_date=datetime.datetime(2020, 1, 1, 8),
    # )

    df = make_synthetic_weather(
        irradiance=[
            [1000, 200, 800, 1000, 200],
            [400, 800, 800, 400, 800],
            [200, 200, 200, 200, 200],
            [600, 200, 20, 600, 200],
        ],
        amb_temperature=[
            [20, 20, 20, 20, 20],
            [25, 25, 25, 25, 25],
            [25, 25, 25, 25, 25],
            [20, 20, 20, 20, 20],
        ],
        size=200,
        initial_date=datetime.datetime(2020, 1, 1, 8),
    )

    df.to_csv(r"data\synthetic_weather.csv")
