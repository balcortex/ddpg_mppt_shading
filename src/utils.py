from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union, Generator
import itertools
import datetime

import pandas as pd


def save_dic_txt(
    dic: Dict[str, str], path: Union[str, Path], sep: str = ":", overwrite: bool = False
) -> None:
    """
    Write a dictionary to a file

    Parameters:
        - dic: the dictionary
        - path: the path of the file
        - sep: the separator character between the key and the value
        - overwrite: whether to append to an existing file or create a new file
    """
    mode = "a" if not overwrite else "w"

    if mode == "a":
        old_dic = read_dic_txt(path)
    elif mode == "w":
        old_dic = {}

    new_keys = dic.keys() - old_dic.keys()
    new_dic = {key: val for key, val in dic.items() if key in new_keys}

    line_gen = (f"{key}{sep}{val}\n" for (key, val) in new_dic.items())
    with open(path, mode) as f:
        for line in line_gen:
            f.write(line)


def read_dic_txt(path: Union[str, Path], sep: str = ":") -> Dict[str, str]:
    """
    Parse a file into a dictionary

    Parameters:
        - path: the path of the file
        - sep: the separator character between the key and the value
    """
    with open(path, "r") as f:
        return {key: val for line in f for key, val in [line.strip("\n").split(sep)]}


def date_steps(
    start: datetime.datetime,
    step: datetime.timedelta,
    steps: int,
    stop: Optional[datetime.datetime] = None,
    as_str: bool = False,
) -> Generator[Union[str, datetime.datetime], None, None]:
    """
    Generate datetime objects

    Parameters:
        - start: the start date
        - step: interval between steps
        - steps: number of steps
        - stop: start a new day if the current datetime is equal to stop
        - as_str: the output is plain str
    """
    start_ = start
    stop_ = stop
    for _ in range(steps):
        yield start if not as_str else str(start)
        start += step
        if start == stop:
            start_ += datetime.timedelta(days=1)
            stop_ += datetime.timedelta(days=1)
            start = start_
            stop = stop_


def csv_to_dataframe(path: Union[Path, str]) -> pd.DataFrame:
    """Read a csv file and return a dataframe with the columm `Date` as the index"""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", drop=True, inplace=True)
    return df


def grid_product(dic: Dict[Any, Any]) -> Generator[Dict[Any, Any], None, None]:
    """
    Perform permutation on the values of a dictionary.
    Works with nested dictionaries.

    Parameters:
        - dic: the dictionary

    Returns:
        a generator that combines the values of the dictionary (sequences)

    Examples:
        dic = {'1': [1, 2], '2': [3, 4]}
        gp = grid_product(dic)
        next(gp)
        -> {'1': 1, '2': 3}
        next(gp)
        -> {'1': 1, '2': 4}
        next(gp)
        -> {'1': 2, '2': 3}
        next(gp)
        -> {'1': 2, '2': 4}

        dic = {'1': {'a': [5, 6], 'b': 7}, '2': [3, 4]}
        gp = grid_product(dic)
        next(gp)
        # {'1': {'a': 5, 'b': 7}, '2': 3}
        next(gp)
        # {'1': {'a': 5, 'b': 7}, '2': 4}
        next(gp)
        # {'1': {'a': 6, 'b': 7}, '2': 3}
        next(gp)
        # {'1': {'a': 6, 'b': 7}, '2': 4}

    """
    if not dic:
        return ({},)

    # Perform a copy of the dic (do not touch the original dic)
    dic = dic.copy()

    # Check if val is a Sequence
    for key, val in dic.items():
        if not isinstance(val, Sequence) or isinstance(val, str):
            dic[key] = [val]

        if isinstance(val, dict):
            dic[key] = list(grid_product(val))

    keys, values = zip(*dic.items())

    return (dict(zip(keys, v)) for v in itertools.product(*values))


def grid_combination(dic: Dict[Any, Any]) -> Generator[Dict[Any, Any], None, None]:
    """
    Combine the elements of the values (sequence) of a dictionary.
    First yield the elements in the position 0, next the elements in the position 1...

    Parameters:
        - dic: the dictionary

    Examples:
        dic = {'1': [1, 2], '2': [3, 4]}
        gc = grid_combination(dic)
        next(gc)
        # {1: 1, 2: 3}
        next(gc)
        # {1: 2, 2: 4}

        dic = {'1': {'a': [5, 6], 'b': 7}, '2': [3, 4]}
        gc = grid_combination(dic)
        next(gc)
        # {'1': {'a': 5, 'b': 7}, '2': 3}
        next(gc)
        # {'1': {'a': 6, 'b': 7}, '2': 4}


    """
    if not dic:
        return ({},)

    # Get the maximum length of the sequence in the dict values
    max_len = 0
    for key, val in dic.items():
        if not isinstance(val, Sequence) or isinstance(val, str):
            continue
        max_len = max(max_len, len(val))

    # Make the sequences of equal length
    for key, val in dic.items():
        if not isinstance(val, Sequence) or isinstance(val, str):
            dic[key] = [val] * max_len
        elif isinstance(val, Sequence) and len(val) != max_len:
            dic[key] = val * max_len

        if isinstance(val, dict):
            dic[key] = list(grid_combination(val))

    lens = {len(val) for val in dic.values()}
    assert (
        len(lens) == 1
    ), "The sequences' length in the dict values are not equal. The dict values may be a scalar, sequence of lenght 1 or sequence of max_len"

    keys, values = zip(*dic.items())

    return (dict(zip(keys, v)) for v in zip(*values))


def copy_dic(dic: Optional[Dict[Any, Any]]) -> Dict[Any, Any]:
    """
    Return a copy of the dictionary or an empty dictionary if no one is provided
    """
    if dic is not None:
        return dic.copy()
    return {}


if __name__ == "__main__":
    # dic = {
    #     "bel": "hola",
    #     # "bal": "(a,b,c)",
    #     # "bol": "(e,f,c)",
    #     "bil": "(j,f,c)",
    #     "bul": "(j,f,c)",
    # }
    # save_dic_txt(dic, "temp.txt", overwrite=False)

    # dic_new = read_dic_txt("temp.txt")

    # dic == dic_new

    for i in date_steps(
        start=datetime.datetime(2020, 1, 1, 8),
        step=datetime.timedelta(minutes=10),
        steps=10,
    ):
        print(i)