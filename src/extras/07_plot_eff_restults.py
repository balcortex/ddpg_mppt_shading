import numbers
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

PATH = Path("default/log.txt")

MAPPING = {
    "test_mean_ep_eff": {
        "ylabel": "Tracking efficiency (%)",
        "title": "Mean episode efficiency on the Test Set",
    },
    "train_mean_ep_eff": {
        "ylabel": "Mean episode efficiency (%)",
        "title": "Mean episode efficiency on the Training Set",
    },
}


def parse_experiment(txt: str) -> Dict[str, Any]:
    """
    txt sample:
        TD3Experience
        Train episodes: 60
        Train mean ep eff: 87.03312429829461
        Train mean ep rew: 295.24693277493054
        Test episodes: 30
        Test mean ep eff: 89.14478593730996
        Test mean ep rew: 231.37040283519985
    """
    dic = {}

    txt_seq = txt.split("\n")
    dic["name"] = txt_seq[0]

    for line in txt.split("\n")[1:]:
        s = line.split(": ")
        key_, val_ = s[0], s[1]
        key = "_".join(key_.lower().split(" "))
        val = float(val_)
        dic[key] = val

    return dic


def merge_exps(
    exps: Sequence[Dict[str, Any]]
) -> Dict[str, Dict[str, Sequence[numbers.Real]]]:
    """
    Merge the results of a sequence of experiments into a dictinary
    """
    d = defaultdict(lambda: defaultdict(list))
    for exp in exps:
        exp = dict(exp)  # Make a copy
        name = exp.pop("name").lower().replace("experience", "exp")

        for k, v in exp.items():
            d[name][k].append(v)

    # Cast the defaultdict to dict
    return {k: {k1: v1 for k1, v1 in v.items()} for k, v in d.items()}


def merged_max(dic: Dict[str, Dict[str, Sequence[numbers.Real]]]) -> int:
    """Return the maximum number of experiments in the merged dict"""
    max_ = 0
    for k, v in dic.items():
        for k_, v_ in v.items():
            max_ = max(max_, len(v_))

    return max_


def plot_bar_mean(
    dic: Dict[str, Dict[str, Sequence[numbers.Real]]], feature: str
) -> Figure:
    fig, ax = plt.subplots()
    for i, (k, v) in enumerate(dic.items()):
        arr = np.array(v[feature])
        ax.bar(i, arr.mean(), yerr=arr.std(), capsize=10)
        ax.text(i - 0.25, 40, f"{arr.mean():0.2f}")
    ax.set_xticks(range(len(dic.keys())))
    ax.set_xticklabels(dic.keys())
    ax.set_ylabel(MAPPING[feature]["ylabel"])
    ax.set_title(MAPPING[feature]["title"])

    return fig


def plot_line(
    dic: Dict[str, Dict[str, Sequence[numbers.Real]]],
    feature: str,
    include_mean: bool = True,
) -> Figure:
    fig, ax = plt.subplots()
    for k, v in dic.items():
        arr = np.array(v[feature])
        if len(arr) == 1:
            mean_ = arr.mean()
            ax.plot((1, merged_max(dic)), (mean_,) * 2, ":", label=f"{k}_mean")
            continue
        ax.plot(range(1, len(arr) + 1), arr, label=k)
        if include_mean:
            mean_ = arr.mean()
            ax.plot((1, merged_max(dic)), (mean_,) * 2, ":", label=f"{k}_mean")
    ax.set_ylabel(MAPPING[feature]["ylabel"])
    ax.set_title(MAPPING[feature]["title"])
    ax.legend()

    return fig


exps = [parse_experiment(exp) for exp in PATH.read_text().split("\n\n")]
merged = merge_exps(exps)

plot_bar_mean(merged, "test_mean_ep_eff")
plot_bar_mean(merged, "train_mean_ep_eff")
plot_line(merged, "test_mean_ep_eff", include_mean=False)
plot_line(merged, "train_mean_ep_eff", include_mean=False)
