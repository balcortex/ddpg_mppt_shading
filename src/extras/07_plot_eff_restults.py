from collections import defaultdict
import numbers
import copy
from pathlib import Path
from typing import Dict, NamedTuple, Sequence, Any


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


# def merge_exps(exps: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
#     dic_total = {}
#     for exp in exps:
#         exp = dict(exp) # Make a copy
#         name = exp.pop("name")
#         d = dic_total.get(name, defaultdict(list))
#         for k, v in exp.items():
#             d[k].append(v)

#         dic_total[name] = d

#     return dic_total


def merge_exps(
    exps: Sequence[Dict[str, Any]]
) -> Dict[str, Dict[str, Sequence[numbers.Real]]]:
    d = defaultdict(lambda: defaultdict(list))
    for exp in exps:
        exp = dict(exp)  # Make a copy
        name = exp.pop("name").lower()

        for k, v in exp.items():
            d[name][k].append(v)

    # Cast the defaultdict to dict
    return {k: {k1: v1 for k1, v1 in v.items()} for k, v in d.items()}


PATH = Path("default/log.txt")

exps = [parse_experiment(exp) for exp in PATH.read_text().split("\n\n")]
merged = merge_exps(exps)

import matplotlib.pyplot as plt
import numpy as np

a = merged["td3experience"]["test_mean_ep_eff"]
b = merged["td3"]["test_mean_ep_eff"]
plt.plot(a, label="td3experience")
plt.plot(b, label="td3")
plt.plot((0, len(a)), (np.mean(a),) * 2, label="td3experience_mean")
plt.plot((0, len(b)), (np.mean(b),) * 2, label="td3_mean")

plt.legend()
plt.show()

a = merged["td3experience"]["train_mean_ep_eff"]
b = merged["td3"]["train_mean_ep_eff"]
plt.plot(a, label="td3experience")
plt.plot(b, label="td3")
plt.plot((0, len(a)), (np.mean(a),) * 2, label="td3experience_mean")
plt.plot((0, len(b)), (np.mean(b),) * 2, label="td3_mean")

plt.legend()
plt.show()
