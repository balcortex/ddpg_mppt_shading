from pathlib import Path
from typing import Sequence

# algs = ["PO", "DDPG", "TD3", "SAC", "A2C", "PPO"]
path = Path("default/TD3")
txts = path.glob("*test\efficiency.txt")

effs = [f.read_text().strip().split("\n") for f in txts]


def max_eff(seq: Sequence[str]) -> float:
    effs = [float(eff) for s in seq for _, eff in [s.split(":")]]
    dates = [date for s in seq for date, _ in [s.split(":")]]
    max_ = max(effs)
    idx = effs.index(max_)

    return max_, dates[idx]


max_effs = [max_eff(eff) for eff in effs]
print(sorted(max_effs, reverse=True))


# Norm Reward, states = ("norm_voltage", "norm_delta_voltage", "norm_power")
# lr = 1e-4, gamma = 0.01
# PO -> 67.47
# DDPG -> 90.93
# TD3 -> 91.13
# SAC -> 73.73
# A2C -> 68.67
# PPO -> 73.13


# DDPG -> 91.35, lr=1e-3, gamma=0.1, noise=0.2
# TD3 -> 92.63, lr=1e-3, gamma=0.1, noise=0.5