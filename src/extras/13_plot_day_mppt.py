from pathlib import Path
from typing import Optional, Sequence
from click import style
import matplotlib

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.axes._axes as axes
import pandas as pd

PATH = Path(r"C:\Users\balco\Downloads\phd_data")
DAY_INDEX = [1, 3, 25, 27]
PO_FOLDER_ID = 0
DDPG_FOLDER_ID = 4
TD3_FOLDER_ID = 6
TD4_FOLDER_ID = 10

YLABEL = {"Power": "Power (W)", "Duty Cycle": "Duty Cycle"}
MPPT = {"po": "P&O", "ddpg": "DDPG", "td3": "TD3", "td3exp": "TD4", "": ""}

CB91_Blue = "#2CBDFE"
CB91_Green = "#47DBCD"
CB91_Pink = "#F3A0F2"
CB91_Purple = "#9D2EC5"
CB91_Violet = "#661D98"
CB91_Amber = "#F5B14C"
C_RED = "#c91711"
C_YELLOW = "#d7c35c"

COLOR_LIST = [
    CB91_Blue,
    C_YELLOW,
    C_RED,
    CB91_Violet,
    CB91_Green,
    CB91_Amber,
    CB91_Pink,
    CB91_Purple,
]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=COLOR_LIST)

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
formatter = mdates.ConciseDateFormatter(locator)


def read_csv(
    *, folder_id: int, file_id: int, root_path: Optional[Path] = None
) -> pd.DataFrame:
    path = root_path or PATH
    path = list(path.glob("*"))[folder_id]
    path = list(path.glob("*.csv"))[file_id]
    print(path)
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", drop=True, inplace=True)

    mppt_name = path.parts[-2].split("_")[-2]

    return df, mppt_name


def combine_df(
    dfs: Sequence[pd.DataFrame], names: Sequence[str], feature: str = "power"
) -> pd.DataFrame:
    """
    Combine multiple dataframes into a single one
    """

    feature = feature.title()
    feat_opt = "Optimum " + feature

    combined_df = {}
    combined_df[names[-1]] = dfs[0][feat_opt]

    for df, name in zip(dfs, names):
        combined_df[name] = df[feature]

    return pd.DataFrame(combined_df)


def plot(df: pd.DataFrame, *, feature: str, mppt_name: str):
    feature = feature.title()
    feat_opt = "Optimum " + feature

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(df.index, df[feat_opt], label=feat_opt)
    ax.plot(df.index, df[feature], label=MPPT[mppt_name])
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.legend(loc="upper left")
    ax.set_ylabel(YLABEL[feature])

    return fig


def plot_combined(df: pd.DataFrame, ylabel: str = "") -> Figure:

    fig = plt.figure()
    ax: axes.Axes = fig.add_subplot(111)

    effs = compute_eff(df).iloc[:, -1].values

    ax.plot(df.index, df.iloc[:, 0], label=df.columns[0], color="k", linestyle="dotted")
    for column, eff in zip(df.columns[1:], effs):
        label = f"{column:<8}({eff*100:.1f}\%)"
        ax.plot(df.index, df[column], label=label)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.legend(loc="upper left")

    ax.set_ylabel(ylabel)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped

    return fig


def compute_eff(df: pd.DataFrame) -> pd.DataFrame:
    sum_ = df.sum()
    max_ = sum_[0]

    names = sum_.index[1:]
    effs = [s / max_ for s in sum_[1:]]

    return pd.DataFrame({"MPPT": names, "Eff": effs})


for i, day in enumerate(DAY_INDEX, start=1):
    # for i, day in enumerate(range(30), start=1):
    df_po, _ = read_csv(folder_id=PO_FOLDER_ID, file_id=day)  # po
    df_ddpg, _ = read_csv(folder_id=DDPG_FOLDER_ID, file_id=day)  # ddpg
    df_td3, _ = read_csv(folder_id=TD3_FOLDER_ID, file_id=day)  # td3
    df_td4, _ = read_csv(folder_id=TD4_FOLDER_ID, file_id=day)  # td4

    df = combine_df(
        [df_po, df_ddpg, df_td3, df_td4],
        ["P\&O", "DDPG", "TD3", "TD4", "Max"],
        feature="power",
    )
    p = plot_combined(df, ylabel="Power (W)")

    p.savefig(f"output\\fig_07_mppt_comparison_{i:02}.pdf", bbox_inches="tight")
    # plt.close(p)


# fig_po = plot(df_po, feature="power", mppt_name="")
# fig_ddpg = plot(df_ddpg, feature="power", mppt_name="")
# fig_td3 = plot(df_td3, feature="power", mppt_name="")
# fig_td4 = plot(df_td4, feature="power", mppt_name="")
