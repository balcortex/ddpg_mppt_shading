from pathlib import Path
from typing import Sequence

import matplotlib.axes._axes as axes
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

PATH = Path(r"C:\Users\balco\Downloads\phd_data")
DAY_INDEX = [1, 3, 25, 27]
FOLDER_INDEX = [1, 2, 3, 4]
DAY_ALONE_INDEX = 5
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

# Pyplot configuration
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=COLOR_LIST)
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
formatter = mdates.ConciseDateFormatter(locator)


def mean_across_dataframes(dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs).groupby(level=0).mean()


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", drop=True, inplace=True)

    return df


def get_dirs(name: str):
    """Get all directories which contain the specified word in their name"""
    return PATH.glob(f"*{name}*")


def get_file_from_dir(dir_: Path, index: int):
    """Get the csv at the specified position"""
    return list(dir_.glob("*.csv"))[index]


def get_files(name: str, index: int):
    """
    Get the csv files at the specified position for all the directories which contain the word `name`
    """
    return (get_file_from_dir(d, index) for d in get_dirs(name))


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

    return fig


def compute_eff(df: pd.DataFrame) -> pd.DataFrame:
    sum_ = df.sum()
    max_ = sum_[0]

    names = sum_.index[1:]
    effs = [s / max_ for s in sum_[1:]]

    return pd.DataFrame({"MPPT": names, "Eff": effs})


def main_plot_day():
    for i, folder in enumerate(FOLDER_INDEX, start=1):
        po_test_files = [list(get_files("po_test", DAY_ALONE_INDEX))[0]]
        ddpg_test_files = [list(get_files("ddpg_test", DAY_ALONE_INDEX))[folder]]
        td3_test_files = [list(get_files("td3_test", DAY_ALONE_INDEX))[folder]]
        td4_test_files = [list(get_files("td3exp_test", DAY_ALONE_INDEX))[folder]]

        assert len(po_test_files) == 1
        assert len(ddpg_test_files) == 1
        assert len(td3_test_files) == 1
        assert len(td4_test_files) == 1

        po_mean = mean_across_dataframes(read_csv(f) for f in po_test_files)
        ddpg_mean = mean_across_dataframes((read_csv(f) for f in ddpg_test_files))
        td3_mean = mean_across_dataframes((read_csv(f) for f in td3_test_files))
        td4_mean = mean_across_dataframes((read_csv(f) for f in td4_test_files))

        df = combine_df(
            [po_mean, ddpg_mean, td3_mean, td4_mean],
            ["P\&O", "DDPG", "TD3", "TD4", "Max"],
            feature="power",
        )

        p = plot_combined(df, ylabel="Power (W)")

        p.savefig(f"output\\fig_07_mppt_comparison_{i:02}.pdf", bbox_inches="tight")


def main_plot_mean():
    for i, day in enumerate(DAY_INDEX, start=1):
        po_test_files = [list(get_files("po_test", day))[0]]
        ddpg_test_files = list(get_files("ddpg_test", day))
        td3_test_files = list(get_files("td3_test", day))
        td4_test_files = list(get_files("td3exp_test", day))

        assert len(po_test_files) == 1
        assert len(ddpg_test_files) == 110
        assert len(td3_test_files) == 110
        assert len(td4_test_files) == 110

        po_mean = mean_across_dataframes(read_csv(f) for f in po_test_files)
        ddpg_mean = mean_across_dataframes((read_csv(f) for f in ddpg_test_files))
        td3_mean = mean_across_dataframes((read_csv(f) for f in td3_test_files))
        td4_mean = mean_across_dataframes((read_csv(f) for f in td4_test_files))

        df = combine_df(
            [po_mean, ddpg_mean, td3_mean, td4_mean],
            ["P\&O", "DDPG", "TD3", "TD4", "Max"],
            feature="power",
        )

        p = plot_combined(df, ylabel="Power (W)")

        p.savefig(
            f"output\\fig_09_mppt_mean_comparison_{i:02}.pdf", bbox_inches="tight"
        )


if __name__ == "__main__":
    main_plot_day()
    main_plot_mean()
