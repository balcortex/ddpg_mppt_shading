from typing import Optional
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PATH = Path("results")
FOLDER_ID = 0
FILE_ID = 3
YLABEL = {"Power": "Power (W)", "Duty Cycle": "Duty Cycle"}
MPPT = {"po": "P&O", "ddpg": "DDPG", "td3": "TD3", "td3exp": "TD4"}

# plt.rc("text", usetex=True)
# plt.rc("font", family="serif")

locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
formatter = mdates.ConciseDateFormatter(locator)


def read_csv(
    *, folder_id: int, file_id: int, root_path: Optional[Path] = None
) -> pd.DataFrame:
    path = root_path or PATH
    path = list(path.glob("*"))[folder_id]
    path = list(path.glob("*.csv"))[file_id]
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", drop=True, inplace=True)

    mppt_name = path.parts[1].split("_")[-2]

    return df, mppt_name


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


# df, name = read_csv(folder_id=0, file_id=3) # po
# df, name = read_csv(folder_id=4, file_id=3) # ddpg
# df, name = read_csv(folder_id=6, file_id=3)  # td3
df, name = read_csv(folder_id=10, file_id=3)  # td4
p = plot(df, feature="power", mppt_name=name)
print("")
