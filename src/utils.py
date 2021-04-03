from pathlib import Path
from typing import Any, Dict, Union


def save_dic_txt(
    dic: Dict[str, str], path: Union[str, Path], sep: str = ":", overwrite: bool = False
) -> None:
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
    with open(path, "r") as f:
        return {key: val for line in f for key, val in [line.strip("\n").split(sep)]}


if __name__ == "__main__":
    dic = {
        "bel": "hola",
        # "bal": "(a,b,c)",
        # "bol": "(e,f,c)",
        "bil": "(j,f,c)",
        "bul": "(j,f,c)",
    }
    save_dic_txt(dic, "temp.txt", overwrite=False)

    dic_new = read_dic_txt("temp.txt")

    dic == dic_new
