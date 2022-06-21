import re
from this import d
import numpy as np
import matplotlib.pyplot as plt


def read_log(file: str) -> dict[str, float]:
    speedup: dict[str, float] = {}
    dset: str = ""
    ref_time: float = None
    opt_time: float = None
    for line in open(file=file, mode="r"):
        dset_pattern: str = r"dset = \"([\w\.]*)\""
        time_pattern: str = r"time = ([\d\.]*) \(double\)"
        match: re.Match = re.search(dset_pattern, line)
        if match:
            dset = match.group(1)
        match: re.Match = re.search(time_pattern, line)
        if match:
            if ref_time:
                opt_time: float = float(match.group(1))
                speedup[dset] = ref_time / opt_time
                print(f"| {dset} | {ref_time} | {opt_time} | {speedup[dset]} |")
                ref_time: float = None
            else:
                ref_time: float = float(match.group(1))
    return speedup


def main():
    print("kLen = 32")
    speedup_32: dict[str, float] = read_log("32.log")
    print("kLen = 256")
    speedup_256: dict[str, float] = read_log("256.log")
    x: np.ndarray = np.arange(len(speedup_32))
    width: float = 0.35
    fig, ax = plt.subplots(figsize=(len(speedup_32), 4.8), dpi=600)
    bar_32 = ax.bar(x - width / 2, speedup_32.values(), width, label="kLen = 32")
    bar_256 = ax.bar(x + width / 2, speedup_256.values(), width, label="kLen = 256")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Speedup")
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x, speedup_32.keys())
    ax.legend()
    ax.bar_label(bar_32, padding=3)
    ax.bar_label(bar_256, padding=3)
    fig.tight_layout()
    plt.savefig("speedup.png")


if __name__ == "__main__":
    main()
