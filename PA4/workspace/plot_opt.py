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
    print("ref")
    ref: dict[str, float] = read_log("ref.log")
    print("phase_1")
    phase_1: dict[str, float] = read_log("phase_1.log")
    print("phase_2")
    phase_2: dict[str, float] = read_log("phase_2.log")
    print("opt")
    opt: dict[str, float] = read_log("32.log")
    x: np.ndarray = np.arange(len(ref))
    width: float = 0.2
    fig, ax = plt.subplots(figsize=(len(ref), 4.8), dpi=600)
    bar_ref = ax.bar(x - width * 1.5, ref.values(), width, label="ref")
    bar_phase_1 = ax.bar(x - width * 0.5, phase_1.values(), width, label="phase_1")
    bar_phase_2 = ax.bar(x + width * 0.5, phase_2.values(), width, label="phase_2")
    bar_opt = ax.bar(x + width * 1.5, opt.values(), width, label="opt")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Speedup")
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x, ref.keys())
    ax.legend()
    ax.bar_label(bar_ref, padding=3)
    ax.bar_label(bar_phase_1, padding=3)
    ax.bar_label(bar_phase_2, padding=3)
    ax.bar_label(bar_opt, padding=3)
    fig.tight_layout()
    plt.savefig("opt.png")


if __name__ == "__main__":
    main()
