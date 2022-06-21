from dataclasses import dataclass
import re

from matplotlib import pyplot as plt
import numpy as np


plt.rcParams["figure.dpi"] = 600
W = 10000
H = 10000
kernel_size = 3
n = (W + kernel_size - 1) * (H + kernel_size - 1)


@dataclass
class Perf:
    block_size_x: int
    block_size_y: int
    time: float


def plot_pcolormesh(data: list[Perf], name: str):
    x: list[int] = np.sort(np.unique([perf.block_size_x for perf in data]))
    y: list[int] = np.sort(np.unique([perf.block_size_y for perf in data]))
    X, Y = np.meshgrid(x, y)
    Z: np.ndarray = np.zeros_like(X) * np.nan
    for perf in data:
        Z[(perf.block_size_x // 32) - 1][perf.block_size_y - 1] = perf.time
    plt.figure()
    plt.pcolormesh(X, Y, Z)
    plt.colorbar()
    plt.xlabel("block_size_x")
    plt.ylabel("block_size_y")
    plt.title(name)
    plt.savefig(f"{name}.png")


def plot_x(
    data: list[Perf], y: int = 1, label: str = None
) -> tuple[np.ndarray, np.ndarray]:
    X = np.sort(np.unique([perf.block_size_x for perf in data]))
    Y = np.zeros_like(X) * np.nan
    for perf in data:
        if perf.block_size_y != y:
            continue
        Y[X.searchsorted(perf.block_size_x)] = perf.time
    plt.plot(X, Y, label=label)
    return X, Y


def plot_y(
    data: list[Perf], x: int = 1, label: str = None
) -> tuple[np.ndarray, np.ndarray]:
    X = np.sort(np.unique([perf.block_size_y for perf in data]))
    Y = np.zeros_like(X) * np.nan
    for perf in data:
        if perf.block_size_x != x:
            continue
        Y[X.searchsorted(perf.block_size_y)] = perf.time
    plt.plot(X, Y, label=label)
    return X, Y


def read_results(file="workspace/results.txt") -> dict[str, list[Perf]]:
    naive: list[Perf] = []
    shared_memory: list[Perf] = []
    with open(file=file, mode="r") as txt:
        for line in txt.readlines():
            match_result = re.match(
                pattern=r"(naive|shared_memory) (\d*) (\d*) Exec-time: ([\d\.]*) ms",
                string=line,
            )
            mode: str = match_result.group(1)
            perf = Perf(
                block_size_x=int(match_result.group(2)),
                block_size_y=int(match_result.group(3)),
                time=float(match_result.group(4)),
            )
            if mode == "naive":
                naive.append(perf)
            elif mode == "shared_memory":
                shared_memory.append(perf)
            else:
                raise ValueError(f"Unknown Mode: {mode}")
    return {"naive": naive, "shared_memory": shared_memory}


def main():
    data = read_results()
    naive = data["naive"]
    shared_memory = data["shared_memory"]
    plot_pcolormesh(data=naive, name="naive")
    plot_pcolormesh(data=shared_memory, name="shared_memory")

    plt.figure(figsize=(16, 12))
    for i, x in enumerate([32, 64, 96, 128]):
        plt.subplot(2, 2, i + 1)
        plot_y(data=naive, x=x, label=f"naive")
        plot_y(data=shared_memory, x=x, label=f"shared_memory")
        plt.xlabel("block_size_y")
        plt.ylabel("time")
        plt.legend(loc="best")
        plt.title(f"block_size_x = {x}")
    plt.savefig("compare-y.png")

    plt.figure(figsize=(16, 12))
    for i, y in enumerate([1, 2, 3, 4]):
        plt.subplot(2, 2, i + 1)
        plot_x(data=naive, y=y, label=f"naive")
        plot_x(data=shared_memory, y=y, label=f"shared_memory")
        plt.xlabel("block_size_x")
        plt.ylabel("time")
        plt.legend(loc="best")
        plt.title(f"block_size_y = {y}")
    plt.savefig("compare-x.png")


if __name__ == "__main__":
    main()
