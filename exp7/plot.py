import os
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

ASSETS_DIR = "report.assets"


def plot_global_memory():
    stride: np.ndarray = np.array([1, 2, 4, 8])
    bandwidth: np.ndarray = np.array([530.015, 182.471, 91.9932, 46.2866])
    x: np.ndarray = np.log2(stride)
    y: np.ndarray = np.log2(bandwidth)
    x_interp: np.ndarray = np.linspace(start=x.min(), stop=x.max(), num=1000)
    y_interp: np.ndarray = scipy.interpolate.interp1d(x=x, y=y, kind="cubic")(x_interp)
    x = 2**x
    y = 2**y
    x_interp = 2**x_interp
    y_interp = 2**y_interp
    plt.figure(dpi=600)
    plt.scatter(x=x, y=y)
    # plt.plot(x_interp, y_interp)
    plt.loglog(x_interp, y_interp, base=2)
    plt.xlabel("STRIDE")
    plt.ylabel("Bandwidth (GB/s)")
    plt.title("Global Memory")
    plt.savefig(os.path.join(ASSETS_DIR, "global_memory.png"))


def plot_shared_memory():
    bitwidth: np.ndarray = np.array(
        [2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8]
    )
    stride: np.ndarray = np.array(
        [1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32]
    )
    bandwidth: np.ndarray = np.array(
        [
            4258.05,
            4270.88,
            2149.69,
            831.405,
            427.135,
            215.022,
            8607.33,
            4315.77,
            2027.46,
            1012.86,
            504.37,
            251.766,
            8657.57,
            4339.44,
            2173.55,
            1087.65,
            544.069,
            544.068,
        ]
    )
    plt.figure(dpi=600)
    for bit in np.unique(bitwidth):
        indices = bitwidth == bit
        x: np.ndarray = np.log2(stride[indices])
        y: np.ndarray = np.log2(bandwidth[indices])
        x_interp: np.ndarray = np.linspace(start=x.min(), stop=x.max(), num=1000)
        y_interp: np.ndarray = scipy.interpolate.interp1d(x=x, y=y, kind="cubic")(
            x_interp
        )
        x = 2**x
        y = 2**y
        x_interp = 2**x_interp
        y_interp = 2**y_interp
        plt.scatter(x=x, y=y)
        plt.loglog(x_interp, y_interp, base=2, label=f"BITWIDTH: {bit}")
    plt.xlabel("STRIDE")
    plt.ylabel("Bandwidth (GB/s)")
    plt.legend(loc="best")
    plt.title("Shared Memory")
    plt.savefig(os.path.join(ASSETS_DIR, "shared_memory.png"))


def main():
    plot_global_memory()
    plot_shared_memory()


if __name__ == "__main__":
    main()
