"""Module to plot GP predictions based on input dimension."""

from colour import Color
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

from ..metrics import GPlibrary


def plot(gplib: GPlibrary, *, show_train_points: bool = True, show_test: bool = True) -> None:
    """Plot GP predictions based on input dimension.

    Args:
        gplib: GPlibrary object containing the GP model and data
        show_train_points: Whether to show training points (default: True)
        show_test: Whether to show test data (for 1D only, default: True)
    """
    dim = gplib.test_x.shape[1]

    match dim:
        case 1:
            __1d(gplib, show_test=show_test)
        case 2:
            __2d(gplib, show_train_points=show_train_points)
            __3d(gplib, show_train_points=show_train_points)
        case _:
            pass


def __1d(gplib: GPlibrary, *, show_test: bool = True) -> None:
    idx = gplib.test_x.argsort(axis=0).flatten()
    pred_y_95_low = gplib.pred_y.flatten() - 1.96 * np.sqrt(gplib.pred_var.flatten())
    pred_y_95_high = gplib.pred_y.flatten() + 1.96 * np.sqrt(gplib.pred_var.flatten())
    plt.figure(figsize=(2 * 6.93 / 2.54, 6.93 / 2.54))
    plt.plot(gplib.test_x[idx], pred_y_95_low[idx], "--", color="#FA9FB5")
    plt.plot(gplib.test_x[idx], pred_y_95_high[idx], "--", color="#FA9FB5")
    plt.fill_between(
        np.ravel(gplib.test_x[idx]),
        np.ravel(pred_y_95_low[idx]),
        np.ravel(pred_y_95_high[idx]),
        alpha=0.1,
        color="#C51C8A",
        label=r"Confidence Interval 95\%",
    )
    plt.plot(gplib.test_x[idx], gplib.pred_y.flatten()[idx], label="Prediction mean", color="#9E166F", linewidth=0.6)
    if show_test:
        plt.plot(gplib.test_x[idx], gplib.test_y.flatten()[idx], "-.", label="True model", color="C0", linewidth=0.6)
    plt.scatter(
        gplib.train_x,
        gplib.train_y.flatten(),
        label="Observed data",
        color="#656565",
        alpha=0.5,
        zorder=10,
        s=15,
        edgecolor="k",
        linewidth=0.5,
    )
    plt.title(f"{gplib.display_name}")
    plt.legend(loc=0)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.tight_layout()


def __2d(gplib: GPlibrary, *, show_train_points: bool = True) -> None:
    assert gplib.test_x.shape[1] == 2, "Only 2D data is supported"

    xs, ys = gplib.test_x[:, 0], gplib.test_x[:, 1]

    xi, yi = np.linspace(min(xs), max(xs), 300), np.linspace(min(ys), max(ys), 300)
    X, Y = np.meshgrid(xi, yi)

    Z1 = griddata((xs, ys), gplib.test_y.flatten(), (X, Y), method="cubic")
    Z2 = griddata((xs, ys), gplib.pred_y.flatten(), (X, Y), method="cubic")
    Z3 = griddata((xs, ys), gplib.pred_var.flatten(), (X, Y), method="cubic")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3.75 * 6.93 / 2.54, 6.93 / 2.54))

    contour1 = ax1.contourf(X, Y, Z1, cmap="turbo", levels=100)
    contour2 = ax2.contourf(X, Y, Z2, cmap="turbo", levels=100)
    contour3 = ax3.contourf(X, Y, Z3, cmap="turbo", levels=100)

    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")
    ax1.set_title("True model")
    cbar = fig.colorbar(contour1, ax=ax1)
    cbar.set_label(r"$y_{\text{test}}$")
    ax1.set_aspect(1.0 / ax1.get_data_ratio(), adjustable="box")

    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$x_2$")
    ax2.set_title(rf"{gplib.display_name}")
    cbar = fig.colorbar(contour2, ax=ax2)
    cbar.set_label(r"$\hat{y}_{\text{pred}}$")
    ax2.set_aspect(1.0 / ax2.get_data_ratio(), adjustable="box")

    ax3.set_xlabel(r"$x_1$")
    ax3.set_ylabel(r"$x_2$")
    ax3.set_title(rf"{gplib.display_name}")
    cbar = fig.colorbar(contour3, ax=ax3)
    cbar.set_label(r"$\hat{\sigma}^2_{\text{pred}}$")
    ax3.set_aspect(1.0 / ax3.get_data_ratio(), adjustable="box")

    if show_train_points:
        ax2.scatter(
            gplib.train_x[:, 0],
            gplib.train_x[:, 1],
            marker=".",
            color="r",
            s=65,
            edgecolors="k",
            linewidths=0.55,
            alpha=0.95,
        )
        ax3.scatter(
            gplib.train_x[:, 0],
            gplib.train_x[:, 1],
            marker=".",
            color="r",
            s=65,
            edgecolors="k",
            linewidths=0.55,
            alpha=0.95,
        )

    plt.subplots_adjust(left=0.031, right=0.954, bottom=0.162, wspace=0.228)


def __3d(gplib: GPlibrary, *, show_train_points: bool = True) -> None:
    assert gplib.test_x.shape[1] == 2, "Only 2D data is supported"

    xs, ys = gplib.test_x[:, 0], gplib.test_x[:, 1]
    xi, yi = np.linspace(min(xs), max(xs), 300), np.linspace(min(ys), max(ys), 300)
    X, Y = np.meshgrid(xi, yi)

    Z1 = griddata((xs, ys), gplib.test_y.flatten(), (X, Y), method="cubic")
    Z2 = griddata((xs, ys), gplib.pred_y.flatten(), (X, Y), method="cubic")

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        subplot_kw={"projection": "3d"},
        figsize=(3 * 6.93 / 2.54, 1.5 * 6.93 / 2.54),
    )

    ax1.plot_surface(
        X,
        Y,
        Z1,
        cmap=__cmap_new(),
        antialiased=False,
        edgecolors="#4E7AA7",
        linewidth=0.002,
        alpha=0.8,
    )
    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")
    ax1.set_zlabel(r"$y_{\text{test}}$")
    ax1.set_title("True model")

    ax2.plot_surface(
        X,
        Y,
        Z2,
        cmap=__cmap_new(),
        antialiased=False,
        edgecolors="#4E7AA7",
        linewidth=0.002,
        alpha=0.8,
    )
    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$x_2$")
    ax2.set_zlabel(r"$\hat{y}_{\text{pred}}$")
    ax2.set_title(f"{gplib.display_name}")

    if show_train_points:
        ax2.scatter(
            gplib.train_x[:, 0],
            gplib.train_x[:, 1],
            gplib.train_y.flatten(),
            marker=".",
            color="r",
            s=120,
            edgecolors="k",
            linewidths=0.55,
            alpha=0.95,
        )


def __cmap_new() -> LinearSegmentedColormap:
    CustomCMAP = ["#E9F0FA", "#6D88A6"]
    return LinearSegmentedColormap.from_list("my_list", [Color(c1).rgb for c1 in CustomCMAP])  # type: ignore
