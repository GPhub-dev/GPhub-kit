"""Radar plots for GP libraries comparison."""

from math import pi
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .utils import save
from ..metrics import GPlibrary

_HUES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
_LINESTYLES = ("solid", (0, (5, 2)), (0, (1, 1.5)))
_KNOWN_LIBRARIES = (
    "DACE",
    "DiceKriging",
    "EGObox",
    "GaussianProcesses",
    "GPflow",
    "GPJax",
    "GPML",
    "GPstuff",
    "GPy",
    "GPyTorch",
    "OpenTURNS",
    "scikitlearn",
    "SMT",
    "STK",
    "tinygp",
    "UQLab",
    "UQpy",
)


def _style(library: str, position: int) -> dict:
    """Color and line style of a library (by name; by position in the chart for an unknown library)."""
    name = library.replace("GaussianProcessesjl", "GaussianProcesses")
    k = _KNOWN_LIBRARIES.index(name) if name in _KNOWN_LIBRARIES else len(_KNOWN_LIBRARIES) + position
    return {"color": _HUES[k % len(_HUES)], "linestyle": _LINESTYLES[(k // len(_HUES)) % len(_LINESTYLES)]}


_OFF_SCALE = 1e6  # |value| above which a metric is not drawn (e.g. NLPD with a numerically zero variance)


def _drawable(values: np.ndarray) -> np.ndarray:
    """Copy of the values with NaN where a metric is undefined (invalid variances) or off the scale."""
    values = np.array(values, dtype=float)
    values[~np.isfinite(values) | (np.abs(values) > _OFF_SCALE)] = np.nan
    return values


def _label(gplib: "GPlibrary", metric: str | None = None) -> str:
    """Display name of a library, with a remark when its NLPD or MSLL cannot be drawn."""
    name = gplib.display_name
    values = (
        [getattr(gplib, metric.lower())]
        if metric in ("NLPD", "MSLL")
        else [gplib.nlpd, gplib.msll]
        if metric is None
        else []
    )
    if any(np.isnan(v) for v in values):
        return rf"{name} (NLPD, MSLL: n/a)" if metric is None else rf"{name} (n/a)"
    if any(abs(v) > _OFF_SCALE for v in values):
        return rf"{name} (NLPD, MSLL $>10^6$)" if metric is None else rf"{name} ($>10^6$)"
    return name


def _fill(ax, angles: list[float], values: list[float], **kwargs) -> None:  # noqa: ANN001, ANN003
    """Fill a closed radar polygon; where values are missing, fill each run of valid values down to the centre."""
    values = np.asarray(values, dtype=float)
    if np.all(np.isfinite(values)):
        ax.fill(angles, values, **kwargs)
        return
    ylim = ax.get_ylim()
    runs, run = [], []
    for k, v in enumerate(values):
        if np.isfinite(v):
            run.append(k)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    for run in runs:
        if len(run) > 1:
            theta = [angles[run[0]], *[angles[k] for k in run], angles[run[-1]]]
            ax.fill(theta, [ylim[0], *values[run], ylim[0]], **kwargs)
    ax.set_ylim(ylim)


def _save_all(path: Path, filename: str, formats: tuple[str, ...]) -> None:
    for fmt in formats:
        save(plt, path=path, filename=filename, format=fmt)


def _adimensional_metrics_by_library(
    gp_libs: dict[str, "GPlibrary"], path: Path, formats: tuple[str, ...] = ("png",)
) -> None:
    gp_libs = {lib: gp_libs[lib] for lib in sorted(gp_libs, key=lambda k: gp_libs[k].display_name.lower())}
    libraries, _ = gp_libs.keys(), len(gp_libs.keys())

    MSE = np.array([gp_libs[lib].mse for lib in gp_libs])
    RMSE = np.array([gp_libs[lib].rmse for lib in gp_libs])
    MAE = np.array([gp_libs[lib].mae for lib in gp_libs])
    R2 = np.array([gp_libs[lib].r2 for lib in gp_libs])
    MedAE = np.array([gp_libs[lib].medae for lib in gp_libs])
    NLPD = _drawable([gp_libs[lib].nlpd for lib in gp_libs])
    MSLL = _drawable([gp_libs[lib].msll for lib in gp_libs])
    train_time = np.array([gp_libs[lib].train_time for lib in gp_libs])
    pred_time = np.array([gp_libs[lib].pred_time for lib in gp_libs])
    train_memory = np.array([gp_libs[lib].train_memory for lib in gp_libs])
    pred_memory = np.array([gp_libs[lib].pred_memory for lib in gp_libs])

    data = [
        MSE / MSE.max(),
        RMSE / RMSE.max(),
        MAE / MAE.max(),
        R2,
        MedAE / MedAE.max(),
        np.abs(NLPD) / np.nanmax(np.abs(NLPD)) if np.any(np.isfinite(NLPD)) else NLPD,
        np.abs(MSLL) / np.nanmax(np.abs(MSLL)) if np.any(np.isfinite(MSLL)) else MSLL,
        train_time / train_time.max(),
        pred_time / pred_time.max(),
        train_memory / train_memory.max(),
        pred_memory / pred_memory.max(),
    ]

    data_labels = [
        r"$MSE^*$",
        r"$RMSE^*$",
        r"$MAE^*$",
        r"$R^{2^*}$",
        r"$MedAE^*$",
        r"$NLPD^*$",
        r"$MSLL^*$",
        r"$t_{\text{train}}^*$",
        r"$t_{\text{pred}}^*$",
        r"$\text{mem.}_{\text{train}}^*$",
        r"$\text{mem.}_{\text{pred}}^*$",
    ]
    num_metrics = len(data_labels)

    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    for position, (lib_name, lib_data) in enumerate(zip(libraries, np.array(data).T, strict=False)):
        lib_data = list(lib_data)
        lib_data += lib_data[:1]
        style = _style(lib_name, position)
        ax.plot(angles, lib_data, linewidth=0.9, label=_label(gp_libs[lib_name]), **style)
        _fill(ax, angles, lib_data, alpha=0.05, color=style["color"])

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(data_labels)
    ax.tick_params(axis="x", pad=17.5)
    plt.yticks(color="grey", size=8)
    plt.legend(loc="upper right", bbox_to_anchor=(1.55, 1.0), handlelength=3.2)
    ax.grid(True)
    plt.tight_layout()
    _save_all(path, "radar_by_library", formats)
    plt.close()


def _metrics(gp_libs: dict, path: Path, formats: tuple[str, ...] = ("png",)) -> None:
    metrics = [
        "MAE",
        "RMSE",
        "MSE",
        "MedAE",
        "R2",
        "NLPD",
        "MSLL",
        "train_time",
        "pred_time",
        "train_memory",
        "pred_memory",
    ]
    libraries = sorted(gp_libs.keys(), key=lambda k: gp_libs[k].display_name.lower())
    num_libs = len(libraries)

    for metric in metrics:
        data = [list(_drawable([getattr(gp_libs[lib], metric.lower()) for lib in libraries]))]

        match metric:
            case "R2":
                data_labels = [r"$R^2$"]
            case "train_time":
                data_labels = [r"$t_{\text{train}}$"]
            case "pred_time":
                data_labels = [r"$t_{\text{pred}}$"]
            case "train_memory":
                data_labels = [r"$\text{Memory}_{\text{train}}\,\,\text{(MB)}$"]
            case "pred_memory":
                data_labels = [r"$\text{Memory}_{\text{pred}}\,\,\text{(MB)}$"]
            case _:
                data_labels = [rf"${metric}$"]

        angles = [n / float(num_libs) * 2 * pi for n in range(num_libs)]
        angles += angles[:1]  # Close the circle

        fig, ax = plt.subplots(subplot_kw=dict(polar=True))
        for idx, (label, lib_data) in enumerate(zip(data_labels, data, strict=False)):
            lib_data += lib_data[:1]  # Close the circle for each country
            ax.plot(angles, lib_data, linewidth=0.75, linestyle="solid", label=label, color="C0")
            _fill(ax, angles, lib_data, alpha=0.2, color="C0")

        # Customize the plot
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([_label(gp_libs[lib], metric) for lib in libraries])
        ax.tick_params(axis="x", pad=30)

        ax.set_rlabel_position(-0.0)
        plt.yticks(color="grey", size=8)

        # Add the legend
        plt.legend(loc="upper right", bbox_to_anchor=(1.43, 1.1))
        ax.grid(True)
        plt.tight_layout()

        _save_all(path, f"radar_{metric}", formats)
        plt.close()
