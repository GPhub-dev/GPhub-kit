"""Module for storing and computing metrics for Gaussian Process Regression libraries."""

from pathlib import Path

from attrs import field, define
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
import numpy as np
import polars as pl

from ..utils import table, console

DISPLAY_NAMES = {
    "EGObox": "egobox-gp",
    "GaussianProcesses": "GaussianProcesses.jl",
    "GaussianProcessesjl": "GaussianProcesses.jl",
    "scikitlearn": "scikit-learn",
}


@define
class GPlibrary:
    """GPlibrary class for storing and computing metrics for Gaussian Process Regression libraries."""

    library: str
    train_x: np.ndarray = field(default=None)
    train_y: np.ndarray = field(default=None)
    test_x: np.ndarray = field(default=None)
    test_y: np.ndarray = field(default=None)
    pred_y: np.ndarray = field(default=None)
    pred_var: np.ndarray = field(default=None)
    train_time: float = field(default=None)
    pred_time: float = field(default=None)
    train_memory: float = field(default=None)
    pred_memory: float = field(default=None)

    def __attrs_post_init__(self) -> None:
        """Initialize the GPlibrary class."""
        self.train_x = pl.read_csv(source=Path("data").resolve() / "train_x.csv", has_header=False).to_numpy()
        self.train_y = pl.read_csv(source=Path("data").resolve() / "train_y.csv", has_header=False).to_numpy()
        self.test_x = pl.read_csv(source=Path("data").resolve() / "test_x.csv", has_header=False).to_numpy()
        self.test_y = pl.read_csv(source=Path("data").resolve() / "test_y.csv", has_header=False).to_numpy()
        results = pl.read_parquet(source=Path("results/storage").resolve() / f"lib_{self.library}.parquet")
        self.pred_y = results["pred_y"].list.explode().to_numpy().reshape(-1, 1)
        self.pred_var = results["pred_var"].list.explode().to_numpy().reshape(-1, 1)
        self.train_time = results["train_time"].item()
        self.pred_time = results["pred_time"].item()
        self.train_memory = results["train_memory"].item()
        self.pred_memory = results["pred_memory"].item()

    @property
    def is_valid(self) -> bool:
        """Whether the stored prediction can be scored (right size, finite values)."""
        return self.pred_y.shape == self.test_y.shape and bool(np.all(np.isfinite(self.pred_y)))

    @property
    def mae(self) -> np.ndarray:
        """Mean Absolute Error."""
        return mean_absolute_error(self.test_y, self.pred_y)  # type: ignore

    @property
    def mse(self) -> np.ndarray:
        """Mean Squared Error."""
        return mean_squared_error(self.test_y, self.pred_y)  # type: ignore

    @property
    def rmse(self) -> np.ndarray:
        """Root Mean Squared Error."""
        return np.sqrt(self.mse)

    @property
    def medae(self) -> np.ndarray:
        """Median Absolute Error."""
        return median_absolute_error(self.test_y, self.pred_y)  # type: ignore

    @property
    def r2(self) -> np.ndarray:
        """R² score."""
        return r2_score(self.test_y, self.pred_y)  # type: ignore

    @property
    def display_name(self) -> str:
        """Name of the library in plots and reports."""
        return DISPLAY_NAMES.get(self.library, self.library)

    @property
    def n_invalid_var(self) -> int:
        """Number of test points whose predictive variance is negative or not finite."""
        if self.pred_var.shape != self.test_y.shape:
            return int(self.test_y.size)
        return int(np.sum(~np.isfinite(self.pred_var) | (self.pred_var < 0)))

    def _nll(self) -> np.ndarray | None:
        """Negative log predictive density of each test point, with the variance as returned by the library.

        No floor is applied to the variance: a floor rewards overconfident models and depends on the scale
        of the output. With a negative or non-finite variance the density is undefined (None). A variance
        that is numerically zero is kept (1e-300 only avoids 0/0): the score is then very large, as it should.
        """
        if self.n_invalid_var:
            return None
        var = np.maximum(self.pred_var, 1e-300)
        return 0.5 * ((self.test_y - self.pred_y) ** 2 / var + np.log(2 * np.pi * var))

    @property
    def nlpd(self) -> float:
        """Negative Log Predictive Density, mean over the test points (NaN if a variance is invalid)."""
        nll = self._nll()
        return float("nan") if nll is None else float(np.mean(nll))

    @property
    def msll(self) -> float:
        """Mean Standardized Log Loss (Rasmussen & Williams, 2006, Sect. 2.5).

        The loss of the model minus the loss of the trivial Gaussian predictor whose mean and variance are
        those of the *training* outputs, averaged over the test points (NaN if a variance is invalid).
        """
        nll = self._nll()
        if nll is None:
            return float("nan")
        mu0, var0 = np.mean(self.train_y), np.var(self.train_y, ddof=1)
        baseline = 0.5 * ((self.test_y - mu0) ** 2 / var0 + np.log(2 * np.pi * var0))
        return float(np.mean(nll - baseline))

    @property
    def residuals(self) -> np.ndarray:
        """Residuals."""
        return self.test_y - self.pred_y

    @property
    def _metrics_row(self) -> list[str]:
        """Metrics row."""
        return [
            f"{self.display_name}",
            f"{self.mae:.4e}",
            f"{self.rmse:.4e}",
            f"{self.mse:.4e}",
            f"{self.medae:.4e}",
            f"{self.r2:.4f}",
            self.__probabilistic(self.nlpd),
            self.__probabilistic(self.msll),
            f"{self.train_time:.4f} s",
            f"{self.pred_time:.4f} s",
            f"{self.train_memory:.2f} MB",
            f"{self.pred_memory:.2f} MB",
        ]

    def __probabilistic(self, value: float) -> str:
        """NLPD or MSLL in the report: 'n/a' when undefined (the report adds a note with the reason)."""
        if np.isnan(value):
            return "n/a"
        return f"{value:.4f}" if abs(value) < 1e6 else f"{value:.2e}"

    @property
    def _metrics_header(self) -> list[str]:
        """Metrics header."""
        return [
            "Library",
            "MAE",
            "RMSE",
            "MSE",
            "MedAE",
            "R²",
            "NLPD",
            "MSLL",
            "t_train",
            "t_pred",
            "train_memory",
            "pred_memory",
        ]

    def print_metrics(self) -> None:
        """Print metrics."""
        tab = table(headers=self._metrics_header, title="Metrics")
        tab.add_row(*self._metrics_row)
        console.log(tab)

    def plot_results(self, path: Path, *, show: bool = False, formats: tuple[str, ...] = ("png",)) -> None:
        """Plot results, one file per format (e.g. ``("png", "pdf")``)."""
        from .. import plotter

        def save(folder: str, filename: str) -> None:
            for fmt in formats:
                plotter.save(plotter.plt, path=path / folder, filename=filename, format=fmt)

        plotter.crossvalidation.plot(self.pred_y, self.test_y, self.display_name)
        save("crossvalidation", f"lib_{self.library}")
        plotter.crossvalidation.plot_density(self.pred_y, self.test_y, self.display_name)
        save("density", f"lib_{self.library}")
        plotter.crossvalidation.plot_residuals(self.residuals, library=self.display_name)
        save("residuals", f"residuals_{self.library}")
        if self.test_x.shape[1] <= 2:  # prediction plots exist for one and two inputs only
            plotter.prediction.plot(gplib=self)
            save("prediction", f"lib_{self.library}")
        plotter.plt.show() if show else plotter.plt.close("all")
