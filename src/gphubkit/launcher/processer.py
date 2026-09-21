"""Post-process module for GPhub-kit."""

from typing import cast
from pathlib import Path
import os
import time

from rich.live import Live
from rich.spinner import Spinner
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, TimeElapsedColumn, TaskProgressColumn

from .. import plotter
from ..utils import Console, table, console, gphubkit_logo, get_main_script_path
from ..metrics import GPlibrary


def __get_script_files(scripts_dir: Path) -> list[str]:
    """Get all library files and organize by language."""
    return [f for f in os.listdir(scripts_dir) if (scripts_dir / f).is_file()]


def __get_libs(all_files: list[str]) -> list[str]:
    """Get all library files and organize by language."""
    return [f for f in all_files if f.endswith(".parquet")]


def __postprocess_library(
    scripts_dir: Path, library: str, *, display: bool = False, formats: tuple[str, ...] = ("png",)
) -> GPlibrary | None:
    """Post-process a library; a stored prediction with non-finite values is reported and skipped."""
    lib = GPlibrary(library=library)
    if not lib.is_valid:
        console.print(
            f"[light_coral]⚠️  Library [blue]{library}[light_coral]: invalid prediction (NaN or wrong size), skipped."
        )
        return None
    lib.print_metrics() if display else None
    lib.plot_results(path=scripts_dir.resolve(), formats=formats)
    return lib


def postprocess(formats: tuple[str, ...] = ("png",), libraries: tuple[str, ...] | None = None) -> None:
    """Post-process the stored results; ``formats`` lists the image formats to write (default: PNG).

    ``libraries`` restricts the plots and the report to the named libraries (script names without the ``lib_`` prefix,
    e.g. ``("DACE", "EGObox")``); by default every library with a stored prediction is post-processed.
    """
    caller_dir = get_main_script_path()
    all_scripts = __get_script_files(caller_dir / "results" / "storage")
    all_libs = sorted(__get_libs(all_scripts), key=str.lower)  # same order in every report and legend
    if libraries is not None:
        wanted = {name.removeprefix("lib_") for name in libraries}
        all_libs = [f for f in all_libs if f.removesuffix(".parquet").removeprefix("lib_") in wanted]

    tab, gp_libs = None, {}

    console.print("[bold yellow]🛠️  Post-processing results...")

    with Progress(
        SpinnerColumn(),
        TextColumn(" "),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        "[progress.description]{task.description}",
        console=console,
    ) as progress:
        task_id = progress.add_task("[bold green]🔄  Running libraries...", total=len(all_libs))
        for script_file in all_libs:
            extension = script_file.split(".")[-1]
            filename = script_file.removesuffix(f".{extension}").removeprefix("lib_")
            scripts_dir = caller_dir / "results" / "img"
            progress.update(task_id, description=f"[cyan] |  [light_coral]Library: [blue]{filename}", cas="cwqd")
            lib = __postprocess_library(scripts_dir, filename, display=False, formats=formats)
            progress.advance(task_id)
            if lib is None:
                continue
            gp_libs[filename] = lib
            tab = table(headers=lib._metrics_header, title="Report Metrics") if tab is None else tab
            tab.add_row(*lib._metrics_row)
        progress.update(task_id, description="[cyan] |  [bold green]✅  Done!\n", cas="cwqd")

    spinner = Spinner(
        "dots",
        text="[bold yellow]  Creating comparative plots and report...[/] [cyan] |  [bold cyan]⏳  Working...[/]",
        speed=2.5,
    )
    with Live(spinner, console=console, refresh_per_second=10):
        # COMPARISON PLOTS
        out_path = caller_dir / "results" / "img"
        plotter.radar._adimensional_metrics_by_library(gp_libs, path=out_path, formats=formats)
        plotter.radar._metrics(gp_libs, path=out_path, formats=formats)

        with Path(os.devnull).open("w") as devnull:
            writing_console = Console(record=True, width=175, log_time=False, log_path=False, file=devnull)
            writing_console.print(gphubkit_logo)
            writing_console.print(tab)
            for lib in gp_libs.values():
                if lib.n_invalid_var:
                    writing_console.print(
                        f"{lib.display_name}: {lib.n_invalid_var} of {lib.test_y.size} predictive variances are negative "
                        "or not finite; NLPD and MSLL are undefined (n/a)."
                    )
            writing_console.save_text((caller_dir / "results" / "report.log").resolve().__str__())
        time.sleep(0.25)
        spinner.update(
            text="[bold yellow]  Creating comparative plots and report...[/] [cyan] |  [bold green]✅  Done!\n"
        )
