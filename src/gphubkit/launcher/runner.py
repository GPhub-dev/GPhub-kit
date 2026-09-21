"""Runner module for GPhub-kit."""

from pathlib import Path
from collections.abc import Callable
import sys
import time
import logging

from rich.console import Console
from rich.progress import (
    TaskID,
    Progress,
    BarColumn,
    TextColumn,
    SpinnerColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)
import numpy as np
import polars as pl

from ..utils import get_logger, get_main_script_path
from .executor import r_executor, julia_executor, matlab_executor, python_executor
from ..benchmark import GPhubkitBenchmark


def __get_script_files(scripts_dir: Path) -> list[str]:
    """Get all library files and organize by language."""
    return [f.name for f in scripts_dir.iterdir() if f.is_file()]


def __get_libs(all_files: list[str]) -> tuple[list[str], ...]:
    """Get all library files and organize by language."""
    python_libs = [f for f in all_files if f.endswith(".py")]
    r_libs = [f for f in all_files if f.endswith(".r")]
    julia_libs = [f for f in all_files if f.endswith(".jl")]
    matlab_libs = [f for f in all_files if f.endswith(".m")]
    return python_libs, r_libs, julia_libs, matlab_libs


def __execute_script(
    script_file: str,
    executor_func: Callable,
    scripts_dir: Path,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    progress: Progress,
    task_id: TaskID,
    filename: str,
    logger: logging.Logger,
) -> None:
    """Execute a script using the specified executor function."""
    try:
        script_path = (scripts_dir / script_file).__str__()
        pred_y, pred_var, train_time, train_memory, pred_time, pred_memory = executor_func(
            script_path,
            train_x,
            train_y,
            test_x,
            logger,
        )

        results = pl.DataFrame(
            {
                "pred_y": [pred_y.flatten()],
                "pred_var": [pred_var.flatten()],
                "train_time": [train_time],
                "train_memory": [train_memory],
                "pred_time": [pred_time],
                "pred_memory": [pred_memory],
            }
        )

        directory = scripts_dir.parent / "results" / "storage"
        directory.mkdir(parents=True, exist_ok=True)

        extension = script_file.split(".")[-1]
        filename = script_file.removesuffix(f".{extension}")
        results.write_parquet(f"{directory}/{filename}.parquet")

    except Exception as e:
        progress.update(
            task_id,
            description=f"[light_coral]⚠️  Running library [blue]{filename} [red]failed with error: {e}",
        )
        time.sleep(1.5)


def run(benchmark: GPhubkitBenchmark) -> None:
    """Run all scripts in the caller's local script directory."""
    train_x = benchmark.train_x
    train_y = benchmark.train_y
    test_x = benchmark.test_x
    caller_dir = get_main_script_path()

    console = Console(file=sys.__stderr__, force_terminal=True, width=90, log_time=False, log_path=False)
    logger = get_logger()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    class LoggingFile:
        def __init__(self, level: int = logging.INFO) -> None:
            self.level = level

        def write(self, text: str) -> None:
            if text.strip():
                logging.log(self.level, text.strip())

        def flush(self) -> None:
            pass

        def isatty(self) -> bool:
            return False

    sys.stdout = LoggingFile(logging.INFO)
    sys.stderr = LoggingFile(logging.WARNING)

    scripts_dir = caller_dir / "scripts"
    all_files = __get_script_files(scripts_dir)
    python_libs, r_libs, julia_libs, matlab_libs = __get_libs(all_files)

    executors = [
        (python_libs, python_executor),
        (r_libs, r_executor),
        (julia_libs, julia_executor),
        (matlab_libs, matlab_executor),
    ]

    total_libs = len(python_libs) + len(r_libs) + len(julia_libs) + len(matlab_libs)

    console.print(f"[bold yellow]🛠️  Building GP models for benchmark: [blue]{benchmark.id}[bold yellow]...")

    with Progress(
        SpinnerColumn(),
        TextColumn(" "),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        "[progress.description]{task.description}",
        console=console,
    ) as progress:
        task_id = progress.add_task("[bold green]🔄  Running libraries...", total=total_libs)
        for libs, executor_func in executors:
            for script_file in libs:
                extension = script_file.split(".")[-1]
                filename = script_file.removesuffix(f".{extension}").removeprefix("lib_")
                progress.update(task_id, description=f"[cyan] |  [light_coral]Library: [blue]{filename}", cas="cwqd")
                __execute_script(
                    script_file,
                    executor_func,
                    scripts_dir,
                    train_x,
                    train_y,
                    test_x,
                    progress,
                    task_id,
                    filename,
                    logger,
                )
                progress.advance(task_id)
        progress.update(task_id, description="[cyan] |  [bold green]✅  Done!\n", cas="cwqd")

    sys.stdout = original_stdout
    sys.stderr = original_stderr
