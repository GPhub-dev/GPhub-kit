"""Command Line Interface for GPhub-kit."""

from pathlib import Path
import sys
import time
import shutil

from rich.live import Live
from rich.prompt import Prompt
from rich.spinner import Spinner
import click

from . import __version__, __app_name__
from .utils import console


def __wd_structure(working_dir: Path) -> None:
    """Create the project structure."""
    working_dir.mkdir(parents=True, exist_ok=True)
    (working_dir / "data").mkdir(parents=True, exist_ok=True)
    (working_dir / "results").mkdir(parents=True, exist_ok=True)
    (working_dir / "results" / "img").mkdir(parents=True, exist_ok=True)
    (working_dir / "results" / "storage").mkdir(parents=True, exist_ok=True)
    (working_dir / "scripts").mkdir(parents=True, exist_ok=True)


def __check_wd_structure() -> None:
    if (
        not Path.cwd().joinpath("data").exists()
        or not Path.cwd().joinpath("results").exists()
        or not Path.cwd().joinpath("scripts").exists()
    ):
        working_dir = Path.cwd()
        console.print(
            f"""[bold red][ERROR][/] [yellow]Current working directory [bold magenta]'{working_dir!s}'[/] is not a valid GPhub-kit project.
        Please navigate to a valid project directory or create a new one using the [bold cyan]'create'[/] command.
            """
        )
        answer = Prompt.ask(
            f"Do you want to initialize a new project in [bold magenta]'{working_dir!s}'[/]? [[green]y[/]/[red]n[/]]"
        )
        if answer.lower() != "y":
            console.print("Operation cancelled by the user.")
            sys.exit(1)
        action_text = "Initializing a new project in"
        spinner = Spinner(
            "dots",
            text=f"{action_text} [bold yellow]'{working_dir!s}'[/] -- ⏳ [cyan]Working...[/]",
            speed=2.5,
        )
        with Live(spinner, console=console, refresh_per_second=10):
            __wd_structure(working_dir)
            time.sleep(0.25)
            spinner.update(text=f"{action_text} [bold yellow]'{working_dir!s}'[/] -- ✅ [bold green]Done![/]")


def __create_library_template(name: str, language: str) -> str:
    """Create a new library template."""
    from .utils.templates import tmpl_m, tmpl_r, tmpl_jl, tmpl_py

    match language.lower():
        case "python" | "py":
            template, ext = tmpl_py, "py"
        case "matlab" | "m":
            template, ext = tmpl_m, "m"
        case "julia" | "jl":
            template, ext = tmpl_jl, "jl"
        case "r":
            template, ext = tmpl_r, "r"
        case _:
            return f"[bold red][ERROR][/] [yellow]Language [bold magenta]'{language}'[/] is not supported yet. Operation cancelled."

    file_path = Path.cwd() / "scripts" / f"lib_{name}.{ext}"
    if file_path.exists():
        return f"⚠️ The template [bold yellow]'{file_path!s}'[/] already exists."
    with open(file_path, "w") as f:
        f.write(template(name))
    return f"✅ [green]Library template [bold yellow]'{file_path!s}'[/] created successfully!\n"


def __check_libraries() -> None:
    """Check if libraries exist in the scripts folder."""
    if not any(Path.cwd().joinpath("scripts").glob("lib_*")):
        console.print(
            f"""\n[bold red][ERROR][/] [yellow]No libraries found in [bold magenta]'{Path.cwd() / "scripts"!s}'[/].
        Please add at least one library before running the benchmark using the [bold cyan]'add'[/] command.
            """
        )
        sys.exit(1)


def __run_benchmark(bm) -> None:
    """Run and postprocess a benchmark."""
    bm.run()
    bm.postprocess()


@click.group()
@click.version_option(version=__version__, prog_name=__app_name__)
def app() -> None:
    """GPhub-kit: A toolkit for benchmarking Gaussian Process Regression libraries."""


@app.command()
@click.option("--project", "-p", type=str, required=True, help="Name of the project")
def create(project: str) -> None:
    """Create a new GPhub-kit project."""
    working_dir = Path.cwd() / project
    if working_dir.exists():
        answer = Prompt.ask(
            f"The directory [bold yellow]'{working_dir!s}'[/] already exists. Do you want to reinitialize it? [[green]y[/]/[red]n[/]]"
        )
        if answer.lower() != "y":
            console.print("Operation cancelled by the user.")
            sys.exit(0)
        shutil.rmtree(str(working_dir), ignore_errors=True)
        action_text = "Overwriting the existing directory"
    else:
        action_text = "Initializing a new project in"
    spinner = Spinner(
        "dots",
        text=f"{action_text} [bold yellow]'{working_dir!s}'[/] -- ⏳ [cyan]Working...[/]",
        speed=2.5,
    )
    with Live(spinner, console=console, refresh_per_second=10):
        __wd_structure(working_dir)
        time.sleep(0.25)
        spinner.update(text=f"{action_text} [bold yellow]'{working_dir!s}'[/] -- ✅ [bold green]Done![/]")
    sys.exit(0)


@app.command()
@click.option("--name", "-n", type=str, required=True, help="Library's name.")
@click.option(
    "--language",
    "-l",
    type=str,
    required=True,
    help="Library's programming language. Supported: Python, Matlab, Julia, R.",
)
def add(name: str, language: str) -> None:
    """Add a new library."""
    __check_wd_structure()
    spinner = Spinner(
        "dots",
        text=f"Adding [magenta]{language} template[/] in [yellow]{Path.cwd() / 'script'!s}[/] for library [blue]{name}[/] -- ⏳ [cyan]Working...[/]",
        speed=2.5,
    )
    with Live(spinner, console=console, refresh_per_second=10):
        msg = __create_library_template(name, language)
        time.sleep(0.75)
        spinner.update(text=msg)
    sys.exit(0)


@app.group()
def run() -> None:
    """Run GPhub-kit benchmarks."""


@app.command()
@click.option(
    "--format",
    "-f",
    "formats",
    multiple=True,
    default=("png",),
    show_default=True,
    help="Image format of the plots; repeat the option for several formats (e.g. -f png -f pdf).",
)
@click.option(
    "--library",
    "-l",
    "libraries",
    multiple=True,
    help="Post-process this library only; repeat the option for several libraries (default: all).",
)
def postprocess(formats: tuple[str, ...], libraries: tuple[str, ...]) -> None:
    """Postprocess GPhub-kit benchmark results."""
    import gphubkit as gpk

    gpk.postprocess(formats=tuple(formats), libraries=tuple(libraries) or None)


@run.command()
@click.option("--id", "-i", type=int, required=True, help="Benchmark ID (1-7)")
def bm(id: int) -> None:
    """Run a standard benchmark (BM01-BM07)."""
    if id not in range(1, 8):
        console.print(
            f"[bold red][ERROR][/] [yellow]Benchmark ID [bold magenta]'{id}'[/] is not supported. Please provide a valid ID (1-7)."
        )
        sys.exit(1)

    __check_libraries()

    import gphubkit as gpk

    bm_class = getattr(gpk.benchmark, f"BM{id:02d}")
    benchmark = bm_class()

    console.print(f"✅  [green]Benchmark [bold blue]'BM{id:02d}'[/] loaded successfully!\n")
    __run_benchmark(benchmark)


@run.command()
@click.option(
    "--dim", "-d", type=click.Choice([2, 4, 8, 16, 32, 48, 64]), required=True, help="Composite shell dimension."
)
@click.option(
    "--size",
    "-s",
    type=click.Choice([100, 200, 500, 700, 1000, 2000, 3000]),
    required=True,
    help="Composite shell train size.",
)
def composite(dim: int, size: int) -> None:
    """Run the Composite Shell benchmark."""
    __check_libraries()

    import gphubkit as gpk

    benchmark = gpk.benchmark.CompositeShell(dim=dim, train_size=size)
    console.print("✅  [green]Benchmark [bold blue]'CompositeShell'[/] loaded successfully!\n")
    __run_benchmark(benchmark)


@run.command()
@click.option(
    "--testsize",
    "-t",
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
    help="Test set size as a fraction of total data (0 < testsize < 1).",
)
def custom(testsize: float) -> None:
    """Run a custom benchmark using data from the 'data' directory."""
    __check_wd_structure()
    __check_libraries()

    # Check if data exists
    if not any(Path.cwd().joinpath("data").glob("*.csv")):
        console.print(
            f"""[bold red][ERROR][/] [yellow]No data found in [bold magenta]'{Path.cwd() / "data"!s}'[/].
        Please add your [cyan]'*.csv'[/] data before running a custom benchmark."""
        )
        sys.exit(1)

    import gphubkit as gpk

    try:
        train_x, test_x, train_y, test_y = gpk.data.load(file_path=Path.cwd().resolve() / "data")
    except FileNotFoundError:
        if testsize is None:
            testsize = float(
                Prompt.ask(
                    "Custom benchmark requires the test size to be specified. Please provide a valid test size (e.g., 0.2 for 20i%)"
                )
            )
            while testsize <= 0 or testsize >= 1:
                console.print(
                    f"[bold red][ERROR][/] [yellow]Test size [bold magenta]'{testsize}'[/] is not valid. Please provide a value between 0 and 1."
                )
                testsize = float(Prompt.ask("Please provide a valid test size between 0 and 1", default="0.2"))

        console.print(f"📊  [cyan]Splitting dataset with test_size={testsize}[/]")
        train_x, test_x, train_y, test_y = gpk.data.split_dataset(
            file_path=Path.cwd().resolve() / "data",
            test_size=testsize,
        )

    benchmark = gpk.benchmark.Custom(
        id="custom",
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        scale_x=True,
        standardize_y=True,
    )

    console.print("✅  [green]Benchmark [bold blue]'Custom'[/] loaded successfully!\n")
    __run_benchmark(benchmark)


if __name__ == "__main__":
    app()
