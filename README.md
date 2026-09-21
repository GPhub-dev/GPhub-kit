<p align="center">
    <img title="GPhub-kit-Logo" alt="GPhub-kit" src="./img/logo/GPhub-kit-Logo.png" width="450">
</p>

<div align="center">


  <a href="https://github.com/alefaraci/GPhub-kit/releases/tag/devs">![version](https://img.shields.io/badge/version-v0.1.0-brightgreen)</a>
  <a href="https://github.com/alefaraci/Kriging-Table-HTML/blob/main/LICENSE">![license](https://img.shields.io/badge/license-MIT-brightgreen.svg)</a>

</div>

#

**GPhub-kit** is a Python toolkit for benchmarking Gaussian Process (GP) libraries across multiple programming languages, enabling reproducible comparisons of implementations.

It provides a unified platform for:

- **Multi-Language Execution**: Run GP libraries in `Python`, `R`, `Julia`, and `MATLAB` from a unified interface
- **Benchmarking**: Selected standard engineering benchmarks
- **Performance Metrics**: Probabilistic metrics and timing/memory profiling
- **Data Management**: Synthetic data generation, dataset splitting


#  Quick Start

## Command Line Interface

**GPhub-kit** provides a CLI for all benchmarking operations.

![Application GIF](./img/cli.gif "Application Demo")

### Project Management

```bash
# Create new project with directory structure
❯ gphubkit create --project PROJECT_NAME

# Add library implementations
❯ gphubkit add --name LIBRARY_NAME --language LANGUAGE
# Supported languages: Python, R, Julia, MATLAB
```

### Running Benchmarks

```bash
# Standard benchmarks (BM01-BM07)
❯ gphubkit run bm --id {1-7}

# Composite Shell benchmark
❯ gphubkit run composite --dim {2,4,8,16,32,48,64} --size {100,200,500,700,1000,2000,3000}

# Custom benchmark with your data
❯ gphubkit run custom [--testsize FRACTION]
```

### Post-processing

```bash
# Generate report and visualizations
❯ gphubkit postprocess
# ... also as PDF
❯ gphubkit postprocess -f png -f pdf
```

## Python API

```python
import gphubkit as gpk
from pathlib import Path

# 1. Load a benchmark - e.g. 2D Branin function
benchmark = gpk.benchmark.BM02()

# 2. Run benchmarking (requires library scripts in ./scripts/)
benchmark.run()

# 3. Postprocess results and generate reports
benchmark.postprocess()
```

## **Setup Project Structure**

```
project/
├── scripts/                     # Library implementations
│   ├── lib_GPyTorch.py          # Python libraries
│   ├── lib_DiceKriging.r        # R libraries
│   ├── lib_GaussianProcesses.jl # Julia libraries
│   ├── lib_UQLab.m              # MATLAB libraries
│   └── ...                      # Additional libraries
├── data/                        # Benchmark data
│   ├── dataset_x.csv            # Dataset x csv file
│   ├── dataset_y.csv            # Dataset y csv file
│   ├── train_x.csv              # Train Set x csv file
│   ├── train_y.csv              # Train Set y csv file
│   ├── test_x.csv               # Test Set x csv file
│   └── test_y.csv               # Test Set y csv file
└── results/                     # Rich console output
    ├── storage/                 # Parquet result files
    │   └── ...
    ├── img/                     # Plots and visualizations
    │   └── ...
    ├── logger.log               # Execution log
    └── report.log               # Analysis report

```

---

# Building and Running

## Installation

The easiest way to install **GPhub-kit** is using `uv`.  The main dependencies are listed in `pyproject.toml` and include libraries for GP modeling, data handling, and plotting.

#### 1. Create a virtual environment

```bash
uv venv --python X.XX.X
```

#### 2. Activate the virtual environment (macOS/Linux)

```bash
source .venv/bin/activate
```

#### 3. Install the dependencies

```bash
uv add gphubkit
```

#### 4. Install additional dependencies

```bash
# For GPy and SMT support
uv add "gphubkit[gpy,smt]"
# or for all optional dependencies
uv add "gphubkit[all]"
```
## Requirements

`MATLAB`, `Julia`, and `R` must be installed on your system to run the libraries written in these languages. The `gphubkit` package interfaces with them through `matlabengine`, `rpy2`, and `juliacall` respectively. Each bridge is imported on demand: when a runtime is missing, only the scripts written in that language fail. `matlabengine` can only be installed next to a MATLAB installation and is therefore an optional dependency:

```bash
uv add "gphubkit[matlab]"
```

A script that raises an error during `init`, `train`, or `test` yields no result file (the error is written to the log). Please refer to each library's documentation for installation instructions.

---

#  Documentation

Documentation for all modules is provided in the **[📚 Wiki](https://github.com/alefaraci/GPhub-kit/wiki)** of the repo:

- **[Benchmark Module](https://github.com/alefaraci/GPhub-kit/wiki/Benchmark-module)**: Standard and custom benchmarks
- **[Data Module](https://github.com/alefaraci/GPhub-kit//wiki/Data-module)**: Dataset management, synthetic generation, file I/O
- **[Metrics Module](https://github.com/alefaraci/GPhub-kit/wiki/Metrics-module)**: Performance metrics and reporting
- **[Launcher Module](https://github.com/alefaraci/GPhub-kit/wiki/Launcher-module)**: Cross-language execution engines and pipeline
- **[Plotter Module](https://github.com/alefaraci/GPhub-kit/wiki/Plotter-module)**: Visualization suite
- **[Utils Module](https://github.com/alefaraci/GPhub-kit/wiki/Utils-module)**: Internal utilities and helpers



---

# Development

To clone and set up the development environment:

```bash
git clone https://github.com/alefaraci/GPhub-kit.git
cd GPhub-kit
uv venv --python X.XX.X
source .venv/bin/activate  # macOS/Linux
uv add --dev "gphubkit[all]"

# Test CLI installation
gphubkit --help
```

##  Contributing

**GPhub-kit** is designed to be extensible and welcomes contributions:

- **New Benchmarks**: Add domain-specific benchmark problems
- **Language Support**: Extend support to additional programming languages
- **Metrics**: Implement new performance and statistical metrics
- **Visualization**: Create specialized plotting functions for specific analyses
- **Documentation**: Improve examples and use-case documentation

---

# Acknowledgments

This project has received funding from the [European Union’s Horizon 2020](https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-2020_en) research and innovation programme under the [Marie Sklodowska-Curie](https://marie-sklodowska-curie-actions.ec.europa.eu) grant agreement No. 955393.

# License

Licensed under the [MIT license](https://github.com/alefaraci/GPhub-kitPro/blob/main/LICENSE).
