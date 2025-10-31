# Matchy

Open software for loudspeaker driver pairing and QC.

![Prepare Tab](images/Prepare%20tab.png "Prepare Tab")

## Overview

Matchy is a desktop application that can:

- Import REW-style frequency response text files.
- Preprocess and trim data by frequency range and optional downsampling.
- Detect outliers and curate sets of drivers.
- Propose optimal monitor pairings (partitions) based on RMS deviation.

The graphical interface is implemented in the class [`MatchyApp`](matchy_ui.py) and processing logic lives in the class [`MatchyLogic`](matchy_logic.py).

## Features

- Import .txt REW exports
- Frequency range filtering and downsampling
- Outlier detection and visualization.
- Partition generation and pairwise deviation metrics.
- CSV export of partition results.

## How to run

Ensure you have Python version 3.10+. Use the command below:

```sh
python --version
```

1. Install packages using the requirements.txt

```sh
pip install -r requirements.txt
```

2. Run Matchy

```sh
python main.py
```

## TODO

- Add "sort by pair average SPL" option in Results 
- Add Percentile Option in Import tab to pre-select outlier tolerance
- Create Python executable script for distribution without needing to run Python (basically, make this application portable)
- Replace executable icon for style points
- Results tab to also show relative difference to mean FR like in Prepare tab
- Confidence intervals in Prepare tab
- Optional ideal FR curve that overrides average curated mean
- Prepare tab showing statistics of batch (max deviation and avg. standard deviation)

## Potential TODO (will think about it)

- More file format support (CSV, FRD, exports from other software)
- Export QC report to PDF
- Pair driver's FR and Impedance Curve for more consistent driver matching in an electrical domain. (Good luck measuring impedance curve) 
- Unit test and coverage test for robustness of software
- GitHub Actions to automate testing. Also for robustness of software
