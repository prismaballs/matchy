# Matchy

Open software for loudspeaker driver pairing and QC.

![Prepare Tab](images/Prepare%20tab.png "Prepare Tab")

## Overview

Matchy is a small desktop tool to:
- Import REW-style frequency response text files.
- Preprocess and trim data by frequency range and optional downsampling.
- Detect outliers and curate sets of drivers.
- Propose optimal monitor pairings (partitions) based on RMS deviation.

The graphical interface is implemented in [`MatchyApp`](matchy_ui.py) and processing logic lives in [`MatchyLogic`](matchy_logic.py).

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
2. Run matchy.py
```sh
python matchy_[check version in file].py
```

## To Do

- Implement weighted blossom algorithm, rip out current algorithm
- Split code into multiple files based on function
- Change calculation for "deviation from curated mean" in Prepare tab from absolute to rms average
- Add "sort by pair average SPL" option in Results tab