<div align="center">
  <img src="images/MatchyHD.png" alt="Matchy Logo" width="250" />
  <h3>Open software for loudspeaker driver pairing and QC.</h3>
</div>

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

- Remove logic of and references to the brute force and balanced algorithms (as they are currently unused)
- Create Python executable script for distribution without needing to run Python (basically, make this application portable)

#### IMPORT

- Move mean or median target selection to the Prepare tab

#### PREPARE

- Optional ideal FR curve that overrides mean/median target
- Show statistics of batch sans outliers (max deviation and avg. standard deviation)
- Add computational time multiplier warning when number of inputs to be matched via blossom is odd

#### RESULTS

- Add "sort by pair average SPL" option
- Make vertical scaling of plots fixed, matching that of the non normalized Prepare tab
- Overlay plot of absolute dB deviation between pairs
- Remove "partition average RMS" and "leftover" columns from table, moving the values to be displayed elsewhere.

## Potential TODO (will think about it)

- Remove average deviation option (alongside related backpropagation code), leaving only RMS deviation. RMS makes the most sense for this application.
- Make blossom the only algorithm? (as it provides the best overall combination of speed and accuracy)
- More file format support (CSV, FRD, exports from other software)
- Results tab to also show relative difference to mean FR like in Prepare tab
- Confidence intervals in Prepare tab
- Export QC report to PDF
- Pair driver's FR and Impedance Curve for more consistent driver matching in an electrical domain.
- Unit test and coverage test for robustness of software
- GitHub Actions to automate testing. Also for robustness of software
