# MTM: Multi-granularity Temporal Modeling for Partially Relevant Video Retrieval

This repository provides the official implementation of **MTM** for partially relevant video retrieval. MTM performs video retrieval by modeling multi-granularity temporal representations and evaluating query-video relevance on TVR, ActivityNet Captions, and Charades-STA.

## Catalogue

- [Getting Started](#getting-started)
- [Run](#run)
- [Results](#results)
- [Acknowledgement](#acknowledgement)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/tianxinlin666/MTM.git
cd MTM
```

### 2. Install Dependencies

We recommend using a conda environment.

```bash
pip install -r requirements.txt
```

A packaged environment can also be downloaded from:

```text
http://120.26.160.25/package
```

### 3. Prepare Datasets

This project supports three PRVR benchmarks:

- TVR
- ActivityNet Captions
- Charades-STA

The pre-extracted features follow the data format used in [MS-SL]. Please place the dataset files under the corresponding paths in `src/data/`, or update the paths in `src/Configs/*.py`.

Expected data structure:

```text
src/
└── data/
    ├── tvr/
    ├── activitynet/
    └── charades/
```

## Run

Train and evaluate MTM on TVR:

```bash
cd src
python main.py -d tvr
```

Train and evaluate MTM on ActivityNet Captions:

```bash
cd src
python main.py -d act
```

Train and evaluate MTM on Charades-STA:

```bash
cd src
python main.py -d cha
```

Training logs and checkpoints will be saved under the corresponding result directory, for example:

```text
src/results-tvr/
src/results-act/
src/results_cha/
```

## Results

The expected retrieval performance is shown below.

| Dataset | R@1 | R@5 | R@10 | R@100 | SumR |
|---|---:|---:|---:|---:|---:|
| TVR | 16.0 | 38.4 | 49.6 | 87.2 | 191.2 |
| ActivityNet Captions | 9.1 | 27.6 | 41.4 | 79.5 | 157.6 |
| Charades-STA | 2.9 | 9.4 | 15.1 | 54.1 | 81.5 |

## Acknowledgement

We thank the authors of [MS-SL] for providing the processed features and benchmark setting.

[MS-SL]: https://github.com/HuiGuanLab/ms-sl
