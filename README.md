# Multi-granularity Temporal Modeling for Partially Relevant Video Retrieval


## Catalogue 
* [1. Getting Started](#getting-started)
* [2. Run](#run)
* [3. Results](#results)


## Getting Started

1. Clone this repository:

```
git clone https://github.com/tianxinlin666/MTM.git
cd MTM
```

2. Create a conda environment and install the dependencies:
> You can download directly through the link We provided. 
>
> > http://120.26.160.25/package


```shell
pip install -r requirements.txt
```

3. Download Datasets: All features of TVR, ActivityNet Captions and Charades-STA are kindly provided by the authors of [MS-SL].

4. Set dataset location

## Run

To train MTM on TVR:
```shell
cd src
python main.py -d tvr
```

To train MTM on ActivityNet Captions:
```shell
cd src
python main.py -d act
```

To train MTM on Charades-STA:
```shell
cd src
python main.py -d cha
```

## Results

### Quantitative Results

For this repository, the expected performance is:

| *Dataset* | *R@1* | *R@5* | *R@10* | *R@100* | *SumR* |
| ---- | ---- | ---- | ---- | ---- | ---- |
| TVR | 16.0 | 38.4 | 49.6 | 87.2 | 191.2 |
| ActivityNet Captions | 9.1 | 27.6 | 41.4 | 79.5 | 157.6 |
| Charades-STA | 2.9 | 9.4 | 15.1 | 54.1 | 81.5 |

[MS-SL]:https://github.com/HuiGuanLab/ms-sl

