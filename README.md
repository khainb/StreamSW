# StreamSW
Official PyTorch implementation for paper: Streaming Sliced Optimal Transport

Details of the model architecture and experimental results can be found in our papers.

```
@article{nguyen2025stream,
  title={Streaming Sliced Optimal Transport},
  author={Khai Nguyen and Nhat Ho},
  year={2023},
  pdf={https://arxiv.org/pdf/2304.13586.pdf}
}
```
Please CITE our paper whenever this repository is used to help produce published results or incorporated into other software.

This implementation is made by [Khai Nguyen](https://khainb.github.io).

## Requirements
To install the required python packages, run
```
pip install -r requirements.txt
```

## What is included?
* Gaussian Comparison
* Mixture of Gaussians Comparison
* Gradient Flow
* Abnormality Detection


## Gaussian (Mixture of Gaussians) Comparison
```
cd Gaussian (MixtureGaussian)
python vary_k.py
python vary_n.py
python plot_figure.py
```

##  Gradient Flow

```
cd GradientFlow
python main.py

```

## Abnormality Detection
```
cd AbnormalityDetection
python main.py

```

## Acknowledgment
The implementation of the KKL sketch is taken from [streaming-quantiles](https://github.com/edoliberty/streaming-quantiles).