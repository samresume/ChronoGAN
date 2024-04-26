
# DualGAN: Dual Adversarial Time Series Generation with Generative Adversarial Networks and Autoencoders

## Introduction
DualGAN is a sophisticated framework that integrates the advantages of Autoencoders and Generative Adversarial Networks (GANs) to address challenges in time series generation, such as slow convergence and information loss.

## Abstract
Existing GAN-based methods for time series generation face challenges such as slow convergence and information loss in embedding spaces. DualGAN introduces a framework that combines the strengths of an Autoencoder-generated embedding space with adversarial training dynamics of GANs. This framework includes specialized loss functions and supervision from a supervisor network, capturing stepwise conditional distributions of the data, operating within the latent space, and receiving crucial feedback from the discriminator based on feature space. The inclusion of an early stopping algorithm and a complex neural network architecture improves stability and effectiveness across varied time series lengths. Currently, the paper describing this framework is under submission.

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/samresume/DualGAN.git
cd DualGAN
pip install -r requirements.txt
```

## Usage
To get started, run the tutorial notebook:
```bash
jupyter notebook tutorial.ipynb
```

## Files
- `dualgan.py`: Main implementation of the DualGAN model.
- `data_loading.py`: Functions for loading and preprocessing data.
- `utils.py`: Helper utilities for the model.


## Contact
For questions or collaborations, contact MohammadReza EskandariNasab as mreskandarinasab@gamil.com.

## Citation
Please note that the paper detailing the DualGAN framework is currently under submission. If you use this framework or any part of it in your work, please cite us appropriately once the paper is published. Check back here for the citation details upon publication.
