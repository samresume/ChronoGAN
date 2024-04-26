
# DualGAN

## Introduction
DualGAN is a sophisticated framework that integrates the advantages of Autoencoders and Generative Adversarial Networks (GANs) to address challenges in time series generation, such as slow convergence and information loss. By leveraging a specialized loss function and supervision from a supervisor network, DualGAN excels in generating high-fidelity time series data, consistently outperforming existing benchmarks.

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

## Contributing
Contributions are welcome. Please fork the repository and submit pull requests, or open an issue to discuss potential changes.

## License
This project is licensed under [License Name]. See the LICENSE file for more details.

## Contact
For questions or collaborations, contact mreskandarinasab@gamil.com.

## Acknowledgments
Credit to all contributors and researchers who have provided insights and feedback on this project.
