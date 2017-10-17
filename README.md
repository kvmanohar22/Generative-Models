# Comparison of Generative Models in Tensorflow

The different generative models considered here are **VAEs** and **GANs**

This experiment is accompanied by blog post at : [https://kvmanohar22.github.io/Generative-Models](https://kvmanohar22.github.io/Generative-Models)

## Usage

- Download the MNIST and CIFAR datasets

#### Train VAE on mnist by running:
```bash

python --model vae --dataset mnist
```
#### Train GAN on mnist by running:

```bash

python --model gan --dataset mnist
```

For the complete list of command line options, run:

```bash
python main.py --help
```

The model generates images at a frequence specified by `generate_frq` which is by default 1.
