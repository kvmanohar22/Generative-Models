# Comparison of Generative Models in Tensorflow

The different generative models considered here are **Variational Autoencoders (VAEs)** and **Generative Adversarial Networks (GANs)**.

This experiment is accompanied by blog post at : [https://kvmanohar22.github.io/Generative-Models](https://kvmanohar22.github.io/Generative-Models)

## Usage

- Download the MNIST and CIFAR datasets

#### Train VAE on mnist by running:
```bash

python main.py --train --model vae --dataset mnist
```
#### Train GAN on mnist by running:

```bash

python main.py --train --model gan --dataset mnist
```

For the complete list of command line options, run:

```bash
python main.py --help
```

The model generates images at a frequence specified by `generate_frq` which is by default 1.

## Results of training GAN on mnist

Sample images from MNIST data is :

<img src="images/target_mnist.jpg" width="55%" align="center">

GIF showing images generated from GAN as a function of epochs is as follows:

![](http://kvfrans.com/content/images/2017/03/Screen-Shot-2017-03-01-at-11-09-09-PM-1.png)

![](http://kvfrans.com/content/images/2017/03/Screen-Shot-2017-03-01-at-11-09-13-PM.png)

<img align="center" src="images/gan.gif">

For examples and explanation, have a look at the [blog post](https://kvmanohar22.github.io/Generative-Models).
