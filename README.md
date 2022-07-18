# Wasserstein GAN (PyTorch Implementation)

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pillow?logo=lol) ![GitHub](https://img.shields.io/github/license/b53k/Wasserstein-GAN--PyTorch)
![GitHub language count](https://img.shields.io/github/languages/count/b53k/Wasserstein-GAN--PyTorch) <br>
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)

An alternative to traditional GAN training, Wasserstein GAN (WGAN) has been showed to improve the stability of learning and get rid of issues like mode collapse. This projects aims to implement WGAN from scratch using PyTorch Framework.

## Prerequisites

The implementation is based on Python3 and PyTorch, check their website [here](https://pytorch.org) for installation instructions. The rest of the requirements is included in the [requirements file](requirements.txt), to install them:
```bash
pip3 install -r requirements.txt
```

## Dataset

[Large-scale CelebFaces Attributes (CelebA) Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), was used to train WGAN. Before training, extracted images need to be in `'./data/img_align_celeba/`

## Training

Training script takes in the following arguments:
* --batch-size    
* --n-epochs 
* --z-dim  
* --lr
* --noreload (optional see help for details)
* --logdir

```bash
python3 train_WGAN.py --help
```
WGAN can be trained using `train_WGAN.py` script as follows (for example):
```bash
python3 train_WGAN.py --batch-size = 128 --n-epochs 10000 --z-dim 200 --lr 1e-4 --noreload --logdir logs
```
Latest models are saved in `./checkpoint/`. 

Training loss and sample images are logged in `./logs/loss_charts/` and `./logs/sample_imgs/` respectively.

## Result
<p align='center'>
<img src="/gifs/cropped_imgs.gif">
<img src="/gifs/chart.gif" width="550" height="350">
</p>

## Author
Bipin Koirala - [b53k](https://github.com/b53k)

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/b53k/Wasserstein-GAN--PyTorch/blob/main/LICENSE) file for details.
