# Score based models

<img src="imgs/ddpm-sampling.gif" alt="diffusion gif" width="768" />

This repo contains some naive implementations of various score-based generative models.

## Installation

To install this library, simply run the following command after cloning the repo :
```shell
pip install -e .
```


## Methods

The supported methods are:
* Noise Conditional Score Networks (NCSN) ([Song and Ermon, 2019](https://arxiv.org/abs/1907.05600))
* Denoising Diffusion Probabilistic Models (DDPM) ([Ho et al., 2020](https://arxiv.org/abs/2006.11239))
* Conditional DDPM - Cross-Attention (CDDPM) ([Rombach et al., 2021](https://arxiv.org/abs/2112.10752))


## Run

### NCSN
* 2D gaussian example: `notebooks/ncsn_2d_example.ipynb`

* CIFAR-10: 

```shell
cd scripts
python ncsn_train_cifar10.py --dataset cifar10
```

You can visualise the CIFAR-10 samples in `notebooks/ncsn_view.ipynb`.


### DDPM
* 2D gaussian example: `notebooks/ddpm_2d_example.ipynb`

* CIFAR-10, [Butterfiles](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset): 

```shell
cd scripts
python ddpm_train.py --dataset [dataset-name]
```
where `[datase-name]` can be `cifar10 | butterflies`.

You can visualise the CIFAR-10 and Butterflies samples in `notebooks/ddpp_view.ipynb`.


### CDDPM

* [Nouns](https://huggingface.co/datasets/m1guelpf/nouns): 

```shell
cd scripts
python ddpm_cond_train.py --dataset m1guelpf/nouns
```
You can visualise the Nouns samples in `notebooks/ddpp_cond_view.ipynb`.


## Results

### NCSN

* 2D Gaussian mixture

<img src="imgs/ncsn-2D-gaussian.png" alt="Gradient field" style="width: 90%;">

* CIFAR-10 (32 x 32)

<img src="imgs/ncsn-cifar-10.png" alt="CIFAR-10" style="width: 50%;">


### DDPM

* CIFAR-10 (32 x 32)

<img src="imgs/ddpm-cifar-10.png" alt="CIFAR-10" style="width: 50%;">

* Butterflies (128 x 128)

<img src="imgs/ddpm-butterflies.png" alt="Butterflies" style="width: 50%;">


### CDDPM

* Nouns (32 x 32)

<img src="imgs/cddpm-nouns.png" alt="Nouns" style="width: 50%;">
