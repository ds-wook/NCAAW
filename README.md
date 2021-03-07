# March Machine Learning Mania 2021 - NCAAW
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
Predict the 2021 NCAAW Basketball Tournament

This is a collection of my code from the [March Machine Learning Mania 2021 - NCAAW](https://www.kaggle.com/c/ncaaw-march-mania-2021) Kaggle competition.

## Code Style
I follow [black](https://pypi.org/project/black/) for code style.

Black is a PEP 8 compliant opinionated formatter.

## Benchmark

#### FE Hyper Parameter Tunning
|method|OOF|Public LB|Private LB|
|------|:---------:|:--------:|:--------:|
|LGBM optuna(4 params)|0.5134(5-fold)|0.41899|-|
|XGB-nomalization optuna(8 params)|0.41756(5-fold)|0.41170|-|
