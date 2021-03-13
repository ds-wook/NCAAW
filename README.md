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
|LGBM-nomalization optuna(5 params)|0.38188(5-fold 2step)|0.37090|-|
|LGBM-nomalization optuna(5 params)|0.39794(5-fold)|0.36249|-|
|LGBM-nomalization optuna(5 params)|0.39988(5-fold)|0.38189|-|
|LGBM-nomalization optuna(7 params)|0.40810(5-fold)|0.40437|-|
|LGBM-leak-nomalization optuna(5 params)|0.40921(5-fold)|0.43557|-|
|XGB-nomalization optuna(6 params)|0.39461(5-fold 2step)|0.40538|-|
|XGB-nomalization optuna(8 params)|0.41756(5-fold)|0.41170|-|
|XGB-leak-nomalization optuna(8 params)|0.41973(5-fold)|0.44573|-|
