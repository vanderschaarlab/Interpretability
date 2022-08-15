# Machine Learning Interpretability Methods

This repository is a collection of different Machine Learning interpretability methods. They 
all aim to provide an insight into why a machine learning model has made a given prediction.
This critical because for a models predictions to be trusted they must be understood.


# Table of Contents
0. [Background](#background)
1. [Explainers By Model Type](#explainers-by-model-type)
    - [Tabular Data Model Explainers](#tabular-data-model-explainers)
    - [Time Series model Explainers](#time-series-model-explainers)
    - [Clustering Model Explainers](#clustering-model-explainers)
2. [Explainers By Model Type2](#explainers-by-model-type2)
    - [Generalized Explainers](#generalized-explainers)
    - [Personalized Explainers](#personalized-explainers)

# Background

The Machine Learning (ML) community has produced incredible models for making highly
accurate predictions and classifications across many fields. However, uptake of these models into
settings outside of the ML community faces a key barrier: Interpretability. If a human cannot
understand why a decision has been made by a machine, they cannot be reasonably expected to act
on that decision with full confidence, particularly in a high-stakes environment. For example,
in a medical setting a doctor is expected to offer the best treatment or most accurate prognosis
for a patient and the decision may be a matter of life and death. Therefore making the decisions
of "Black-Box" models more transparent is of vital importance.

This GitHub Aims to act as a home for interpretability methods, where the state-of-the-art models
can be found for every application.

# Explainers By Model Type

Different model architectures can require different interpretability models, or "Explainers". The
following sections detail different methods broken down by the type of model that they explain.

## Tabular Data Model Explainers

The following Explainers work with models for making predictions from static tabular data.

| Explainer | Affiliation | GitHub | Paper |
| ----------- | ----------- | ----------- | ----------- |
| SimplEx | [van der Schaar Lab](https://www.vanderschaar-lab.com/)  | [SimplEx Source Code](https://github.com/JonathanCrabbe/Label-Free-XAI) | [SimplEx Paper](https://papers.nips.cc/paper/2021/hash/65658fde58ab3c2b6e5132a39fae7cb9-Abstract.html) |
| Symbolic Pursuit | [van der Schaar Lab](https://www.vanderschaar-lab.com/)  | [Symbolic Pursuit Source Code](https://github.com/JonathanCrabbe/Symbolic-Pursuit) | [Symbolic Pursuit Paper](https://arxiv.org/abs/2011.08596#:~:text=Learning%20outside%20the%20Black%2DBox%3A%20The%20pursuit%20of%20interpretable%20models,-Jonathan%20Crabb%C3%A9%2C%20Yao&text=Machine%20Learning%20has%20proved%20its,difficulties%20of%20interpreting%20these%20models.) |

## Time Series model Explainers

The following Explainers work with models for making predictions from time series data.

| Explainer | Affiliation | GitHub | Paper |
| ----------- | ----------- | ----------- | ----------- |
| Dynamask | [van der Schaar Lab](https://www.vanderschaar-lab.com/) | [Dynamask Source Code](https://github.com/JonathanCrabbe/Dynamask) | [Dynamask Paper](https://arxiv.org/abs/2106.05303) |

## Clustering Model Explainers

The following Explainers work with unsupervised clustering ML models, that is to say those without labelled data in the training set.

| Explainer | Affiliation | GitHub | Paper |
| ----------- | ----------- | ----------- | ----------- |
| Label-Free XAI | [van der Schaar Lab](https://www.vanderschaar-lab.com/)  | [Label-Free XAI Source Code](https://github.com/vanderschaarlab/Simplex) | [Label-Free XAI Paper](https://arxiv.org/abs/2203.01928) |


