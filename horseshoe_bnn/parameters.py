"""
File: parameters.py
Author: Anna-Lena Popkes, Hiske Overweg
Description: Contains the parameter classes, such as the hyperparameters
             used in Bayesian Linear Regression.
"""

import numpy as np
import ipdb
import torch

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from datetime import datetime

class EvaluationParameters:
    """
    Parameters used by the evaluator

    Args:
        n_splits: int, number of splits to perform k-fold cross validation
        scaler: scaler applied to normalized the features
        n_epochs: int, number of epochs for which the model is trained
        poly_features: bool, indicates whether polynomial features should be calculated
        poly_degree: int, degree of the polynomial features to be calculated
        interaction_only: type of polynomial features to be calculated (see sklearn.preprocessing.Polynomialfeatures)
        optimizer: optimizer used for neural nets
        learning_rate: float, learning rate of the optimizer
    """

    def __init__(
        self,
        n_splits=5,
        scaler=None,
        n_epochs=1,
        poly_features=False,
        poly_degree=1,
        interaction_only=True,
        optimizer=None,
        learning_rate=0.001,
        normalize=False,
    ):
        if not isinstance(poly_degree, int):
            raise TypeError("Poly_degree should be integer")
        if poly_degree < 1:
            raise ValueError("Poly_degree should be at least 1")
        if not isinstance(interaction_only, bool):
            raise TypeError("Interaction_only should be boolean")

        self.n_splits = n_splits
        self.scaler = scaler
        self.n_epochs = n_epochs
        self.poly_features = poly_features
        self.poly_degree = poly_degree
        # set boolean for sklearn.preprocessing.Polynomialfeatures
        self.interaction_only = interaction_only
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.normalize = normalize

@dataclass
class BNNRegressionHyperparameters:
    """
    Hyperparameter class for Bayes By Backprop Bayesian Neural Network for regression

    Args:
        batch_size: int, number of examples in a training batch
        n_samples: int, number of Monte Carlo samples during training
        n_samples_testing: int, number of Monte Carlo samples during testing
        n_hidden_units: int, number of hidden units in the hidden layer

        classification: bool, True when performing binary classification
        n_features: int, number of features in the dataset
        dataset_name: Name of dataset the model is trained on
        timestamp: current timestamp, used for naming the log directory

        Bayesian layer parameters:
        mixing coefficient: float, needed to initialize Mixture of Gaussian prior
        sigma1: Pytorch FloatTensor, needed to initialize Mixture of Gaussian prior
        sigma2: Pytorch FloatTensor, needed to initialize Mixture of Gaussian prior
        bayesian_weight_rho_scale: float, scales rho of the weights where rho determines
                                   the standard deviation of the weights: std = log(1 + exp(rho))
        bayesian_bias_rho_scale: float, scales rho of bias, where rho determines the standard
                                 deviation of the bias: std = log(a + exp(rho))
        bayesian_scale: float, scale to initialize the mean of the weights

    """
    batch_size: int
    mixing_coefficient: float
    sigma1: torch.FloatTensor
    sigma2: torch.FloatTensor
    bayesian_weight_rho_scale: float
    bayesian_bias_rho_scale: float
    bayesian_scale: float
    var_noise: float
    learning_rate: float

    n_samples: int
    n_samples_testing: int
    n_hidden_units: int

    classification: bool
    n_features: int
    dataset_name: str
    timestamp: datetime.timestamp

    def __post_init__(self):
        self.sigma1 = torch.Tensor([self.sigma1])
        self.sigma2 = torch.Tensor([self.sigma2])


@dataclass
class LinearBNNHyperparameters:
    """
    Hyperparameter class for Bayes By Backprop Bayesian Neural Network for regression

    Args:
        batch_size: int, number of examples in a training batch
        n_samples: int, number of Monte Carlo samples during training
        n_samples_testing: int, number of Monte Carlo samples during testing
        n_features: int, number of features in the dataset
        dataset_name: Name of dataset the model is trained on
        timestamp: current timestamp, used for naming the log directory

        Bayesian layer parameters:
        mixing coefficient: float, needed to initialize Mixture of Gaussian prior
        sigma1: Pytorch FloatTensor, needed to initialize Mixture of Gaussian prior
        sigma2: Pytorch FloatTensor, needed to initialize Mixture of Gaussian prior
        bayesian_weight_rho_scale: float, scales rho of the weights where rho determines
                                   the standard deviation of the weights: std = log(1 + exp(rho))
        bayesian_bias_rho_scale: float, scales rho of bias, where rho determines the standard
                                 deviation of the bias: std = log(a + exp(rho))
        bayesian_scale: float, scale to initialize the mean of the weights
    """

    batch_size: int
    mixing_coefficient: float
    sigma1: torch.FloatTensor
    sigma2: torch.FloatTensor
    bayesian_weight_rho_scale: float
    bayesian_bias_rho_scale: float
    bayesian_scale: float
    var_noise: float
    learning_rate: float

    n_samples: int
    n_samples_testing: int

    classification: bool
    n_features: int
    dataset_name: str
    timestamp: datetime.timestamp

    def __post_init__(self):
        self.sigma1 = torch.Tensor([self.sigma1])
        self.sigma2 = torch.Tensor([self.sigma2])


@dataclass
class HorseshoeHyperparameters:
    """
    Hyperparameter class for Bayes By Backprop Bayesian Neural Network for regression

    Args:
        batch_size: int, number of examples in a training batch
        n_samples: int, number of Monte Carlo samples during training
        n_samples_testing: int, number of Monte Carlo samples during testing
        n_hidden_units: int, number of hidden units in the hidden layer
        n_features: int, number of features in the dataset
        dataset_name: Name of dataset the model is trained on
        timestamp: current timestamp, used for naming the log directory
        var_noise: float, variance of noise over the outputs
        learning_rate: float, learning rate used to train the models

        Bayesian layer parameters:
        mixing coefficient: float, needed to initialize Mixture of Gaussian prior
        sigma1: Pytorch FloatTensor, needed to initialize Mixture of Gaussian prior
        sigma2: Pytorch FloatTensor, needed to initialize Mixture of Gaussian prior
        bayesian_weight_rho_scale: float, scales rho of the weights where rho determines
                                   the standard deviation of the weights: std = log(1 + exp(rho))
        bayesian_bias_rho_scale: float, scales rho of bias, where rho determines the standard
                                 deviation of the bias: std = log(a + exp(rho))
        bayesian_scale: float, scale to initialize the mean of the weights

        Horseshoe layer parameters:
        weight_cauchy_scale: float, scale parameter of the half-Cauchy distribution of the
                             shrinkage parameter of the weights
        global_cauchy_scale: float, scale parameter of the half-Cauchy distribution fo the
                             global shrinkage parameter
        horseshoe_scale: float, scale to initialize the mean of the variational distribution
                         over the weight parameters
        beta_rho_scale: float, scales rho of the variational distribution over the weight
                        parameters where std = log(1 + exp(rho))
        log_tau_mean: torch nn.Parameter, default None. If None this is initialized by
                      drawing samples from a HalfCauchy distribution
        log_tau_rho_scale: float, scales rho of the variational distribution over the log_tau
                           parameters where std = log(1 + exp(rho))
        bias_rho_scale: float, same purpose as other rho scale values
        log_v_mean: torch nn.Parameter, default None. If None this is initialized by
                    drawing samples from a HalfCauchy distribution
        log_v_rho_scale: float, same purpose as other rho scale values
    """

    batch_size: int
    n_samples: int
    n_samples_testing: int
    n_hidden_units: int
    n_features: int
    dataset_name: str
    timestamp: datetime.timestamp
    classification: bool
    var_noise: float
    learning_rate: float

    # Parameters of Bayesian layer
    mixing_coefficient: float
    sigma1: torch.FloatTensor
    sigma2: torch.FloatTensor
    bayesian_weight_rho_scale: float
    bayesian_bias_rho_scale: float
    bayesian_scale: float

    # Parameters of Horseshoe layer
    horseshoe_scale: float
    weight_cauchy_scale: float
    global_cauchy_scale: float
    beta_rho_scale: float
    log_tau_mean: float
    log_tau_rho_scale: float
    bias_rho_scale: float
    log_v_mean: float
    log_v_rho_scale: float

    def __post_init__(self):
        self.sigma1 = torch.Tensor([self.sigma1])
        self.sigma2 = torch.Tensor([self.sigma2])


@dataclass
class LinearHorseshoeHyperparameters:
    """
    Hyperparameter class for Neural Network with Horseshoe prior for regression

    Args:
        batch_size: int, number of examples in a training batch
        n_samples: int, number of Monte Carlo samples during training
        n_samples_testing: int, number of Monte Carlo samples during testing
        n_features: int, number of features in the dataset
        dataset_name: Name of dataset the model is trained on
        timestamp: current timestamp, used for naming the log directory

        Horseshoe layer parameters:
        weight_cauchy_scale: float, scale parameter of the half-Cauchy distribution of the
                             shrinkage parameter of the weights
        global_cauchy_scale: float, scale parameter of the half-Cauchy distribution fo the
                             global shrinkage parameter
        horseshoe_scale: float, scale to initialize the mean of the variational distribution
                         over the weight parameters
        beta_rho_scale: float, scales rho of the variational distribution over the weight
                        parameters where std = log(1 + exp(rho))
        log_tau_mean: torch nn.Parameter, default None. If None this is initialized by
                      drawing samples from a HalfCauchy distribution
        log_tau_rho_scale: float, scales rho of the variational distribution over the log_tau
                           parameters where std = log(1 + exp(rho))
        bias_rho_scale: float, same purpose as other rho scale values
        log_v_mean: torch nn.Parameter, default None. If None this is initialized by
                    drawing samples from a HalfCauchy distribution
        log_v_rho_scale: float, same purpose as other rho scale values
    """

    batch_size: int
    n_samples: float
    n_samples_testing: float
    n_features: int
    dataset_name: str
    classification: bool
    timestamp: datetime.timestamp
    var_noise: float
    learning_rate: float

    # Parameters of Horseshoe layer
    horseshoe_scale: float
    weight_cauchy_scale: float
    global_cauchy_scale: float
    beta_rho_scale: float
    log_tau_mean: float
    log_tau_rho_scale: float
    bias_rho_scale: float
    log_v_mean: float
    log_v_rho_scale: float

