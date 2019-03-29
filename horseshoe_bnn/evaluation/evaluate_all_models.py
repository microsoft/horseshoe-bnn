"""
File: evaluate_all_models.py
Author: Anna-Lena Popkes, Hiske Overweg
Description: Trains models LinearGaussian, LinearHorseshoe, GaussianBNN and HorseshoeBNN on a dataset and prints the results
"""

import torch
import pickle
import os
import numpy as np
import torch.optim as optim
import math
import ipdb
import pandas as pd
import datetime
import yaml

from horseshoe_bnn.parameters import BNNRegressionHyperparameters, LinearBNNHyperparameters, LinearHorseshoeHyperparameters, HorseshoeHyperparameters, EvaluationParameters
from horseshoe_bnn.models import LinearGaussian, GaussianBNN, LinearHorseshoe, HorseshoeBNN
from horseshoe_bnn.metrics import AllMetrics
from horseshoe_bnn.data_handling.dataset import Dataset
from horseshoe_bnn.evaluation.evaluator import evaluate

from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

root = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

"""
Set up the datasets.
When training on a different dataset, please change the code below
"""
boston = load_boston()
features, labels = boston.data, boston.target
boston_dataset = Dataset(features, labels, 'boston')

"""
Choose dataset for training/testing.
When training on a different dataset, please change the code below
"""
dataset = boston_dataset

"""
Set number of epochs the models should be trained for
"""
n_epochs = 5

def run_evaluation(config_path, create_hyperparameters, model_instance, metrics):
    with open(config_path) as c:
        config = yaml.load(c)
        config['n_features'] = dataset.features.shape[1]
        config['timestamp'] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config['dataset_name'] = dataset.name
        hyperparams = create_hyperparameters(**config)

    model = model_instance(device, hyperparams).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams.learning_rate)
    evaluationparams = EvaluationParameters(n_splits=10,
                                            scaler=False,
                                            normalize=True,
                                            n_epochs=n_epochs,
                                            poly_features=False,
                                            learning_rate = hyperparams.learning_rate,
                                            optimizer=optimizer)

    results, _ =  evaluate(model, dataset, metrics, evaluationparams, config, save=True)

    print('###################################################')
    print(f"RESULTS {model.__class__.__name__} MODEL")
    print('###################################################')
    results.print()
    print()


"""
Evaluate all models
"""
config_horseshoeBNN = os.path.join(root, 'configs/horseshoeBNN_config.yaml')
config_linearHorseshoe = os.path.join(root, 'configs/linear_horseshoe.yaml')
config_gaussianBNN = os.path.join(root, 'configs/bnn_config.yaml')
config_linearGaussian = os.path.join(root, 'configs/linearBNN_config.yaml')

metrics = [AllMetrics.mae, AllMetrics.rmse, AllMetrics.logprob]

run_evaluation(config_horseshoeBNN, HorseshoeHyperparameters, HorseshoeBNN, metrics)
run_evaluation(config_linearHorseshoe, LinearHorseshoeHyperparameters, LinearHorseshoe, metrics)
run_evaluation(config_gaussianBNN, BNNRegressionHyperparameters, GaussianBNN, metrics)
run_evaluation(config_linearGaussian, LinearBNNHyperparameters, LinearGaussian, metrics)

