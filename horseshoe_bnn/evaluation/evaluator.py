"""
File: evaluator.py
Author: Anna-Lena Popkes, Hiske Overweg
Description: Method for evaluation a given model on a given dataset.
             Evaluation is performed according to a given list of metrics.
"""

import numpy as np
import yaml
import ipdb
import torch
import torch.optim as optim
import math
import datetime
import pickle
import os

from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.datasets import load_boston
from collections import namedtuple

from horseshoe_bnn.metrics import AllMetrics
from horseshoe_bnn.data_handling.dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

class EvaluationResults:
    def __init__(self):
        self.results = {}

    def print(self):
        """
        Prints for each metric the metric name and corresponding metric value
        """
        for metric, result in self.results.items():
            metric.print(result)

def normalize_dataset(model, evaluationparams, dataset_train, dataset_test):
    """
    Normalizes the dataset
    """
    if evaluationparams.normalize:
        std_X_train = np.std(dataset_train.features, 0)
        std_X_train[std_X_train == 0] = 1

        mean_X_train = np.mean(dataset_train.features, 0)
        dataset_train.features = (
            dataset_train.features - mean_X_train
        ) / std_X_train
        dataset_test.features = (dataset_test.features - mean_X_train) / std_X_train

        if not model.hyperparams.classification:
            mean_y_train = np.mean(dataset_train.labels)
            std_y_train = np.std(dataset_train.labels)
            dataset_train.labels = (dataset_train.labels - mean_y_train) / std_y_train
        else:
            mean_y_train = 0.0
            std_y_train = 1.0

    return dataset_train, dataset_test, mean_y_train, std_y_train

def save_params(model, evaluationparams, config):
    """
    Saves hyperparameters and evaluation parameters
    """
    dir_ = os.getcwd().split("horseshoe_bnn")[0]
    path = (
        f"{dir_}models/{model.hyperparams.dataset_name}/{model.name}"
        f"/{model.hyperparams.timestamp}"
    )
    if not os.path.exists(path):
        os.makedirs(path)

    # Get evaluation parameters except for Adam optimizer instance (can't be serialized like this')
    eval_params = {key:value for key, value in evaluationparams.__dict__.items() if key != 'optimizer'}
    # Add learning rate field
    if evaluationparams.__dict__['optimizer'] is not None:
        eval_params['learning_rate'] = float(evaluationparams.__dict__['optimizer'].param_groups[0]['lr'])
    with open(f"{path}/evaluationparams.yaml", "w") as f:
        yaml.dump(eval_params, f, default_flow_style=False)

    # Save hyperparameters. Some of them are numpy floats and need to be converted
    hp = {}
    for key, value in config.items():
        if isinstance(value, float):
            hp[key] = float(value)
        else:
            hp[key] = value
    with open(f'{path}/model_config.yaml', 'w') as f:
        yaml.dump(hp, f, default_flow_style=False)

        return path

def save_predictions(path, split, evaluationparams, model, test_idx, predicted_distribution):
    """
    Saves model parameters, indices of test set and predictive distribution
    """
    subpath = os.path.join(path, f'split_{split}')
    if not os.path.exists(subpath):
        os.makedirs(subpath)

    if evaluationparams.n_epochs > 1:
        torch.save(model.state_dict(), f"{path}/split_{split}/params.pt")
    else:
        with open(f"{path}/split_{split}/params.pkl", 'wb') as f:
            pickle.dump(model.param_distribution, f)

    # save indices of test set
    with open(f"{path}/split_{split}/indices.pkl", 'wb') as f:
        pickle.dump(test_idx, f)

    # save predictions
    with open(f"{path}/split_{split}/predictions.pkl", 'wb') as f:
            pickle.dump(predicted_distribution, f)


def evaluate(model, dataset, list_of_metrics, evaluationparams, config=None, save=False):
    """
    Evaluates a model using a list of metrics

    Args:
        model: instance of Model class
        dataset: instance of Dataset class
        list_of_metrics: list of metrics that should be computed
        evaluationparams: instance of EvaluationParameters class,
                          holds parameters used in the evaluation.
        config: model config file
        save: bool, whether to save the model

    Returns:
        evaluation_results: instance of EvaluationResults class,
                            contains all computed metric and corresponding values
        model: trained model
    """

    kf = KFold(n_splits=evaluationparams.n_splits, shuffle=True, random_state=42)

    # dictionary to store metrics obtained in different folds
    results = {metric: [] for metric in list_of_metrics}
    split = 0

    if save:
        path = save_params(model, evaluationparams, config)

    for train_idx, test_idx in kf.split(dataset.features):
        # Keep track of the split in order to control visualization of Pytorch models
        split += 1
        visualize_errors = True if split == 1 else False

        dataset_train = Dataset(dataset.features[train_idx], dataset.labels[train_idx])
        dataset_test = Dataset(dataset.features[test_idx], dataset.labels[test_idx])

        # scale the data
        if evaluationparams.scaler:
            scaler = evaluationparams.scaler()
            scaler.fit(dataset_train.features)
            dataset_train.normalize(scaler=scaler)
            dataset_test.normalize(scaler=scaler)

        mean_y_train = 0
        std_y_train = 1
        dataset_train, dataset_test, mean_y_train, std_y_train = normalize_dataset(model, evaluationparams, dataset_train, dataset_test)

        # create polynomial features
        if evaluationparams.poly_features:
            dataset_train.compute_polynomial_features(
                poly_degree=evaluationparams.poly_degree,
                interaction_only=evaluationparams.interaction_only,
            )
            dataset_test.compute_polynomial_features(
                poly_degree=evaluationparams.poly_degree,
                interaction_only=evaluationparams.interaction_only,
            )

        # For each fold initialize/reset the model. When training a neural net,
        # the optimizer needs to be re-initialized, too. In case of a simple net
        # we don't need an optimizer at all
        if evaluationparams.n_epochs > 1:
            model = model.initialize(n_features=dataset_train.features.shape[1])
            evaluationparams.optimizer = optim.Adam(
                model.parameters(), lr=evaluationparams.learning_rate
            )

        else:
            model.initialize(n_features=dataset_train.features.shape[1])

        for epoch in range(evaluationparams.n_epochs):

            if evaluationparams.n_epochs > 1: # Only NN's are trained multiple epochs
                model.train_model(
                    dataset_train, epoch, evaluationparams.optimizer, visualize_errors
                )
                predicted_distribution, *rest = model.predict(
                    dataset_test,
                    epoch=epoch,
                    mean_y_train=mean_y_train,
                    std_y_train=std_y_train,
                    visualize_errors=visualize_errors,
                )
            else:
                model.train_model(dataset_train)
                predicted_distribution = model.predict(
                    dataset_test, mean_y_train=mean_y_train, std_y_train=std_y_train
                )

        if save:
            save_predictions(path, split, evaluationparams, model, test_idx, predicted_distribution)

        for metric in list_of_metrics:
            result = metric.compute(dataset_test.labels, predicted_distribution)
            results[metric].append(result)

    assert len(results) == len(
        list_of_metrics
    ), "Length of results dictionary and list of metrics must be the same!"

    evaluationResults = EvaluationResults()
    for metric in list_of_metrics:
        aggregation_result = metric.aggregate(results[metric])
        evaluationResults.results[metric] = aggregation_result

    if save:
    # Save final results
        with open(f"{path}/evaluation_results.pkl", 'wb') as f:
            pickle.dump(evaluationResults, f)

    return evaluationResults, model


