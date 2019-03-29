"""
File: model.py
Author: Anna-Lena Popkes, Hiske Overweg
Description: All model classes
"""

import numpy as np
import math
import scipy
import ipdb
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from scipy.misc import logsumexp
from scipy.special import psi
from abc import ABCMeta, abstractmethod

from horseshoe_bnn.metrics import AllMetrics
from horseshoe_bnn.network_layers import BayesianLayer, HorseshoeLayer
from horseshoe_bnn.data_handling.dataset_to_dataloader import dataset_to_dataloader
from horseshoe_bnn.distributions import PredictiveDistribution, SampleDistribution, BinarySampleDistribution

from tensorboardX import SummaryWriter


def compute_log_likelihoods(classification, outputs, target, n_samples, var_noise):
    if classification:
        log_likelihoods = - F.binary_cross_entropy_with_logits(outputs, target.repeat(n_samples, 1), reduction='none')
        log_likelihoods = torch.sum(log_likelihoods, dim = 1)
    else:
        log_likelihoods =  torch.sum(-0.5 * torch.log(2 * math.pi * var_noise) - 0.5 * (target.repeat(n_samples, 1) - outputs) ** 2 / var_noise, dim=1)

    return log_likelihoods

def update_mse_mae(mse, mae, device_type, mean_output, target):
    if device_type == 'cuda':
        mse += F.mse_loss(mean_output, target, reduction='sum').cpu().data.numpy()
        mae += F.l1_loss(mean_output, target, reduction='sum').cpu().data.numpy()
    else:
        mse += F.mse_loss(mean_output, target, reduction='sum').detach().numpy()
        mae += F.l1_loss(mean_output, target, reduction='sum').detach().numpy()

    return mse, mae

class Model(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def predict(self):
        pass

class GaussianBNN(nn.Module, Model):
    def __init__(self, device, hyperparameters):
        super(GaussianBNN, self).__init__()
        self.name = 'GaussianBNN'
        self.device = device
        self.hyperparams = hyperparameters

        dir_ = os.getcwd().split('horseshoe_bnn')[0]
        path = f"{dir_}models/{self.hyperparams.dataset_name}/{self.name}" \
               f"/{self.hyperparams.timestamp}"
        self.train_writer = SummaryWriter(path + '/train')
        self.test_writer = SummaryWriter(path + '/test')

        self.l1 = BayesianLayer(self.hyperparams.n_features, self.hyperparams.n_hidden_units, self.hyperparams, device)
        self.l2 = BayesianLayer(self.hyperparams.n_hidden_units, 1, self.hyperparams, device)
        self.log_var_noise = torch.log(torch.Tensor([self.hyperparams.var_noise]))

    def initialize(self, n_features):
        """
        Reset model parameters
        """
        self.__init__(self.device, self.hyperparams)
        return self

    def forward(self, x, sample, n_samples):
        x = F.relu(self.l1.forward(x, n_samples=n_samples))
        x = self.l2.forward(x, n_samples=n_samples)
        return x

    def log_prior(self):
        """
        Calculates the logarithm of the current
        value of the prior distribution over the weights
        """
        return self.l1.log_prior \
               + self.l2.log_prior

    def log_variational_posterior(self):
        """
        Calculates the logarithm of the current value
        of the variational posterior distribution over the weights
        """

        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior

    def sample_elbo(self, input_, target, dataset_size):
        """
        Computes an estimate of the evidence lower bound using Monte Carlo sampling.

        The evidence lower bound is approximated as follows:
        Multiple samples are drawn from the variational posterior over the weights.

        For each sample:
        - The given input batch is forwarded through the resulting network
        - The current value of the prior distribution is computed
        - The current value of the variational posterior distribution is computed

        The final approximation of the lower bound is given by
        1. Averaging over the computed prior distribution values and variational posterior values
        2. Computing the value of the log likelihood
        3. Computing the value of the ELBO
        """

        batch_size = target.size()[0]
        n_samples = self.hyperparams.n_samples

        outputs = self.forward(input_, sample=True, n_samples=n_samples)
        outputs = outputs.reshape(n_samples, batch_size)
        log_prior = self.log_prior() / n_samples
        log_variational_posterior = self.log_variational_posterior() / n_samples

        var_noise = torch.exp(self.log_var_noise)
        if self.device.type == 'cuda':
            var_noise = var_noise.cuda()

        log_likelihoods = compute_log_likelihoods(self.hyperparams.classification, outputs, target, n_samples, var_noise)

        outputs = outputs.t()

        log_likelihood = log_likelihoods.mean()

        loss = (log_variational_posterior - log_prior) * batch_size / dataset_size
        if self.device.type == 'cuda':
            loss = loss.cuda()

        loss -= log_likelihood

        return loss, log_prior, log_variational_posterior, log_likelihood, outputs


    def train_model(self, dataset, epoch, optimizer, visualize_errors=False):
        """
        Trains a given model for a given number of epochs on a given dataset.
        """
        # self.train()
        super().train()

        dataset_size = dataset.features.shape[0]
        batch_size = self.hyperparams.batch_size

        # transform dataset to dataloader
        train_loader = dataset_to_dataloader(dataset, batch_size=batch_size)
        n_train_samples = len(train_loader.dataset)
        n_batches = len(train_loader)

        mse = 0
        mae = 0
        total_loss = 0
        total_log_likelihood = 0
        total_kl_divergence = 0

        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(self.device), target.to(self.device)
            self.zero_grad()
            loss, log_prior, log_variational_posterior, log_likelihood, outputs = self.sample_elbo(data, target, dataset_size)

            loss.backward()
            optimizer.step()

            # Given all model outputs, compute the mean output of the ensemble
            mean_output = outputs.mean(dim=1)

            if not self.hyperparams.classification:
                device_type = self.device.type
                mse, mae = update_mse_mae(mse, mae, device_type, mean_output, target)

            total_loss += loss
            total_log_likelihood += -log_likelihood
            total_kl_divergence += (log_variational_posterior - log_prior) * target.size()[0] / dataset_size

        rmse = np.sqrt(mse / dataset_size)
        mae /= dataset_size

        if visualize_errors:
            self.train_writer.add_scalar('loss__training loss', total_loss.item(), epoch)
            self.train_writer.add_scalar('loss__kl term' , total_kl_divergence.item(), epoch)
            self.train_writer.add_scalar('loss__log_likelihood term', total_log_likelihood.item(), epoch)
            if not self.hyperparams.classification:
                self.train_writer.add_scalar('errors__mae', mae, epoch)
                self.train_writer.add_scalar('errors__rmse', rmse, epoch)

        return loss, rmse, mae


    def predict(self, dataset, epoch=1, mean_y_train=0, std_y_train=1, visualize_errors=False):
        """
        Evaluates an ensemble of networks on a given test dataset.

        Because the Bayesian Neural Network has a distribution over the weights,
        we basically have an infinite number of different neural networks. We can
        take advantage of that by using an ensemble of networks during prediction.
        Each model in the ensemble performs a prediction. The different predictions
        are then averaged to give a final output.

        A model can be obtained by sampling weights from the distribution. Note: for
        each input batch, new models will be sampled.
        """
        n_samples_testing = self.hyperparams.n_samples_testing
        dataset_size = dataset.features.shape[0]
        test_batch_size = dataset_size

        # transform dataset to dataloader
        test_loader = dataset_to_dataloader(dataset, batch_size=test_batch_size, shuffle=False)
        n_test_samples = len(test_loader.dataset)
        n_test_batches = len(test_loader)

        super(GaussianBNN, self).eval()
        mse = 0
        rmse = 0
        mae = 0
        loglike = 0

        all_predicted_distributions = []
        means = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Each batch is forwarded through each model in the ensemble
                # and the model outputs are saved.
                ensemble_outputs = self.forward(data, sample=True, n_samples=n_samples_testing) * std_y_train + mean_y_train
                ensemble_outputs = ensemble_outputs.reshape(n_samples_testing, test_batch_size).t()

                # calculation of the predictive log likelihood of a batch, see notes from 18.12.18
                var_noise = np.exp(self.log_var_noise.detach().numpy()) * std_y_train ** 2

                if self.hyperparams.classification:
                    loglike_factor = - F.binary_cross_entropy_with_logits(ensemble_outputs, target.reshape(-1,1).repeat(1,n_samples_testing), reduction='none')
                    loglike = torch.sum(torch.logsumexp(loglike_factor - math.log(n_samples_testing), 1))
                    # Given all model outputs, compute the mean output of the ensemble
                    mean_output = ensemble_outputs.mean(1)
                else:
                    if self.device.type == 'cuda':
                        target = target.cpu().numpy()
                        ensemble_outputs = ensemble_outputs.cpu().numpy()
                    log_factor = -0.5 * np.log(2 * math.pi * var_noise) - (np.tile(target.reshape(-1, 1), (1, n_samples_testing)) - np.array(ensemble_outputs))**2 / (2* var_noise)
                    loglike += np.sum(logsumexp(log_factor - np.log(n_samples_testing), 1))
                    # Given all model outputs, compute the mean output of the ensemble
                    mean_output = ensemble_outputs.mean(1)
                    if self.device.type == 'cuda':
                        target = torch.from_numpy(target)
                        mean_output = torch.from_numpy(mean_output)

                if self.hyperparams.classification:
                    distributions = [BinarySampleDistribution(1 / (1 + np.exp(-e))) for e in ensemble_outputs.cpu().detach().numpy()]

                else:
                    mse += F.mse_loss(mean_output, target, reduction='sum')
                    mae += F.l1_loss(mean_output, target, reduction='sum')

                    distributions = [SampleDistribution(ensemble_outputs.cpu().detach().numpy()[i], var_noise)
                                        for i in range(test_batch_size)]

                all_predicted_distributions.extend(distributions)

        predicted_distr = PredictiveDistribution(all_predicted_distributions)

        loglike /= dataset_size
        rmse = np.sqrt(mse / dataset_size)
        mae /= dataset_size

        if self.hyperparams.classification:
            zero_one = AllMetrics.zero_one_loss.compute(target.cpu().detach().numpy(), predicted_distr)

        if visualize_errors:
            self.test_writer.add_scalar('errors__predictive log likelihood', loglike, epoch)
            if self.hyperparams.classification:
                self.test_writer.add_scalar('errors__zero_one', zero_one, epoch)
            else:
                self.test_writer.add_scalar('errors__mae', mae, epoch)
                self.test_writer.add_scalar('errors__rmse', rmse, epoch)


        return predicted_distr, rmse, mae, -loglike


class HorseshoeBNN(GaussianBNN):
    def __init__(self, device, hyperparameters):
        super(GaussianBNN, self).__init__()
        self.name = 'HorseshoeBNN'
        self.device = device
        self.hyperparams = hyperparameters

        dir_ = os.getcwd().split('horseshoe_bnn')[0]
        path = f"{dir_}models/{self.hyperparams.dataset_name}/{self.name}" \
               f"/{self.hyperparams.timestamp}"
        self.train_writer = SummaryWriter(path + '/train')
        self.test_writer = SummaryWriter(path + '/test')

        self.l1 = HorseshoeLayer(self.hyperparams.n_features, self.hyperparams.n_hidden_units, self.hyperparams, device)
        self.l2 = BayesianLayer(self.hyperparams.n_hidden_units, 1, self.hyperparams, device)
        self.log_var_noise = torch.log(torch.Tensor([self.hyperparams.var_noise]))

    def initialize(self, n_features):
        """
        Reset model parameters
        """
        self.__init__(self.device, self.hyperparams)
        return self

    def log_prior(self, n_samples):
        """
        Calculates the logarithm of the current
        value of the prior distribution over the weights
        """
        return self.l1.log_prior() \
               + self.l2.log_prior / n_samples

    def log_variational_posterior(self, n_samples):
        """
        Calculates the logarithm of the current value
        of the variational posterior distribution over the weights
        """
        return self.l1.log_variational_posterior() \
               + self.l2.log_variational_posterior / n_samples \

    def analytic_update(self):
        """
        Calculates the update of the model parameters with
        analytic update equations
        """
        return self.l1.analytic_update()

    def sample_elbo(self, input_, target, dataset_size):
        """
        Computes an estimate of the evidence lower bound

        The evidence lower bound is calculated as follows:
         - compute the log of the prior and the variational posterior

         - To calculate the log likelihood Monte-Carlo sampling is performed:
         mulltiple samples are drawn from the variational posterior over the weights.

        For each sample:
        - The given input batch is forwarded through the resulting network
        - the corresponding output is stored

        The log likelihood is calculated by averaging the log likelihood of each sample

        Finally the terms are added to calculate the complete elbo
        """
        batch_size = target.size()[0]
        n_samples = self.hyperparams.n_samples

        outputs = self.forward(input_, sample=True, n_samples=n_samples)
        outputs = outputs.reshape(n_samples, batch_size)

        log_prior = self.log_prior(n_samples)
        log_variational_posterior = self.log_variational_posterior(n_samples)

        var_noise = torch.exp(self.log_var_noise)
        if self.device.type == 'cuda':
            var_noise = var_noise.cuda()

        log_likelihoods = compute_log_likelihoods(self.hyperparams.classification, outputs, target, n_samples, var_noise)

        outputs = outputs.t()

        log_likelihood = log_likelihoods.mean()

        loss = (log_variational_posterior - log_prior) * batch_size / dataset_size
        if self.device.type == 'cuda':
            loss = loss.cuda()
        loss -= log_likelihood

        return loss, log_prior, log_variational_posterior, log_likelihood, outputs

    def train_model(self, dataset, epoch, optimizer, visualize_errors=False):
        """
        Trains a given model for a given number of epochs on a given dataset.
        """
        # self.train()
        super().train()

        dataset_size = dataset.features.shape[0]
        batch_size = self.hyperparams.batch_size # train using the specified batch size

        # Transform dataset to dataloader
        train_loader = dataset_to_dataloader(dataset, batch_size=batch_size)
        n_train_samples = len(train_loader.dataset)
        n_batches = len(train_loader)

        mse = 0
        mae = 0
        total_loss = 0
        total_log_likelihood = 0
        total_kl_divergence = 0

        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(self.device), target.to(self.device)
            self.zero_grad()
            loss, log_prior, log_variational_posterior, log_likelihood, outputs = self.sample_elbo(data, target, dataset_size)

            loss.backward()
            optimizer.step()

            # perform analytic update of remaining model parameters
            self.analytic_update()

            # Given all model outputs, compute the mean output of the ensemble
            mean_output = outputs.mean(dim=1)

            mse, mae = update_mse_mae(mse, mae, self.device.type, mean_output, target)

            total_loss += loss
            total_log_likelihood += -log_likelihood
            total_kl_divergence += (log_variational_posterior - log_prior) * target.size()[0] / dataset_size

        rmse = np.sqrt(mse / dataset_size)
        mae /= dataset_size

        if visualize_errors:
            self.train_writer.add_scalar('loss__training loss', total_loss.item(), epoch)
            self.train_writer.add_scalar('loss__kl term' , total_kl_divergence.item(), epoch)
            self.train_writer.add_scalar('loss__log_likelihood term', total_log_likelihood.item(), epoch)
            if not self.hyperparams.classification:
                self.train_writer.add_scalar('errors__mae', mae, epoch)
                self.train_writer.add_scalar('errors__rmse', rmse, epoch)

        return loss, rmse, mae

class LinearHorseshoe(GaussianBNN):
    def __init__(self, device, hyperparameters):
        super(GaussianBNN, self).__init__()
        self.name  = 'LinearHorseshoe'
        self.device = device
        self.hyperparams = hyperparameters

        dir_ = os.getcwd().split('horseshoe_bnn')[0]
        path = f"{dir_}models/{self.hyperparams.dataset_name}/{self.name}" \
               f"/{self.hyperparams.timestamp}"
        self.train_writer = SummaryWriter(path + '/train')
        self.test_writer = SummaryWriter(path + '/test')

        self.layer = HorseshoeLayer(self.hyperparams.n_features, 1, self.hyperparams, device)
        self.log_var_noise = torch.log(torch.Tensor([self.hyperparams.var_noise]))

    def initialize(self, n_features):
        """
        Reset model parameters
        """
        self.__init__(self.device, self.hyperparams)
        return self

    def forward(self, x, sample, n_samples):
        x = self.layer.forward(x, n_samples=n_samples)
        return x

    def log_prior(self):
        """
        Calculates the logarithm of the current
        value of the prior distribution over the weights
        """
        return self.layer.log_prior()

    def log_variational_posterior(self):
        """
        Calculates the logarithm of the current value
        of the variational posterior distribution over the weights
        """
        return self.layer.log_variational_posterior()

    def analytic_update(self):
        """
        Calculates the update of the model parameters with
        analytic update equations
        """
        return self.layer.analytic_update()

    def sample_elbo(self, input_, target, dataset_size):
        """
        Computes an estimate of the evidence lower bound

        The evidence lower bound is calculated as follows:
         - compute the log of the prior and the variational posterior

         - To calculate the log likelihood Monte-Carlo sampling is performed:
         mulltiple samples are drawn from the variational posterior over the weights.

        For each sample:
        - The given input batch is forwarded through the resulting network
        - the corresponding output is stored

        The log likelihood is calculated by averaging the log likelihood of each sample

        Finally the terms are added to calculate the complete elbo
        """
        batch_size = target.shape[0]
        n_samples = self.hyperparams.n_samples

        log_prior = self.log_prior()
        log_variational_posterior = self.log_variational_posterior()

        outputs = self.forward(input_, sample=True, n_samples=n_samples)
        outputs = outputs.reshape(n_samples, batch_size)

        var_noise = torch.exp(self.log_var_noise)
        if self.device.type == 'cuda':
            var_noise = var_noise.cuda()

        log_likelihoods = compute_log_likelihoods(self.hyperparams.classification, outputs, target, n_samples, var_noise)

        outputs = outputs.t()

        log_likelihood = log_likelihoods.mean()

        loss = (log_variational_posterior - log_prior) * batch_size / dataset_size

        if self.device.type == 'cuda':
            loss = loss.cuda()

        loss -= log_likelihood

        return loss, log_prior, log_variational_posterior, log_likelihood, outputs

    def train_model(self, dataset, epoch, optimizer, visualize_errors=False):
        """
        Trains a given model for a given number of epochs on a given dataset.
        """
        # self.train()
        super().train()

        dataset_size = dataset.features.shape[0]
        # batch_size = dataset_size # train on entire data in one batch
        batch_size = self.hyperparams.batch_size # train using the specified batch size

        # transform dataset to dataloader
        train_loader = dataset_to_dataloader(dataset, batch_size=batch_size)
        n_train_samples = len(train_loader.dataset)
        n_batches = len(train_loader)

        mse = 0
        mae = 0
        total_loss = 0
        total_log_likelihood = 0
        total_kl_divergence = 0

        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(self.device), target.to(self.device)
            self.zero_grad()
            loss, log_prior, log_variational_posterior, log_likelihood, outputs = self.sample_elbo(data, target, dataset_size)

            loss.backward()
            optimizer.step()

            # perform analytic update of remaining model parameters
            self.analytic_update()

            # Given all model outputs, compute the mean output of the ensemble
            mean_output = outputs.mean(dim=1)

            device_type = self.device.type
            mse, mae = update_mse_mae(mse, mae, device_type, mean_output, target)

            total_loss += loss
            total_log_likelihood += -log_likelihood
            total_kl_divergence += (log_variational_posterior - log_prior) * target.size()[0] / dataset_size

        rmse = np.sqrt(mse / dataset_size)
        mae /= dataset_size

        if visualize_errors:
            self.train_writer.add_scalar('loss__training loss', total_loss.item(), epoch)
            self.train_writer.add_scalar('loss__kl term' , total_kl_divergence.item(), epoch)
            self.train_writer.add_scalar('loss__log_likelihood term', total_log_likelihood.item(), epoch)
            if not self.hyperparams.classification:
                self.train_writer.add_scalar('errors__mae', mae, epoch)
                self.train_writer.add_scalar('errors__rmse', rmse, epoch)

        return loss, rmse, mae

class LinearGaussian(GaussianBNN):
    def __init__(self, device, hyperparameters):
        super(GaussianBNN, self).__init__()
        self.name = 'LinearGaussian'
        self.device = device
        self.hyperparams = hyperparameters

        dir_ = os.getcwd().split('horseshoe_bnn')[0]
        path = f"{dir_}models/{self.hyperparams.dataset_name}/{self.name}" \
               f"/{self.hyperparams.timestamp}"
        self.train_writer = SummaryWriter(path + '/train')
        self.test_writer = SummaryWriter(path + '/test')

        self.layer = BayesianLayer(self.hyperparams.n_features, 1, self.hyperparams, device)
        self.log_var_noise = torch.log(torch.Tensor([self.hyperparams.var_noise]))

    def forward(self, x, sample=False, n_samples=1):
        x = self.layer.forward(x, sample, n_samples)
        return x

    def log_prior(self):
        """
        Calculates the logarithm of the current
        value of the prior distribution over the weights
        """
        return self.layer.log_prior

    def log_variational_posterior(self):
        """
        Calculates the logarithm of the current value
        of the variational posterior distribution over the weights
        """
        return self.layer.log_variational_posterior



