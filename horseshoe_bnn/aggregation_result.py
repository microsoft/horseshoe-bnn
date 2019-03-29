"""
File: aggregation_result.py
Author: Anna-Lena Popkes, Hiske Overweg
Description: Classes that store the results of the evaluation procedure.
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class AggregationResult(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def print():
        pass

class AveragedResult(AggregationResult):
    def __init__(self, mean, std):
        self.mean = mean
        self.std= std

    def print(self):
        print(f'{self.mean:.3e} +/- {self.std:.3e}')

class CalibrationResult(AggregationResult):
    def __init__(self, variances, errors):
        self.variances = variances
        self.errors = errors
        self.runmean = []
        for i in range(1, len(self.errors)):
            rm = np.mean(self.errors[:i])
            self.runmean.append(rm)

    def print(self, axis_labels, title):
        # Constant parameters
        black = 'k'
        gray = '0.5'
        label_runmean = 'run mean'
        label_sigma = '2 sigma'
        dotted = '.'
        noline = ''
        xlabel = axis_labels[0]
        ylabel = axis_labels[1]

        plt.figure()
        plt.title(title)
        plt.plot(self.errors, marker=dotted, linestyle=noline)
        plt.plot(self.runmean, color=black, label=label_runmean)
        plt.plot(np.sqrt(self.variances)*2, color=gray, label=label_sigma)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc=1, prop={'size':12})
        plt.show()


class HistogramResult(AggregationResult):
    def __init__(self, ratios):
        self.ratios = ratios

    def print(self, axis_labels, title):
        # Constant parameters
        xlabel = axis_labels[0]
        ylabel = axis_labels[1]
        bins = 50

        plt.figure()
        plt.title(title)
        plt.hist(self.ratios, bins=bins)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
