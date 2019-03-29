"""
File: dataset.py
Author: Anna-Lena Popkes, Hiske Overweg
Description: the Dataset class
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pandas as pd
import csv
import pickle

class Dataset():
    def __init__(self, features=None, labels=None, name=''):
        """
        Class constructor, sets parameters

        Args:
            features: np array, features of the dataset
            labels: np array, labels of the dataset
            name: string, name of the dataset

        Raises:
            TypeError: if features/labels are not np array or None
            TypeError: if features is None and labels is not or vice versa
            ValueError: if features/labels have different first dimension
        """
        if not (isinstance(features, np.ndarray) or features is None):
            raise TypeError('Features should be numpy array or None')
        if not (isinstance(labels, np.ndarray) or labels is None):
            raise TypeError('Labels should be numpy array or None')
        if ((features is None and not(labels is None)) or \
            (not(features is None) and labels is None)):
            raise TypeError('Features and labels should be of same type')
        elif not(labels is None):
            if features.shape[0] != labels.shape[0]:
                raise ValueError('Features and labels should have same first dimension')

        self.features = features
        self.labels = labels
        self.name = name

    def load_from_csv(self, path_to_csv, feature_columns, label_columns):
        """
        Loads a dataset from a csv file and sets
        the dataset features and targets

        Args:
            path_to_csv: string, path to the csv file
            feature_columns: list of ints, indices of feature columns
            label_columns: list of ints, indices of label columns

        Raises:
            ValueError: if requested feature columns do not exist in file
            ValueError: if requested label columns do not exist in file
        """
        df = pd.read_csv(path_to_csv, encoding="utf-8")

        # check that all requested feature columns exist
        max_idx = np.max(feature_columns)
        if max_idx > df.values.shape[1]:
            raise ValueError('File does not contain requested feature columns')
        # check that all requested label columns exist
        max_idx = np.max(label_columns)
        if max_idx > df.values.shape[1]:
            raise ValueError('File does not contain requested label columns')

        df_features = df.iloc[:, feature_columns]
        df_labels = df.iloc[:, label_columns]

        self.features = df_features.values
        self.labels = df_labels.values

    def save_to_csv(self, path_to_save):
        """
        Saves targets and features to csv file

        Args:
            path_to_save: string, path to save the csv file
        """
        if self.labels.ndim == 1:
            data = np.concatenate((self.features, self.labels.reshape(-1, 1)), axis=1)
        else:
            data = np.concatenate((self.features, self.labels), axis=1)

        df = pd.DataFrame(data)
        df.to_csv(path_to_save, index=False)

    def load_from_pickle(self, path_to_pickle, feature_columns, label_columns):
        """
        Loads a dataset from a pickle file and sets
        the dataset features and targets

        Args:
            path_to_pickle: string, path to the pickle file
            feature_columns: list of ints, indices of feature columns
            label_columns: list of ints, indices of label columns

        Raises:
            ValueError: if requested feature columns do not exist in file
            ValueError: if requested label columns do not exist in file
        """
        with open(path_to_pickle, 'rb') as f:
            data = pickle.load(f)

        # check that all requested feature columns exist
        max_idx = np.max(feature_columns)
        if max_idx > data.shape[1]:
            raise ValueError('File does not contain requested feature columns')
        # check that all requested label columns exist
        max_idx = np.max(label_columns)
        if max_idx > data.shape[1]:
            raise ValueError('File does not contain requested label columns')

        self.features = data[:, feature_columns]
        self.labels = data[:, label_columns]

    def save_to_pickle(self, path_to_save):
        """
        Saves targets and features to pickle file

        Args:
            path_to_save: string, path to save the pickle file
        """
        if self.labels.ndim == 1:
            data = np.concatenate((self.features, self.labels.reshape(-1, 1)), axis=1)
        else:
            data = np.concatenate((self.features, self.labels), axis=1)

        with open(path_to_save, 'wb') as f:
            pickle.dump(data, f)

    def split(self, test_size=0.25):
        """
        Splits the dataset into a train and test set

        Returns:
            train_dataset: Dataset
            test_dataset: Dataset

        Raises:
            ValueError: if Dataset is empty
        """
        if self.features is None:
            raise ValueError('Cannot split empty dataset')

        train_features, test_features, train_labels, test_labels \
                = train_test_split(self.features, self.labels, test_size=test_size)

        self.train_dataset = Dataset(train_features, train_labels)
        self.test_dataset = Dataset(test_features, test_labels)

        return self.train_dataset, self.test_dataset

    def compute_polynomial_features(self, poly_degree=1, interaction_only=False):
        """
        Computes polynomial features of the dataset.

        Args:
            poly_degree: int, polynomial degree

        Raises:
            TypeError: if poly_degree is not integer
            ValueError: if poly_degree is smaller than 1
            TypeError: if interaction_only is not a boolean
            ValueError: if dataset is empty
        """
        if not isinstance(poly_degree, int):
            raise TypeError('Poly_degree should be integer')
        if poly_degree < 1:
            raise ValueError('Poly_degree should be at least 1')
        if not isinstance(interaction_only, bool):
            raise TypeError('Interaction_only should be boolean')
        if self.features is None:
            raise ValueError('Cannot compute polynomial features for empty dataset')

        poly = PolynomialFeatures(poly_degree, interaction_only=interaction_only)
        self.features = poly.fit_transform(self.features)

    def normalize(self, scaler=None):
        """
        Normalizes the dataset features given a Scaler.

        Args:
            scaler: Scaler instance, e.g. scikit-learn StandardScaler

        Raises:
            ValueError: if dataset is empty
        """
        if self.features is None:
            raise ValueError('Cannot normalize empty dataset')
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(self.features)

        self.features = scaler.transform(self.features)

    def add_bias(self):
        """
        Adds a bias of 1 to each input example.

        Raises:
            ValueError: if dataset is empty
        """
        if self.features is None:
            raise ValueError('Cannot add bias to empty dataset')
        n_samples = self.features.shape[0]
        self.features = np.column_stack((np.ones(n_samples), self.features))

    def remove_single_feature(self):
        """
        Removes the first feature from the original dataset.
        Returns a Dataset instance which is a copy of the
        original dataset but with the first feature/column removed

        Raises:
            ValueError: if dataset is empty/contains a single feature
        """
        if self.features is None:
            raise ValueError('Cannot remove feature from empty dataset')
        if self.features.shape[1] == 1:
            raise ValueError("Dataset only has a single feature left! You can't remove further features")

        # Remove feature column
        new_features = np.delete(self.features, 0, axis=1)

        return Dataset(new_features, self.labels)



