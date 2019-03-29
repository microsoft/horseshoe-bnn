"""
File: mimic_preprocessing.py
Author: Anna-Lena Popkes, Hiske Overweg
Description: This script preprocesses the MIMIC3 mortality and length of stay subsets.
             It replaces missing values, converts Glascow coma scale values and saves
             the resulting features and targets as Python pickle files.

BEFORE USING THIS SCRIPT, PLEASE TAKE A LOOK AT THE README CONTAINED IN THIS FOLDER
"""

import os
import numpy as np
import ipdb
import pickle
import yaml
import pandas as pd
from mimic_readers import LengthOfStayReader, InHospitalMortalityReader

def create_glasgow_dict():
    """
    Create dictionaries that map the different parts of the
    Glasgow Coma Scale onto numbers
    """

    glasgow_eye = {'4 Spontaneously': 4, 'Spontaneously': 4, '3 To speech': 3, 'To Speech': 3,
                    'To speech': 3, '2 To pain': 2, 'To Pain': 2, 'To pain': 2, 'No Response': 1}

    glasgow_verbal = {'5 Oriented': 5, 'Oriented': 5, '4 Confused': 4, 'Confused': 4,
                        '3 Inappropriate Words': 3, '3 Inapprop words': 3, '3 Inappropriate words': 3,
                        'Inappropriate Words': 3,  'Inappropriate words': 3, '2 Incomp sounds': 2,
                        'Incomp sounds': 2, 'Incomprehensible sounds': 2, '1 No Response': 1,
                        'No Response': 1, '1.0 ET/Trach': 1, 'None': 1, 'No response': 1,
                        'No Response-ETT': 1}

    glasgow_motor = {'6 Obeys Commands': 6, 'Obeys Commands': 6, '5 Localizes Pain': 5, 'Localizes Pain': 5,
                        '4 Flex-withdraws': 4, 'Flex-withdraws': 4, '3 Abnorm flexion': 3,
                        '3 Abnorm Flexion': 3, '3 Abnormal Flexion': 3, 'Abnormal Flexion': 3,
                        'Abnormal flexion': 3, '2 Abnormal extension': 2, 'Abnormal Extension': 2,
                        'Abnormal extension': 2}

    numbers = {'15': 15, '14': 14, '13': 13, '12': 12, '11': 11, '10': 10,
                '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2, '1': 1, '0': 0}

    glasgow_dict = {**glasgow_eye, **glasgow_verbal, **glasgow_motor, **numbers}

    return glasgow_dict

def correct_glasgow_features(features, glasgow_dict):
    """
    Check all entries in the feature array for a match with the glasgow_dict
    If a match is found, the entry is replaced by the corresponding value
    If no match is found, the entry is replaced by NaN

    Args:
        features: np array, contains the raw data of a single patient
        glagow_dict: dictionary containing the glasgow coma scale

    Returns:
        glasgow_features: np array in which incorrect entries have been replaced
    """
    # Select columns with Glascow scales/ text entries
    glasgow_index = range(4, 8)
    features_glasgow = features[:, glasgow_index]
    # Transform text to features, replace missing values
    glasgow_features = [[float(glasgow_dict.get(entry, np.nan))
                        if len(entry) > 0
                        else np.nan
                        for entry in row]
                        for row in features_glasgow]

    return np.array(glasgow_features)

# Define the upper limit for the different features
max_values = {'Capillary refill rate': 100,
            'Diastolic blood pressure': 150,
            'Fraction inspired oxygen': 1,
            'Glucose': 1250,
            'Heart Rate': 300,
            'Height': 250,
            'Mean blood pressure': 190,
            'Oxygen saturation': 100,
            'Respiratory rate': 150,
            'Systolic blood pressure': 275,
            'Temperature': 49,
            'Weight': 250}

# Specify for which features values of zero should be removed
remove_zeros = {'Capillary refill rate': False,
            'Diastolic blood pressure': True,
            'Fraction inspired oxygen': False,
            'Glucose': False,
            'Heart Rate': False,
            'Height': True,
            'Mean blood pressure': True,
            'Oxygen saturation': True,
            'Respiratory rate': False,
            'Systolic blood pressure': True,
            'Temperature': True,
            'Weight': True}

def correct_num_features(features, header):
    """
    Replace all entries in the feature array which are not floats
    or which fall outside the physical limits of that feature by NaN

    Args:
        features: np array, contains the raw data of a single patient
        header: list, contains the names of the different columns

    Returns:
        num_features: np array in which incorrect entries have been replaced
    """

    # indices of columns with numberical features, pH is treated separately
    num_index = list(range(1 ,4)) + list(range(8 ,17))
    num_features = np.zeros((features.shape[0], len(num_index)))

    for j, index in enumerate(num_index):
        feature = header[index]
        upper_limit = max_values[feature]
        if remove_zeros[feature]:
            num_features[:, j] = [float(entry)
                            if len(entry) > 0 and float(entry) < upper_limit\
                            and float(entry) > 0
                            else np.nan
                            for entry in features[:, index]]
        else:
            num_features[:, j] = [float(entry)
                            if len(entry) > 0 and float(entry) < upper_limit\
                            and float(entry) >= 0
                            else np.nan
                            for entry in features[:, index]]

    return num_features


def correct_pH(feature):
    """
    Replace all entries in the feature array which are not floats
    or which fall outside the physical limits of pH values by NaN

    Args:
        feature: np array, contains the raw pH values of a single patient

    Returns:
        pH_lim: np array in which incorrect entries have been replaced
    """
    pH_float = [float(entry)
                            if len(entry) > 0
                            else np.nan
                            for entry in feature]

    # Correct the pH values
    # Values higher than 14  are replaced by - log( value* 1E-9)/log(10)
    # because they are assumed to be hydrogen concentration in n mole/L
    # Values which are lower than 6 or higher than 8 are discarded
    hydrogen = lambda x: - np.log(x * 1E-9) / np.log(10)
    def check_pH(x):
        if x > 6 and x < 8:
            return(x)
        elif x > 14:
            y = hydrogen(x)
            if y > 6 and y < 8:
                return(y)
        return(np.nan)

    pH_lim = np.array([list(map(check_pH, pH_float))]).T

    return pH_lim


def process_MIMIC_dataset(mimic_subset_name, reader, yaml_file):
    """
    Preprocesses a given MIMIC subset.

    Args:
        mimic_subset_name: str, name of MIMIC subset to be processed (e.g. 'test' or 'train')

    Returns:
        Pandas DataFrame of the training examples, with features as columns
        Numpy array with target labels
    """

    with open(yaml_file, 'r') as c:
        config = yaml.load(c)
        test_dir = os.path.expanduser(config[mimic_subset_name + '_dir'])
        test_listfile = os.path.expanduser(config[mimic_subset_name + '_listfile'])

    # Set up the reader for the dataset.
    # The reader class is provided in this GitHub repo:
    # https://github.com/YerevaNN/mimic3-benchmarks/tree/master/mimic3benchmark
    reader =reader(test_dir, test_listfile)
    n_samples = reader.get_number_of_examples()

    # get Glascow dictionary
    glasgow_dict = create_glasgow_dict()

    # Preprocess the dataset
    features_all_samples = []
    targets = []

    for i in range(0, n_samples):
        dict_ = reader.read_example(i)
        features = dict_['X']
        target = dict_['y']

        # filter acceptable glasgow coma scale values
        glasgow_features = correct_glasgow_features(features, glasgow_dict)

        # filter acceptable values for numerical features
        header = dict_['header']
        num_features = correct_num_features(features, header)

        # filter acceptable pH values
        pH_feature = correct_pH(features[:, -1])

        # Combine the three arrays into one
        features_combined = np.concatenate((num_features, pH_feature, glasgow_features), axis=1)

        # Compute mean over each column
        features_mean = np.nanmean(features_combined, axis=0)

        # Add to result array
        features_all_samples.append(features_mean)
        targets.append(target)

        if i % 10000 == 0:
            print(f"{i} samples have been processed")

    return pd.DataFrame(features_all_samples), np.array(targets)

def save_data(features_arr, targets, name, yaml_file, replaced_nan):
    """
    Saves preprocessed array of features and vector of targets
    to Python pickle files.
    """
    ipdb.set_trace()
    with open(yaml_file, 'r') as c:
        config = yaml.load(c)
        if replaced_nan:
            features_path = os.path.expanduser(config[f"{name}_features_path"])
            targets_path = os.path.expanduser(config[f"{name}_labels_path"])
        else:
            features_path = os.path.expanduser(config[f"{name}_features_path_with_nan"])
            targets_path = os.path.expanduser(config[f"{name}_labels_path_with_nan"])

    with open(features_path, 'wb') as f:
        pickle.dump(features_arr, f)

    with open(targets_path, 'wb') as f:
        pickle.dump(targets, f)


if __name__ == "__main__":

    # task = 'mortality'
    task = 'length_of_stay'

    reader = {'mortality': InHospitalMortalityReader, 'length_of_stay':LengthOfStayReader}
    yaml_file = {'mortality': 'mimic_config_mortality.yaml',  'length_of_stay': 'mimic_config.yaml'}

    name = 'train'
    df_train, targets_train = process_MIMIC_dataset(name, reader[task], yaml_file[task])
    save_data(df_train.values, targets_train, name, yaml_file[task], replaced_nan=False)

    # calculate feature means
    means = df_train.mean()

    # replace nan values with mean of each feature
    df_train = df_train.fillna(means)
    save_data(df_train.values, targets_train, name, yaml_file[task], replaced_nan=True)

    name = 'test'
    df_test, targets_test = process_MIMIC_dataset(name, reader[task], yaml_file[task])
    save_data(df_test.values, targets_test, name, yaml_file[task], replaced_nan=False)

    # replace nan values with mean from training
    df_test = df_test.fillna(means)
    save_data(df_test.values, targets_train, name, yaml_file[task], replaced_nan=True)
