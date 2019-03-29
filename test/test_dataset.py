import pytest
from horseshoe_bnn.data_handling.dataset import Dataset
import numpy as np

@pytest.fixture
def saved_random_dataset():
    n_features = 5
    n_samples = 10
    features = np.random.rand(n_samples, n_features)
    labels = np.random.rand(n_samples)
    path_to_save = 'test_pickle_save.pkl'
    dataset = Dataset(features, labels)
    dataset.save_to_pickle(path_to_save)
    return path_to_save, dataset

@pytest.fixture
def generate_random_dataset():
    n_features = 5
    n_samples = 10
    features = np.random.rand(n_samples, n_features)
    labels = np.random.rand(n_samples)
    dataset = Dataset(features, labels)
    return dataset

def test_constructor_with_valid_arguments():
    n_features = 5
    n_samples = 10
    features = np.random.rand(n_samples, n_features)
    labels = np.random.rand(n_samples)
    dataset = Dataset(features, labels)
    assert np.array_equal(features, dataset.features)
    assert np.array_equal(labels, dataset.labels)

def test_constructor_with_invalid_arguments_types_raises_type_error():
    features = 'a'
    labels = 'b'

    with pytest.raises(TypeError):
        Dataset(features, labels)

def test_constructor_with_non_matching_argument_types_raises_type_error():
    features = None
    labels = np.linspace(0, 9, 10)

    with pytest.raises(TypeError):
        Dataset(features, labels)

def test_constructor_with_non_matching_argument_sizes_raises_value_error():
    n_features = 5
    n_samples = 10
    wrong_n_samples = n_samples - 2
    features = np.random.rand(n_samples, n_features)
    labels = np.random.rand(wrong_n_samples)

    with pytest.raises(ValueError):
        Dataset(features, labels)

def test_save_and_load_to_pickle_identical_dataset(generate_random_dataset):
    path_to_save = 'test_pickle_save.pkl'

    dataset = generate_random_dataset
    n_features = dataset.features.shape[1]
    dataset.save_to_pickle(path_to_save)

    loaded_dataset = Dataset()
    loaded_dataset.load_from_pickle(path_to_save, range(n_features), range(n_features, n_features + 1))

    assert np.array_equal(loaded_dataset.features, dataset.features)
    assert np.array_equal(loaded_dataset.labels, dataset.labels.reshape(-1, 1))

def test_save_and_load_to_csv_identical_dataset(generate_random_dataset):
    path_to_save = 'test_csv_save.csv'

    dataset = generate_random_dataset
    n_features = dataset.features.shape[1]
    dataset.save_to_csv(path_to_save)

    loaded_dataset = Dataset()
    loaded_dataset.load_from_csv(path_to_save, range(n_features), range(n_features, n_features + 1))

    assert np.all(np.isclose(loaded_dataset.features, dataset.features))
    assert np.all(np.isclose(loaded_dataset.labels, dataset.labels.reshape(-1, 1)))

def test_save_and_load_to_pickle_identical_file_multiple_labels():
    n_features = 5
    n_samples = 10
    n_labels = 3
    features = np.random.rand(n_samples, n_features)
    labels = np.random.rand(n_samples, n_labels)
    path_to_save = 'test_pickle_save.pkl'

    dataset = Dataset(features, labels)
    dataset.save_to_pickle(path_to_save)

    loaded_dataset = Dataset()
    loaded_dataset.load_from_pickle(path_to_save, range(n_features), range(n_features, n_features + n_labels))

    assert np.array_equal(loaded_dataset.features, dataset.features)
    assert np.array_equal(loaded_dataset.labels, dataset.labels)

def test_load_nonexisting_file_raises_filenotfound_error():
    path_to_file = 'nonexisting_file.pkl'
    dataset = Dataset()

    with pytest.raises(FileNotFoundError):
        dataset.load_from_pickle(path_to_file, range(1),range(1,2))

def test_load_with_too_many_label_columns_raises_value_error(saved_random_dataset):
    path, dataset = saved_random_dataset
    n_features = dataset.features.shape[1]
    wrong_n_labels = n_features + 3

    dataset = Dataset()
    with pytest.raises(ValueError):
        dataset.load_from_pickle(path, range(n_features), range(n_features, wrong_n_labels))

def test_load_with_too_many_feature_columns_raises_value_error(saved_random_dataset):
    path, dataset = saved_random_dataset
    n_features = dataset.features.shape[1]
    wrong_n_features = n_features + 3

    dataset = Dataset()
    with pytest.raises(ValueError):
        dataset.load_from_pickle(path, range(wrong_n_features), range(n_features, n_features + 1))

def test_compute_polynomial_features_does_not_change_n_samples(generate_random_dataset):
    dataset = generate_random_dataset
    old_n_samples = dataset.features.shape[0]
    dataset.compute_polynomial_features(poly_degree=2)
    new_n_samples = dataset.features.shape[0]

    assert old_n_samples == new_n_samples

def test_compute_polynomial_features_does_increase_n_features(generate_random_dataset):
    dataset = generate_random_dataset
    old_n_features = dataset.features.shape[1]
    dataset.compute_polynomial_features(poly_degree=2)
    new_n_features = dataset.features.shape[1]

    assert old_n_features < new_n_features

def test_compute_polynomial_features_with_empty_dataset_raises_value_error():
    dataset = Dataset()
    with pytest.raises(ValueError):
        dataset.compute_polynomial_features()

def test_compute_polynomial_features_with_wrong_poly_degree_type_raises_type_error(generate_random_dataset):
    dataset = generate_random_dataset
    poly_degree = 'a'
    with pytest.raises(TypeError):
        dataset.compute_polynomial_features(poly_degree=poly_degree)

def test_compute_polynomial_features_with_negative_poly_degree_raises_value_error(generate_random_dataset):
    dataset = generate_random_dataset
    poly_degree = -2
    with pytest.raises(ValueError):
        dataset.compute_polynomial_features(poly_degree=poly_degree)

def test_compute_polynomial_features_with_wrong_interaction_only_type_raises_type_error(generate_random_dataset):
    dataset = generate_random_dataset
    interaction_only = 'a'
    with pytest.raises(TypeError):
        dataset.compute_polynomial_features(interaction_only=interaction_only)

def test_remove_last_feature_raises_value_error():
    n_features = 1
    n_samples = 10
    features = np.random.rand(n_samples, n_features)
    labels = np.random.rand(n_samples)
    dataset = Dataset(features, labels)
    with pytest.raises(ValueError):
        dataset.remove_single_feature()

def test_remove_single_feature_removes_first_column(generate_random_dataset):
    dataset = generate_random_dataset
    old_columns = dataset.features[:, 1:]
    reduced_dataset = dataset.remove_single_feature()
    reduced_columns = reduced_dataset.features

    assert np.array_equal(old_columns, reduced_columns)

def test_add_bias_increases_number_of_columns_by_one(generate_random_dataset):
    dataset = generate_random_dataset
    old_n_features = dataset.features.shape[1]
    dataset.add_bias()
    new_n_features = dataset.features.shape[1]

    assert new_n_features == old_n_features + 1

def test_add_bias_creates_column_of_ones(generate_random_dataset):
    dataset = generate_random_dataset
    n_samples = dataset.features.shape[0]
    column_of_ones = np.ones(n_samples)
    dataset.add_bias()
    first_column = dataset.features[:, 0]

    assert np.array_equal(first_column, column_of_ones)

def test_add_bias_with_empty_dataset_raises_value_error():
    dataset = Dataset()
    with pytest.raises(ValueError):
        dataset.add_bias()

def test_normalize_does_not_change_size_of_features(generate_random_dataset):
    dataset = generate_random_dataset
    old_shape = dataset.features.shape
    dataset.normalize()
    new_shape = dataset.features.shape

    assert old_shape == new_shape

def test_normalize_does_not_change_size_of_labels(generate_random_dataset):
    dataset = generate_random_dataset
    old_shape = dataset.labels.shape
    dataset.normalize()
    new_shape = dataset.labels.shape

    assert old_shape == new_shape

def test_normalize_with_empty_dataset_raises_value_error():
    dataset = Dataset()
    with pytest.raises(ValueError):
        dataset.normalize()

def test_split_yields_correct_number_of_columns_in_train_set(generate_random_dataset):
    dataset = generate_random_dataset
    old_n_features =  dataset.features.shape[1]
    dataset_train, _ = dataset.split()
    new_n_features = dataset_train.features.shape[1]

    assert old_n_features == new_n_features

def test_split_yields_correct_number_of_samples_in_train_and_test_set(generate_random_dataset):
    dataset = generate_random_dataset
    old_n_samples =  dataset.features.shape[0]
    dataset_train, dataset_test = dataset.split()
    new_n_samples = dataset_train.features.shape[1] + dataset_test.features.shape[1]

    assert old_n_samples == new_n_samples

def test_split_with_empty_dataset_raises_value_error():
    dataset = Dataset()
    with pytest.raises(ValueError):
        dataset.split()

