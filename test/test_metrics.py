import pytest
import os
import sys
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from horseshoe_bnn.metrics import AllMetrics, Metric, MeanAbsoluteError, RootMeanSquaredError, PredictiveLogLikelihood, CalibrationPlot, ZeroOneLoss, F1Score, GlobalF1Score
from horseshoe_bnn.aggregation_result import AggregationResult
import horseshoe_bnn.distributions as distr

@pytest.fixture
def generate_random_labels_and_predictions_and_predictive_distribution():
    np.random.seed(1)
    ground_truth = np.random.rand(30)
    predicted_targets = np.random.rand(30)
    distributions = [distr.ReparametrizedGaussian(target, 0.) for target in predicted_targets]
    predictiveDistribution = distr.PredictiveDistribution(distributions)
    return ground_truth, predicted_targets, predictiveDistribution

@pytest.fixture
def generate_positive_ground_truth_predicted_targets():
    ground_truth = np.ones(4)
    predicted_targets = np.array([1., 2., 3., 4.])
    distributions = [distr.ReparametrizedGaussian(y, 0.) for y in predicted_targets]
    predictiveDistribution = distr.PredictiveDistribution(distributions)
    return ground_truth, predictiveDistribution

@pytest.fixture
def generate_negative_ground_truth_predicted_targets():
    ground_truth = -np.ones(4)
    predicted_targets = np.array([-1., -2., -3., -4.])
    distributions = [distr.ReparametrizedGaussian(y, 0.) for y in predicted_targets]
    predictiveDistribution = distr.PredictiveDistribution(distributions)
    return ground_truth, predictiveDistribution

@pytest.fixture
def generate_incompatible_shape_ground_truth_predicted_targets():
    ground_truth = np.array([-1., -2., -3., -4.])
    predicted_targets = ground_truth[:-1]
    distributions = [distr.ReparametrizedGaussian(y, 0.) for y in predicted_targets]
    predictiveDistribution = distr.PredictiveDistribution(distributions)
    return ground_truth, predictiveDistribution

@pytest.fixture
def generate_identical_ground_truth_predicted_targets():
    ground_truth = np.array([-1., -2., -3., -4.])
    predicted_targets = ground_truth
    distributions = [distr.ReparametrizedGaussian(y, 0.) for y in predicted_targets]
    predictiveDistribution = distr.PredictiveDistribution(distributions)
    return ground_truth, predictiveDistribution

def test_base_class_raises_TypeError_when_instantiated():
    with pytest.raises(TypeError):
        m = Metric()

def test_aggregate_raises_TypeError_when_called_with_arguments_of_wrong_type():
    m = MeanAbsoluteError()
    with pytest.raises(TypeError):
        m.aggregate('a')

def test_aggregate_raises_ValueError_when_called_with_empty_list():
    m = MeanAbsoluteError()
    with pytest.raises(ValueError):
        m.aggregate([])

def test_aggregate_yields_correct_output_type():
    m = MeanAbsoluteError()
    temp = [1., 2., 3., 4.]
    assert isinstance(m.aggregate(temp), AggregationResult)

def test_base_aggregate_yields_correct_output_values():
    m = MeanAbsoluteError()
    temp = [1., 2., 3., 4.]
    result = m.aggregate(temp)
    assert result.mean == 2.5
    assert round(result.std, 4) == 1.1180





def test_MAE_compute_raises_TypeError_when_called_with_ground_truths_of_wrong_type(generate_positive_ground_truth_predicted_targets):
    m = MeanAbsoluteError()
    _, preditive_distribution = generate_positive_ground_truth_predicted_targets
    with pytest.raises(TypeError):
        m.compute('a', preditive_distribution)

def test_RMSE_compute_raises_TypeError_when_called_with_ground_truths_of_wrong_type(generate_positive_ground_truth_predicted_targets):
    m = RootMeanSquaredError()
    _, preditive_distribution = generate_positive_ground_truth_predicted_targets
    with pytest.raises(TypeError):
        m.compute('a', preditive_distribution)

def test_loglikelihood_compute_raises_TypeError_when_called_with_ground_truths_of_wrong_type(generate_positive_ground_truth_predicted_targets):
    m = PredictiveLogLikelihood()
    _, preditive_distribution = generate_positive_ground_truth_predicted_targets
    with pytest.raises(TypeError):
        m.compute('a', preditive_distribution)

def test_calibrationPlot_compute_raises_TypeError_when_called_with_ground_truths_of_wrong_type(generate_positive_ground_truth_predicted_targets):
    m = CalibrationPlot()
    _, preditive_distribution = generate_positive_ground_truth_predicted_targets
    with pytest.raises(TypeError):
        m.compute('a', preditive_distribution)


def test_MAE_compute_raises_TypeError_when_called_with_predictiveDistribution_of_wrong_type(generate_positive_ground_truth_predicted_targets):
    m = MeanAbsoluteError()
    ground_truth, _ = generate_positive_ground_truth_predicted_targets
    with pytest.raises(TypeError):
        m.compute(ground_truth, 'a')

def test_RMSE_compute_raises_TypeError_when_called_with_predictiveDistribution_of_wrong_type(generate_positive_ground_truth_predicted_targets):
    m = RootMeanSquaredError()
    ground_truth, _ = generate_positive_ground_truth_predicted_targets
    with pytest.raises(TypeError):
        m.compute(ground_truth, 'a')

def test_LogLikelihood_compute_raises_TypeError_when_called_with_predictiveDistribution_of_wrong_type(generate_positive_ground_truth_predicted_targets):
    m = PredictiveLogLikelihood()
    ground_truth, _ = generate_positive_ground_truth_predicted_targets
    with pytest.raises(TypeError):
        m.compute(ground_truth, 'a')

def test_CalibrationPlot_compute_raises_TypeError_when_called_with_predictiveDistribution_of_wrong_type(generate_positive_ground_truth_predicted_targets):
    m = CalibrationPlot()
    ground_truth, _ = generate_positive_ground_truth_predicted_targets
    with pytest.raises(TypeError):
        m.compute(ground_truth, 'a')

def test_rmse_compute_with_correct_inputs(generate_positive_ground_truth_predicted_targets):
    ground_truth, predictiveDistribution = generate_positive_ground_truth_predicted_targets
    assert AllMetrics.rmse.compute(ground_truth, predictiveDistribution) == 1.8708286933869707

def test_mae_compute_with_correct_inputs(generate_positive_ground_truth_predicted_targets):
    ground_truth, predictiveDistribution = generate_positive_ground_truth_predicted_targets
    assert AllMetrics.mae.compute(ground_truth, predictiveDistribution) == 1.5

# def test_CalibrationPlot_compute_returns_correct_output_type(generate_random_labels_and_predictions_and_predictive_distribution):
#     c = CalibrationPlot()
#     ground_truth, _ , predictiveDistribution = \
#                             generate_random_labels_and_predictions_and_predictive_distribution

#     result = c.compute(ground_truth, predictiveDistribution)
#     assert isinstance(result, dict)

def test_CalibrationPlot_aggregate_yields_correct_outputs():
    metric = CalibrationPlot()
    temp = [{'var': np.array([1., 2., ]), 'err': np.array([1., 1.])}, {'var': np.array([1., 2.,]), 'err': np.array([1., 1.])}]
    result = metric.aggregate(temp)

    assert np.array_equal(result.variances, np.array([2., 2., 1., 1.]))
    assert np.array_equal(result.errors, np.array([1., 1., 1., 1.]))


def test_mae_raises_ValueError_with_wrong_shapes(generate_incompatible_shape_ground_truth_predicted_targets):
    ground_truth, predictiveDistribution = generate_incompatible_shape_ground_truth_predicted_targets

    with pytest.raises(ValueError):
        AllMetrics.mae.compute(ground_truth, predictiveDistribution)

def test_rmse_raises_ValueError_with_wrong_shapes(generate_incompatible_shape_ground_truth_predicted_targets):
    ground_truth, predictiveDistribution = generate_incompatible_shape_ground_truth_predicted_targets

    with pytest.raises(ValueError):
        AllMetrics.rmse.compute(ground_truth, predictiveDistribution)

def test_mae_gives_zero_for_identical_input_and_targets(generate_identical_ground_truth_predicted_targets):
    ground_truth, predictiveDistribution = generate_identical_ground_truth_predicted_targets

    assert AllMetrics.mae.compute(ground_truth, predictiveDistribution) == 0

def test_rmse_gives_zero_for_identical_input_and_targets(generate_identical_ground_truth_predicted_targets):
    ground_truth, predictiveDistribution = generate_identical_ground_truth_predicted_targets

    assert AllMetrics.rmse.compute(ground_truth, predictiveDistribution) == 0

def test_mae_gives_positive_value_for_negative_inputs_and_targets(generate_negative_ground_truth_predicted_targets):
    ground_truth, PredictiveDistribution = generate_negative_ground_truth_predicted_targets
    assert AllMetrics.mae.compute(ground_truth, PredictiveDistribution) > 0

def test_rmse_gives_positive_value_for_negative_inputs_and_targets(generate_negative_ground_truth_predicted_targets):
    ground_truth, PredictiveDistribution = generate_negative_ground_truth_predicted_targets
    assert AllMetrics.rmse.compute(ground_truth, PredictiveDistribution) > 0

def test_mae_same_as_sklearn(generate_random_labels_and_predictions_and_predictive_distribution):
    ground_truth, predicted_targets, predictiveDistribution = \
                                    generate_random_labels_and_predictions_and_predictive_distribution
    assert AllMetrics.mae.compute(ground_truth, predictiveDistribution) == mean_absolute_error(ground_truth, predicted_targets)

def test_rmse_same_as_sklearn(generate_random_labels_and_predictions_and_predictive_distribution):
    ground_truth, predicted_targets, predictiveDistribution = \
                                    generate_random_labels_and_predictions_and_predictive_distribution
    assert AllMetrics.rmse.compute(ground_truth, predictiveDistribution) == np.sqrt(mean_squared_error(ground_truth, predicted_targets))

def test_f1_compute_with_correct_inputs(generate_positive_ground_truth_predicted_targets):
    ground_truth, predictiveDistribution = generate_positive_ground_truth_predicted_targets
    assert AllMetrics.f1_score.compute(ground_truth, predictiveDistribution) == 0.4

def test_global_f1_compute_with_correct_inputs(generate_positive_ground_truth_predicted_targets):
    ground_truth, predictiveDistribution = generate_positive_ground_truth_predicted_targets
    assert AllMetrics.global_f1_score.compute(ground_truth, predictiveDistribution) == 0.25

def test_zero_one_loss_compute_with_correct_inputs(generate_positive_ground_truth_predicted_targets):
    ground_truth, predictiveDistribution = generate_positive_ground_truth_predicted_targets
    assert AllMetrics.zero_one_loss.compute(ground_truth, predictiveDistribution) == 0.75
