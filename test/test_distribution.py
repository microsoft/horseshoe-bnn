import numpy as np
import torch
import pytest
import math
from scipy.stats import norm
from scipy.stats import bernoulli as bernoulli_sc
from horseshoe_bnn.distributions import Bernoulli, PredictiveDistribution, ReparametrizedGaussian, Gamma, InverseGamma, ScaleMixtureGaussian

@pytest.fixture
def generate_Bernoulli():
    return Bernoulli(0.6)

@pytest.fixture
def generate_InverseGamma():
    return InverseGamma(torch.Tensor([1]), torch.Tensor([1]))

@pytest.fixture
def generate_ScaleMixtureGaussian():
    return ScaleMixtureGaussian(torch.Tensor([1]), torch.Tensor([1]), torch.Tensor([1]))

@pytest.fixture
def generate_ReparametrizedGaussian():
    return ReparametrizedGaussian(torch.Tensor([1]), torch.Tensor([0]))

def test_ReparametrizedGaussian_logprop(generate_ReparametrizedGaussian):
    target = torch.Tensor([1])
    logprob = generate_ReparametrizedGaussian.logprob(target)
    logprob =  round(logprob.item(), 4)
    assert logprob == -0.5524


def test_ScaleMixtureGaussian_logprop(generate_ScaleMixtureGaussian):
    target = torch.Tensor([1])
    logprob = generate_ScaleMixtureGaussian.logprob(target)
    logprob =  round(logprob.item(), 4)
    assert logprob == -1.4189

def test_InverseGamma_logprob(generate_InverseGamma):
    target = torch.Tensor([1])
    logprob = generate_InverseGamma.logprob(target)
    logprob =  logprob.item()
    assert logprob == -1.

@pytest.mark.parametrize("target", [0, 1, 5])
def test_logprob_bernoulli_same_as_scipy(target):
    probability = 0.2

    # calculate logprob using custom bernoulli class
    bernoulli = Bernoulli(probability)
    logprob = bernoulli.logprob(target)

    # calculate logprob using scipy.stats.norm
    bernoulli_scipy = bernoulli_sc(probability)
    logprob_scipy = bernoulli_scipy.logpmf(target)

    assert math.isclose(logprob, logprob_scipy)

def test_Bernoulli_constructor_raises_TypeError_when_called_with_wrong_input_type():
    probability = 'prob'

    with pytest.raises(TypeError):
        Bernoulli(probability)

def test_Bernoulli_constructor_raises_ValueError_when_called_with_input_larger_one():
    probability = 2

    with pytest.raises(ValueError):
        Bernoulli(probability)

def test_Bernoulli_constructor_raises_ValueError_when_called_with_input_smaller_zero():
    probability = -1

    with pytest.raises(ValueError):
        Bernoulli(probability)

def test_Bernoulli_logprob_raises_TypeError_when_called_with_wrong_input_type(generate_Bernoulli):
    target = 'target'

    with pytest.raises(TypeError):
        generate_Bernoulli.logprob(target)

def test_Bernoulli_logprob_computes_inf_for_large_target(generate_Bernoulli):
    result = generate_Bernoulli.logprob(5)
    assert result == - np.inf

def test_Bernoulli_logprob_computes_correct_value_for_input_one(generate_Bernoulli):
    result = generate_Bernoulli.logprob(1)
    assert round(result, 4) == -0.5108

def test_Bernoulli_logprob_computes_correct_value_for_input_zero(generate_Bernoulli):
    result = generate_Bernoulli.logprob(0)
    assert round(result, 4) == -0.9163




