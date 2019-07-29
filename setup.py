import setuptools

NAME = 'horseshoe_bnn'
VERSION = '1.0'
AUTHOR = 'Anna-Lena Popkes, Hiske Overweg'
DESCRIPTION = 'Horseshoe BNN model for performing feature selection'
PACKAGES = setuptools.find_packages()
LICENSE = 'MICROSOFT RESEARCH LICENSE'
LONG_DESCRIPTION = 'This package includes code for training and evaluating Baysian models for performing '\
                   'feature selection. Four models are included: two (linear and non-linear) using '\
                   'Gaussian prior distributions and two (linear and non-linear) using Horseshoe prior. '\
                   'distributions. The horseshoe distribution introduces sparsity in the latter two models '\
                   'and allows for performing feature selection. '

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    packages=PACKAGES,
    license=LICENSE,
    long_description=LONG_DESCRIPTION,
)
