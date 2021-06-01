#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from typing import *
from functools import update_wrapper
from gpflow import config as gpflow_config

# ---- Exports
__all__ = (
  'UnitTestConfig',
  'prepare_test_env',
  'allclose',
  'draw_samples_mvn',
  'empirical_covariance',
  'generate_covariance',
)


# ==============================================
#                                          utils
# ==============================================
class UnitTestConfig:
  def __init__(self,
               seed: int,
               dtype: Any = None,
               jitter: float = None,
               num_samples: int = None,
               rtol: float = None,
               atol: float = 1e-8):

    if dtype is None:
      dtype = gpflow_config.default_float()

    if jitter is None:
      jitter = gpflow_config.default_jitter()

    if rtol is None:  # Gaussian approximation to Monte Carlo error
      rtol = 1e-5 if (num_samples is None) else 10 * (num_samples ** -0.5)

    self.seed = seed
    self.dtype = dtype
    self.jitter = jitter
    self.num_samples = num_samples
    self.rtol = rtol
    self.atol = atol


def prepare_test_env(test):
  def wrapped_test(config, **kwargs):
    # Common setup for unit tests
    tf.random.set_seed(config.seed)
    gpflow_config.set_default_float(config.dtype)
    gpflow_config.set_default_jitter(config.jitter)

    # Run the test
    return test(config, **kwargs)
  return update_wrapper(wrapped_test, test)


def allclose(a, b, rtol=1e-5, atol=1e-8):
  return tf.reduce_all(tf.abs(a - b) <= rtol + atol * tf.abs(b))


def draw_samples_mvn(mu, cov, sample_shape, sqrt=None):
  """
  Generate draws from multivariate normal distributions.
  """
  if sqrt is None:
    sqrt = tf.linalg.cholesky(cov)

  ndims = mu.shape[-1]

  rvs_shape = list(sample_shape) + list(cov.shape[:-2]) + [ndims]
  rvs = tf.random.normal(shape=rvs_shape, dtype=mu.dtype)
  return mu + tf.linalg.matvec(sqrt, rvs)


def empirical_covariance(a: tf.Tensor,
                         b: tf.Tensor = None,
                         center: bool = True):
  """
  Estimate the (cross-)covariance of a collection of samples.
  [!] Improve or deprecate me
  """
  _a = a - tf.reduce_mean(a, axis=0, keepdims=True) if center else a
  if b is None:
    _b = _a
  else:
    _b = b - tf.reduce_mean(b, axis=0, keepdims=True) if center else b
  return 1/(len(a) - 1) * tf.einsum('ni,nj->ij', _a, _b)


def generate_covariance(ndims: int,
                        sample_shape: List[int] = [],
                        scale: float = None,
                        dtype: Any = None):
  """
  Draw samples from a prior over covariance matrices.
  """
  if dtype is None:
    dtype = gpflow_config.default_float()

  eigen_vals = -tf.math.log(tf.random.uniform(list(sample_shape) + [1, ndims],
                                              dtype=dtype))

  orthog_mat = tf.linalg.svd(tf.random.normal(list(sample_shape) + [ndims, ndims],
                                              dtype=dtype), full_matrices=True)[1]

  sqrt_cov = tf.sqrt(eigen_vals) * orthog_mat
  cov = tf.matmul(sqrt_cov, sqrt_cov, transpose_b=True)
  if scale is not None:
    istd = tf.math.rsqrt(tf.linalg.diag_part(cov))
    cov = (scale ** 2) * cov * istd[..., None] * istd[..., None, :]

  return cov

