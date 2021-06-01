#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import logging
import tensorflow as tf
import gpflow

from gpflow_pilco.utils.kernel_expectation import kernel_expectation
from math import log
from random import randint
from tests.utils import *
from typing import *

from importlib import reload; reload(logging)
logger = logging.getLogger('test_kernel_expectation')


# ==============================================
#                        test_kernel_expectation
# ==============================================
class UnitTestConfigExpectation(UnitTestConfig):
  def __init__(self,
               seed: int,
               ndims_x: int,
               scale_x: float = 0.10,
               scale_f: float = 0.89,  # large values should break Monte Carlo tests
               num_samples: int = int(1e6),
               num_inducing: int = 32,
               active_dims: Tuple[int] = None,
               anisotropic: bool = True,
               lengthscale_bounds: Tuple[float] = (0.1, 10.0),
               **kwargs):
    super().__init__(seed=seed, num_samples=num_samples, **kwargs)
    if active_dims is None:
      active_dims = tuple(range(ndims_x))

    self.ndims_x = ndims_x
    self.scale_x = scale_x
    self.scale_f = scale_f
    self.num_inducing = num_inducing
    self.active_dims = active_dims
    self.anisotropic = anisotropic
    self.lengthscale_bounds = lengthscale_bounds


@prepare_test_env
def test_expectation_squaredExp(config: UnitTestConfigExpectation):
  def get_kernel_and_inducing(mx, Sxx):
    log_ls = tf.random.uniform([len(config.active_dims)] if config.anisotropic else [],
                               *(log(bound) for bound in config.lengthscale_bounds),
                               dtype=config.dtype)

    kernel = gpflow.kernels.SquaredExponential(variance=config.scale_f ** 2,
                                             active_dims=config.active_dims,
                                             lengthscales=tf.exp(log_ls))

    # Sample some points near the mode of $p(x)$ and others uniformly. This
    # helps prevent the test trivial passing because values became too small.
    Z1 = draw_samples_mvn(mx, 0.1 * Sxx, sample_shape=[config.num_inducing//2])
    Z2 = tf.random.uniform([config.num_inducing - len(Z1), config.ndims_x],
                            dtype=Z1.dtype)
    return kernel, tf.concat([Z1, Z2], axis=0)

  # Mean and covariance of input $x \sim N(mx, Sxx)$
  mx = tf.random.normal([config.ndims_x], dtype=config.dtype)
  Sxx = generate_covariance(ndims=config.ndims_x, scale=config.scale_x)

  # Generate squared exponential kernels and inducing points
  K2, A_raw = get_kernel_and_inducing(mx, Sxx)
  K3, B_raw = get_kernel_and_inducing(mx, Sxx)

  # Compute closed-form expressions for kernel expectations
  A, B = map(gpflow.inducing_variables.InducingPoints, (A_raw, B_raw))
  px = gpflow.probability_distributions.Gaussian(mu=mx[None], cov=Sxx[None])
  eK2_ax = tf.linalg.adjoint(kernel_expectation(px, (K2, A)))
  eK3_bx = tf.linalg.adjoint(kernel_expectation(px, (K3, B)))
  eK6_axxb = tf.squeeze(kernel_expectation(px, (K2, A), (K3, B)))

  # Monte Carlo estimates to kernel expectations
  X = draw_samples_mvn(mx, Sxx, sample_shape=[config.num_samples])
  _K2_ax = K2(A_raw, X)
  _K3_bx = K3(B_raw, X)
  _eK2_ax = tf.reduce_mean(_K2_ax, axis=-1, keepdims=True)
  _eK3_bx = tf.reduce_mean(_K3_bx, axis=-1, keepdims=True)
  _eK6_axxb = (1/X.shape[0]) * tf.matmul(_K2_ax, _K3_bx, transpose_b=True)

  assert allclose(eK2_ax, _eK2_ax, rtol=config.rtol, atol=config.atol)
  assert allclose(eK3_bx, _eK3_bx, rtol=config.rtol, atol=config.atol)
  assert allclose(eK6_axxb, _eK6_axxb, rtol=config.rtol, atol=config.atol)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO,
                      datefmt='%Y-%m-%d %H:%M:%S',
                      format='%(asctime)-4s %(levelname)s:%(name)s:%(message)s')

  config = UnitTestConfigExpectation(seed=randint(0, 2 ** 31), ndims_x=2)
  logger.info(f'Test seed: {config.seed}')
  test_expectation_squaredExp(config)
