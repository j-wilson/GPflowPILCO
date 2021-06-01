#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import logging
import tensorflow as tf
import gpflow

from gpflow_pilco.moment_matching import GaussianMoments, moment_matching
from math import log
from random import randint
from tests.utils import *
from typing import *

from importlib import reload; reload(logging)
logger = logging.getLogger('test_expectations')


# ==============================================
#                           test_moment_matching
# ==============================================
class UnitTestConfigMomentMatching(UnitTestConfig):
  def __init__(self,
               seed: int,
               ndims_x: int,
               scale_x: float = 0.01,
               scale_f: float = 0.89,  # large values should break Monte Carlo tests
               num_cond: int = 16,
               num_eval: int = 2,
               active_dims: Tuple[int] = None,
               anisotropic: bool = True,
               lengthscale_bounds: Tuple[float] = (0.01, 10.0),
               min_lengthscale: float = 0.01,
               max_lengthscale: float = 10.0,
               num_samples: int = int(1e6),
               **kwargs):

    if active_dims is None:
      active_dims = tuple(range(ndims_x))

    super().__init__(seed=seed, num_samples=num_samples, **kwargs)
    self.ndims_x = ndims_x
    self.scale_f = scale_f
    self.scale_x = scale_x
    self.num_cond = num_cond
    self.num_eval = num_eval
    self.active_dims = active_dims
    self.anisotropic = anisotropic
    self.lengthscale_bounds = lengthscale_bounds
    self.min_lengthscale = min_lengthscale
    self.max_lengthscale = max_lengthscale


def monte_carlo_estimator(model,
                          mx,
                          Sxx,
                          num_samples=int(1e6),
                          active_dims=None):
  X = draw_samples_mvn(mx, Sxx, [num_samples])
  F_mu, F_sigma2 = model.predict_f(tf.reshape(X, [-1, mx.shape[-1]]),
                                   full_cov=False,
                                   full_output_cov=True)

  ndims_f = F_mu.shape[-1]
  F_mu = tf.reshape(F_mu, list(X.shape[:-1]) + [ndims_f])
  F_sigma2 = tf.reshape(F_sigma2, list(X.shape[:-1]) + [ndims_f, ndims_f])
  mf = tf.reduce_mean(F_mu, 0)

  dF_mu = F_mu - mf
  Sff = (1/num_samples) * tf.einsum('sni,snj->nij', dF_mu, dF_mu)\
            + tf.reduce_mean(F_sigma2, 0)

  if active_dims is None:
    A, a_mu = X, mx
  else:
    A, a_mu = map(lambda arr: tf.gather(arr, active_dims, axis=-1), (X, mx))

  Saf = (1/num_samples) * tf.einsum('sni,snj->nij', A, F_mu)\
        - tf.expand_dims(a_mu, -1) * tf.expand_dims(mf, -2)

  return mf, Sff, Saf


@prepare_test_env
def test_moment_matching_gpr(config):
  log_ls = tf.random.uniform([len(config.active_dims)] if config.anisotropic else [],
                             *(log(bound) for bound in config.lengthscale_bounds),
                             dtype=config.dtype)

  kernel = gpflow.kernels.SquaredExponential(variance=config.scale_f ** 2,
                                             active_dims=config.active_dims,
                                             lengthscales=tf.exp(log_ls))

  f_mu = 1 + tf.random.normal([1], dtype=config.dtype)
  mean_function = gpflow.mean_functions.Constant(c=f_mu)

  X = tf.random.uniform([config.num_cond, config.ndims_x], dtype=config.dtype)
  Y = config.scale_f * tf.random.normal([config.num_cond, 1], dtype=config.dtype)

  model = gpflow.models.GPR(data=(X, Y),
                            kernel=kernel,
                            mean_function=mean_function,
                            noise_variance=1e-5)

  mx = tf.random.uniform([config.num_eval, config.ndims_x], dtype=config.dtype)
  Sxx = generate_covariance(ndims=config.ndims_x,
                            scale=config.scale_x,
                            sample_shape=[config.num_eval])

  _mf, _Sff, _Sxf = monte_carlo_estimator(model=model,
                                          mx=mx,
                                          Sxx=Sxx,
                                          active_dims=config.active_dims,
                                          num_samples=config.num_samples)

  x = GaussianMoments(moments=(mx, Sxx), centered=True)
  match_full = moment_matching(x, model)
  assert allclose(match_full.y.mean(), _mf, rtol=config.rtol, atol=config.atol)
  assert allclose(match_full.y.covariance(), _Sff, rtol=config.rtol,
                  atol=config.atol)
  assert allclose(match_full.cross_covariance(), _Sxf, rtol=config.rtol,
                  atol=config.atol)

  match_diag = moment_matching(x, model, full_output_cov=False)
  assert allclose(match_diag.y.mean(), match_full.y.mean(), rtol=1e-12, atol=0)
  assert allclose(tf.linalg.diag_part(match_diag.y.covariance()),
                  tf.linalg.diag_part(match_full.y.covariance()),
                  rtol=1e-12,
                  atol=0)
  assert allclose(match_diag.cross_covariance(),
                  match_full.cross_covariance(),
                  rtol=1e-12,
                  atol=0)


@prepare_test_env
def test_moment_matching_svgp(config):
  log_ls = tf.random.uniform([len(config.active_dims)] if config.anisotropic else [],
                             *(log(bound) for bound in config.lengthscale_bounds),
                             dtype=config.dtype)

  kernel = gpflow.kernels.SquaredExponential(variance=config.scale_f ** 2,
                                             active_dims=config.active_dims,
                                             lengthscales=tf.exp(log_ls))

  Z = tf.random.uniform([config.num_cond, config.ndims_x], dtype=config.dtype)
  points = gpflow.inducing_variables.InducingPoints(Z)

  q_mu = config.scale_f * tf.random.normal([config.num_cond, 1], dtype=config.dtype)
  q_cov = generate_covariance(ndims=config.num_cond, scale=config.scale_f)[None]

  f_mu = 1 + tf.random.normal([1], dtype=config.dtype)
  mean_function = gpflow.mean_functions.Constant(c=f_mu)

  model = gpflow.models.SVGP(kernel=kernel,
                             q_mu=q_mu,
                             q_sqrt=tf.linalg.cholesky(q_cov),
                             inducing_variable=points,
                             likelihood=gpflow.likelihoods.Gaussian(variance=1e-5),
                             mean_function=mean_function,
                             whiten=False)

  mx = tf.random.uniform([config.num_eval, config.ndims_x], dtype=config.dtype)
  Sxx = generate_covariance(ndims=config.ndims_x,
                              scale=config.scale_x,
                              sample_shape=[config.num_eval])

  _mf, _Sff, _Sxf = monte_carlo_estimator(model=model,
                                          mx=mx,
                                          Sxx=Sxx,
                                          active_dims=config.active_dims,
                                          num_samples=config.num_samples)

  x = GaussianMoments(moments=(mx, Sxx), centered=True)
  match_full = moment_matching(x, model)
  assert allclose(match_full.y.mean(), _mf, rtol=config.rtol, atol=config.atol)
  assert allclose(match_full.y.covariance(), _Sff, rtol=config.rtol,
                  atol=config.atol)
  assert allclose(match_full.cross_covariance(), _Sxf, rtol=config.rtol,
                  atol=config.atol)

  match_diag = moment_matching(x, model, full_output_cov=False)
  assert allclose(match_diag.y.mean(), match_full.y.mean(), rtol=1e-12, atol=0)
  assert allclose(tf.linalg.diag_part(match_diag.y.covariance()),
                  tf.linalg.diag_part(match_full.y.covariance()),
                  rtol=1e-12,
                  atol=0)
  assert allclose(match_diag.cross_covariance(),
                  match_full.cross_covariance(),
                  rtol=1e-12,
                  atol=0)



@prepare_test_env
def test_moment_matching_svgp_mo(config, ndims_f=2, ndims_y=3):
  inducing_vars = []
  latent_kernels = []
  for _ in range(ndims_f):
    Z = tf.random.uniform([config.num_cond, config.ndims_x], dtype=config.dtype)
    log_ls = tf.random.uniform([len(config.active_dims)] if config.anisotropic else [],
                               *(log(bound) for bound in config.lengthscale_bounds),
                               dtype=config.dtype)

    k = gpflow.kernels.SquaredExponential(variance=config.scale_f ** 2,
                                          active_dims=config.active_dims,
                                          lengthscales=tf.exp(log_ls))

    inducing_vars.append(gpflow.inducing_variables.InducingPoints(Z))
    latent_kernels.append(k)

  W = tf.math.l2_normalize(tf.random.uniform([ndims_y, ndims_f], dtype=config.dtype), axis=-1)
  kernel = gpflow.kernels.LinearCoregionalization(kernels=latent_kernels, W=W)
  inducing_var = gpflow.inducing_variables.SeparateIndependentInducingVariables(inducing_vars)

  f_mu = 1 + tf.random.normal([ndims_y], dtype=config.dtype)
  mean_function = gpflow.mean_functions.Constant(c=f_mu)

  q_mu = config.scale_f * tf.random.normal([config.num_cond, ndims_f], dtype=config.dtype)
  q_cov = generate_covariance(ndims=config.num_cond,
                              scale=config.scale_f,
                              sample_shape=[ndims_f])

  model = gpflow.models.SVGP(kernel=kernel,
                             q_mu=q_mu,
                             q_sqrt=tf.linalg.cholesky(q_cov),
                             num_latent_gps=kernel.num_latent_gps,
                             inducing_variable=inducing_var,
                             likelihood=gpflow.likelihoods.Gaussian(variance=1e-5),
                             mean_function=mean_function,
                             whiten=False)

  mx = tf.random.uniform([config.num_eval, config.ndims_x], dtype=config.dtype)
  Sxx = generate_covariance(ndims=config.ndims_x,
                            scale=config.scale_x,
                            sample_shape=[config.num_eval])

  _mf, _Sff, _Sxf = monte_carlo_estimator(model=model,
                                          mx=mx,
                                          Sxx=Sxx,
                                          active_dims=config.active_dims,
                                          num_samples=config.num_samples)

  x = GaussianMoments(moments=(mx, Sxx), centered=True)
  match_full = moment_matching(x, model)
  assert allclose(match_full.y.mean(), _mf, rtol=config.rtol, atol=config.atol)
  assert allclose(match_full.y.covariance(), _Sff, rtol=config.rtol,
                  atol=config.atol)
  assert allclose(match_full.cross_covariance(), _Sxf, rtol=config.rtol,
                  atol=config.atol)

  match_diag = moment_matching(x, model, full_output_cov=False)
  assert allclose(match_diag.y.mean(), match_full.y.mean(), rtol=1e-12, atol=0)
  assert allclose(tf.linalg.diag_part(match_diag.y.covariance()),
                  tf.linalg.diag_part(match_full.y.covariance()),
                  rtol=1e-12,
                  atol=0)
  assert allclose(match_diag.cross_covariance(),
                  match_full.cross_covariance(),
                  rtol=1e-12,
                  atol=0)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO,
                      datefmt='%Y-%m-%d %H:%M:%S',
                      format='%(asctime)-4s %(levelname)s:%(name)s:%(message)s')

  config = UnitTestConfigMomentMatching(seed=randint(0, 2 ** 31), ndims_x=4)
  logger.info(f'Test seed: {config.seed}')
  test_moment_matching_gpr(config)
  test_moment_matching_svgp(config)
  test_moment_matching_svgp_mo(config)
