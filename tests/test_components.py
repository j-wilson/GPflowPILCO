#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import logging
import tensorflow as tf

from gpflow_pilco.components import GaussianObjective, TrigonometricEncoder
from gpflow_pilco.moment_matching import GaussianMoments, moment_matching
from itertools import combinations_with_replacement
from random import randint
from scipy.spatial.distance import cdist
from tests.utils import *
from typing import *

from importlib import reload; reload(logging)
logger = logging.getLogger("test_components")


# ==============================================
#                                test_components
# ==============================================
class UnitTestConfigPILCO(UnitTestConfig):
  def __init__(self,
               seed: int,
               ndims_x: int,
               scale_x: float = 0.1,
               num_samples: int = int(1e6),
               **kwargs):
    super().__init__(seed=seed, num_samples=num_samples, **kwargs)
    self.ndims_x = ndims_x
    self.scale_x = scale_x


@prepare_test_env
def test_objective_gaussian(config: UnitTestConfigPILCO):
  # Mean and covariance of input $x \sim N(mx, Sxx)$
  mx = tf.random.normal([config.ndims_x], dtype=config.dtype)
  Sxx = generate_covariance(ndims=config.ndims_x, scale=config.scale_x)
  x = GaussianMoments(moments=(mx, Sxx), centered=True)

  # Mean and covariance of target $t \sim N(mt, Stt)$
  mt = mx + 0.1 * tf.random.normal([config.ndims_x], dtype=config.dtype)  # [!] improve me
  Stt = generate_covariance(ndims=config.ndims_x, scale=config.scale_x)
  iStt = tf.linalg.inv(Stt)

  # Cost function $cost(x)_= -exp(-0.5(x - u)^{T} W (x - u))$
  objective = GaussianObjective(target=mt, precis=iStt)

  # Monte carlo estimates of cost
  X = draw_samples_mvn(mx, Sxx, sample_shape=[config.num_samples])
  sample_losses = objective(X)

  dist2 = tf.squeeze(cdist(X, mt[None], metric='mahalanobis', VI=iStt), -1)**2
  test_costs = -tf.exp(-0.5 * dist2)
  assert allclose(sample_losses, test_costs, rtol=1e-12, atol=0.0)

  # Closed-form expression for expected cost
  expected_loss = objective(x)
  assert allclose(expected_loss,
                  tf.reduce_mean(test_costs),
                  rtol=config.rtol,
                  atol=config.atol)


@prepare_test_env
def test_encoder_trig(config: UnitTestConfigPILCO,
                      active_dims: Tuple[int] = None):

  if active_dims is None:
    active_dims = tuple(range(config.ndims_x//2, config.ndims_x))
  inactive_dims = tuple(set(range(config.ndims_x)) - set(active_dims))

  # Mean and covariance of input $x \sim N(mx, Sxx)$
  mx = tf.random.normal([config.ndims_x], dtype=config.dtype)
  Sxx = generate_covariance(ndims=config.ndims_x, scale=config.scale_x)
  x = GaussianMoments(moments=(mx, Sxx), centered=True)

  # Joint statistics of e = [encoder(a), b] where encoder(a) = [sin(a), cos(a)]
  encoder = TrigonometricEncoder(active_dims=active_dims)
  match = moment_matching(x, encoder)

  X = draw_samples_mvn(mx, Sxx, sample_shape=[config.num_samples])
  A = tf.gather(X, active_dims, axis=-1)
  B = tf.gather(X, inactive_dims, axis=-1)
  E = tf.concat([tf.math.sin(A), tf.math.cos(A), B], axis=-1)

  assert allclose(match.y.mean(),
                  tf.reduce_mean(E, axis=0),
                  rtol=config.rtol,
                  atol=config.atol)

  assert allclose(match.y.covariance(),
                  empirical_covariance(E, E),
                  rtol=config.rtol,
                  atol=config.atol)

  assert allclose(match.cross_covariance(),
                  empirical_covariance(X, E),
                  rtol=config.rtol,
                  atol=config.atol)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO,
                      datefmt='%Y-%m-%d %H:%M:%S',
                      format='%(asctime)-4s %(levelname)s:%(name)s:%(message)s')

  config = UnitTestConfigPILCO(seed=randint(0, 2 ** 31), ndims_x=2)
  logger.info(f'Test seed: {config.seed}')
  test_objective_gaussian(config)
  test_encoder_trig(config)
