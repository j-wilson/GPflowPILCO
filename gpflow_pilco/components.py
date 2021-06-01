#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf
from gpflow.config import default_float
from gpflow_pilco.moment_matching import GaussianMoments
from gpflow_pilco.moment_matching.maths import sincos
from typing import Callable, Union, Tuple

# ---- Exports
__all__ = ('GaussianObjective', "Encoder", "TrigonometricEncoder")


# ==============================================
#                                     components
# ==============================================
class GaussianObjective:
  def __init__(self, target: tf.Tensor, precis: tf.Tensor):
    self.target = target
    self.precis = precis

  def __call__(self, x: Union[tf.Tensor, GaussianMoments], t: tf.Tensor = None):
    """
    Compute the Gaussian objective: $-exp(-0.5 * (x - x*)^{T} W (x - x*))$.
    """
    if isinstance(x, GaussianMoments):
      # Expected value: $E_{x ~ p(x)}[-exp(-0.5 * (x - x*)^{T} W (x - x*))]$
      I = tf.eye(self.precis.shape[-1], dtype=default_float())
      IpSW = I + x.covariance() @ self.precis
      iSpW = tf.matmul(self.precis, tf.linalg.inv(IpSW))
      err = x.mean() - self.target  # NxD
      dist2 = tf.reduce_sum(err * tf.linalg.matvec(iSpW, err), -1)
      return -tf.math.rsqrt(tf.linalg.det(IpSW)) * tf.exp(-0.5 * dist2)  # N

    err = x - self.target
    dist2 = tf.reduce_sum(err * tf.linalg.matvec(self.precis, err), -1)
    return -tf.exp(-0.5 * dist2)


class Encoder:
  def __init__(self, transform: Callable, active_dims: Tuple[int]):
    self._transform = transform
    self.active_dims = active_dims

  def __call__(self, x: tf.Tensor, append_inactive: bool = True) -> tf.Tensor:
    active, inactive = self.get_partition_indices(ndims=x.shape[-1])

    x_active = tf.gather(x, active, axis=-1)
    retvals = self.transform(x_active)
    if append_inactive and len(inactive):
      retvals = tf.concat([retvals, tf.gather(x, inactive, axis=-1)], axis=-1)
    return retvals

  def get_partition_indices(self, ndims: int) -> Tuple[Tuple[int], Tuple[int]]:
    """
    Return indices for active and inactive splits of input features.
    """
    indices_x = tuple(range(ndims))
    indices_a = tuple(indices_x[dim] for dim in self.active_dims)
    set_a = set(indices_a)
    assert len(indices_a) == len(set_a)  # enforce uniqueness
    return indices_a, tuple(set(indices_x) - set_a)

  @property
  def transform(self):
    return self._transform


class TrigonometricEncoder(Encoder):
  def __init__(self, active_dims: Tuple[int]):
    super().__init__(transform=sincos, active_dims=active_dims)
