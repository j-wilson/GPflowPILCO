#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from dataclasses import dataclass
from gpflow_pilco.moment_matching.core import *
from typing import Optional, Tuple, Union

# ---- Exports
__all__ = (
  "GaussianMoments",
  "GaussianMatch",
)

# ==============================================
#                                       gaussian
# ==============================================
class GaussianMoments(Moments):
  pass  # here for multiple dispatching


@dataclass
class GaussianMatch(MomentMatch):
  x: GaussianMoments
  y: GaussianMoments
  cross: Tuple[Union[tf.Tensor, tf.linalg.LinearOperator], bool]

  def cross_covariance(self,
                       dense: Optional[bool] = None,
                       preinv: bool = False):
    Sxy, is_preinv = self.cross
    if not preinv and is_preinv:
      Sxy = self.x.covariance() @ Sxy
    elif preinv and not is_preinv:
      Sxx = self.x.covariance()
      if isinstance(Sxx, tf.linalg.LinearOperator):
        Sxy = Sxx.solve(Sxy)
      elif isinstance(Sxy, tf.linalg.LinearOperator):
        Sxy = tf.linalg.cholesky_solve(tf.linalg.cholesky(Sxx), Sxy.to_dense())
      else:
        Sxy = tf.linalg.cholesky_solve(tf.linalg.cholesky(Sxx), Sxy)

    if dense and isinstance(Sxy, tf.linalg.LinearOperator):
      Sxy = Sxy.to_dense()

    return Sxy

  def joint(self) -> GaussianMoments:
    """
    Returns (a Gaussian approximation to) the joint distribution of `x` and `y`.
    """
    m = tf.concat([self.x.mean(), self.y.mean()], axis=-1)
    Sxx = self.x.covariance(dense=True)
    Sxy = self.cross_covariance(dense=True, preinv=False)
    Syy = self.y.covariance(dense=True)
    S = tf.concat([tf.concat([Sxx, Sxy], axis=-1),
                   tf.concat([tf.linalg.adjoint(Sxy), Syy], axis=-1)], axis=-2)
    return GaussianMoments(moments=(m, S), centered=True)


@dispatcher.register(GaussianMoments, Chain)
def _mm_gauss_chain(x: GaussianMoments, chain: Chain):
  """
  Linear approximation to the moments of Gaussian rvs
  pushed forward through a sequence of transformations.
  """
  state = x
  preinv = None
  cross_covariance = None  # may be premultiplied by Cov(x, x)^{-1}
  for i, op in enumerate(reversed(chain)):
    match = moment_matching(state, op)
    state = match.y
    if i:
      cross_covariance = cross_covariance @ match.cross_covariance(preinv=True)
    else:
      cross_covariance, preinv = match.cross

  return GaussianMatch(x=x, y=state, cross=(cross_covariance, preinv))
