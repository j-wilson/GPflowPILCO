#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from gpflow_pilco.moment_matching.core import *
from gpflow_pilco.moment_matching.gaussian import *
from gpflow_pilco.utils.bvn import bvn, ndtr
from math import pi
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.math.special import owens_t


# ==============================================
#                                      bijectors
# ==============================================
@dispatcher.register(Moments, tfb.Chain)
def _mm_chain(x: Moments, bijector: tfb.Chain, /, **kwargs):
  chain = Chain(*bijector.bijectors)
  return moment_matching(x, chain, **kwargs)


@dispatcher.register(Moments, tfb.Shift)
def _mm_shift(x: Moments, bijector: tfb.Shift, /, **kwargs):
  return moment_matching(x, tf.math.add, bijector.shift, **kwargs)


@dispatcher.register(Moments, tfb.Scale)
def _mm_scale(x: Moments, bijector: tfb.Scale, /, **kwargs):
  return moment_matching(x, tf.math.multiply, bijector.scale, **kwargs)


@dispatcher.register(GaussianMoments, tfb.NormalCDF)
def _mm_gauss_ndtr(x: GaussianMoments, _):
  """
  Let $zi, zj ~ N(0, I)$. Then,

    E[Phi(xi) Phi(xj)] = E[1_{zi <= xi} 1_{zj <= xj}]
                       = P(wi <= 0, wj <= 0)
                       = BVN(-inf, 0, -inf, 0, Corr[wi, wj])

  where $wi = zi - xi$ and $wj = zj - xj$.
  """
  x1 = x.mean()
  Sxx = x.covariance(dense=True)

  vx = tf.linalg.diag_part(Sxx)
  vw = vx + 1  # variance of $w = z - x$
  isq_vw = tf.math.rsqrt(vw)
  isqvw_x1 = isq_vw * x1

  y1 = ndtr(isqvw_x1)
  if x.ndim == 1:
    y2 = y1 - 2 * owens_t(isqvw_x1, tf.math.rsqrt(1 + 2 * vx))
  else:  # lower should be -float("inf"), but gradients become unstable
    lower = tf.fill([x.ndim, x.ndim], tf.cast(-9.0, vx.dtype))
    upper = tf.stack(x.ndim * [isqvw_x1], axis=-1)
    rho = Sxx * tf.expand_dims(isq_vw, -1) * tf.expand_dims(isq_vw, -2)
    y2 = bvn(lower, upper, lower, tf.linalg.adjoint(upper), rho)

  vxy = isq_vw * vx * ((2 * pi) ** -0.5) * tf.exp(-0.5 * tf.square(isqvw_x1))
  iSxx_Sxy = tf.linalg.LinearOperatorDiag(diag=vxy/vx)

  y = GaussianMoments(moments=(y1, y2), centered=False)
  return GaussianMatch(x=x, y=y, cross=(iSxx_Sxy, True))
