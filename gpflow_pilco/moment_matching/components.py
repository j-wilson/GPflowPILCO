#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from gpflow_pilco.components import Encoder
from gpflow_pilco.moment_matching import (dispatcher,
                                          GaussianMatch,
                                          GaussianMoments,
                                          moment_matching)

# ==============================================
#                                     components
# ==============================================
@dispatcher.register(GaussianMoments, Encoder)
def _mm_gauss_encoder(x: GaussianMoments,
                      encoder: Encoder,
                      append_inactive: bool = True) -> GaussianMatch:

  # Extract moments for (active features of) x
  x1 = x.mean()
  active, inactive = encoder.get_partition_indices(ndims=x1.shape[-1])
  a1 = tf.gather(x1, active, axis=-1)
  Sxx = x.covariance(dense=True)
  Sxa = tf.gather(Sxx, active, axis=-1)
  Saa = tf.gather(Sxa, active, axis=-2)
  moments_a = GaussianMoments(moments=(a1, Saa), centered=True)

  # Moment match transformation of active features
  match_part = moment_matching(moments_a, encoder.transform)

  moments_y = match_part.y
  iSaa_Say = match_part.cross_covariance(preinv=True)
  Sxy = tf.matmul(Sxa, iSaa_Say)

  # Augment partial match to incl. inactive features
  if append_inactive:
    b1 = tf.gather(x1, inactive, axis=-1)
    y1 = tf.concat([moments_y.mean(), b1], -1)

    Sxb = tf.gather(Sxx, inactive, axis=-1)
    Sbb = tf.gather(Sxb, inactive, axis=-2)
    Sby = tf.gather(Sxy, inactive, axis=-2)
    Syy = moments_y.covariance(dense=True)
    Syy = tf.concat([tf.concat([Syy, tf.linalg.adjoint(Sby)], axis=-1),
                     tf.concat([Sby, Sbb], axis=-1)], axis=-2)

    moments_y = GaussianMoments(moments=(y1, Syy), centered=True)
    Sxy = tf.concat([Sxy, Sxb], axis=-1)  # incl. inactive terms in cross-covar
  else:
    moments_y = match_part.y

  return GaussianMatch(x=x, y=moments_y, cross=(Sxy, False))
