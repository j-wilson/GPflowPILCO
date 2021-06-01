#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from gpflow_pilco.moment_matching.core import *
from gpflow_pilco.moment_matching.gaussian import *
from numpy import ndarray
from typing import Union


# ==============================================
#                                          maths
# ==============================================
NumericalTypes = (int, float, complex) + ArrayTypes


def sincos(x: NumericalTypes, axis: int = -1) -> tf.Tensor:
  return tf.concat([tf.math.sin(x), tf.math.cos(x)], axis=axis)


# Register custom types used for dispatching against tensorflow functions
_type_tfIdendity = register_type(tf.identity)
_type_tfAdd = register_type(tf.math.add)
_type_tfSub = register_type(tf.math.subtract)
_type_tfMul = register_type(tf.math.multiply)
_type_tfMatvec = register_type(tf.linalg.matvec)
_type_tfCos = register_type(tf.math.cos)
_type_tfSin = register_type(tf.math.sin)
_type_SinCos = register_type(sincos)


def _outer_op(op, a, b):  # helper method for, e.g., outer products
  return op(tf.expand_dims(a, -1), tf.expand_dims(b, -2))


@dispatcher.register(GaussianMoments, _type_tfIdendity, NumericalTypes)
def _mm_gauss_identity(x: GaussianMoments, _, c: Union[NumericalTypes]):
  iSxx_Sxy = tf.linalg.LinearOperatorIdentity(num_rows=x.ndim, dtype=x.dtype)
  return GaussianMatch(x=x, y=x, cross=(iSxx_Sxy, True))


@dispatcher.register(GaussianMoments, _type_tfAdd, NumericalTypes)
def _mm_gauss_add(x: GaussianMoments, _, c: Union[NumericalTypes]):
  iSxx_Sxy = tf.linalg.LinearOperatorIdentity(num_rows=x.ndim, dtype=x.dtype)
  y = GaussianMoments(moments=(x.mean() + c, x.covariance()), centered=True)
  return GaussianMatch(x=x, y=y, cross=(iSxx_Sxy, True))


@dispatcher.register(GaussianMoments, _type_tfSub, NumericalTypes)
def _mm_gauss_sub(x: GaussianMoments, _, c: Union[NumericalTypes]):
  iSxx_Sxy = tf.linalg.LinearOperatorIdentity(num_rows=x.ndim, dtype=x.dtype)
  y = GaussianMoments(moments=(x.mean() - c, x.covariance()), centered=True)
  return GaussianMatch(x=x, y=y, cross=(iSxx_Sxy, True))


@dispatcher.register(GaussianMoments, _type_tfMul, NumericalTypes)
def _mm_gauss_mul(x: GaussianMoments, _, c: Union[NumericalTypes]):
  if isinstance(c, (int, float, complex)):
    c = tf.convert_to_tensor(c, dtype=x.dtype)

  y1 = c * x[0]
  if isinstance(x[1], tf.linalg.LinearOperator):
    c2 = tf.linalg.LinearOperatorScaledIdentity(num_rows=x.ndim,
                                                multiplier=c ** 2)
    y2 = tf.linalg.LinearOperatorComposition((c2, x[1]))
  else:
    y2 = (c ** 2) * x[1]

  iSxx_Sxy = tf.linalg.LinearOperatorScaledIdentity(num_rows=x.ndim,
                                                    multiplier=c)

  y = GaussianMoments(moments=(y1, y2), centered=x.centered)
  return GaussianMatch(x=x, y=y, cross=(iSxx_Sxy, True))


@dispatcher.register(GaussianMoments, _type_tfMatvec, ArrayTypes)
def _mm_gauss_matvec(x: GaussianMoments,
                     _,
                     a: Union[ndarray, tf.Tensor],
                     adjoint_a: bool = False):

  y1 = tf.linalg.matvec(a, x[0], adjoint_a=adjoint_a)
  y2 = tf.linalg.matmul(a,
                        tf.linalg.matmul(x[1], a, adjoint_b=not adjoint_a),
                        adjoint_a=adjoint_a)

  iSxx_Sxy = a if adjoint_a else tf.linalg.adjoint(a)
  y = GaussianMoments(moments=(y1, y2), centered=x.centered)
  return GaussianMatch(x=x, y=y, cross=(iSxx_Sxy, True))


@dispatcher.register(GaussianMoments, _type_tfCos)
def _mm_gauss_cos(x: GaussianMoments, _):
  x1 = x.mean()
  Sxx = x.covariance(dense=True)

  vx = tf.linalg.diag_part(Sxx)
  vx_add_vxT = _outer_op(tf.add, vx, vx)
  Sxx_add_SxxT = Sxx + tf.linalg.adjoint(Sxx)

  A = tf.math.exp(-0.5 * (vx_add_vxT + Sxx_add_SxxT))
  B = tf.math.exp(-0.5 * (vx_add_vxT - Sxx_add_SxxT))
  A_mul_cosAdd = A * tf.math.cos(_outer_op(tf.add, x1, x1))
  B_mul_cosSub = B * tf.math.cos(_outer_op(tf.subtract, x1, x1))

  evx = tf.math.exp(-0.5 * vx)
  y1 = evx * tf.math.cos(x1)
  y2 = 0.5 * (B_mul_cosSub + A_mul_cosAdd)

  iSxx_Sxy = tf.linalg.LinearOperatorDiag(diag=-tf.math.sin(x1) * evx)
  y = GaussianMoments(moments=(y1, y2), centered=False)
  return GaussianMatch(x=x, y=y, cross=(iSxx_Sxy, True))


@dispatcher.register(GaussianMoments, _type_tfSin)
def _mm_gauss_sin(x: GaussianMoments, _):
  x1 = x.mean()
  Sxx = x.covariance(dense=True)

  vx = tf.linalg.diag_part(Sxx)
  vx_add_vxT = _outer_op(tf.add, vx, vx)
  Sxx_add_SxxT = Sxx + tf.linalg.adjoint(Sxx)

  A = tf.math.exp(-0.5 * (vx_add_vxT + Sxx_add_SxxT))
  B = tf.math.exp(-0.5 * (vx_add_vxT - Sxx_add_SxxT))
  A_mul_cosAdd = A * tf.math.cos(_outer_op(tf.add, x1, x1))
  B_mul_cosSub = B * tf.math.cos(_outer_op(tf.subtract, x1, x1))

  evx = tf.math.exp(-0.5 * vx)
  y1 = evx * tf.math.sin(x1)
  y2 = 0.5 * (B_mul_cosSub - A_mul_cosAdd)

  iSxx_Sxy = tf.linalg.LinearOperatorDiag(diag=tf.math.cos(x1) * evx)
  y = GaussianMoments(moments=(y1, y2), centered=False)
  return GaussianMatch(x=x, y=y, cross=(iSxx_Sxy, True))


@dispatcher.register(GaussianMoments, _type_SinCos)
def _mm_gauss_sincos(x: GaussianMoments, _):
  x1 = x.mean()
  Sxx = x.covariance(dense=True)

  vx = tf.linalg.diag_part(Sxx)
  vx_add_vxT = _outer_op(tf.add, vx, vx)
  Sxx_add_SxxT = Sxx + tf.linalg.adjoint(Sxx)

  A = tf.math.exp(-0.5 * (vx_add_vxT + Sxx_add_SxxT))
  B = tf.math.exp(-0.5 * (vx_add_vxT - Sxx_add_SxxT))
  A_mul_cosAdd = A * tf.math.cos(_outer_op(tf.add, x1, x1))
  B_mul_cosSub = B * tf.math.cos(_outer_op(tf.subtract, x1, x1))

  evx = tf.math.exp(-0.5 * vx)
  cos_x1 = tf.math.cos(x1)
  sin_x1 = tf.math.sin(x1)

  c1 = evx * cos_x1
  c2 = 0.5 * (B_mul_cosSub + A_mul_cosAdd)

  s1 = evx * sin_x1
  s2 = 0.5 * (B_mul_cosSub - A_mul_cosAdd)

  sx1_cx1T = _outer_op(tf.multiply, sin_x1, cos_x1)
  sc = 0.5 * (sx1_cx1T * (B + A) - tf.linalg.adjoint(sx1_cx1T) * (B - A))

  y1 = tf.concat([s1, c1], axis=-1)
  y2 = tf.concat([tf.concat([s2, sc], axis=-1),
                  tf.concat([tf.linalg.adjoint(sc), c2], axis=-1)], axis=-2)

  iSxx_Sxy = tf.concat([tf.linalg.diag(c1), tf.linalg.diag(-s1)], axis=-1)
  y = GaussianMoments(moments=(y1, y2), centered=False)
  return GaussianMatch(x=x, y=y, cross=(iSxx_Sxy, True))
