#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
"""
Tensorflow implementation of Alan Genz's BVN method for
quasi-analytically solving bivariate normal probabilities.
"""

# ---- Imports
import tensorflow as tf
from math import nan, pi as _pi
from typing import Any, Callable, Iterable, NamedTuple, Optional, Tuple

# ---- Exports
__all__ = ("bvn", "ndtr")


# ==============================================
#                                            bvn
# ==============================================
_inf = float('inf')
_2pi = 2 * _pi
_neg_inv_sqrt2 = -1/(2 ** 0.5)


class _bvnuThresholds(NamedTuple):  # controls branching of _bvnu method
  """
  Helper class for controlling how <_bvnu> branches internally
  """
  r: float = 0.925
  hk: float = -100.0
  asr: float = -100.0


def ndtr(x: tf.Tensor) -> tf.Tensor:
  """
  Standard normal CDF. Called <phid> in Genz's original code.
  """
  return 0.5 * tf.math.erfc(_neg_inv_sqrt2 * x)


def case_nd(pred_fn_pairs: Iterable,
            default: Optional[Callable] = None) -> tf.Tensor:

  accum = None
  active = None
  for i, (mask, fn) in enumerate(pred_fn_pairs):
    vals = fn()
    if i == 0:
      accum = tf.fill(vals.shape, tf.cast(nan, vals.dtype))
      active = tf.fill(mask.shape, True)

    mask = tf.logical_and(active, mask)
    accum = tf.where(mask, vals, accum)
    active = tf.logical_and(active, tf.logical_not(mask))

  return tf.where(active, default(), accum)


def apply_mask(pred: tf.Tensor, src: tf.Tensor, val: float = 0.0):
  return tf.where(pred, src, val)


def bvn(xl: tf.Tensor,
        xu: tf.Tensor,
        yl: tf.Tensor,
        yu: tf.Tensor,
        r: tf.Tensor) -> tf.Tensor:
  """
  BVN
    A function for computing bivariate normal probabilities.
    bvn calculates the probability that
      xl < x < xu and yl < y < yu,
    with correlation coefficient r.
     p = bvn(xl, xu, yl, yu, r)
     Author
        Alan Genz, Department of Mathematics
        Washington State University, Pullman, Wa 99164-3113
        Email : alangenz@wsu.edu
  """
  p = bvnu(xl, yl, r) - bvnu(xu, yl, r) - bvnu(xl, yu, r) + bvnu(xu, yu, r)
  return tf.clip_by_value(p, 0, 1)


def bvnu(dh: tf.Tensor, dk: tf.Tensor, r: tf.Tensor) -> tf.Tensor:
  # Special cases admitting closed-form solutions
  empty = tf.logical_or(tf.equal(dh, _inf), tf.equal(dk, _inf))
  indefinite_dh = tf.equal(dh, -_inf)
  indefinite_dk = tf.equal(dk, -_inf)
  indefinite = tf.logical_and(indefinite_dh, indefinite_dk)
  independent = tf.equal(r, 0)
  return case_nd([
    (empty, lambda: tf.cast(0, dh.dtype)),
    (indefinite, lambda: tf.cast(1, dh.dtype)),
    (indefinite_dh, lambda: ndtr(-dk)),
    (indefinite_dk, lambda: ndtr(-dh)),
    (independent, lambda: ndtr(-dh) * ndtr(-dk)),
  ], default=lambda: _bvnu(dh, dk, r))



def _bvnu(dh: tf.Tensor,
          dk: tf.Tensor,
          r: tf.Tensor,
          thresholds: _bvnuThresholds = None,
          dtype: Any = None) -> tf.Tensor:
  """
  Primary subroutine for bvnu()
  """
  if dtype is None:
    dtype = dh.dtype

  if thresholds is None:
    thresholds = _bvnuThresholds()

  # Precompute some terms
  h = dh
  k = dk
  hk = h * k
  tp = tf.cast(_2pi, dtype)
  itp = 1/tp
  x, w = gauss_legendre(r, dtype=dtype)

  def moderate_corr():
    asr = 0.5 * tf.math.asin(r)
    sn = tf.math.sin(asr[..., None] * x)
    res = (sn * hk[..., None] - 0.5 * (h**2 + k**2)[..., None])/(1 - sn ** 2)
    res = tf.reduce_sum(w * tf.exp(res), axis=-1)
    res = res * itp * asr + ndtr(-h) * ndtr(-k)
    return res

  def strong_corr():
    corr_sign = tf.sign(r)
    _k = k * corr_sign
    _hk = hk * corr_sign

    def partial_corr():
      _as = 1 - tf.square(r)
      a = tf.sqrt(_as)
      bs = tf.square(h - _k)
      asr = -0.5 * (bs / _as + _hk)
      c = 0.125 * (4 - _hk)
      d = 0.0125 * (12 - _hk)

      def asr_gt_threshold():
        return a * tf.exp(asr)*(1 - c*(bs-_as)*(1-d*bs)/3 + c*d*_as**2)
      res = apply_mask(asr > thresholds.asr, asr_gt_threshold())

      def hk_gt_threshold():
        b = tf.sqrt(bs)
        sp = tf.sqrt(tp) * ndtr(-b / a)
        return tf.exp(-0.5 * _hk) * sp * b * (1 - c * bs * (1 - d * bs) / 3)
      res -= apply_mask(_hk > thresholds.hk, hk_gt_threshold())

      a *= 0.5
      xs = tf.square(a[..., None] * x)
      asr = -0.5 * (bs[..., None] / xs + _hk[..., None])
      sp = 1 + c[..., None] * xs * (1 + 5 * d[..., None] * xs)
      rs = tf.sqrt(1 - xs)

      ep = tf.exp(-0.5 * _hk[..., None] * xs / tf.square(1 + rs)) / rs
      deltas = apply_mask(asr > thresholds.asr, w * tf.exp(asr) * (sp - ep))
      return itp * (a * tf.reduce_sum(deltas, axis=-1) - res)

    res = apply_mask(tf.abs(r) < 1, partial_corr())
    return case_nd([
      (r > 0, lambda: res + ndtr(-tf.maximum(h, _k))),
      (h >= _k, lambda: -res),
      (h < 0, lambda: ndtr(_k) - ndtr(h) - res),
    ], default=lambda: ndtr(-h) - ndtr(-_k) - res)

  res = tf.where(tf.abs(r) < thresholds.r, moderate_corr(), strong_corr())
  return tf.clip_by_value(res, 0, 1)


def gauss_legendre(corr: tf.Tensor, dtype: Any) -> Tuple[tf.Tensor, tf.Tensor]:
  """
  Returns Gauss-Legendre abscissae and weights for fixed
  order polynomials.
  """
  def order6():
    _abscissae = tf.constant([
      0.9324695142031522, 0.6612093864662647, 0.2386191860831970
    ], dtype=dtype)

    _weights = tf.constant([
      0.1713244923791705, 0.3607615730481384, 0.4679139345726904
    ], dtype=dtype)
    return _abscissae, _weights

  def order12():
    _abscissae = tf.constant([
      0.9815606342467191, 0.9041172563704750, 0.7699026741943050,
      0.5873179542866171, 0.3678314989981802, 0.1252334085114692
    ], dtype=dtype)

    _weights = tf.constant([
      0.04717533638651177, 0.1069393259953183, 0.1600783285433464,
      0.2031674267230659, 0.2334925365383547, 0.2491470458134029
    ], dtype=dtype)
    return _abscissae, _weights

  def order20():
    _abscissae = tf.constant([
      0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
      0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
      0.5108670019508271, 0.3737060887154196, 0.2277858511416451,
      0.07652652113349733
    ], dtype=dtype)

    _weights = tf.constant([
      0.01761400713915212, .04060142980038694, .06267204833410906,
      0.08327674157670475, 0.1019301198172404, 0.1181945319615184,
      0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
      0.1527533871307259
    ], dtype=dtype)
    return _abscissae, _weights

  abs_corr = tf.abs(corr)
  if tf.reduce_all(abs_corr < 0.3):
    _abscissae, _weights = order6()
  elif tf.reduce_all(abs_corr < 0.75):
    _abscissae, _weights = order12()
  else:
    _abscissae, _weights = order20()

  abscissae = tf.concat([1.0 - _abscissae, 1.0 + _abscissae], axis=0)
  weights = tf.concat([_weights, _weights], axis=0)
  return abscissae, weights
