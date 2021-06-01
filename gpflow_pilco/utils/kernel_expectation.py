#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import itertools
import tensorflow as tf

from functools import lru_cache
from gpflow.kernels import (Kernel,
                            LinearCoregionalization,
                            MultioutputKernel,
                            SeparateIndependent,
                            SharedIndependent,
                            SquaredExponential)
from gpflow.utilities import Dispatcher
from gpflow.utilities.ops import square_distance
from gpflow.inducing_variables import (InducingPoints,
                                       InducingVariables,
                                       SeparateIndependentInducingVariables,
                                       SharedIndependentInducingVariables)
from gpflow.probability_distributions import Gaussian, DiagonalGaussian
from gpflow.expectations.dispatch import expectation as dispatcher
from gpflow.expectations import expectation as kernel_expectation
from typing import Tuple


# ---- Exports
unpack_multioutput = Dispatcher("unpack_multioutput")
__all__ = ("kernel_expectation", "unpack_multioutput")

# ==============================================
#                             kernel_expectation
# ==============================================
NoneType = type(None)
GaussianType = Gaussian, DiagonalGaussian


@unpack_multioutput.register(
    (LinearCoregionalization, SeparateIndependent),
    (InducingVariables, NoneType))
def _unpack(kernel, inducing_variable):
  return unpack_multioutput(list(kernel.kernels), inducing_variable)


@unpack_multioutput.register(SharedIndependent, (InducingVariables, NoneType))
def _unpack(kernel, inducing_variable):
  base_kernels = list(kernel.num_latent_gps * (kernel.kernel,))
  return unpack_multioutput(base_kernels, inducing_variable)


@unpack_multioutput.register(list, NoneType)
def _unpack(kernels, _):
  return kernels, None


@unpack_multioutput.register(list, SeparateIndependentInducingVariables)
def _unpack(kernels, inducing_variable):
  inducing_variables = inducing_variable.inducing_variables
  assert len(kernels) == len(inducing_variables)
  return kernels, list(inducing_variables)


@unpack_multioutput.register(list, SharedIndependentInducingVariables)
def _unpack(kernels, inducing_variable):
  inducing_variables = len(kernels) * inducing_variable.inducing_variables
  return kernels, list(inducing_variables)


@dispatcher.register(GaussianType,
                     SquaredExponential,
                     InducingPoints,
                     SquaredExponential,
                     InducingPoints)
def _E(p, kern1, feat1, kern2, feat2, nghp=None):
  """
  Compute the expectation:
  expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n)
      - Ka_{.,.}, Kb_{.,.} :: RBF kernels

  :return: NxM1xM2
  """
  if kern1.on_separate_dims(kern2) and \
    isinstance(p, DiagonalGaussian):  # no joint expectations required
    eKxz1 = kernel_expectation(p, (kern1, feat1))
    eKxz2 = kernel_expectation(p, (kern2, feat2))
    return eKxz1[:, :, None] * eKxz2[:, None, :]

  if kern1.on_separate_dims(kern2):
    raise NotImplementedError("The expectation over two kernels only has an "
                              "analytical implementation if both kernels have "
                              "the same active features.")

  is_same_kern = (kern1 == kern2)  # code branches by case for
  is_same_feat = (feat1 == feat2)  # computational efficiency

  mx = kern1.slice(p.mu)[0]
  if isinstance(p, DiagonalGaussian):
    Sxx = kern1.slice_cov(tf.linalg.diag(p.cov))
  else:
    Sxx = kern1.slice_cov(p.cov)

  N = tf.shape(mx)[0]  # num. random inputs $x$
  D = tf.shape(mx)[1]  # dimensionality of $x$

  # First Gaussian kernel $k1(x, z) = exp(-0.5*(x - z) V1^{-1} (x - z))$
  V1 = kern1.lengthscales ** 2  # D|1
  z1 = kern1.slice(feat1.Z)[0]  # M1xD
  iV1_z1 = (1/V1) * z1

  # Second Gaussian kernel $k2(x, z) = exp(-0.5*(x - z) V2^{-1} (x - z))$
  V2 = V1 if is_same_kern else kern2.lengthscales ** 2  # D|1
  z2 = z1 if is_same_feat else kern2.slice(feat2.Z)[0]  # M2xD
  iV2_z2 = iV1_z1 if (is_same_kern and is_same_feat) else (1/V2) * z2

  # Product of Gaussian kernels is another Gaussian kernel $k = k1 * k2$
  V = 0.5 * V1 if is_same_kern else (V1 * V2)/(V1 + V2)  # D|1
  if not (kern1.ard or kern2.ard):
      V = tf.fill((D,), V)  # D

  # Product of Gaussians is an unnormalized Gaussian; compute determinant of
  # this new Gaussian (and the Gaussian kernel) in order to normalize
  S = Sxx + tf.linalg.diag(V)
  L = tf.linalg.cholesky(S)  # NxDxD
  half_logdet_L = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=1)
  sqrt_det_iL = tf.exp(-half_logdet_L)
  sqrt_det_L = tf.sqrt(tf.reduce_prod(V))
  determinant = sqrt_det_L * sqrt_det_iL  # N

  # Solve for linear systems involving $S = LL^{T}$ where $S$
  # is the covariance of an (unnormalized) Gaussian distribution
  iL_mu = tf.linalg.triangular_solve(L,
                                     tf.expand_dims(mx, 2),
                                     lower=True)  # NxDx1

  V_iV1_z1 = tf.expand_dims(tf.transpose(V * iV1_z1), 0)
  iL_z1 = tf.linalg.triangular_solve(L,
                                     tf.tile(V_iV1_z1, [N, 1, 1]),
                                     lower=True)  # NxDxM1

  z1_iS_z1 = tf.reduce_sum(tf.square(iL_z1), axis=1)  # NxM1
  z1_iS_mu = tf.squeeze(tf.linalg.matmul(iL_z1, iL_mu, transpose_a=True), 2)  # NxM1
  if is_same_kern and is_same_feat:
    iL_z2 = iL_z1
    z2_iS_z2 = z1_iS_z1
    z2_iS_mu = z1_iS_mu
  else:
    V_iV2_z2 = tf.expand_dims(tf.transpose(V * iV2_z2), 0)
    iL_z2 = tf.linalg.triangular_solve(L,
                                       tf.tile(V_iV2_z2, [N, 1, 1]),
                                       lower=True)  # NxDxM2

    z2_iS_z2 = tf.reduce_sum(tf.square(iL_z2), 1)  # NxM2
    z2_iS_mu = tf.squeeze(tf.matmul(iL_z2, iL_mu, transpose_a=True), 2)  # NxM2

  z1_iS_z2 = tf.linalg.matmul(iL_z1, iL_z2, transpose_a=True)  # NxM1xM2
  mu_iS_mu = tf.expand_dims(tf.reduce_sum(tf.square(iL_mu), 1), 2)  # Nx1x1

  # Gram matrix from Gaussian integral of Gaussian kernel $k = k1 * k2$
  exp_mahalanobis = tf.exp(-0.5 * (mu_iS_mu + 2 * z1_iS_z2
    + tf.expand_dims(z1_iS_z1 - 2 * z1_iS_mu, axis=-1)
    + tf.expand_dims(z2_iS_z2 - 2 * z2_iS_mu, axis=-2)
  ))  # NxM1xM2

  # Part of $E_{p(x)}[k1(z1, x) k2(x, z2)]$ that is independent of $x$
  if is_same_kern:
    ampl2 = kern1.variance ** 2
    sq_iV = tf.math.rsqrt(V)
    if is_same_feat:
      matrix_term = ampl2 * tf.exp(-0.125 * square_distance(sq_iV * z1, None))
    else:
      matrix_term = ampl2 * tf.exp(-0.125 * square_distance(sq_iV * z1, sq_iV * z2))
  else:
    z1_iV1_z1 = tf.reduce_sum(z1 * iV1_z1, axis=-1)  # M1
    z2_iV2_z2 = tf.reduce_sum(z2 * iV2_z2, axis=-1)  # M2
    z1_iV1pV2_z1 = tf.reduce_sum(iV1_z1 * V * iV1_z1, axis=-1)
    z2_iV1pV2_z2 = tf.reduce_sum(iV2_z2 * V * iV2_z2, axis=-1)
    z1_iV1pV2_z2 = tf.matmul(iV1_z1, V * iV2_z2, transpose_b=True)  # M1xM2
    matrix_term = kern1.variance * kern2.variance * tf.exp(0.5 * (
        2 * z1_iV1pV2_z2  # implicit negative
        + tf.expand_dims(z1_iV1pV2_z1 - z1_iV1_z1, axis=-1)
        + tf.expand_dims(z2_iV1pV2_z2 - z2_iV2_z2, axis=-2)
    ))

  return tf.reshape(determinant, [N, 1, 1]) * matrix_term * exp_mahalanobis


@dispatcher.register(GaussianType, list, NoneType, NoneType, NoneType)
def _eKff(p: GaussianType, K: Tuple[Kernel], _, __, ___, nghp=None):
  @lru_cache(maxsize=len(K))
  def _expectation(p, k, nghp=nghp):
      return dispatcher(p, k, None, None, None, nghp=nghp)

  # TODO: Do we wrap this with a call to tf.linalg.diag?
  return tf.stack([_expectation(p, k) for k in K], axis=-1)  # NxL


@dispatcher.register(GaussianType, list, list, NoneType, NoneType)
def _eKfu(p: GaussianType,
          K: Tuple[Kernel],
          Z: Tuple[InducingVariables],
          _,
          __,
          nghp=None):

  L = len(K)
  assert len(Z) == L

  @lru_cache(maxsize=L)
  def _expectation(p, k, z, nghp=nghp):
      return dispatcher(p, k, z, None, None, nghp=nghp)
  return tf.stack([_expectation(p, k, z) for k, z in zip(K, Z)], axis=-1) # NxMxL


@dispatcher.register(GaussianType, list, list, list, list)
def _eKuffu(p: GaussianType,
            K1: Tuple[Kernel],
            Z1: Tuple[InducingVariables],
            K2: Tuple[Kernel],
            Z2: Tuple[InducingVariables],
            nghp=None):

  L1 = len(K1)
  assert L1 == len(Z1)

  L2 = len(K2)
  assert L2 == len(Z2)

  @lru_cache(maxsize=L1 * L2)
  def _expectation(p, k1, z1, k2, z2, nghp=nghp):
    return dispatcher(p, k1, z1, k2, z2, nghp=nghp)

  eKuffu = [L2 * [None] for _ in range(L1)]
  group1 = zip(range(L1), K1, Z1)
  group2 = zip(range(L2), K2, Z2)
  for (i, k1, z1), (j, k2, z2) in itertools.product(group1, group2):
    id1 = hash(k1), hash(z1)
    id2 = hash(k2), hash(z2)
    if id1 <= id2:
      eKuffu[i][j] = _expectation(p, k1, z1, k2, z2)
    else:  # reuse cache where possible
      eKuffu[i][j] = tf.linalg.adjoint(_expectation(p, k2, z2,  k1, z1))

  # Output shape: NxL1xM1xL2xM2
  return tf.stack([tf.stack(eK, axis=-2) for eK in eKuffu], axis=-4)


@dispatcher.register(
    GaussianType,
    MultioutputKernel,
    NoneType,
    NoneType,
    NoneType)
def _E(p, kernel, inducing_var, _, __, nghp=None):
  return dispatcher(p,
                    *unpack_multioutput(kernel, inducing_var),
                    None,
                    None,
                    nghp=nghp)


@dispatcher.register(
    GaussianType,
    MultioutputKernel,
    InducingVariables,
    NoneType,
    NoneType)
def _E(p, kernel, inducing_var, _, __, nghp=None):
  return dispatcher(p,
                    *unpack_multioutput(kernel, inducing_var),
                    None,
                    None,
                    nghp=nghp)


@dispatcher.register(
    GaussianType,
    MultioutputKernel,
    InducingVariables,
    MultioutputKernel,
    InducingVariables)
def _E(p, kern1, inducing1, kern2, inducing2, nghp=None):
  return dispatcher(p,
                    *unpack_multioutput(kern1, inducing1),
                    *unpack_multioutput(kern2, inducing2),
                    nghp=nghp)
