#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf
import gpflow

from tensorflow import linalg as tfla
from functools import partial, update_wrapper
from gpflow.mean_functions import Zero, Constant
from gpflow_pilco.models import InverseLinkWrapper, KernelRegressor
from gpflow_pilco.moment_matching.core import Chain, dispatcher
from gpflow_pilco.moment_matching.gaussian import *
from gpflow_pilco.utils.kernel_expectation import kernel_expectation

DiagonalLinearOperators = (tf.linalg.LinearOperatorDiag,
                           tf.linalg.LinearOperatorIdentity,
                           tf.linalg.LinearOperatorScaledIdentity)


# ==============================================
#                                         models
# ==============================================
@dispatcher.register(GaussianMoments, InverseLinkWrapper)
def _mm_gauss_invlink(x: GaussianMoments, wrapper: InverseLinkWrapper, /, **kw):
  base = update_wrapper(partial(wrapper.model, **kw), wrapper.model)
  chain = Chain(wrapper.invlink, base)
  return dispatcher(x, chain)


@dispatcher.register(GaussianMoments, KernelRegressor)
def _mm_gauss_kr(x: GaussianMoments, regressor: KernelRegressor, /, **kwargs):
  uncertainty = kwargs.pop("model_uncertainty", False)
  assert not uncertainty, ValueError("Kernel regressors have no uncertainty.")
  return dispatcher(x,
                    regressor.model,
                    model_uncertainty=False,
                    **kwargs)


@dispatcher.register(GaussianMoments, gpflow.models.GPR)
def _mm_gauss_gpr(x: GaussianMoments,
                  model: gpflow.models.GPR,
                  /,
                  full_output_cov: bool = True,
                  model_uncertainty: bool = True,
                  jitter: float = 0.0):

  kernel, (X, Y) = model.kernel, model.data
  if isinstance(model.mean_function, Constant):
    Y = Y - model.mean_function(X)
  elif not isinstance(model.mean_function, Zero):
    raise NotImplementedError

  # Compute Gaussian integrals of Gaussian kernels (and products thereof)
  p = _moments_to_gpflow_distrib(x)
  Z = gpflow.inducing_variables.InducingPoints(X)
  eKff = kernel_expectation(p, kernel)
  eKfu = kernel_expectation(p, (kernel, Z))
  eKuffu = kernel_expectation(p, (kernel, Z), (kernel, Z))

  # Solve for linear systems involving $(K_{n,n} + \sigma^{2}I)^{-1}$
  Knn = model.kernel(X, full_cov=True)
  Kyy = tfla.set_diag(Knn, tfla.diag_part(Knn) + model.likelihood.variance)
  Lyy = tfla.cholesky(Kyy)

  iLyy_y = tfla.triangular_solve(Lyy, Y)
  iLyy_eKuffu = tfla.triangular_solve(Lyy, eKuffu)
  iLyy_eKuffu_Lit = tfla.triangular_solve(Lyy, tfla.adjoint(iLyy_eKuffu))

  # First moment (sans mean function)
  iKyy_y = tfla.triangular_solve(Lyy, iLyy_y, adjoint=True)
  f1 = tf.matmul(eKfu, iKyy_y)

  # (Centered) second moment
  if full_output_cov:
    f2 = tf.matmul(iLyy_y, tf.matmul(iLyy_eKuffu_Lit, iLyy_y), transpose_a=True)
    Sff = f2 - tf.expand_dims(f1, -1) * tf.expand_dims(f1, -2)  # Nx1x1
  else:
    f2 = tf.reduce_sum(iLyy_y * tf.matmul(iLyy_eKuffu_Lit, iLyy_y), -2)
    Sff = f2 - tf.square(f1)

  if model_uncertainty:  # factor in expected value of covariance
    e_cov = eKff - tfla.trace(iLyy_eKuffu_Lit)
    Sff += e_cov[..., None, None] if full_output_cov else e_cov[..., None]

  # Cross-covariance term $cov(f, x)$
  dX = kernel.slice(X)[0] - tf.expand_dims(kernel.slice(x.mean())[0], axis=-2)
  Sxx = kernel.slice_cov(x.covariance(dense=True))
  V_sqrt = tfla.cholesky(Sxx + _get_lengthscales_matrix(kernel, Sxx.shape[-1]))
  iV_dXt = tfla.cholesky_solve(V_sqrt, tfla.adjoint(dX))
  iSxx_Sxf = tf.reduce_sum(tfla.adjoint(iKyy_y)
                           * tf.expand_dims(eKfu, axis=-2)
                           * iV_dXt,
                           axis=-1, keepdims=True)

  if isinstance(model.mean_function, Constant):
    f1 += model.mean_function(x.mean())
  elif not isinstance(model.mean_function, Zero):
    raise NotImplementedError

  if full_output_cov:
    Sff = tf.linalg.set_diag(Sff, tf.linalg.diag_part(Sff) + jitter)
  else:
    Sff = tf.linalg.LinearOperatorDiag(Sff + jitter)

  f = GaussianMoments(moments=(f1, Sff), centered=True)
  return GaussianMatch(x=x, y=f, cross=(iSxx_Sxf, True))


@dispatcher.register(GaussianMoments, gpflow.models.SVGP)
def _mm_gauss_svgp(x: GaussianMoments,
                   model: gpflow.models.SVGP,
                   /,
                   **kwargs):
  """
  TODO: Keep or deprecate and change function signature.
  """
  if isinstance(model.kernel, gpflow.kernels.MultioutputKernel):
    subroutine = _mm_gauss_svgp_mo  # multiple outputs
  else:
    subroutine = _mm_gauss_svgp_so  # single output
  return subroutine(x=x, model=model, **kwargs)


def _mm_gauss_svgp_so(x: GaussianMoments,
                      model: gpflow.models.SVGP,
                      full_output_cov: bool = True,
                      model_uncertainty: bool = True,
                      jitter: float = 0.0):

  assert not isinstance(model.kernel, gpflow.kernels.MultioutputKernel)
  kernel, Z = model.kernel, model.inducing_variable

  # Compute Gaussian integrals of Gaussian kernels (and products thereof)
  p = _moments_to_gpflow_distrib(x)
  eKff = kernel_expectation(p, kernel)
  eKfu = kernel_expectation(p, (kernel, Z))
  eKuffu = kernel_expectation(p, (kernel, Z), (kernel, Z))

  # Solve for linear systems involving $K_{u,u}^{-1}$
  Kuu = gpflow.covariances.Kuu(Z, kernel, jitter=gpflow.config.default_jitter())
  Luu = tfla.cholesky(Kuu)
  iLuu_eKuffu = tfla.triangular_solve(Luu, eKuffu)
  iLuu_eKuffu_iLuut = tfla.triangular_solve(Luu, tfla.adjoint(iLuu_eKuffu))

  iLuu_qmu = model.q_mu
  iLuu_qsqrt = tfla.band_part(model.q_sqrt, -1, 0)
  if not model.whiten:
    iLuu_qmu = tfla.triangular_solve(Luu, iLuu_qmu)
    iLuu_qsqrt = tfla.triangular_solve(Luu, iLuu_qsqrt)

  # First moment
  iKuu_qmu = tfla.triangular_solve(Luu, iLuu_qmu, adjoint=True)
  f1 = tf.matmul(eKfu, iKuu_qmu)

  # (Centered) second moment of conditional expectation
  if full_output_cov:
    f2 = tf.matmul(iLuu_qmu, iLuu_eKuffu_iLuut @ iLuu_qmu, transpose_a=True)
    Sff = f2 - tf.expand_dims(f1, -1) * tf.expand_dims(f1, -2)  # Nx1x1
  else:
    f2 = tf.reduce_sum(iLuu_qmu * (iLuu_eKuffu_iLuut @ iLuu_qmu), -2)
    Sff = f2 - tf.square(f1)  # Nx1

  if model_uncertainty:  # factor in expected value of covariance
    Li_qcov_LiT = tf.matmul(iLuu_qsqrt, iLuu_qsqrt, transpose_b=True)
    e_cov = (eKff - tfla.trace(iLuu_eKuffu_iLuut)
             + tf.reduce_sum(
                  tf.reduce_sum(iLuu_eKuffu_iLuut * Li_qcov_LiT, -1), -1))
    Sff += e_cov[..., None, None] if full_output_cov else e_cov[..., None]

  # Cross-covariance term $cov(f, x)$
  x1 = kernel.slice(x.mean())[0]
  dX = kernel.slice(tf.stack(Z.variables))[0] - tf.expand_dims(x1, axis=-2)
  Sxx = kernel.slice_cov(x.covariance(dense=True))
  V_sqrt = tfla.cholesky(Sxx + _get_lengthscales_matrix(kernel, Sxx.shape[-1]))
  iV_dXt = tfla.cholesky_solve(V_sqrt, tfla.adjoint(dX))
  iSxx_Sxf = tf.reduce_sum(tfla.adjoint(iKuu_qmu)
                           * tf.expand_dims(eKfu, axis=-2)
                           * iV_dXt,
                           axis=-1, keepdims=True)

  if isinstance(model.mean_function, Constant):
    f1 += model.mean_function(x.mean())
  elif not isinstance(model.mean_function, Zero):
    raise NotImplementedError

  if full_output_cov:
    Sff = tf.linalg.set_diag(Sff, tf.linalg.diag_part(Sff) + jitter)
  else:
    Sff = tf.linalg.LinearOperatorDiag(Sff + jitter)

  y = GaussianMoments(moments=(f1, Sff), centered=True)
  return GaussianMatch(x=x, y=y, cross=(iSxx_Sxf, True))


def _mm_gauss_svgp_mo(x: GaussianMoments,
                      model: gpflow.models.SVGP,
                      full_output_cov: bool = True,
                      model_uncertainty: bool = True,
                      jitter: float = 0.0):

  assert isinstance(model.kernel, gpflow.kernels.MultioutputKernel)
  kernel, Z = model.kernel, model.inducing_variable

  # Compute Gaussian integrals of Gaussian kernels (and products thereof)
  p = _moments_to_gpflow_distrib(x)
  eKff = tf.linalg.diag(kernel_expectation(p, kernel))
  eKfu = kernel_expectation(p, (kernel, Z))
  eKuffu = kernel_expectation(p, (kernel, Z), (kernel, Z))

  # Solve for linear systems involving block diagonal matrices $K_{u,u}^{-1}$
  Kuu = gpflow.covariances.Kuu(Z, kernel, jitter=gpflow.config.default_jitter())
  Luu = tfla.cholesky(Kuu)

  ndims = eKuffu.shape.ndims
  swaps = {-3: -2}, {-4: -3}, {-4: -3, -2: -1}
  perm0 = _get_permutation(_get_permutation(ndims, swaps[0]), swaps[1])
  perm1 = _get_permutation(ndims, swaps[2])
  perm2 = _get_permutation(_get_permutation(perm1.copy(), swaps[1]), swaps[0])
  iLuu_eKuffu = tfla.triangular_solve(Luu, tf.transpose(eKuffu, perm0))  # NxL2xL1xM1xM2 (denoted as L1 & L2 for clarity)
  iLuu_eKuffu_iLuut = tfla.triangular_solve(Luu, tf.transpose(iLuu_eKuffu, perm1))
  iLuu_eKuffu_iLuut = tf.transpose(iLuu_eKuffu_iLuut, perm2)  # NxL1xM1xL2xM2

  iLuu_qmu = tf.expand_dims(tfla.adjoint(model.q_mu), -1)  # LxMx1
  iLuu_qsqrt = tfla.band_part(model.q_sqrt, -1, 0)
  if not model.whiten:
    iLuu_qmu = tfla.triangular_solve(Luu,  iLuu_qmu)
    iLuu_qsqrt = tfla.triangular_solve(Luu, iLuu_qsqrt)

  # First moment
  iKuu_qmu = tfla.triangular_solve(Luu, iLuu_qmu, adjoint=True)
  f1 = tf.reduce_sum(tfla.adjoint(eKfu) * iKuu_qmu[..., 0], axis=-1)

  # Maybe precompute a term
  if model_uncertainty or not full_output_cov:
    Li_eKuffu_Lit_blkdiag = tf.stack([iLuu_eKuffu_iLuut[..., i, :, i, :] for i
                                      in range(kernel.num_latent_gps)], axis=-3)

  # (Centered) second moment
  if full_output_cov or isinstance(kernel, gpflow.kernels.LinearCoregionalization):
    f2 = tf.reduce_sum(iLuu_qmu * tf.reduce_sum(
                       iLuu_eKuffu_iLuut * tf.squeeze(iLuu_qmu, -1), -1),
                       axis=-2)
    Sff = f2 - tf.expand_dims(f1, -1) * tf.expand_dims(f1, -2)
  else:
    Sff = tf.reduce_sum(tf.squeeze(iLuu_qmu, -1) *
                        tf.reduce_sum(Li_eKuffu_Lit_blkdiag * iLuu_qmu, -2),
                        axis=-1) - tf.square(f1)

  if model_uncertainty:  # add in expected value of covariance
    Li_qcov_Lit = tf.matmul(iLuu_qsqrt, iLuu_qsqrt, transpose_b=True)  # LxM1xM2
    trace = tfla.trace(Li_eKuffu_Lit_blkdiag)  # NxL
    matmul = tf.reduce_sum(tf.reduce_sum(Li_eKuffu_Lit_blkdiag * Li_qcov_Lit, -1), -1)
    if full_output_cov or isinstance(kernel, gpflow.kernels.LinearCoregionalization):
      Sff += eKff + tfla.diag(matmul - trace)  # NxLxL
    else:
      Sff += tfla.diag_part(eKff) + matmul - trace  # NxL

  # Cross-covariance term $cov(f, x)$
  x1 = x.mean()
  x1 = tf.stack([k.slice(x1)[0] for k in kernel.kernels], axis=-2)  # NxLxD
  Zs = tf.stack([k.slice(z)[0] for k, z in zip(kernel.kernels, Z.variables)])
  dX = Zs - tf.expand_dims(x1, -2)  # NxLxMxD

  Sxx = x.covariance(dense=True)
  Sxx = tf.stack([k.slice_cov(Sxx) for k in kernel.kernels], axis=-3)  # NxLxDxD
  V_sqrt = tfla.cholesky(Sxx + _get_lengthscales_matrix(kernel, Sxx.shape[-1]))
  iV_dXt = tfla.cholesky_solve(V_sqrt, tfla.adjoint(dX))
  perm_iV_dXt = _get_permutation(iV_dXt.shape.ndims, {-2: -3})
  iSxx_Sxf = tf.reduce_sum(tf.squeeze(iKuu_qmu, -1)  # NxDxL
                            * tf.expand_dims(tfla.adjoint(eKfu), axis=-3)
                            * tf.transpose(iV_dXt, perm_iV_dXt),
                            axis=-1)

  if isinstance(kernel, gpflow.kernels.LinearCoregionalization):
    f1 = tf.matmul(f1, kernel.W, transpose_b=True)  # NxP
    iSxx_Sxf = tf.matmul(iSxx_Sxf, kernel.W, transpose_b=True)

    if full_output_cov:
      Sff = kernel.W @ tf.matmul(Sff, kernel.W, transpose_b=True)  # NxPxP
    else:
      Sff = tf.reduce_sum(kernel.W * tf.matmul(kernel.W, Sff, transpose_b=True), -1)

  if isinstance(model.mean_function, Constant):
    f1 += model.mean_function(x.mean())
  elif not isinstance(model.mean_function, Zero):
    raise NotImplementedError

  if full_output_cov:
    Sff = tf.linalg.set_diag(Sff, tf.linalg.diag_part(Sff) + jitter)
  else:
    Sff = tf.linalg.LinearOperatorDiag(Sff + jitter)

  y = GaussianMoments(moments=(f1, Sff), centered=True)
  return GaussianMatch(x=x, y=y, cross=(iSxx_Sxf, True))


def _moments_to_gpflow_distrib(x: GaussianMoments, diag: bool = None):
  if diag is None:  # attempt to auto-detect diagonal covariances
    diag = x.centered and isinstance(x[1], DiagonalLinearOperators)

  m = x.mean()
  S = x.covariance()
  if diag:
    v = tf.linalg.diag_part(S)
    return gpflow.probability_distributions.DiagonalGaussian(m, v)
  return gpflow.probability_distributions.Gaussian(m, S)


def _get_permutation(ndims_or_perm, pairs: dict):
  if isinstance(ndims_or_perm, int):
    perm = indices = list(range(ndims_or_perm))
  else:
    perm = ndims_or_perm
    indices = list(range(len(perm)))

  pairs = {indices[k]: indices[v] for k, v in pairs.items()}  # normalize axes
  assert len(pairs.keys() - pairs.values()) == len(pairs)  # test for duplicates
  for i, j in pairs.items():
    tmp = perm[i]
    perm[i] = perm[j]
    perm[j] = tmp

  return perm


def _get_lengthscales_matrix(kernel,
                             ndims,
                             square=True,
                             invert=False,
                             diag_part=False):
  if isinstance(kernel, gpflow.kernels.MultioutputKernel):
    return tf.stack([_get_lengthscales_matrix(kernel=k,
                                              ndims=ndims,
                                              square=square,
                                              invert=invert)
                     for k in kernel.kernels], axis=-3)

  if kernel.ard:
    diag = kernel.lengthscales
  else:
    diag = tf.fill([ndims], kernel.lengthscales)

  if square:
    diag = tf.math.square(diag)

  if invert:
    diag = tf.math.reciprocal(diag)

  return diag if diag_part else tfla.diag(diag)
