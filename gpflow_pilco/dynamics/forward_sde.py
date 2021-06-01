#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from gpflow.utilities import Dispatcher
from gpflow_pilco.moment_matching import (GaussianMatch,
                                          GaussianMoments,
                                          moment_matching)
from typing import Callable, Union
NoneType = type(None)
MaybeObject = object, NoneType
forward_sde = Dispatcher("forward_sde")


# ==============================================
#                                    forward_sde
# ==============================================
@forward_sde.register(tf.Tensor, object, MaybeObject, MaybeObject, MaybeObject)
def _forward(x: tf.Tensor,
             drift: Callable,
             noise: Union[Callable, NoneType],  # a.k.a. diffusion
             policy: Union[Callable, NoneType],
             encoder: Union[Callable, NoneType]):
  e = x if (encoder is None) else encoder(x)
  eu = e if (policy is None) else tf.concat([e, policy(e)], axis=-1)
  return drift(eu), None if (noise is None) else noise(e)


@forward_sde.register(GaussianMoments,
                      object,
                      MaybeObject,
                      NoneType,
                      NoneType)
def forward(x: GaussianMoments,
            drift: Callable,
            noise: Union[Callable, NoneType],
            _,
            __):
  match_drift = moment_matching(x, drift)
  match_noise = None if (noise is None) else moment_matching(x, noise)
  return match_drift, match_noise


@forward_sde.register(GaussianMoments, object, MaybeObject, object, NoneType)
def _forward(x: GaussianMoments,
             drift: Callable,
             noise: Union[Callable, NoneType],
             policy: Callable,
             _):
  match_policy = moment_matching(x, policy)
  match_drift = moment_matching(match_policy.joint(), drift)

  # Approx. Cov(x, f) by Cov(x, d) Cov(d, d)^{-1} Cov(d, f) where d = (x, u)
  if match_drift.cross[1]:  # is Cov(d, f) premultiplied by Cov(d, d)^{-1}?
    preinv = match_policy.cross[1]  # try to avoid multiplying by Cov(x, x)^{-1}
    cross = match_policy.cross_covariance(preinv=preinv)\
            @ match_drift.cross_covariance(preinv=True), preinv
  else:
    cross = match_drift.cross_covariance()[..., :x.mean().shape[-1], :], False

  chain_match_drift = GaussianMatch(x=x, y=match_drift.y, cross=cross)
  match_noise = None if (noise is None) else moment_matching(x, noise)
  return chain_match_drift, match_noise


@forward_sde.register(GaussianMoments, object, MaybeObject, NoneType, object)
def _forward(x: GaussianMoments,
             drift: Callable,
             noise: Union[Callable, NoneType],
             _,
             encoder: Callable):
  match_encoder = moment_matching(x, encoder)
  match_drift = moment_matching(match_encoder.y, drift)

  # Approx. Cov(x, f) by Cov(x, e) Cov(e, e)^{-1} Cov(e, f) where e = encoder(x)
  preinv = match_encoder.cross[1]  # can we avoid multiplying by Cov(x, x)^{-1}?
  Sxe = match_encoder.cross_covariance(preinv=preinv)
  cross = Sxe @ match_drift.cross_covariance(preinv=True), preinv
  chain_match_drift = GaussianMatch(x=x, y=match_drift.y, cross=cross)
  if noise is None:
    chain_match_noise = None
  else:
    match_noise = moment_matching(match_encoder.y, noise)
    # Approx. Cov(x, z) by Cov(x, e) Cov(e, e)^{-1} Cov(e, z) where z is noise
    cross = Sxe @ match_noise.cross_covariance(preinv=True), preinv
    chain_match_noise = GaussianMatch(x=x, y=match_noise.y, cross=cross)
  return chain_match_drift, chain_match_noise


@forward_sde.register(GaussianMoments, object, object, object, object)
def _forward(x: GaussianMoments,
             drift: Callable,
             noise: Union[Callable, NoneType],
             policy: Callable,
             encoder: Callable):
  match_encoder = moment_matching(x, encoder)
  match_policy = moment_matching(match_encoder.y, policy)
  match_drift = moment_matching(match_policy.joint(), drift)

  # Get shape and partition info
  ndims_x = x.mean().shape[-1]
  ndims_u = match_policy.y[0].shape[-1]
  ndims_b = ndims_x - len(encoder.active_dims)
  active, inactive = encoder.get_partition_indices(ndims_x)

  # Approx. Cov(a, u) by Cov(a, e) Cov(e, e)^{-1} Cov(e, u) where e = encoder(x)
  if match_encoder.cross[1]:  # is Cov(x, e) premultiplied by Cov(x, x)^{-1}?
    Sax = tf.gather(x.covariance(dense=True), active, axis=-2)
    Sae = Sax @ match_encoder.cross_covariance(preinv=True)
  else:
    Sxe = match_encoder.cross_covariance(dense=True)
    Sae = tf.gather(Sxe, active, axis=-2)
  Sau = Sae @ match_policy.cross_covariance(preinv=True)

  # Approx. Cov(x, f) by Cov(x, d) Cov(d, d)^{-1} Cov(d, f) where d = (e, u)
  _, perm = zip(*sorted(zip(active + inactive, range(ndims_x))))
  Sad = tf.concat([Sae, Sau], axis=-1)
  Sbd = match_drift.x.covariance()[..., -ndims_b - ndims_u: -ndims_u, :]
  Sxd = tf.gather(tf.concat([Sad, Sbd], axis=-2), perm, axis=-2)
  Sxf = Sxd @ match_drift.cross_covariance(preinv=True)
  chain_match_drift = GaussianMatch(x=x, y=match_drift.y, cross=(Sxf, False))

  if noise is None:
    chain_match_noise = None
  else:
    preinv = match_encoder.cross[1]
    match_noise = moment_matching(match_encoder.y, noise)
    # Approx. Cov(x, z) by Cov(x, e) Cov(e, e)^{-1} Cov(e, z) where z is noise
    Sxz = match_encoder.cross_covariance(preinv=preinv)\
          @ match_noise.cross_covariance(preinv=True)
    chain_match_noise = GaussianMatch(x=x, y=match_noise.y, cross=(Sxz, preinv))
  return chain_match_drift, chain_match_noise
