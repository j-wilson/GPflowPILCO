#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from abc import abstractmethod
from gpflow.likelihoods import Gaussian as GaussianLikelihood
from gpflow.kernels import SharedIndependent, SeparateIndependent, LinearCoregionalization
from gpflow.models import GPModel

# ---- Exports
__all__ = ("AbstractSNR", "PilcoPenaltySNR")


# ==============================================
#                                         priors
# ==============================================
class AbstractSNR:
  def get_log_snr(self, model: GPModel):
    assert isinstance(model.likelihood, GaussianLikelihood)
    log_noise = tf.math.log(model.likelihood.variance)

    if isinstance(model.kernel, SharedIndependent):
      log_signals = tf.fill(model.num_latent_gps,
                            tf.math.log(model.kernel.kernel.variance))
    elif isinstance(model.kernel, SeparateIndependent):
      log_signals = tf.math.log(
          tf.stack([k.variance for k in model.kernel.kernels]))
    elif isinstance(model.kernel, LinearCoregionalization):
      log_signals = tf.math.log(
          tf.linalg.matvec(model.kernel.W ** 2,
                           [k.variance for k in model.kernel.kernels]))
    else:
      log_signals = tf.math.log(model.kernel.variance)

    return log_signals - log_noise

  @abstractmethod
  def __call__(self, model: GPModel) -> tf.Tensor:
    raise NotImplementedError


class PilcoPenaltySNR(AbstractSNR):
  def __init__(self, threshold: float, power: float):
    self.threshold = threshold
    self.power = power

  def __call__(self, model: GPModel):
    log_snr = self.get_log_snr(model)
    log_thresh = tf.math.log(tf.cast(self.threshold, log_snr.dtype))
    return -tf.reduce_sum(tf.math.pow(log_snr/log_thresh, self.power))









