#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import gpflow
import tensorflow as tf

from gpflow.config import default_float
from gpflow.inducing_variables import SeparateIndependentInducingVariables
from gpflow.kernels import (Kernel,
                            SquaredExponential,
                            SeparateIndependent,
                            LinearCoregionalization)
from gpflow.likelihoods import Likelihood
from gpflow.utilities import deepcopy
from gpflow_pilco.models import initializers
# from gpflow_pilco.models.core import CallableModel, LabelledData
from gpflow_pilco.models.core import LabelledData
from gpflow_pilco.models.mean_functions import Constant
from gpflow_sampling import models as pathwise_models
from typing import Callable, List

# ---- Exports
__all__ = ("SVGP", "PathwiseSVGP")


# ==============================================
#                                          svgp
# ==============================================
class SVGP(Callable, gpflow.models.SVGP):
  def __init__(self, *args, prior: Callable = None, **kwargs):
    super().__init__(*args, **kwargs)
    self.prior = prior

  def __call__(self, x: tf.Tensor, **kwargs):
    return self.predict_f(x, **kwargs)

  def maximum_log_likelihood_objective(self, data: LabelledData):
    objective = self.elbo(data)
    if self.prior is not None:
      objective += self.prior(self)
    return objective

  @classmethod
  def initialize(cls,
                 data: LabelledData,
                 num_inducing: int,
                 likelihood: Likelihood = None,
                 mean_function: Callable = "default",
                 kernels: List[Kernel] = None,
                 coregionalize: bool = None,
                 num_latent_gps: int = None,
                 max_corr: float = 1.0,
                 **kwargs):

    x, y = data
    num_data, num_output_dims = y.shape
    if num_latent_gps is None:
      num_latent_gps = num_output_dims

    if coregionalize is None:
      coregionalize = num_output_dims != num_latent_gps

    # Construct mean function
    if mean_function == "default":
      mean_function = Constant(c=num_output_dims * [0])

    # Initialize kernel covariance functions
    if kernels is None:
      kernels = []
      for i in range(num_latent_gps):
        kern = SquaredExponential()
        kern.lengthscales = initializers.lengthscales_median(x)
        kernels.append(kern)

    # Find starting positions for inducing points
    def keygen(kern):
      if max_corr == 1:  # no replacement, so treat all kernels as the same
        return None

      class_id = kern.__class__
      variance = tuple(kernel.variance.numpy())
      lenscale = tuple(kern.lengthscales.numpy())
      return (class_id, variance, lenscale)

    cache = dict()
    points = []
    for i, kern in enumerate(kernels):
      key = keygen(kern)
      if key in cache:
        pts = deepcopy(cache[key])
      else:
        pts = cache[key]\
            = initializers.inducingPoints_kmeans(x=x,
                                                 num_inducing=num_inducing,
                                                 kernel_and_tol=(kern, max_corr))
      points.append(pts)

    # Maybe correlate latent GPs
    if coregionalize:
      if num_output_dims == num_latent_gps:
        W = tf.eye(num_output_dims, dtype=default_float())
      else:
        W = tf.linalg.l2_normalize(
            tf.random.normal(shape=[num_output_dims, num_latent_gps],
                             dtype=default_float()), axis=-1)

      kernel = LinearCoregionalization(kernels=kernels, W=W)
    else:
      assert num_output_dims == num_latent_gps
      kernel = SeparateIndependent(kernels=kernels)

    return cls(kernel=kernel,
               likelihood=likelihood,
               mean_function=mean_function,
               inducing_variable=SeparateIndependentInducingVariables(points),
               num_latent_gps=num_latent_gps,
               **kwargs)


class PathwiseSVGP(pathwise_models.PathwiseSVGP, SVGP):
  def __init__(self, *args, prior: Callable = None, **kwargs):
      pathwise_models.PathwiseSVGP.__init__(self, *args, **kwargs)
      self.prior = prior

  def __call__(self, x: tf.Tensor, **kwargs):
    return self.predict_f_samples(x, **kwargs)