#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import numpy as np
import tensorflow as tf

from gpflow.base import Parameter, TensorLike
from gpflow.config import default_float
from gpflow.kernels import Kernel
from gpflow.inducing_variables import InducingPoints
from gpflow.utilities import positive
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import pdist
from tensorflow_probability.python.bijectors import Bijector, Sigmoid
from typing import *
from warnings import warn

# ---- Exports
__all__ = ("lengthscales_median", "inducingPoints_kmeans", "replace_duplicates")


# ==============================================
#                                   initializers
# ==============================================
def lengthscales_median(x: np.ndarray,
                        transform: Bijector = "default",
                        lower: float = 0.01,
                        upper: float = 100.0):
  """
  Initialize lengthscales using the median heuristic.
  """
  if transform == "default":
    if upper is None:
      transform = positive(lower=lower)
    else:
      transform = Sigmoid(low=tf.cast(lower, default_float()),
                          high=tf.cast(upper, default_float()))

  # TODO: Improve me, avoid initializing at the boundaries
  _lower = None if (lower is None) else 1.1 * lower
  _upper = None if (upper is None) else 0.9 * upper

  dist = pdist(x, metric='euclidean')
  init = np.full(x.shape[-1],
                 np.clip(np.sqrt(0.5) * np.median(dist), _lower, _upper))
  return Parameter(init, transform=transform)


def inducingPoints_kmeans(x: np.ndarray,
                          num_inducing: int,
                          batch_size: int = 1024,
                          kernel_and_tol: Tuple[Kernel, float] = None,
                          init: str = "k-means++",
                          seed: Optional[int] = None,
                          name: Optional[str] = None,
                          **kwargs) -> InducingPoints:
  """
  Initialize inducing points using k-means.
  """
  assert len(x.shape) == 2  # TODO improve me
  n = x.shape[0]
  if n <= num_inducing:
    points = np.array(x)
  else:
    kmeans = MiniBatchKMeans(n_clusters=num_inducing,
                             batch_size=min(n, batch_size),
                             random_state=seed,
                             init=init)
    kmeans.fit(x)
    points = kmeans.cluster_centers_
    nmissing = num_inducing - len(points)
    if nmissing > 0:
      rvs = tf.random.randn([nmissing, points.shape[-1]])
      points = np.vstack([points, rvs])

  if kernel_and_tol is not None and kernel_and_tol[1] < 1:
    points = replace_duplicates(points, *kernel_and_tol)

  Z = Parameter(value=points, **kwargs)
  return InducingPoints(Z=Z, name=name)


def replace_duplicates(points: TensorLike,
                       kernel: Kernel,
                       tol: float,
                       num_attempts: int = 32):
  """
  Attempt to replace excessively correlated points.
  """
  if tol == 1:
    return points

  assert 0 < tol < 1
  ivar = tf.math.reciprocal(kernel.variance).numpy()
  corr = np.array(ivar * kernel(points, full_cov=True))
  assert corr.ndim == 2

  corr[np.diag_indices_from(corr)] = -np.inf  # ignore auto-correlation
  hits = np.sum(corr > tol, axis=-1)
  while np.any(hits > 0):
    index = np.argmax(hits)
    original = points[index]
    for attempt in range(num_attempts):
      rvs = tf.random.normal(original.shape, dtype=default_float())
      alt = original + 1e-3 * (1.1 ** attempt) * rvs  # perturb original
      xorr = ivar * np.squeeze(kernel(points, alt[None]), axis=-1)
      xorr[index] = -np.inf
      if not np.any(xorr >= tol):
        points[index] = alt
        corr[index, :] = xorr
        corr[:, index] = xorr
        break

      if attempt + 1 == num_attempts:  # skip this point
        warn("Failed to replace an overly correlated point; continuing...")
        corr[index, :] = -np.inf
        corr[:, index] = -np.inf

    hits = np.sum(corr > tol, axis=-1)
  return points
