#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import numpy as np
import tensorflow as tf
from gpflow import mean_functions

# ---- Exports
__all__ = ('Zero', 'Constant')


# ==============================================
#                                 mean_functions
# ==============================================
class Zero(mean_functions.Zero):
  """
  Modified version of <gpflow.mean_functions.Constant>,
  supporting n-dimensional input tensors X.
  """
  def __call__(self, X):
    return tf.zeros_like(X[..., :1])  # TODO: improve me...


class Constant(mean_functions.Constant):
    """
    Modified version of <gpflow.mean_functions.Constant>,
    supporting n-dimensional input tensors X.
    """
    def __call__(self, X):
      shape_x = tuple(X.shape)
      ndims_x = len(shape_x)
      ndims_c = len(tuple(self.c.shape))
      return tf.tile(self.c[(ndims_x - ndims_c) * (np.newaxis,)],
                     shape_x[:ndims_x - ndims_c] + ndims_c * (1,))
