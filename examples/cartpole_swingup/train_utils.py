#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from examples.cartpole_swingup.settings import (drift_spec,
                                                policy_spec,
                                                resolve_default_keywords)
from gpflow.models.training_mixins import InternalDataTrainingLossMixin,\
                                          ExternalDataTrainingLossMixin
from gpflow.utilities import Dispatcher
from gpflow_pilco.loops import ModelBasedRL, PathwisePILCO
from gpflow_pilco.utils.optimizers import GradientDescent
from gpflow_sampling.models import PathwiseGPModel
from gpflow_sampling.sampling.core import AbstractSampler
from tensorflow.python.data.ops.iterator_ops import OwnedIterator
from typing import List

# ---- Exports
dynamics = Dispatcher("train_utils_dynamics")
policy = Dispatcher("train_utils_policy")
__all__ = ("dynamics", "policy")


# ==============================================
#                                    train_utils
# ==============================================
@dynamics.register(object, ModelBasedRL, InternalDataTrainingLossMixin)
def _train_dynamics_internal(optimizer: object,
                             loop: ModelBasedRL,
                             drift: InternalDataTrainingLossMixin,
                             /,
                             variables: List[tf.Variable] = None,
                             compile: bool = True,
                             **kwargs):
  assert loop.drift == drift
  if variables is None:
    variables = drift.trainable_variables

  closure = loop.dynamics_loss_closure(compile=compile)
  return optimizer.minimize(closure, variables, **kwargs)


@dynamics.register(object, ModelBasedRL, ExternalDataTrainingLossMixin)
def _train_dynamics_external_fallback(optimizer: object,
                                      loop: ModelBasedRL,
                                      drift: ExternalDataTrainingLossMixin,
                                      /,
                                      variables: List[tf.Variable] = None,
                                      compile: bool = True,
                                      **kwargs):
  assert loop.drift == drift
  if variables is None:
    variables = drift.trainable_variables

  data = loop.get_data_dynamics(flatten=True)
  closure = loop.dynamics_loss_closure(data, compile=compile)
  return optimizer.minimize(closure, variables, **kwargs)


@dynamics.register(GradientDescent, ModelBasedRL, ExternalDataTrainingLossMixin)
@resolve_default_keywords(defaults_factory=drift_spec, prefix="@")
def _train_dynamics_external_gd(optimizer: object,
                                loop: ModelBasedRL,
                                drift: object,
                                /,
                                variables: List[tf.Variable] = None,
                                compile: bool = True,
                                batch_size: int = "@batch_size",
                                **kwargs):
  assert loop.drift == drift
  if variables is None:
    variables = drift.trainable_variables

  data = loop.get_data_dynamics(flatten=True)
  num_data = data[0].shape[0]
  if batch_size is not None and num_data > batch_size:
    data = OwnedIterator(tf.data.Dataset.from_tensor_slices(data)
                           .shuffle(num_data, reshuffle_each_iteration=True)
                           .batch(batch_size, drop_remainder=True)
                           .repeat(-1))

  closure = loop.dynamics_loss_closure(data, compile=compile)
  return optimizer.minimize(closure, variables, **kwargs)


@policy.register(object, ModelBasedRL, object, object)
def _train_policy_fallback(optimizer: object,
                           loop: ModelBasedRL,
                           drift: object,
                           policy: object,
                           /,
                           variables: List[tf.Variable] = None,
                           compile: bool = True,
                           **kwargs):
  assert loop.policy == policy
  if variables is None:
    variables = policy.trainable_variables

  closure = loop.policy_loss_closure(compile=compile)
  return optimizer.minimize(closure, variables, **kwargs)


@policy.register(object, PathwisePILCO, PathwiseGPModel, object)
@resolve_default_keywords(defaults_factory=policy_spec, prefix="@")
def _train_policy_pathwise(optimizer: object,
                           loop: PathwisePILCO,
                           drift: PathwiseGPModel,
                           policy: object,
                           /,
                           variables: List[tf.Variable] = None,
                           compile: bool = True,
                           batch_size: int = "@batch_size",
                           num_bases: int = "@num_bases",
                           paths: AbstractSampler = None,
                           **kwargs):

  assert loop.policy == policy
  if variables is None:
    variables = policy.trainable_variables

  batch_closure = loop.policy_loss_closure(compile=False,
                                           batch_size=batch_size,
                                           num_bases=num_bases,
                                           paths=paths)

  def mean_closure():
    return tf.reduce_mean(batch_closure())

  closure = tf.function(mean_closure) if compile else mean_closure
  return optimizer.minimize(closure, variables, **kwargs)
