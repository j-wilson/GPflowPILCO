#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import pickle

import gpflow.kernels
import tensorflow as tf

from functools import partial, update_wrapper
from gpflow.config import default_float
from gpflow.likelihoods import Likelihood, Gaussian as GaussianLikelihood
from gpflow.utilities import set_trainable
from gpflow_pilco.loops.core import ArrayLike, EpisodeSpec
from gpflow_pilco.loops.model_based_rl import CheckpointedModelBasedRL
from gpflow_pilco.models import (InverseLinkWrapper,
                                    KernelRegressor,
                                    PathwiseSVGP,
                                    SVGP)

from gpflow_pilco.moment_matching.gaussian import GaussianMoments
from gpflow_pilco.moment_matching import moment_matching
from gpflow_pilco.dynamics.solvers import MomentMatchingEuler
from gpflow_sampling.sampling.core import AbstractSampler
from numpy import arange
from pathlib import Path
from tensorflow_probability.python.distributions import Distribution
from typing import Callable, Tuple, Type, Union

# ---- Exports
__all__ = ("AbstractPILCO", "MomentMatchingPILCO", "PathwisePILCO")


# ==============================================
#                                          pilco
# ==============================================
class AbstractPILCO(CheckpointedModelBasedRL):
  def __init__(self, *args, diffusion: Callable = None, **kwargs):
    assert diffusion is None, NotImplementedError
    super().__init__(*args, **kwargs)

  def build_dynamics(self,
                     cls: Type,
                     num_centers: int,
                     data: Tuple = None,
                     likelihood: Likelihood = "default",
                     model_uncertainty: bool = True,
                     invlink: Callable = None,
                     **kwargs):
    """
    Initializes a model-based drift function.
    """
    if data is None:
      data = self.get_data_dynamics(flatten=True)

    if likelihood == "default":
      likelihood = GaussianLikelihood()

    drift = cls.initialize(data=data,
                           num_inducing=num_centers,
                           likelihood=likelihood,
                           **kwargs)

    if drift.q_mu.shape[-2] >= len(data[0]):
      set_trainable(drift.inducing_variable, False)

    if not model_uncertainty:
      drift = KernelRegressor(model=drift)
      set_trainable(drift.q_sqrt, False)
      for kernel in drift.kernel.kernels:
        set_trainable(kernel.variance, False)

    if invlink is not None:
      drift = InverseLinkWrapper(model=drift, invlink=invlink)

    return drift, None

  def build_policy(self,
                   cls: Type,
                   num_centers: int,
                   data: Tuple = None,
                   likelihood: Likelihood = None,
                   mean_function: Callable = None,
                   model_uncertainty: bool = False,
                   invlink: Callable = None,
                   **kwargs):
    if data is None:
      data = self.get_data_policy(flatten=True)

    model = cls.initialize(data=data,
                           num_inducing=num_centers,
                           likelihood=likelihood,
                           mean_function=mean_function,
                           **kwargs)

    if not model_uncertainty:
      model = KernelRegressor(model=model)
      set_trainable(model.q_sqrt, False)
      for kernel in model.kernel.kernels:
        set_trainable(kernel.variance, False)

    if invlink is not None:
      model = InverseLinkWrapper(model=model, invlink=invlink)

    return model

  def restore_or_initialize(self,
                            filepath: Union[Path, str] = None,
                            build_dynamics_kwargs: dict = None,
                            build_policy_kwargs: dict = None):

    if build_dynamics_kwargs is None:
      build_dynamics_kwargs = dict()

    if build_policy_kwargs is None:
      build_policy_kwargs = dict()

    if filepath is None:
      if len(self.checkpoints) == 0:
        return  # nothing to restore
      filepath = self.latest_checkpoint

    # Fetch episode data
    step_count = self.read_checkpoint("^step_counter", filepath)[0]
    with self.directory.joinpath("episodes.pkl").open('rb') as f:
      episodes = pickle.load(f)[:step_count]

    # Maybe build models
    if step_count > 1:
      self.episodes = episodes[:-1]  # at this point, models unaware of last ep
      self.drift, _ = self.build_dynamics(**build_dynamics_kwargs)
      self.policy = self.build_policy(**build_policy_kwargs)
      self.checkpoint.restore(filepath)  # restore variables
    self.episodes = episodes


class MomentMatchingPILCO(AbstractPILCO):
  def __init__(self, *args, solver: Callable = None, **kwargs):
    if solver is None:
      solver = MomentMatchingEuler()
    super().__init__(*args, solver=solver, **kwargs)

  def build_dynamics(self, num_centers: int, **kwargs):
    return super().build_dynamics(cls=SVGP, num_centers=num_centers, **kwargs)

  def build_policy(self,
                   num_centers: int,
                   data: Tuple = None,
                   q_mu: Union[tf.Tensor, tf.Variable] = None,
                   **kwargs):

    if data is None:
      data = self.get_data_policy(flatten=True)

    if q_mu is None:
      '''
      The moment matched covariance of a kernel regressor with constant-valued
      targets q_mu is zero, which can lead to numerical issues. Hence, we
      randomly initialize to something small.
      '''
      q_mu = 1e-3 * tf.random.normal([num_centers, data[1].shape[-1]],
                                     dtype=default_float())

    return super().build_policy(cls=SVGP,
                                num_centers=num_centers,
                                data=data,
                                q_mu=q_mu,
                                **kwargs)

  def dynamics_loss_closure(self, *args, **kwargs):
    return self.drift.training_loss_closure(*args, **kwargs)

  def policy_loss_closure(self,
                          episode_spec: EpisodeSpec = None,
                          state_initializer: Callable = None,
                          **kwargs):
    if episode_spec is None:
      episode_spec = self.episode_spec

    if state_initializer is None:
      state_initializer = self.get_state_initializer(episode_spec.state_distrib)

    solution_times = arange(1, 1 + episode_spec.num_steps, dtype=default_float())
    return self._policy_loss_closure(state_initializer=state_initializer,
                                     initial_time=episode_spec.initial_time,
                                     solution_times=solution_times,
                                     **kwargs)

  def _policy_loss_closure(self,
                           state_initializer: Callable,
                           initial_time: float,
                           solution_times: ArrayLike,
                           compile: bool = True,
                           **kwargs):

    def _accumulate_loss(t: tf.Tensor,
                         state: Tuple[tf.Tensor, tf.Tensor],
                         loss: tf.Tensor):
      x = GaussianMoments(moments=state, centered=True)
      if self.encoder is not None:
        x = moment_matching(x, self.encoder).y
      return loss + self.objective(x=x, t=t)

    def _closure(state_initializer):
      mx, Sxx = state_initializer()
      loss = tf.zeros(mx.shape[:-1], dtype=default_float())
      calls_and_inits = (_accumulate_loss, loss),
      _, loss = self.solve_forward(iterator=tf.foldl,
                                   initial_time=initial_time,
                                   initial_state=(mx, Sxx),
                                   solution_times=solution_times,
                                   callbacks_and_initializers=calls_and_inits,
                                   **kwargs)
      return loss

    wrapper = update_wrapper(partial(_closure, state_initializer), _closure)
    return tf.function(wrapper) if compile else wrapper

  def get_state_initializer(self, p: Distribution):
    mx = tf.cast(p.mean(), default_float())[None]
    Sxx = tf.cast(p.covariance(), default_float())[None]
    def _initializer():  # represent initial state in terms of moments
      return mx, Sxx
    return _initializer


class PathwisePILCO(AbstractPILCO):
  def build_dynamics(self, num_centers: int, **kwargs):
    return super().build_dynamics(cls=PathwiseSVGP,
                                  num_centers=num_centers,
                                  **kwargs)

  def build_policy(self, num_centers: int, **kwargs):
    return super().build_policy(cls=SVGP,
                                num_centers=num_centers,
                                **kwargs)

  def dynamics_loss_closure(self, *args, **kwargs):
    return self.drift.training_loss_closure(*args, **kwargs)

  def policy_loss_closure(self,
                          episode_spec: EpisodeSpec = None,
                          state_initializer: Callable = None,
                          batch_size: int = 128,
                          **kwargs):

    if episode_spec is None:
      episode_spec = self.episode_spec

    if state_initializer is None:
      state_initializer = self.get_state_initializer(episode_spec.state_distrib,
                                                     batch_size=batch_size)

    solution_times = arange(1, 1 + episode_spec.num_steps, dtype=default_float())
    return self._policy_loss_closure(state_initializer=state_initializer,
                                     initial_time=episode_spec.initial_time,
                                     solution_times=solution_times,
                                     **kwargs)

  def _policy_loss_closure(self,
                           state_initializer: Callable,
                           initial_time: float,
                           solution_times: ArrayLike,
                           compile: bool = True,
                           num_bases: int = 1024,
                           paths: AbstractSampler = None,
                           **kwargs):

    def _accumulate_loss(t: tf.Tensor, state: tf.Tensor, loss: tf.Tensor):
      if self.encoder is not None:
        state = self.encoder(state)
      return loss + self.objective(x=state, t=t)

    def _closure(state_initializer, paths):
      state = state_initializer()
      loss = tf.zeros(state.shape[:-1], dtype=default_float())
      calls_and_inits = (_accumulate_loss, loss),
      if paths is None:  # generate new sample paths with each call of _closure
        _paths = self.drift.generate_paths(num_samples=state.shape[0],
                                           num_bases=num_bases,
                                           sample_axis=0)
      else:
        _paths = paths

      with self.drift.set_temporary_paths(_paths):
        _, loss = self.solve_forward(iterator=tf.foldl,
                                     initial_time=initial_time,
                                     initial_state=state,
                                     solution_times=solution_times,
                                     callbacks_and_initializers=calls_and_inits,
                                     **kwargs)
      return loss

    wrapper = update_wrapper(partial(_closure, state_initializer, paths), _closure)
    return tf.function(wrapper) if compile else wrapper

  def get_state_initializer(self, p: Distribution, batch_size: int = 128):
    def _initializer():  # Propagate initial states 1-to-1 with paths
      return p.sample(sample_shape=[batch_size])
    return _initializer
