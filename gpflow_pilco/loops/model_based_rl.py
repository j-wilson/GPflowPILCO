#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import pickle
import re
import tensorflow as tf

from abc import abstractmethod
from gpflow.config import default_float
from gpflow_pilco.loops.core import AbstractLoop, EpisodeSpec
from gpflow_pilco.dynamics.dynamical_system import DynamicalSystem
from gym import Env
from pathlib import Path
from tensorflow.python.training.py_checkpoint_reader import NewCheckpointReader
from typing import Callable, Union

# ---- Exports
__all__ = ("ModelBasedRL", "CheckpointedModelBasedRL")


# ==============================================
#                                 model_based_rl
# ==============================================
class ModelBasedRL(AbstractLoop, DynamicalSystem):
  def __init__(self,
               env: Env,
               episode_spec: EpisodeSpec,
               objective: Callable,
               drift: Callable = None,
               diffusion: Callable = None,
               policy: Callable = None,
               encoder: Callable = None,
               solver: Callable = None,
               **kwargs):

    AbstractLoop.__init__(self=self,
                          env=env,
                          episode_spec=episode_spec,
                          **kwargs)

    DynamicalSystem.__init__(self=self,
                             drift=drift,
                             diffusion=diffusion,
                             policy=policy,
                             encoder=encoder,
                             solver=solver)

    self.objective = objective

  @abstractmethod
  def dynamics_loss_closure(self, *args, **kwargs):
    raise NotImplementedError

  @abstractmethod
  def policy_loss_closure(self, *args, **kwargs):
    raise NotImplementedError

  def policy_closure(self, compile: bool = True):
    if self.policy is None:
      def _closure(state):  # random initial policy
        return self.env.action_space.sample()
      return _closure

    def _closure(state):  # model-based policy
      _state = tf.cast(state, dtype=default_float())[None]
      _feats = self.featurize_states(_state)
      return tf.squeeze(self.policy(_feats), axis=0)

    return tf.function(_closure) if compile else _closure

  def get_data_dynamics(self, flatten: bool = False):
    x, u = self.get_state_action_pairs()
    z = x if (self.encoder is None) else self.encoder(x)
    zu = tf.concat([z[:, :-1, :], u], axis=-1)
    dx = x[:, 1:, :] - x[:, :-1, :]
    if flatten:
      zu = tf.reshape(zu, (-1, zu.shape[-1]))
      dx = tf.reshape(dx, (-1, dx.shape[-1]))
    return zu, dx

  def get_data_policy(self, flatten: bool = False):
    x, u = self.get_state_action_pairs()
    z = x if (self.encoder is None) else self.encoder(x)
    if flatten:
      z = tf.reshape(z, (-1, z.shape[-1]))
      u = tf.reshape(u, (-1, u.shape[-1]))
    return z, u

  def featurize_states(self, x: tf.Tensor) -> tf.Tensor:
    return x if (self.encoder is None) else self.encoder(x)


class CheckpointedModelBasedRL(tf.train.CheckpointManager, ModelBasedRL):
  def __init__(self,
               directory: Union[Path, str],
               env,
               episode_spec: EpisodeSpec,
               objective: Callable,
               checkpoint: tf.train.Checkpoint = None,
               step_counter: tf.Variable = None,
               max_to_keep: int = None,
               **kwargs):

    if isinstance(directory, str):
      directory = Path(directory)

    if step_counter is None:
      step_counter = tf.Variable(tf.zeros([], tf.int64), trainable=False)

    tf.train.CheckpointManager.__init__(self=self,
                                        directory=directory,
                                        checkpoint=None,
                                        max_to_keep=max_to_keep,
                                        step_counter=step_counter)

    ModelBasedRL.__init__(self=self,
                          env=env,
                          episode_spec=episode_spec,
                          objective=objective,
                          **kwargs)

    self._checkpoint = checkpoint
    self._step_counter = step_counter

  @abstractmethod
  def restore_or_initialize(self, filepath: Union[Path, str] = None, **kwargs):
    raise NotImplementedError

  @property
  def checkpoint(self):
    # TODO: Improve me
    if self._checkpoint is None:  # lazy initialization
      self._checkpoint = tf.train.Checkpoint(step_counter=self._step_counter)

    self._checkpoint.drift = self.drift
    self._checkpoint.policy = self.policy
    return self._checkpoint

  def read_checkpoint(self, pattern: str, filepath: Union[Path, str] = None):
    if filepath is None:
      filepath = self.latest_checkpoint

    reader = NewCheckpointReader(filepath)
    dtype_map = reader.get_variable_to_dtype_map()
    retvals = list()
    for name, dtype in dtype_map.items():
      if re.search(pattern, name) is not None:
        retvals.append(reader.get_tensor(name))

    return retvals if len(retvals) else None

  def save(self, step_count: int, **kwargs):
    self.checkpoint.step_counter.assign(step_count)
    filepath = super().save(**kwargs)
    with self.directory.joinpath("episodes.pkl").open('wb') as f:
      pickle.dump(self.episodes, f)
    return filepath

