#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import numpy as np
import tensorflow as tf

from abc import abstractmethod
from gpflow.config import default_float
from gym import Env
from math import ceil
from tensorflow_probability.python.distributions import Distribution
from typing import Callable, Dict, Iterable, List, NamedTuple, Union

# ---- Exports
__all__ = ("EpisodeData", "EpisodeSpec", "AbstractLoop",)

# ==============================================
#                                           core
# ==============================================
ArrayLike = Union[np.ndarray, tf.Tensor]


class EpisodeSpec(NamedTuple):
  state_distrib: Distribution
  horizon: float
  step_size: float
  initial_time: float = 0.0

  @property
  def num_steps(self):
    return int(ceil(self.horizon / self.step_size))


class EpisodeData(NamedTuple):
  states: ArrayLike
  actions: ArrayLike
  metrics: dict


class AbstractLoop:
  def __init__(self,
               env: Env,
               episode_spec: EpisodeSpec,
               metrics: Dict[str, Callable] = None,
               episodes: List[EpisodeData] = None,
               step_callbacks: List[Callable] = None,
               unroll_callbacks: List[Callable] = None):

    if metrics is None:
      metrics = dict()

    if episodes is None:
      episodes = list()

    if step_callbacks is None:
      step_callbacks = list()

    if unroll_callbacks is None:
      unroll_callbacks = list()

    self.env = env
    self.episode_spec = episode_spec
    self.metrics = metrics
    self.episodes = episodes
    self.step_callbacks = step_callbacks
    self.unroll_callbacks = unroll_callbacks

  @abstractmethod
  def policy_closure(self, *args, **kwargs):
    raise NotImplementedError

  def step(self,
           policy: Callable = None,
           initial_state: ArrayLike = None,
           callbacks: Iterable[Callable] = "default"):

    if policy is None:
      policy = self.policy_closure()

    if callbacks == "default":
      callbacks = self.step_callbacks

    states, actions = self.unroll(policy=policy, initial_state=initial_state)
    metrics = {name: fn(states, actions) for name, fn in self.metrics.items()}
    episode = EpisodeData(states=states, actions=actions, metrics=metrics)
    if callbacks is not None:
      for callback in callbacks:
        callback(step=len(self.episodes), episode=episode)

    self.episodes.append(episode)
    return episode

  def unroll(self,
             policy: Callable,
             initial_state: ArrayLike = None,
             callbacks: Iterable[Callable] = "default"):

    if initial_state is None:
      initial_state = self.episode_spec.state_distrib.sample()

    if callbacks == "default":
      callbacks = self.unroll_callbacks

    _ = self.env.reset(state=initial_state)
    state = initial_state
    states = [initial_state]
    actions = []
    with self.env:
      for step in range(self.episode_spec.num_steps):
        action = policy(state)
        state, *_ = self.env.step(action)
        if callbacks is not None:
          for callback in callbacks:
            callback(state=state, action=action)

        states.append(state)
        actions.append(action)

    states = np.asarray(states, dtype=default_float())
    actions = np.asarray(actions, dtype=default_float())
    return states, actions

  def get_state_action_pairs(self):
    states = []
    actions = []
    for episode in self.episodes:
      states.append(episode.states)
      actions.append(episode.actions)

    states = tf.convert_to_tensor(states, dtype=default_float())
    actions = tf.convert_to_tensor(actions, dtype=default_float())
    return states, actions
