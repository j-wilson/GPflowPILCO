#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import numpy as np
import tensorflow as tf
from itertools import groupby


# ==============================================
#                                        metrics
# ==============================================
class Metrics:
  def __init__(self, loop):
    """
    Helper class for defining the metrics that are tracked
    at each step of the outer-loop.
    """
    self.loop = loop

  def rewards(self, states: np.ndarray, actions: np.ndarray, **ignore):
    feats = self.loop.featurize_states(states)
    return -self.loop.objective(feats)

  def success(self,
              states: np.ndarray,
              actions: np.ndarray,
              radius: float = None,
              prox_threshold: float = 0.2,
              num_consecutive: int = 10,
              **ignore):

    if radius is None:
      radius = self.loop.env.pole.height

    x, y = self.loop.env.get_tip_coordinates(states)
    prox = np.sqrt(x ** 2 + (y - radius) ** 2) < (prox_threshold * radius)
    for _, group in filter(lambda kg: kg[0] == 1, groupby(prox)):
      if sum(1 for _ in group) >= num_consecutive:
        return True  # consecutively within a prescribed distance from goal
    return False

  def expected_reward(self,
                      states: np.ndarray,
                      actions: np.ndarray,
                      **ignore):
      if self.loop.drift is None:
        return np.nan

      closure = self.loop.policy_loss_closure(compile=False)
      return -tf.reduce_mean(closure())

  def validation_reward(self,
                        states: np.ndarray,
                        actions: np.ndarray,
                        num_samples: int = 100,
                        **ignore):

    times = np.arange(0, 1 + self.loop.episode_spec.num_steps)
    policy = self.loop.policy_closure()
    reward = 0.0

    for k in range(num_samples):
      states, actions = self.loop.unroll(policy, callbacks=None)
      feats = self.loop.featurize_states(states)
      reward -= tf.reduce_sum(self.loop.objective(x=feats, t=times))
    return float(reward / num_samples)

  def validation_success(self,
                         states: np.ndarray,
                         actions: np.ndarray,
                         num_samples: int = 100,
                         **ignore):

    policy = self.loop.policy_closure()
    count = 0
    for k in range(num_samples):
      states, actions = self.loop.unroll(policy, callbacks=None)
      count += self.success(states, actions)
    return count / num_samples
