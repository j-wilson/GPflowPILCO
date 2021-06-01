#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf
from gpflow_pilco.dynamics.forward_sde import forward_sde
from gpflow_pilco.dynamics.solvers import Euler
from typing import Any, Callable


# ==============================================
#                               dynamical_system
# ==============================================
class DynamicalSystem:
  def __init__(self,
               drift: Callable,
               diffusion: Callable = None,
               policy: Callable = None,
               encoder: Callable = None,
               solver: Callable = None):

    if solver is None:
      solver = Euler()

    self._drift = drift
    self._diffusion = diffusion
    self._policy = policy
    self.encoder = encoder
    self.solver = solver

  def forward(self, t: Any, x: Any) -> Any:
    return forward_sde(x,
                       self.drift,
                       self.diffusion,
                       self.policy,
                       self.encoder)

  def solve_forward(self,
                    initial_time: tf.Tensor,
                    initial_state: tf.Tensor,
                    solution_times: tf.Tensor,
                    **kwargs) -> Any:

    return self.solver(func=self.forward,
                       initial_time=initial_time,
                       initial_state=initial_state,
                       solution_times=solution_times,
                       **kwargs)

  def solve_forward_closure(self,
                            initial_time: tf.Tensor,
                            state_initializer: Callable,
                            solution_times: tf.Tensor,
                            compile: bool = True,
                            **kwargs) -> Callable:

    def closure(state_initializer=state_initializer):
      return self.solve_forward(initial_time=initial_time,
                                initial_state=state_initializer(),
                                solution_times=solution_times,
                                **kwargs)

    return tf.function(closure) if compile else closure

  @property
  def drift(self):
    return self._drift

  @drift.setter
  def drift(self, drift: Callable):
    self._drift = drift

  @property
  def diffusion(self):
    return self._diffusion

  @diffusion.setter
  def diffusion(self, diffusion: Callable):
    self._diffusion = diffusion

  @property
  def policy(self):
    return self._policy

  @policy.setter
  def policy(self, policy: Callable):
    self._policy = policy
