#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import numpy as np

from abc import abstractmethod
from gym import Env
from gym.spaces import Space
from gym.utils.seeding import np_random
from gpflow_pilco.dynamics.solvers import ScipyODE
from typing import Callable


# ==============================================
#                      ordinary_differential_env
# ==============================================
class OrdinaryDifferentialEnv(Env):
  def __init__(self,
               observation_space: Space,
               action_space: Space,
               ode_solver: Callable = None,
               time_per_step: float = 1.0):

    if ode_solver is None:
      ode_solver = ScipyODE()

    self.observation_space = observation_space
    self.action_space = action_space
    self.ode_solver = ode_solver
    self.time_per_step = time_per_step
    self.viewer = None
    self._state = None

  @abstractmethod
  def ode_fn(self, time, state_action, **kwargs):
    raise NotImplementedError

  def solve_ode(self, action, initial_time: float = 0.0, **kwargs):
    assert self.action_space.contains(action)
    state_vec = np.ravel(self.state)
    action_vec = np.ravel(action)
    state_action = np.hstack([state_vec, action_vec])
    solution_time = np.array([initial_time + self.time_per_step])
    solution_state = self.ode_solver(func=self.ode_fn,
                                     initial_time=initial_time,
                                     initial_state=state_action,
                                     solution_times=solution_time,
                                     **kwargs)[0, :state_vec.size]
    return np.reshape(solution_state, np.shape(self.state))

  def step(self, action, **kwargs):
    self.state = self.solve_ode(action=action, **kwargs)
    return self.state, 0.0, False, {}

  def close(self):
    if self.viewer:
      import pyglet  # schedule the viewer's window to immediately close
      pyglet.clock.schedule_once(lambda dt: self.viewer.close(), 0)
      pyglet.app.run()
      self.viewer = None

  def seed(self, seed: int = None):
    self.np_random, seed = np_random(seed)
    return [seed]

  @property
  def state(self):
    return self._state

  @state.setter
  def state(self, state):
    self._state = state
