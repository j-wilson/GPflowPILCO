#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import numpy as np

from scipy.linalg import cholesky, cho_solve
from gym.spaces import Box
from gpflow_pilco.envs.utils import RectangleTuple
from gpflow_pilco.envs.ordinary_differential_env import OrdinaryDifferentialEnv


# ==============================================
#                                double_pendulum
# ==============================================
class DoublePendulum(OrdinaryDifferentialEnv):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50
  }

  def __init__(self,
               observation_space: Box = None,
               action_space: Box = None,
               time_per_step: float = 0.1,
               link0: RectangleTuple = None,
               link1: RectangleTuple = None,
               **kwargs):

    if observation_space is None:
      observation_space = Box(low=np.finfo(np.float32).min,
                              high=np.finfo(np.float32).max,
                              dtype=np.float32,
                              shape=(4,))

    if action_space is None:
      action_space = Box(low=-2, high=2, dtype=np.float32, shape=(2,))

    if link0 is None:
      link0 = RectangleTuple(mass=0.5, height=0.5)

    if link1 is None:
      link1 = RectangleTuple(mass=0.5, height=0.5)

    super().__init__(observation_space=observation_space,
                     action_space=action_space,
                     time_per_step=time_per_step,
                     **kwargs)
    self.link0 = link0
    self.link1 = link1
    self.seed()

  def ode_fn(self, t, state_action):
    # Unpack terms
    g = self.link0.gravity
    assert g == self.link1.gravity  # [!] improve me

    l0 = self.link0
    l1 = self.link1
    a0, a1, d_a0, d_a1, *F = state_action
    f0, f1 = np.clip(F, self.action_space.low, self.action_space.high)

    # Construct equations of motion
    z = a0 - a1
    c = np.cos(z)
    s = np.sin(z)

    a00 = (l0.height ** 2) * (l0.mass/3 + l1.mass)
    a01 = a10 = 0.5 * l0.height * l1.height * l1.mass * c
    a11 = (l1.height ** 2) * l1.mass/3
    A = np.array([[a00, a01], [a10, a11]])

    b0 = f0 - l0.friction * d_a0 + l0.height * (
            (0.5 * l0.mass + l1.mass) * g * np.sin(a0)
            - 0.5 * l1.mass * l1.height * s * (d_a1 ** 2))

    b1 = f1 - l1.friction * d_a1 + l1.height * (0.5 * l1.mass * (
            g * np.sin(a1) + l0.height * s * (d_a0 ** 2)))
    b = np.array([b0, b1])

    # Solve for linear system $Ax = b$
    U = cholesky(A, lower=False)
    dd_a0, dd_a1 = cho_solve((U, False), b)

    # Return time derivatives of constrained system
    derivatives = np.clip((d_a0, d_a1, dd_a0, dd_a1),
                          np.subtract(self.observation_space.low, self.state),
                          np.subtract(self.observation_space.high, self.state))

    return np.pad(derivatives, [0, 2])

  def reset(self, state: np.ndarray = None) -> np.ndarray:
    if state is None:
      loc = np.array([np.pi, np.pi, 0.0, 0.0], dtype=np.float32)
      scale = np.array([0.01, 0.01, 0.1, 0.1], dtype=np.float32)
      state = self.np_random.normal(loc=loc, scale=scale)
    self.state = state
    return np.array(self.state)

  def get_vertex_coordinates(self, state):
    """
    Return the Cartesian coordinates of all vertices.
    """
    a0, a1, *_ = state  # absolute angles

    # Tip of first link
    x0 = self.link0.height * -np.sin(a0)
    y0 = self.link0.height * np.cos(a0)

    # Tip of second link
    x1 = x0 + self.link1.height * -np.sin(a1)
    y1 = y0 + self.link1.height * np.cos(a1)
    return (0, 0), (x0, y0), (x1, y1)

  def render(self, mode='human'):
    if self.state is None:
      return None

    if self.viewer is None:
      self.viewer = self.create_viewer()

    a0, a1, d_a0, d_a1 = self.state
    angles = a0 + 0.5 * np.pi, a1 + 0.5 * np.pi  # change in coordinates
    vertices = self.get_vertex_coordinates(self.state)
    transforms = [link.attrs[1] for link in self.viewer.geoms[1::2]]
    for ((x, y), rot, transform) in zip(vertices, angles, transforms):
      transform.set_rotation(rot)
      transform.set_translation(x, y)

    return self.viewer.render(return_rgb_array=mode == 'rgb_array')

  def create_viewer(self):
    from gym.envs.classic_control import rendering
    viewer = rendering.Viewer(600, 400)
    bound = 1.2 * (self.link0.height + self.link1.height)
    viewer.set_bounds(-bound, bound, -bound, bound)

    a0, a1, d_a0, d_a1 = np.pi, np.pi, 0, 0  # initial state
    angles = a0 + 0.5 * np.pi, a1 + 0.5 * np.pi  # change in coordinates
    vertices = self.get_vertex_coordinates((a0, a1))
    heights = self.link0.height, self.link1.height
    bound = 1.2 * sum(heights)

    track = rendering.Line((-bound, 0), (bound, 0))
    viewer.add_geom(track)
    for ((x, y), rot, height) in zip(vertices, angles, heights):
      l, r, b, t = 0, height, -0.025, 0.025  # corner coordinate values
      transform = rendering.Transform(rotation=rot, translation=(x, y))
      link = rendering.make_polygon(((l, b), (l, t), (r, t), (r, b)))
      link.add_attr(transform)
      link.set_color(0.8, 0.6, 0.4)
      viewer.add_geom(link)

      circle = rendering.make_circle(radius=0.05)
      circle.set_color(0.2, 0.2, 0.2)
      circle.add_attr(transform)
      viewer.add_geom(circle)

    return viewer
