#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import numpy as np

from gym.spaces import Box
from gpflow_pilco.envs.utils import RectangleTuple
from gpflow_pilco.envs.ordinary_differential_env import OrdinaryDifferentialEnv


# ==============================================
#                                      cart_pole
# ==============================================
class CartPole(OrdinaryDifferentialEnv):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50
  }

  def __init__(self,
               observation_space: Box = None,
               action_space: Box = None,
               time_per_step: float = 0.01,
               cart: RectangleTuple = None,
               pole: RectangleTuple = None,
               **kwargs):

    if observation_space is None:
      observation_space = Box(low=np.finfo(np.float32).min,
                              high=np.finfo(np.float32).max,
                              dtype=np.float32,
                              shape=(4,))

    if action_space is None:
      action_space = Box(low=-10, high=10, dtype=np.float32, shape=(1,))

    if cart is None:
      cart = RectangleTuple(mass=0.5, height=0.125, width=0.25, friction=0.1)

    if pole is None:
      pole = RectangleTuple(mass=0.5, height=0.5, width=0.05)

    super().__init__(observation_space=observation_space,
                     action_space=action_space,
                     time_per_step=time_per_step,
                     **kwargs)
    self.cart = cart
    self.pole = pole
    self.seed()

  def ode_fn(self, t, state_action):
    assert self.pole.friction == 0.0, NotImplementedError

    # Unpack terms
    g = self.pole.gravity
    h = self.pole.height
    m = self.pole.mass
    M = self.cart.mass

    state, action = np.split(state_action, [4], axis=-1)
    x, a, d_x, d_a = np.split(state, 4, axis=-1)
    f = np.clip(action, self.action_space.low, self.action_space.high)

    # Compute accelerations
    s = np.sin(a)
    c = np.cos(a)
    drag = -self.cart.friction * d_x
    dd_x = np.divide(
        f + drag + 0.5 * s * m * (h * (d_a ** 2) + 1.5 * g * c),
        (M + m) - 0.75 * m * (c ** 2))  # factor out 4

    dd_a = np.divide(
        c * (f + drag + 0.5 * s * m * h * (d_a ** 2)) + (M + m) * g * s,
        2 / 3 * h * (M + m) - 0.5 * m * h * (c ** 2))  # factor out 6

    # Return time derivatives of constrained system
    derivatives = np.clip(np.concatenate((d_x, d_a, dd_x, dd_a), axis=-1),
                          np.subtract(self.observation_space.low, state),
                          np.subtract(self.observation_space.high, state))

    return np.pad(derivatives, (derivatives.ndim - 1) * [[0, 0]] + [[0, 1]])

  def get_tip_coordinates(self, states):
    """
    Returns the Cartesian coordinates of the pole's tip (sans cart heigth).
    """
    cart_x = states[..., 0]
    pole_a = states[..., 1]
    x = cart_x - self.pole.height * np.sin(pole_a)
    y = self.pole.height * np.cos(pole_a)
    return x, y

  def reset(self, state: np.ndarray = None) -> np.ndarray:
    if state is None:
      loc = np.array([0.0, np.pi, 0.0, 0.0], dtype=np.float32)
      scale = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float32)
      state = self.np_random.normal(loc=loc, scale=scale)
    self.state = state
    return np.array(self.state)

  def render(self, mode='human'):
    if self.state is None:
      return None

    if self.viewer is None:
      self.viewer = self.create_viewer()

    cart, pole, axle, track = self.viewer.geoms
    x, a, *_ = self.state
    scale = self.viewer._coordinate_scale
    x_cart = scale * x + 0.5 * self.viewer.width  # center of cart
    y_cart = track.start[1]  # top of track
    cart.attrs[1].set_translation(x_cart, y_cart)
    pole.attrs[1].set_rotation(a)
    return self.viewer.render(return_rgb_array=mode == 'rgb_array')

  def create_viewer(self):
    from gym.envs.classic_control import rendering
    viewer = rendering.Viewer(600, 400)
    scale = viewer._coordinate_scale = viewer.width / 4

    h_pole = scale * self.pole.height
    w_pole = scale * self.pole.width
    h_cart = scale * self.cart.height
    w_cart = scale * self.cart.width

    l, r = np.multiply((-0.5, 0.5), w_cart)
    b, t = np.multiply((-0.5, 0.5), h_cart)
    cart = rendering.make_polygon(((l, b), (l, t), (r, t), (r, b)))
    cart_trans = rendering.Transform()
    cart.add_attr(cart_trans)
    viewer.add_geom(cart)

    l, r = np.multiply((-0.5, 0.5), w_pole)
    b, t = -0.5 * w_pole, h_pole - 0.5 * w_pole
    pole = rendering.make_polygon(((l, b), (l, t), (r, t), (r, b)))
    pole_trans = rendering.Transform(translation=(0, 0.25 * h_cart))
    pole.add_attr(pole_trans)
    pole.add_attr(cart_trans)
    pole.set_color(0.8, 0.6, 0.4)
    viewer.add_geom(pole)

    axle = rendering.make_circle(radius=0.5 * w_pole)
    axle.add_attr(pole_trans)
    axle.add_attr(cart_trans)
    axle.set_color(0.5, 0.5, 0.8)
    viewer.add_geom(axle)

    h_track = h_pole + 0.5 * h_cart
    track = rendering.Line((0, h_track), (viewer.width, h_track))
    track.set_color(0, 0, 0)
    viewer.add_geom(track)
    return viewer
