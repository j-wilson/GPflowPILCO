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
#                                   mountain_car
# ==============================================
class MountainCar(OrdinaryDifferentialEnv):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second': 50
  }

  def __init__(self,
               observation_space: Box = None,
               action_space: Box = None,
               time_per_step: float = 0.01,
               car: RectangleTuple = None):

    if observation_space is None:
      observation_space = Box(low=np.array([-1.5, np.finfo(np.float32).min]),
                              high=np.array([1.5, np.finfo(np.float32).max]),
                              dtype=np.float32)

    if action_space is None:
      action_space = Box(low=-4, high=4, dtype=np.float32, shape=(1,))

    if car is None:
      car = RectangleTuple(mass=1.0, height=0.1, width=0.2, friction=0.0)

    super().__init__(observation_space=observation_space,
                     action_space=action_space,
                     time_per_step=time_per_step)

    self.car = car

  def height_fn(self, x):
    x2 = np.square(x)
    x_neg = x + x2
    x_pos = x / np.sqrt(1 + 5 * x2)
    return np.where(x < 0, x_neg, x_pos) + 0.5

  def slope_fn(self, x):
    dx_neg = 1 + 2 * x
    dx_pos = np.power(1 + 5 * np.square(x), -1.5)
    return np.where(x < 0, dx_neg, dx_pos)

  def ode_fn(self, t, state_action):
    assert self.car.friction == 0.0, NotImplementedError

    x, d_x, _f = state_action
    f = np.clip(_f, self.action_space.low, self.action_space.high)

    slope = self.slope_fn(x)
    inv_slope2p1 = np.reciprocal(np.square(slope) + 1)
    dd_x = f / self.car.mass * np.sqrt(inv_slope2p1) \
           - self.car.gravity * slope * inv_slope2p1

    # Return time derivatives of constrained system
    derivatives = np.clip((d_x, dd_x),
                          np.subtract(self.observation_space.low, (x, d_x)),
                          np.subtract(self.observation_space.high, (x, d_x)))

    return np.pad(derivatives, [0, 1])

  def reset(self, state: np.ndarray = None) -> np.ndarray:
    if state is None:
      state = np.array([self.np_random.uniform(low=-0.4, high=-0.6), 0])
    self.state = state
    return np.array(self.state)

  def render(self, mode='human'):
    if self.state is None:
      return None

    if self.viewer is None:
      self.viewer = self.create_viewer()

    x = self.state[0]
    car = self.viewer.geoms[0]
    trans = car.attrs[-1]
    x_low = self.observation_space.low[0]
    scale = self.viewer.width/(self.observation_space.high[0] - x_low)
    trans.set_translation(scale * (x - x_low), scale * self.height_fn(x))
    trans.set_rotation(self.slope_fn(x))
    return self.viewer.render(return_rgb_array=mode == 'rgb_array')

  def create_viewer(self):
    from gym.envs.classic_control import rendering
    viewer = rendering.Viewer(600, 400)

    x_goal = 0.6
    x_range = self.observation_space.low[0], self.observation_space.high[0]
    x_width = np.diff(x_range)

    scale = viewer.width / x_width
    h_car = scale * self.car.height
    w_car = scale * self.car.width

    y_offset = 10  # vertical offset from track
    l, r = -0.5 * w_car, 0.5 * w_car
    b, t = 0, h_car

    car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    car.add_attr(rendering.Transform(translation=(0, y_offset)))
    car_trans = rendering.Transform()
    car.add_attr(car_trans)
    viewer.add_geom(car)

    def create_wheel(center, radius):
      wheel = rendering.make_circle(radius)
      trans = rendering.Transform(translation=center)
      wheel.add_attr(trans)
      wheel.add_attr(car_trans)  # move along with car
      wheel.set_color(0.5, 0.5, 0.5)
      return wheel

    wheel_l = create_wheel((-0.25 * w_car, y_offset),  h_car/3)
    wheel_r = create_wheel((0.25 * w_car, y_offset), h_car/3)
    viewer.add_geom(wheel_l)
    viewer.add_geom(wheel_r)

    x = np.linspace(*x_range, 128)
    y = self.height_fn(x)
    coords = tuple(zip(scale * (x - x_range[0]), scale * y))
    track = rendering.make_polyline(coords)
    track.set_linewidth(4)
    track.set_color(0.0, 0.0, 0.0)

    viewer.add_geom(track)

    x_flag = scale * (x_goal - x_range[0])
    b_flag = scale * self.height_fn(x_goal)
    t_flag = b_flag + 50

    pole = rendering.Line((x_flag, b_flag), (x_flag, t_flag))
    flag = rendering.FilledPolygon(
        [(x_flag, t_flag), (x_flag, t_flag - 10), (x_flag + 25, t_flag - 5)]
    )
    flag.set_color(0.8, 0.8, 0.0)
    viewer.add_geom(pole)
    viewer.add_geom(flag)

    return viewer
