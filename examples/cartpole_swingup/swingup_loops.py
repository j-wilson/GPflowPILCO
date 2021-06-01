#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from examples.cartpole_swingup import train_utils
from examples.cartpole_swingup.settings import (drift_spec,
                                                policy_spec,
                                                resolve_default_keywords)
from gpflow.config import default_float
from gpflow.optimizers import Scipy
from gpflow_pilco.components import GaussianObjective, TrigonometricEncoder
from gpflow_pilco.envs import CartPole
from gpflow_pilco.loops import (EpisodeSpec,
                                MomentMatchingPILCO,
                                PathwisePILCO)
from gpflow_pilco.models.priors import PilcoPenaltySNR
from gpflow_pilco.utils.optimizers import GradientDescent
from tensorflow_probability.python import bijectors
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

# ---- Exports
__all__ = (
  "SwingupWrapper",
  "SwingupMomentMatchingPILCO",
  "SwingupPathwisePILCO",
)


# ==============================================
#                                          loops
# ==============================================
class SwingupWrapper:
  def build_task_components(self, episode_spec: EpisodeSpec):
    # Create cart-pole enviroment
    env = CartPole(time_per_step=episode_spec.step_size)

    # Encode pole orientation as sin and cosine of the angle
    encoder = TrigonometricEncoder(active_dims=(1,))

    # Build Gaussian objective
    target = encoder(tf.zeros([4], dtype=default_float()))
    height = env.pole.height
    precis = 16 * tf.convert_to_tensor([[height ** 2, 0, -height, 0, 0],
                                        [0, height ** 2, 0, 0, 0],
                                        [-height, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]], dtype=default_float())
    objective = GaussianObjective(target=target, precis=precis)

    return env, objective, encoder

  @resolve_default_keywords(defaults_factory=drift_spec, prefix="@")
  def update_dynamics(self,
                      reinitialize: bool = "@reinitialize",
                      build_kwargs: dict = "@build_kwargs",
                      train_kwargs: dict = "@train_kwargs"):
    if self.drift is None or reinitialize:
      prior = build_kwargs.pop("prior", "default")
      if prior == "default":  # penalize very large signal-to-noise ratios
        prior = PilcoPenaltySNR(threshold=1e5, power=30)

      self.drift, _ = super().build_dynamics(prior=prior, **build_kwargs)

    return train_utils.dynamics(Scipy(),
                                self,
                                self.drift,
                                **train_kwargs)

  @resolve_default_keywords(defaults_factory=policy_spec, prefix="@")
  def update_policy(self,
                    reinitialize: bool = "@reinitialize",
                    build_kwargs: dict = "@build_kwargs",
                    train_kwargs: dict = "@train_kwargs",
                    step_limit: int = "@step_limit",
                    initial_learning_rate: float = "@initial_learning_rate",
                    global_clipnorm: float = "@global_clipnorm"):

    if self.policy is None or reinitialize:
      invlink = build_kwargs.pop("invlink", "default")
      if invlink == "default":
        invlink = bijectors.Chain(bijectors=[
            bijectors.Scale(scale=tf.cast(x=20 - 1e-5, dtype=default_float())),
            bijectors.Shift(shift=tf.cast(x=-0.5, dtype=default_float())),
            bijectors.NormalCDF()])
      self.policy = super().build_policy(invlink=invlink, **build_kwargs)

    values = tuple((0.1 ** k) * initial_learning_rate for k in range(3))
    bounds = tuple(k * step_limit // len(values) for k in range(1, len(values)))
    schedule = PiecewiseConstantDecay(boundaries=bounds, values=values)

    adam = Adam(learning_rate=schedule, global_clipnorm=global_clipnorm)
    optimizer = GradientDescent(optimizer=adam, step_limit=step_limit)
    return train_utils.policy(optimizer,
                              self,
                              self.drift,
                              self.policy,
                              **train_kwargs)


class SwingupMomentMatchingPILCO(SwingupWrapper, MomentMatchingPILCO):
  def __init__(self, directory: str, episode_spec: EpisodeSpec, **kwargs):
    env, objective, encoder = self.build_task_components(episode_spec)
    MomentMatchingPILCO.__init__(self=self,
                                 directory=directory,
                                 env=env,
                                 episode_spec=episode_spec,
                                 objective=objective,
                                 encoder=encoder,
                                 **kwargs)


class SwingupPathwisePILCO(SwingupWrapper, PathwisePILCO):
  def __init__(self, directory: str, episode_spec: EpisodeSpec, **kwargs):
    env, objective, encoder = self.build_task_components(episode_spec)
    PathwisePILCO.__init__(self=self,
                           directory=directory,
                           env=env,
                           episode_spec=episode_spec,
                           objective=objective,
                           encoder=encoder,
                           **kwargs)
