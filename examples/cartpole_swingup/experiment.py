#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import gpflow
import logging
import numpy as np
import tensorflow as tf

from examples.cartpole_swingup.metrics import Metrics
from gpflow.config import default_float
from gpflow.utilities.traversal import tabulate_module_summary
from gpflow_pilco.loops import EpisodeData, EpisodeSpec, CheckpointedModelBasedRL
from scipy.optimize import OptimizeResult
from tensorflow_probability.python.distributions import MultivariateNormalTriL
from typing import Callable, Iterable

default_logger = logging.getLogger(__name__)


# ==============================================
#                                     experiment
# ==============================================
def log_module_summary(module: tf.Module,
                       fmt: str = None,
                       logger: logging.Logger = default_logger):
  """
  Same as <gpflow.utilities.print_summary> but using a logger.
  """
  fmt = gpflow.config.default_summary_fmt() if (fmt is None) else fmt
  logger.info("\n" + tabulate_module_summary(module, fmt))


def build_loop(cls,
               directory: str,
               episode_spec: EpisodeSpec,
               step_callbacks: Iterable[Callable] = None,
               unroll_callbacks: Iterable[Callable] = None) \
    -> CheckpointedModelBasedRL:

  # Construct outer-loop
  loop = cls(directory=directory, episode_spec=episode_spec)

  # Define tracked metrics
  metrics = Metrics(loop=loop)
  loop.metrics.update(rewards=metrics.rewards,
                      success=metrics.success,
                      eReward=metrics.expected_reward,
                      vReward=metrics.validation_reward,
                      vSuccess=metrics.validation_success)

  # Include any additional callbacks
  if step_callbacks is not None:
    # Called at the end of each outer-loop step: callback(step, episode)
    loop.step_callbacks.extend(step_callbacks)

  if unroll_callbacks is not None:
    # Called at the iteration during unrolling: callback(state, action)
    loop.unroll_callbacks.extend(unroll_callbacks)

  # Maybe restore from checkpoint
  loop.restore_or_initialize()
  return loop


def outer_loop(loop: CheckpointedModelBasedRL,
               seed: int,
               num_episodes: int = 10,
               num_episodes_init: int = 1,
               logger: logging.Logger = default_logger):

  def _set_seeds(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    loop.env.seed(seed)
    loop.env.action_space.seed(seed)

  while len(loop.episodes) < num_episodes:
    # We set all seeds at the start of each loop iteration to ensure
    # consistent results even when restoring from checkpoints.
    _set_seeds(seed + len(loop.episodes) + 1)

    if len(loop.episodes) >= num_episodes_init:
      logger.info("Updating dynamics model...")
      result = loop.update_dynamics()
      if isinstance(result, OptimizeResult):
        logger.info("\n - ".join(("Dynamics OptimizeResult:",
                                  f"fun={np.mean(result.fun)}",
                                  f"nit={result.nit}",
                                  f"nfev={result.nfev}",
                                  f"status={result.status}",
                                  f"success={result.success}",
                                  f"message={result.message}")))
      log_module_summary(loop.drift, logger=logger)

      logger.info("Updating policy...")
      result = loop.update_policy()
      if isinstance(result, OptimizeResult):
        logger.info("\n - ".join(("Policy OptimizeResult:",
                                  f"fun={np.mean(result.fun)}",
                                  f"nit={result.nit}",
                                  f"nfev={result.nfev}",
                                  f"status={result.status}",
                                  f"success={result.success}",
                                  f"message={result.message}")))
      log_module_summary(loop.policy, logger=logger)

    # Use current policy to generate a new episode
    _ = loop.step()

    # Save episodes and maybe create a checkpoint
    loop.save(step_count=len(loop.episodes))


def main(dest: str,
         seed: int,
         loop_constructor: type,
         time_horizon: float = 3.0,
         time_step_size: float = 0.1,
         state_scale: tf.Tensor = None,
         num_episodes: int = 10,
         num_episodes_init: int = 1,
         step_callbacks: Iterable[Callable] = None,
         unroll_callbacks: Iterable[Callable] = None,
         logger: logging.Logger = default_logger) -> CheckpointedModelBasedRL:

  # Define initial state distribution
  dtype = default_float()
  if state_scale is None:
    state_scale = tf.linalg.diag(0.1 + tf.zeros([4], dtype=dtype))
  state_loc = tf.convert_to_tensor(value=(0.0, np.pi, 0.0, 0.0), dtype=dtype)
  state_distrib = MultivariateNormalTriL(loc=state_loc, scale_tril=state_scale)

  # Organize episode metadata
  episode_spec = EpisodeSpec(state_distrib=state_distrib,
                             horizon=time_horizon,
                             step_size=time_step_size)

  # Construct RL outer-loop class
  loop = build_loop(cls=loop_constructor,
                    directory=dest,
                    episode_spec=episode_spec,
                    step_callbacks=step_callbacks,
                    unroll_callbacks=unroll_callbacks)

  # Callback to log metrics recorded at the end of each outer-loop step
  def callback_logMetrics(step: int, episode: EpisodeData):
    logger.info(
        f"Round {step} metrics: " + ', '.join(
            f"{metric_id}={np.sum(val) if isinstance(val, Iterable) else val}"
            for metric_id, val in episode.metrics.items())
    )
  loop.step_callbacks.append(callback_logMetrics)

  #### UNCOMMENT ME FOR VIDEOS
  # # Callback to visualze the training episode generated at each outer-loop step
  # from time import sleep
  # def callback_render(state: np.ndarray, action: np.ndarray, tic: float = 0.05):
  #   loop.env.render()
  #   sleep(tic)
  # loop.unroll_callbacks.append(callback_render)

  # Run the experiment
  outer_loop(loop=loop,
             seed=seed,
             num_episodes=num_episodes,
             num_episodes_init=num_episodes_init,
             logger=logger)

  return loop
