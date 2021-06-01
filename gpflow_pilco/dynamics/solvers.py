#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from typing import *
from numpy import ndarray, concatenate
from scipy.integrate import solve_ivp
from gpflow_pilco.moment_matching.gaussian import GaussianMoments

# ---- Exports
__all__ = (
  "ScipyODE",
  "Euler",
  "MomentMatchingEuler"
)


# ==============================================
#                                        solvers
# ==============================================
class ScipyODE:
  @classmethod
  def __call__(cls: "ScipyODE",
               func: Callable,
               initial_time: float,
               initial_state: Any,
               solution_times: tf.Tensor,
               callbacks_and_initializers: List[Tuple[Callable, Any]] = None,
               **kwargs) -> ndarray:
    assert callbacks_and_initializers is None
    t_span = min(initial_time, min(solution_times)), \
             max(initial_time, max(solution_times))

    result = solve_ivp(fun=func,
                       t_span=t_span,
                       t_eval=solution_times,  # T
                       y0=initial_state,  # D
                       **kwargs)

    return result.y.T  # TxD


class Euler:
  @classmethod
  def step(cls: "Euler",
           func: Callable,
           t: float,
           dt: float,
           x: tf.Tensor) -> tf.Tensor:
    """
    Euler-Maruyama method for approximately solving SDEs.
    """
    dx_dt, sqrt_cov = func(t, x)

    _x = x + dt * dx_dt
    if sqrt_cov is None:
      return _x

    rvs = tf.random.normal(_x.shape, dtype=_x.dtype)
    return _x + tf.linalg.matvec((dt ** 0.5) * sqrt_cov, rvs)

  @classmethod
  def __call__(cls,
               func: Callable,
               initial_time: float,
               initial_state: Any,
               solution_times: tf.Tensor,
               callbacks_and_initializers: List[Tuple[Callable, Any]] = None,
               iterator: Callable = tf.scan) -> Any:
    """
    Solve for states $x_{t}$ with $t=1,...,T$, given an initial state $x_{0}$.
    """
    if callbacks_and_initializers is None:
      initializer = initial_state
    else:
      callbacks, callback_initializers = zip(*callbacks_and_initializers)
      initializer = (initial_state,) + tuple(callback_initializers)

    def body(state_and_maybe_callback_args, elems):
      t, dt = elems
      if callbacks_and_initializers is None:
        state = state_and_maybe_callback_args
      else:
        state, *callback_args = state_and_maybe_callback_args

      new_state = cls.step(func=func, t=t, dt=dt, x=state)
      if callbacks_and_initializers is None:
        return new_state

      callback_retvals = []
      for (callback_i, args_i) in zip(callbacks, callback_args):
        callback_retvals.append(callback_i(t, new_state, args_i))
      return (new_state,) + tuple(callback_retvals)

    step_sizes = concatenate([solution_times[:1] - initial_time,
                              solution_times[1:] - solution_times[:-1]], axis=0)

    return iterator(fn=body,
                    elems=(solution_times, step_sizes),
                    initializer=initializer)


class MomentMatchingEuler(Euler):
  @classmethod
  def step(cls: "MomentMatchingEuler",
           func: Callable,
           t: float,
           dt: float,
           x: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Moment matching variant of Euler-Maruyama solver.
    """
    x = GaussianMoments(moments=x, centered=True)
    match_drift, match_noise = func(t, x)

    mx = x.mean()
    Sxx = x.covariance()

    mf = match_drift.y.mean()
    Sxf = match_drift.cross_covariance()
    Sff = match_drift.y.covariance()

    _mx = mx + dt * mf
    _Sxx = Sxx + dt * (Sxf + tf.linalg.adjoint(Sxf)) + (dt ** 2) * Sff
    if match_noise is not None:
      Sxz = match_drift.cross_covariance()
      Szz = match_drift.y.covariance()
      _Sxx += (dt ** 0.5) * (Sxz + tf.linalg.adjoint(Sxz)) + dt * Szz

    return _mx, _Sxx
