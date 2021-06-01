#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from tqdm import tqdm
from tensorflow.keras.optimizers import Optimizer
from typing import List, Callable

# ---- Exports
__all__ = ("GradientDescent",)


# ==============================================
#                                     optimizers
# ==============================================
class GradientDescent:
  """
  TODO: Move gradient clipping into trasnform and progress bar into callbacks
  """
  def __init__(self,
               step_limit: int,
               optimizer: Optimizer = None,
               callbacks: List[Callable] = None,
               transform: Callable = None,
               show_progress: bool = True,
               ema_const: float = 0.6):

    if callbacks is None:
      callbacks = list()

    if optimizer is None:
      optimizer = tf.keras.optimizers.Adam()

    self.step_limit = step_limit
    self.optimizer = optimizer
    self.transform = transform
    self.callbacks = callbacks
    self.show_progress = show_progress
    self.ema_const = ema_const

  def minimize(self, closure: Callable, variables: List[tf.Variable]):
    iterator = range(self.step_limit)
    if self.show_progress:
      iterator = tqdm(iterator)

    for step in iterator:
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(variables)
        loss = closure()

      grads = tape.gradient(loss, variables)
      if self.transform is not None:
        grads = self.transform(*grads)

      if self.show_progress:
        _loss = tf.reduce_mean(loss)
        _norm = tf.linalg.global_norm(grads)
        if self.ema_const == 1:
          postfix = f"loss={_loss:.2e}, norm={_norm:.2e}"
        else:
          if step:
            ema_loss += self.ema_const * (_loss - ema_loss)
            ema_norm += self.ema_const * (_norm - ema_norm)
          else:
            ema_loss = _loss
            ema_norm = _norm
          postfix = f"EMA(loss)={ema_loss:.2e}, EMA(norm)={ema_norm:.2e}"
        iterator.set_postfix_str(postfix)

      grads_and_vars = tuple(zip(grads, variables))
      self.optimizer.apply_gradients(grads_and_vars)
      for callback in self.callbacks:
        callback(step, loss, grads_and_vars)
