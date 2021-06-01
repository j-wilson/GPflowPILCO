#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from gpflow.utilities import Dispatcher
from numpy import ndarray
from typing import *

# ---- Exports
__all__ = (
  "ArrayTypes",
  "Chain",
  "dispatcher",
  "get_type",
  "moment_matching",
  "Moments",
  "MomentMatch",
  "register_type",
)


# ==============================================
#                                           core
# ==============================================
_MomentMatchingCustomTypes = dict()
ArrayTypes = ndarray, tf.Tensor, tf.Variable, tf.linalg.LinearOperator
dispatcher = Dispatcher("moment_matching")


def get_type(obj: Hashable) -> Type:
  return _MomentMatchingCustomTypes[obj]


def _get_object_name(obj: Any) -> str:
  return f"{obj.__module__}.{obj.__name__}"


def register_type(obj: Hashable,
                  name: str = None,
                  bases: Tuple = tuple(),
                  dict: Dict = None,
                  exist_ok: bool = False) -> Type:
  """
  Create a dedicated type to represent a given object and add it to the
  register. Mainly used for multiple dispatching against functions, such
  as <tf.math.add>.
  """
  if obj in _MomentMatchingCustomTypes and not exist_ok:
    raise ValueError("Attempted to register a preexisting custom type")

  if name is None:
    name = _get_object_name(obj)

  if dict is None:
    dict = {}

  new_type = _MomentMatchingCustomTypes[obj] = type(name, bases, dict)
  return new_type


@dataclass
class Moments:
  moments: Union[List, Tuple]
  centered: bool

  def __getitem__(self, index: int) -> Union[ArrayTypes]:
    return self.moments[index]

  def mean(self) -> Union[ArrayTypes]:
    return self[0]

  def covariance(self, dense: Optional[bool] = None) -> Union[ArrayTypes]:
    m1, m2 = self[:2]
    if self.centered:
      Syy = m2
    elif isinstance(m2, tf.linalg.LinearOperator):
      diag = tf.cast([-1], m2.dtype)
      Syy = tf.linalg.LinearOperatorLowRankUpdate(base_operator=m2,
                                                  u=tf.expand_dims(m1, -1),
                                                  diag_update=diag,
                                                  is_square=True,
                                                  is_self_adjoint=True,
                                                  is_non_singular=True,
                                                  is_positive_definite=True)
    else:
      Syy = m2 - tf.expand_dims(m1, -1) * tf.expand_dims(m1, -2)

    if dense and isinstance(Syy, tf.linalg.LinearOperator):
      Syy = Syy.to_dense()

    return Syy

  @property
  def ndim(self) -> int:
    return self[0].shape[-1]

  @property
  def dtype(self):
    dtype = self[0].dtype
    for moment in self[1:]:
      assert moment.dtype == dtype, ValueError("dtype of moments do not match")
    return dtype


@dataclass
class MomentMatch:
  x: Moments  # moments of base measure
  y: Moments  # moments of push-forward


class Chain(tuple):
  def __new__(cls, *ops: Iterable[Callable]):
    return super().__new__(cls, ops)

  def __call__(self, x: tf.Tensor) -> tf.Tensor:
    for op in reversed(self):
      x = op(x)
    return x


@dispatcher.register(Moments, partial)
def _mm_partial(x: Moments, op: partial):
  return moment_matching(x, op.func, *op.args, **op.keywords)


def moment_matching(x: Moments, obj: Any, *args, **kwargs) -> MomentMatch:
  if isinstance(obj, (partial, Chain)):
    return dispatcher(x, obj, *args, **kwargs)

  if isinstance(obj, Hashable) and obj in _MomentMatchingCustomTypes:
    obj = get_type(obj)()  # represent object as an instance of a dedicated type

  return dispatcher(x, obj, *args, **kwargs)
