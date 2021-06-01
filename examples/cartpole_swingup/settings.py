#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
from dataclasses import dataclass, field
from functools import update_wrapper
from inspect import signature
from typing import Callable

# ---- Exports
__all__ = (
  "resolve_default_keywords",
  "drift_spec",
  "set_drift_spec",
  "update_drift_spec",
  "policy_spec",
  "set_policy_spec",
  "update_policy_spec",
)

# ==============================================
#                                       settings
# ==============================================
@dataclass
class DriftSpec:
  """
  Options for building and training GPs to model black-box drift functions
  """
  reinitialize: bool = True  # should we train drift model from scratch?
  build_kwargs: dict = field(default_factory=lambda: dict(num_centers=256))
  train_kwargs: dict = field(default_factory=dict)

  # Some additional defaults used when initializing the drift model
  num_centers: int = 256  # we actually use min(num_centers, num_data)
  batch_size: int = 1024  # only relevant if we train using SGD


@dataclass
class PolicySpec:
  """
  Options for building and training kernel regressors as policies
  """
  reinitialize: bool = False  # should we train policy model from scratch?
  build_kwargs: dict = field(default_factory=lambda: dict(num_centers=30))
  train_kwargs: dict = field(default_factory=dict)

  # Additional settings used during training
  step_limit: int = 5000
  global_clipnorm: float = 1.0
  initial_learning_rate: float = 0.01

  # Settings used by pathwise PILCO when estimating expected losses
  batch_size: int = 1024
  num_bases: int = 1024


def resolve_default_keywords(defaults_factory: Callable, prefix: str) -> Callable:
  """
  Find and replace all of a function's keyword arguments whose string-valued
  defaults begin with a chosen prefix. The remainder of this string is used
  to lookup the true (and possibly dynamic) default value by accessing the
  attributes of the object returned by the getter.
  """
  def _outer_wrapper(func: Callable):
    _targets = dict()
    _pre_len = len(prefix)
    for name, param in signature(func).parameters.items():
      if isinstance(param.default, str) and param.default[:_pre_len] == prefix:
        _targets[param.name] = param.default[_pre_len:]

    def _inner_wrapper(*args, **kwargs):
      defaults = defaults_factory()
      for arg_id, attr_id in _targets.items():
        kwargs.setdefault(arg_id, getattr(defaults, attr_id))
      return func(*args, **kwargs)

    return update_wrapper(_inner_wrapper, func)
  return _outer_wrapper


def drift_spec():
  return _DRIFT_SPEC


def set_drift_spec(spec: DriftSpec):
  global _DRIFT_SPEC
  _DRIFT_SPEC = spec


def update_drift_spec(**updates):
  spec = drift_spec()
  for key, val in updates.items():
    setattr(spec, key, val)


def policy_spec():
  return _POLICY_SPEC


def set_policy_spec(spec: PolicySpec):
  global _POLICY_SPEC
  _POLICY_SPEC = spec


def update_policy_spec(**updates):
  spec = policy_spec()
  for key, val in updates.items():
    setattr(spec, key, val)


_DRIFT_SPEC = DriftSpec()
_POLICY_SPEC = PolicySpec()
