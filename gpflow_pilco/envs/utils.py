#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
from typing import NamedTuple

# ---- Exports
__all__ = ('RectangleTuple',)


# ==============================================
#                                          utils
# ==============================================
class RectangleTuple(NamedTuple):
  mass: float = 1.0  # mass
  width: float = 1.0  # pole width
  height: float = 1.0  # pole height
  gravity: float = 9.81  # gravitational constant
  friction: float = 0.0  # friction coefficient
