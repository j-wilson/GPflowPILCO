#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from examples.cartpole_swingup.experiment import main
from examples.cartpole_swingup.swingup_loops import SwingupMomentMatchingPILCO
from random import randint
from tempfile import TemporaryDirectory

logger = logging.getLogger("MomentMatchingPILCO")
logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='%(asctime)-4s %(levelname)s:%(name)s:%(message)s')

# # Example of some additional options
# from examples.cartpole_swingup.settings import update_drift_spec, update_policy_spec
# update_drift_spec(train_kwargs=dict(options=dict(maxiter=1000)))  # BFGS options
# update_policy_spec(step_limit=500)  # num. param updates when optimizing policy

with TemporaryDirectory() as dest:  # examples save to temporary directories
  seed = randint(0, 2 ** 31)
  logger.info(f"Files will be saved to: {dest}")
  logger.info(f"Experiment seed: {seed}")
  loop = main(dest=dest,
              seed=seed,
              loop_constructor=SwingupMomentMatchingPILCO,
              logger=logger)
