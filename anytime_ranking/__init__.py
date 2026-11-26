"""Bayesian active ranking simulations."""
import os

# Limit each worker to one BLAS thread.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

__version__ = "2.0.0"
