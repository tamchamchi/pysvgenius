"""
_config.py

This module provides a global configuration handler for the project.
It supports loading configuration from a YAML file and accessing/modifying
config values at runtime.

Typical use cases include setting project directories, model parameters,
and runtime flags.
"""

import os
import threading

# Fixed project directories
PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(PROJ_DIR, "models")
DATA_DIR = os.path.join(PROJ_DIR, "data")

# Global configuration dictionary
_global_config = {}

_threadlocal = threading.local()


def _get_threadlocal_config():
    """Get a threadlocal **mutable** configuration. If the configuration
    does not exist, copy the default global configuration."""
    if not hasattr(_threadlocal, "global_config"):
        _threadlocal.global_config = _global_config.copy()
    return _threadlocal.global_config


def get_config():
    """Retrieve current values for configuration set by :func:`set_config`.

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    See Also
    --------
    config_context : Context manager for global scikit-learn configuration.
    set_config : Set global scikit-learn configuration.
    """
    return _get_threadlocal_config().copy()
