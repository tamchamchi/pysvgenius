"""
_config.py

This module provides a global configuration handler for the project.
It supports loading configuration from a YAML file and accessing/modifying
config values at runtime.

Typical use cases include setting project directories, model parameters,
and runtime flags.
"""

import os

# Fixed project directories
PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(PROJ_DIR, "models")
DATA_DIR = os.path.join(PROJ_DIR, "data")

AESTHETIC_MODEL_PATH = os.path.join(MODEL_DIR, "sac+logos+ava1-l14-linearMSE.pth")
CLIP_MODEL_PATH = os.path.join(MODEL_DIR, "ViT-L-14.pt")
