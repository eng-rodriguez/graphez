"""
GraphEZ: EEG Data Analysis Package

A Python package for loading, processing, and visualizing EEG data using MNE-Python.
"""

__version__ = "0.1.0"
__author__ = "EEG Research Team"

from . import utils
from . import plotting
from . import dataloader
from . import processing
from . import connectivity

__all__ = ["dataloader", "plotting", "processing", "utils", "connectivity"]