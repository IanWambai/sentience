"""
Sentience - A production-grade cognition engine for Apple Silicon MacBooks.

This package loads the int4-quantized Gemma 3n model and runs continuous inference
on webcam frames, providing autonomous perception and recommendations.
"""

from .core.runtime import run

__version__ = '0.1.0'
__all__ = ['run']
