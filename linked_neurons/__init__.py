# -*- coding: utf-8 -*-

"""Top-level package for Linked Neurons."""

__author__ = """Carles R. Riera Molina"""
__email__ = 'carlesrieramolina@gmail.com'
__version__ = '0.1.0'

from .linked_neurons import LKReLU, LKPReLU, LKSELU, LKSwish, swish
__all__ = ['LKReLU', 'LKPReLU', 'LKSELU', 'LKSwish', 'swish']
