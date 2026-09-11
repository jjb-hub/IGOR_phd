"""Compatibility imports for figure/analysis classes.

New code should import from the smaller modules directly. Existing notebooks can
keep using ``from module.Figure import ...`` while migration happens.
"""

from module.figure_common import ROOT
from module.selection import DataSelection
from module.stats import MixedLMStatsMixin
from module.figure_base import Figure
from module.response import ResponseCharecterisation
from module.if_curve import IF_curve
from module.histograms import Histogram, CellHistogram, SubjectHistogram, ResponseHistogram
from module.application import AggregateApplication, Application
from module.ra_ap_analysis import RA_AP_analysis

__all__ = [
    "ROOT",
    "DataSelection",
    "MixedLMStatsMixin",
    "Figure",
    "ResponseCharecterisation",
    "IF_curve",
    "Histogram",
    "CellHistogram",
    "SubjectHistogram",
    "ResponseHistogram",
    "AggregateApplication",
    "Application",
    "RA_AP_analysis",
]
