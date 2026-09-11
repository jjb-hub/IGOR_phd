import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field
from typing import ClassVar
import itertools
# from Cachable import Cachable
from itertools import cycle
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import os
import textwrap
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_ind
import seaborn as sns
from typing import Optional
from statsmodels.tools.sm_exceptions import ConvergenceWarning
# from module.utils import  subselectDf, saveFigure, getCache, isCached, cache, cache_excel #should become Cashable class
from module.constants import CACHE_DIR, color_dict, unit_dict
# from module.Ephys import Ephys, APP, FP, EphysData # I THINK THIS IS OLD?
from module.Ephys_Project import Ephys, APP_IC, Project, IF_IC
from module.Cachable import Cachable
from collections import defaultdict
#Readapting
from module.action_potential_functions import ap_characteristics_extractor_main, select_protocol_array #should become ActionPotential class
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

# from module.Stats import Stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import mixedlm
import itertools
from patsy import build_design_matrices
from statsmodels.stats.multitest import multipletests
import importlib



# Root directory for projects #HACKY SHIT should have a project or filesystem class to prevent dupicate code
ROOT = f"{os.getcwd()}/PROJECTS"
if not os.path.exists(ROOT):
    os.mkdir(ROOT)

