from dataclasses import dataclass, field
from scipy.stats import ttest_ind
import numpy as np
import warnings

@dataclass
class Stats:
    p_thresh: float = 0.05
    diff_thresh: float = 0
    percentage_threshold: bool = field(kw_only=True, default=False)

    def welchs_t_test(self, group1, group2):
        """
        Perform Welch's t-test and classify response based on significance and effect size.

        Parameters:
        - group1: array-like numeric data (PRE or BASELINE data)
        - group2: array-like numeric data
        - labels: tuple of two strings (positive_diff_label, negative_diff_label), optional

        Returns:
        dict with keys:
            'response': label or None
            'mean_diff': float
            'p_val': float
        """
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)


        if np.var(group1) == 0 and np.var(group2) == 0: # no vairability in either group
            return {'response': np.nan, 'mean_diff': 0.0, 'p_val': 1.0}
        
        t_stat, p_val = ttest_ind(group1, group2, equal_var=False)


        mean_diff = np.mean(group2) - np.mean(group1)  # POST - PRE
        percent_diff = 100 * mean_diff / np.mean(group1)
        percent_diff = float(f"{percent_diff:.3g}")

        diff_vairable = abs(percent_diff) if self.percentage_threshold  else  abs(mean_diff)
        if p_val < self.p_thresh and diff_vairable >= self.diff_thresh:
            response = 'increase' if mean_diff > 0 else 'decrease'
        else:
            response = np.nan

        return {'response': response, 'mean_diff': mean_diff, 'p_val': p_val, 'percent_diff': percent_diff}
