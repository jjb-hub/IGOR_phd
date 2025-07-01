from dataclasses import dataclass
from scipy.stats import ttest_ind
import numpy as np
import warnings

@dataclass
class Stats:
    p_thresh: float = 0.05
    diff_thresh: float = 0

    def welchs_t_test(self, group1, group2, labels=['+', '-']):
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

        # if len(group1) < 2 or len(group2) < 2:
        #     return {'response': np.nan, 'mean_diff': np.nan, 'p_val': np.nan}

        if np.var(group1) == 0 and np.var(group2) == 0: # no vairability in either group
            return {'response': np.nan, 'mean_diff': 0.0, 'p_val': 1.0}
        
        t_stat, p_val = ttest_ind(group1, group2, equal_var=False)

        # elif np.var(group1) == 0 and np.mean(group2) > 0: # PRE events ==[0] but events in POST
        #     return {'response': labels[0], 'mean_diff': np.mean(group2), 'p_val': None}  

        # else:
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("error", category=RuntimeWarning)
        #         try:
        #             t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
        #         except RuntimeWarning:
        #             print(f" RuntimeWarning: {RuntimeWarning}")
        #             return {'response': np.nan, 'mean_diff': np.nan, 'p_val': np.nan}

        mean_diff = np.mean(group2) - np.mean(group1)  # POST - PRE

        if labels is not None and p_val < self.p_thresh and abs(mean_diff) >= self.diff_thresh:
            response = labels[0] if mean_diff > 0 else labels[1]
        else:
            response = np.nan

        return {'response': response, 'mean_diff': mean_diff, 'p_val': p_val}
