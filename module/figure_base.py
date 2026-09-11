from module.figure_common import *
from module.selection import DataSelection

@dataclass
class Figure(DataSelection):

    '''
    Base class for all figures. Handles loading, saving, and plotting of figures. #TODO loading form cache / genergic function to be redefined in child classes ? #REMI

        Attributes:
        - data_type (str): The data type (e.g., 'APP' or 'FP').
        - cell_type (str | list): The type of cell to filter on (optional) / can inout list 
        - treatment (str | list): The treatment to filter on, i.e. drug applied (optional).
        - cell_subtype (str | list): The subtype of cell to filter on (optional).
        - I_set (str | list): The I_set to filter on (optional).
        - threshold_access_change (float): The threshold for access change filtering (optional, default 30).
        - extension (str): The file extension for the figure. Defaults to "png". #TODO
    '''
  
    # extension: ClassVar[str] = "png" # if I make a good casheable class

    def __post_init__(self):
        # DataSelection.__post_init__(self)
        super().__post_init__()

    def filter_n_minimum(self, df):
        """
        Filters a dataframe to enforce a minimum number of samples per group.

        If 'time' column exists, groups by ['treatment', 'time'].
        Otherwise, groups only by ['treatment'].
        """
        df = df.dropna(subset=[self.dependant_var]).reset_index(drop=True)

        if hasattr(self, 'first_factor'):
            group_cols = [self.first_factor]
        else:
            group_cols = ['treatment']  # default grouping for response-style figures without first_factor

        if 'time' in df.columns:
            group_cols.append('time')

        group_sizes = df.groupby(group_cols).size()
        insufficient_groups = group_sizes[group_sizes < self.n_minimum]

        if not insufficient_groups.empty:
            print(f"Warning: The following groups have less than {self.n_minimum} samples and will be excluded:")
            print(insufficient_groups)
            df = df[~df[group_cols].apply(tuple, axis=1).isin(insufficient_groups.index)].reset_index(drop=True)

        if df.empty:
            print("No groups meet the minimum sample size requirement. Statistical analysis will not be performed.")
            return None
        else:
            return df
        
    def safe_str(self, x):
        """Convert attribute to clean string for filename."""
        if x is None:
            return None
        if isinstance(x, (list, tuple, set)):
            return ", ".join(map(str, x))
        return str(x)
    
    def check_valid_dependant_var(self):
        if self.dependant_var not in self.agg_df.columns:
            dvs = [
                col for col in self.agg_df.columns
                if col not in self.project_obj.data_independant_columns()
            ]
            raise ValueError(f"Invalid dependant variable: {self.dependant_var}. Valid dv's : {dvs}")
            
    def sweep_dependant_var(self):
        if self.dependant_var.startswith("sweep_"):
            return self.dependant_var
        return f"sweep_{self.dependant_var}"

    def sweep_count(self, value):
        if isinstance(value, (list, np.ndarray, pd.Series)):
            return len(value)
        return 0

    def numeric_sweeps(self, value):
        if isinstance(value, pd.Series):
            value = value.tolist()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, list):
            return np.array([], dtype=float)
        return pd.to_numeric(pd.Series(value, dtype="object"), errors="coerce").dropna().to_numpy()

    def mean_or_nan(self, value):
        values = self.numeric_sweeps(value)
        return float(np.nanmean(values)) if len(values) > 0 else np.nan

    def get_pre_post_sweep_windows(self,
        df: pd.DataFrame,
        dependant_var: str, #should have sweep_{dv}
        pre_sweep_window: int | None = None,
        post_sweep_window: int | None = None,
        verbose: bool = True
        ):
        """
        Determine PRE and POST sweep windows and filter cells with enough sweeps.

        POST is APP + WASH, or rows already labelled POST. Missing WASH is valid
        and simply contributes zero sweeps.
        If pre or post sweep window is not provided, the minimum available for all cells is used.
        """
        filtered_df = df.copy()

        if dependant_var not in filtered_df.columns:
            raise ValueError(f"{dependant_var} is missing; cannot build PRE/POST sweep windows.")

        def count_pre(group):
            return group.loc[group["time"] == "PRE", dependant_var].map(self.sweep_count).sum()

        def count_post(group):
            post_mask = group["time"].isin(["APP", "WASH", "POST"])
            return group.loc[post_mask, dependant_var].map(self.sweep_count).sum()

        pre_counts = filtered_df.groupby("cell_id").apply(count_pre)
        post_counts = filtered_df.groupby("cell_id").apply(count_post)

        if pre_sweep_window is None:
            valid_pre_counts = pre_counts[pre_counts > 0]
            pre_sweep_window = int(valid_pre_counts.min()) if not valid_pre_counts.empty else 0
            if verbose:
                print(f"pre_sweep_window set to {pre_sweep_window}")

        sufficient_pre_cells = pre_counts[pre_counts >= pre_sweep_window].index.tolist()

        if post_sweep_window is None:
            valid_post_counts = post_counts[post_counts > 0]
            post_sweep_window = int(valid_post_counts.min()) if not valid_post_counts.empty else 0
            if verbose:
                print(f"post_sweep_window default calculated: {post_sweep_window}")

        sufficient_post_cells = post_counts[post_counts >= post_sweep_window].index.tolist()
        valid_cells = sorted(set(sufficient_pre_cells) & set(sufficient_post_cells))

        all_cells = filtered_df["cell_id"].unique().tolist()
        dropped_cells = sorted(set(all_cells) - set(valid_cells))
        if verbose and dropped_cells:
            print(f"Dropping {len(dropped_cells)} cells due to insufficient PRE or POST sweeps: {dropped_cells}")

        filtered_df = filtered_df[filtered_df["cell_id"].isin(valid_cells)]

        return filtered_df, pre_sweep_window, post_sweep_window

    def build_pre_post_df(self, df: Optional[pd.DataFrame] = None, slice: bool = True, verbose: bool = True) -> pd.DataFrame:
        '''
        Builds a DataFrame with PRE and POST sweeps for each cell_id.

        Parameters:
            df (pd.DataFrame, optional): DataFrame to process. If None, uses self.data.
            slice (bool, optional): If True, slices to uniform pre/post sweep window sizes. 
                                    If False, uses full APP+WASH as POST, but still slices PRE.

        Returns: 
            pd.DataFrame with columns: cell_id, PRE_sweeps, POST_sweeps
        '''
        sweep_dv = self.sweep_dependant_var()
        
        if df is None:
            df = self.data

        if df is None or df.empty:
            return pd.DataFrame()

        requested_pre_sweep_window = self.pre_sweep_window
        requested_post_sweep_window = self.post_sweep_window
        use_all_post_sweeps = requested_post_sweep_window == "all"
        post_sweep_window_for_counts = None if use_all_post_sweeps else self.post_sweep_window

        if slice:
            df, self.pre_sweep_window, self.post_sweep_window = self.get_pre_post_sweep_windows(
                df, 
                dependant_var=sweep_dv, 
                pre_sweep_window=self.pre_sweep_window, 
                post_sweep_window=post_sweep_window_for_counts,
                verbose=verbose,
            )
        else:
            _, self.pre_sweep_window, self.post_sweep_window = self.get_pre_post_sweep_windows( #return but dont use filtered_df
                df, 
                dependant_var=sweep_dv, 
                pre_sweep_window=self.pre_sweep_window, 
                post_sweep_window=post_sweep_window_for_counts,
                verbose=verbose,
            )
        if use_all_post_sweeps:
            self.post_sweep_window = "all"

        metadata_cols = [
            col for col in df.columns
            if col not in {"time", sweep_dv, self.dependant_var}
            and (not col.startswith("sweep_") or col == "sweep_duration_s")
        ]

        rows = []
        for cell_id, sub_df in df.groupby('cell_id'):
            pre_parts = [
                self.numeric_sweeps(value)
                for value in sub_df.loc[sub_df['time'] == 'PRE', sweep_dv]
            ]
            post_parts = [
                self.numeric_sweeps(value)
                for value in sub_df.loc[sub_df['time'].isin(["APP", "WASH", "POST"]), sweep_dv]
            ]
            pre_vals = np.concatenate(pre_parts) if pre_parts else np.array([], dtype=float)
            post_vals = np.concatenate(post_parts) if post_parts else np.array([], dtype=float)

            pre = pre_vals[-self.pre_sweep_window:] if self.pre_sweep_window is not None else pre_vals

            if use_all_post_sweeps:
                post = post_vals
            elif slice:
                post = post_vals[:self.post_sweep_window] if self.post_sweep_window is not None else post_vals
            else:
                post = (
                    post_vals[:requested_post_sweep_window]
                    if requested_post_sweep_window is not None
                    else post_vals
                )

            if len(pre) == 0 or len(post) == 0:
                print(f"Skipping cell {cell_id} due to empty PRE or POST sweeps.")
                continue

            row = {
                'cell_id': cell_id,
                'dependant_var': self.dependant_var,
                'PRE_sweeps': pre,
                'POST_sweeps': post,
                'requested_pre_sweep_window': requested_pre_sweep_window,
                'requested_post_sweep_window': requested_post_sweep_window,
                'pre_sweep_window_used': len(pre),
                'post_sweep_window_used': len(post),
                'post_sweep_zero': 'drug_in',
            }

            for col in metadata_cols:
                vals = sub_df[col].dropna()
                if len(vals) > 0:
                    row[col] = vals.iloc[0]

            rows.append(row)

        result_df = pd.DataFrame(rows)
        if use_all_post_sweeps and verbose and not result_df.empty:
            post_lengths = result_df["post_sweep_window_used"].dropna().unique()
            if len(post_lengths) > 1:
                print(
                    "Warning: post_sweep_window='all' uses unequal POST lengths "
                    f"across cells: {sorted(post_lengths.tolist())}"
                )

        return result_df

    def build_pre_post_plot_df(self, df: Optional[pd.DataFrame] = None, slice: bool = True) -> pd.DataFrame:
        """
        Build a long PRE/POST dataframe for plotting and mixed-model stats.

        The original APP/WASH rows are not modified; callers keep them in agg_df.
        """
        if df is None:
            df = self.agg_df
        if df is None or df.empty or "time" not in df.columns:
            return df

        sweep_dv = self.sweep_dependant_var()
        if sweep_dv not in df.columns:
            plot_df = df.copy()
            plot_df["time"] = np.where(plot_df["time"] == "PRE", "PRE", "POST")
            return plot_df

        pre_post_df = self.build_pre_post_df(df, slice=slice)
        rows = []
        for _, row in pre_post_df.iterrows():
            base_row = row.drop(labels=["PRE_sweeps", "POST_sweeps"], errors="ignore").to_dict()
            for label, sweep_col in [("PRE", "PRE_sweeps"), ("POST", "POST_sweeps")]:
                sweeps = row[sweep_col]
                plot_row = base_row.copy()
                plot_row["time"] = label
                plot_row[sweep_dv] = sweeps
                plot_row[self.dependant_var] = self.mean_or_nan(sweeps)
                rows.append(plot_row)

        return pd.DataFrame(rows)

    def response_stats(self, pre, post, percentage_threshold=False):
        pre = self.numeric_sweeps(pre)
        post = self.numeric_sweeps(post)
        pre_mean = np.nanmean(pre) if len(pre) > 0 else np.nan
        post_mean = np.nanmean(post) if len(post) > 0 else np.nan
        mean_diff = post_mean - pre_mean

        if pd.isna(pre_mean) or pre_mean == 0:
            percent_diff = np.nan
        else:
            percent_diff = (mean_diff / abs(pre_mean)) * 100

        if len(pre) >= 2 and len(post) >= 2:
            _, p_val = ttest_ind(pre, post, equal_var=False, nan_policy="omit")
        else:
            p_val = np.nan

        effect = percent_diff if percentage_threshold else mean_diff
        passes_diff = pd.notna(effect) and abs(effect) >= self.diff_thresh
        passes_p = pd.isna(p_val) or p_val <= self.p_thresh
        response = "no response"
        if passes_diff and passes_p:
            response = "increase" if effect > 0 else "decrease"

        return {
            "response": response,
            "baseline_mean": pre_mean,
            "event_mean": post_mean,
            "mean_diff": mean_diff,
            "percent_diff": percent_diff,
            "p_val": p_val,
        }

    def group_consecutive_responses(self, bin_results: list[dict]) -> list[dict]:
        # Drop NaN responses
        bin_results = [br for br in bin_results if pd.notna(br['response'])]

        # Sort by bin start index
        bin_results.sort(key=lambda x: x['range_sweeps'][0])
        grouped = []

        for _, group in itertools.groupby(
            enumerate(bin_results),
            key=lambda x: (x[0] - x[1]['range_sweeps'][0], x[1]['response'])  # group by consecutive and same response
        ):
            group = [x[1] for x in group]
            cell_id = group[0]['cell_id']
            pre = group[0]['PRE_sweeps']
            post = group[0]['POST_sweeps']
            response_type = group[0]['response']

            start_sweep = group[0]['range_sweeps'][0]
            end_sweep = group[-1]['range_sweeps'][1]

            # Deltas
            deltas = [g['delta'] for g in group]
            mean_delta = np.mean(deltas)
            signed_peak_delta = self.signed_peak_delta(deltas)
            event_sweeps = post[start_sweep:end_sweep]
            baseline_mean = np.nanmean(pre) if len(pre) > 0 else np.nan
            event_mean = np.nanmean(event_sweeps) if len(event_sweeps) > 0 else np.nan

            # Handle p-values
            p_vals = [g['p_val'] for g in group if pd.notna(g['p_val'])]
            if p_vals:
                p_val = np.min(p_vals)
                # Latency is first bin with p ≤ threshold
                latency_sweep = next(
                    (g['range_sweeps'][0] for g in group if pd.notna(g['p_val']) and g['p_val'] <= self.p_thresh),
                    start_sweep
                )
            else:
                # No valid p-values: fallback to start of group
                p_val = None
                latency_sweep = start_sweep

            grouped.append({
                'cell_id': cell_id,
                'dependant_var': self.dependant_var,
                'PRE_sweeps': pre,
                'POST_sweeps': post,
                'response': response_type,
                'baseline_mean': baseline_mean,
                'event_mean': event_mean,
                'delta': mean_delta,
                'mean_delta': mean_delta,
                'signed_peak_delta': signed_peak_delta,
                'p_val': p_val,
                'latency_sweeps': latency_sweep,
                'range_sweeps': (start_sweep, end_sweep)
            })

        return grouped

    def response_metadata_from_row(self, row: pd.Series) -> dict:
        """Metadata carried from PRE/POST rows into response tables."""
        return row.drop(labels=["PRE_sweeps", "POST_sweeps"], errors="ignore").to_dict()

    def add_response_timing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add compact response timing columns.

        ``latency``, ``start_s``, and ``end_s`` are seconds relative to
        ``drug_in``/POST sweep zero. They are NaN if sweep duration was not
        available in the extracted APP_IC dataframe. Sweep-index timing is kept
        as a fallback for plotting/debugging when old caches lack seconds.
        """
        if df is None or df.empty:
            return df

        df = df.copy()

        def range_start(value):
            if isinstance(value, tuple) and len(value) == 2:
                return value[0]
            if isinstance(value, list) and len(value) == 2:
                return value[0]
            return np.nan

        def range_end(value):
            if isinstance(value, tuple) and len(value) == 2:
                return value[1]
            if isinstance(value, list) and len(value) == 2:
                return value[1]
            return np.nan

        sweep_duration_s = pd.to_numeric(
            df.get("sweep_duration_s", np.nan),
            errors="coerce",
        )
        latency_sweeps = pd.to_numeric(df.get("latency_sweeps", np.nan), errors="coerce")
        start_sweeps = pd.to_numeric(df["range_sweeps"].apply(range_start), errors="coerce")
        end_sweeps = pd.to_numeric(df["range_sweeps"].apply(range_end), errors="coerce")

        df["latency"] = latency_sweeps * sweep_duration_s
        df["start_s"] = start_sweeps * sweep_duration_s
        df["end_s"] = end_sweeps * sweep_duration_s
        df["start_sweeps"] = start_sweeps
        df["end_sweeps"] = end_sweeps
        df["duration_sweeps"] = end_sweeps - start_sweeps
        df["duration_s"] = df["duration_sweeps"] * sweep_duration_s
        if "post_sweep_window_used" in df.columns:
            post_sweeps = pd.to_numeric(df["post_sweep_window_used"], errors="coerce")
            df["recording_end_s"] = post_sweeps * sweep_duration_s
        if "response" in df.columns:
            active_response = df["response"].isin(["increase", "decrease", "biphasic"])
            df.loc[~active_response, "latency"] = np.nan
        return df

    def get_response_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return one row per checked POST bin, before grouping consecutive bins.
        """
        result_rows = []

        for _, row in df.iterrows():
            pre = row["PRE_sweeps"]
            post = row["POST_sweeps"]
            cell_id = row["cell_id"]
            metadata = self.response_metadata_from_row(row)

            if self.dynamic_search:
                bin_results = self.analyze_post_bins(pre, post, cell_id)
            else:
                if "_count" in self.dependant_var:
                    pre_mean = np.mean(pre)
                    post_mean = np.mean(post)
                    mean_diff = post_mean - pre_mean
                    response = "no response"
                    if abs(mean_diff) >= self.diff_thresh:
                        response = "increase" if mean_diff > 0 else "decrease"
                    result = {
                        "response": response,
                        "baseline_mean": pre_mean,
                        "event_mean": post_mean,
                        "mean_diff": mean_diff,
                        "percent_diff": np.nan,
                        "p_val": np.nan,
                    }
                else:
                    result = self.response_stats(
                        pre,
                        post,
                        percentage_threshold=self.sweep_dependant_var() == "sweep_inputR_MOhm",
                    )

                bin_results = [{
                    "cell_id": cell_id,
                    "PRE_sweeps": pre,
                    "POST_sweeps": post,
                    "response": result["response"],
                    "baseline_mean": result["baseline_mean"],
                    "event_mean": result["event_mean"],
                    "delta": result["mean_diff"],
                    "p_val": result["p_val"],
                    "latency_sweeps": 0,
                    "range_sweeps": (0, len(post)),
                }]

            for result in bin_results:
                result_rows.append({**metadata, **result})

        return self.add_response_timing_columns(pd.DataFrame(result_rows))

    def response_excel_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop long sweep arrays and stringify list-like metadata for Excel."""
        if df is None:
            return pd.DataFrame()

        excel_df = df.copy()
        for sweep_col in ["PRE_sweeps", "POST_sweeps"]:
            if sweep_col not in excel_df.columns:
                continue
            excel_df[f"{sweep_col}_n"] = excel_df[sweep_col].apply(self.sweep_count)
            excel_df[f"{sweep_col}_mean"] = excel_df[sweep_col].apply(self.mean_or_nan)
            excel_df = excel_df.drop(columns=[sweep_col])

        def clean_value(value):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, tuple):
                return ":".join(map(str, value))
            if isinstance(value, list):
                return ", ".join(map(str, value))
            return value

        for col in excel_df.columns:
            excel_df[col] = excel_df[col].apply(clean_value)
        return excel_df

    def save_response_workbook(self, filename: str, tables: dict):
        """Save response tables into one Excel workbook with one sheet per table."""
        filename = self.sanitize_filename(filename)
        if not filename.endswith('.xlsx'):
            filename += '.xlsx'
        filepath = os.path.join(self.output_dir, filename)
        os.makedirs(self.output_dir, exist_ok=True)
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, df in tables.items():
                self.response_excel_df(df).to_excel(
                    writer,
                    sheet_name=str(sheet_name)[:31],
                    index=False,
                )
        print(f'CREATED {filepath} RESPONSE WORKBOOK')

    def get_responses(self, df) -> pd.DataFrame:  #take in pre_post_bins filtered or not but df with just cell_id pre post
        result_rows = []

        for _, row in df.iterrows():
            pre = row['PRE_sweeps']
            post = row['POST_sweeps']
            cell_id = row['cell_id']
            metadata = self.response_metadata_from_row(row)

            if self.dynamic_search:
                bin_results = self.analyze_post_bins(pre, post, cell_id) #sliding window analysis, t-test and filter on diff_threshold
                grouped = self.group_consecutive_responses(bin_results)
                if grouped:
                    result_rows.extend({**metadata, **result} for result in grouped)
                else:
                    # print(f"No significant bins found for {cell_id}")
                    result_rows.append({
                        **metadata,
                        'cell_id': cell_id,
                        'dependant_var': self.dependant_var,
                        'PRE_sweeps': pre,
                        'POST_sweeps': post, 
                        'response': 'no response',
                        'delta': np.nan,
                        'p_val': 1.0,
                        'latency_sweeps': None,
                        'range_sweeps': None
                    })
            else:
                if '_count' in self.dependant_var: # no stats for cpunts to avoid 0 lists
                    pre_mean = np.mean(pre)
                    post_mean = np.mean(post)
                    mean_diff = post_mean - pre_mean
                    response = 'no response'
                    if abs(mean_diff) >= self.diff_thresh:
                        response = 'increase' if mean_diff > 0 else 'decrease'

                    result = {
                        'response': response,
                        'baseline_mean': pre_mean,
                        'event_mean': post_mean,
                        'mean_diff': mean_diff,
                        'p_val': np.nan
                    }
                else:
                    result = self.response_stats(
                        pre,
                        post,
                        percentage_threshold=self.sweep_dependant_var() == "sweep_inputR_MOhm",
                    )

                result_rows.append({
                    **metadata,
                    'cell_id': cell_id,
                    'dependant_var': self.dependant_var,
                    'PRE_sweeps': pre,
                    'POST_sweeps': post,
                    'response': result['response'],
                    'baseline_mean': result['baseline_mean'],
                    'event_mean': result['event_mean'],
                    'delta': result['mean_diff'],
                    'p_val': result['p_val'],
                    'latency_sweeps': 0,
                    'range_sweeps': (0, len(post))
                })

        return self.add_response_timing_columns(pd.DataFrame(result_rows))
    
    def get_plot_param(self, key, default=None):
        """
        Read optional plotting parameters from self.plot_params.

        Child classes can define:
            plot_params: dict = field(default_factory=dict)
        """
        return getattr(self, "plot_params", {}).get(key, default)


    def format_param_label(self, name, value, for_filename=False):
        """
        Format one optional parameter for a filename or title.
        """
        if value is None:
            return None

        if isinstance(value, (list, tuple)):
            value = "-".join(map(str, value)) if for_filename else ", ".join(map(str, value))

        if for_filename:
            return f"{name}_{value}"

        return f"{name} = {value}"


    def optional_param_labels(self, params, for_filename=False):
        """
        Build labels for optional parameters.

        params should be:
            {"I_range_pA": self.I_range_pA, "n_minimum": self.n_minimum}
        """
        labels = []
        for name, value in params.items():
            label = self.format_param_label(name, value, for_filename=for_filename)
            if label is not None:
                labels.append(label)
        return labels
    
    
    
    def get_unit(self): 
        base_var = self.dependant_var.replace("sweep_", "")
        if base_var == 'inputR_MOhm':
            return '%'
        elif base_var == 'RMP_mV':
            return 'mV'
        elif '_count' in base_var:
            return 'APs/sweep'
        else:
            return unit_dict.get(base_var, base_var)
        
    def analyze_post_bins(self, pre: np.ndarray, post: np.ndarray, cell_id: str) -> list[dict]:
        results = []
        n_bins = len(post) - self.bin_width + 1
        pre_mean = np.mean(pre)

        for i in range(n_bins):
            bin_post = post[i:i + self.bin_width]  # sliding window of bin_width
            if '_count' in self.dependant_var: #HARD HACKY TODO  
                post_mean = np.mean(bin_post)
                mean_diff = post_mean - pre_mean

                if abs(mean_diff) >= self.diff_thresh:
                    response = 'increase' if mean_diff > 0 else 'decrease'
                else:
                    response = 'no response'

                result = {
                    'response': response,
                    'baseline_mean': pre_mean,
                    'event_mean': post_mean,
                    'mean_diff': mean_diff,
                    'p_val': None
                }

            else:
                result = self.response_stats(
                    pre,
                    bin_post,
                    percentage_threshold=self.sweep_dependant_var() == "sweep_inputR_MOhm",
                )

            results.append({
                'cell_id': cell_id,
                'PRE_sweeps': pre,
                'POST_sweeps': post,
                'response': result['response'],
                'baseline_mean': result['baseline_mean'],
                'event_mean': result['event_mean'],
                'delta': result['mean_diff'],
                'p_val': result['p_val'],
                'latency_sweeps': i,  # start sweep index of the bin
                'range_sweeps': (i, i + self.bin_width)  
            })

        return results
    
    def save_plot(self, fig, filename: str, formats=('png', 'svg')):
        """
        Saves a plot in specified formats to the figure directory.

        Parameters:
        ----------
        fig : matplotlib.figure.Figure
            The matplotlib figure to save.
        filename : str
            The base name of the file (without extension).
        formats : tuple, optional
            Formats to save the plot (default: ('png', 'svg')).
        """
        for fmt in formats:
            filepath = os.path.join(self.figure_output_dir, f"{filename}.{fmt}")
            fig.savefig(filepath, format=fmt, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved figure: {filename} in formats: {formats}")

    def build_name(self, *args, sep="_", titlecase=False):
        """
        Build a clean, flattened name string from any mix of strings, lists, tuples, or None.

        Parameters
        ----------
        *args : str | list | tuple | None
            Any number of items to join together. Nested lists/tuples are flattened.
        sep : str, optional
            Separator used to join the strings (default: '_').
        titlecase : bool, optional
            If True, capitalizes each word (useful for figure titles).

        Returns
        -------
        str
            A single cleaned string joined by the given separator.
        """
        def flatten(items):
            """Recursively flatten nested structures."""
            for i in items:
                if i is None:
                    continue
                elif isinstance(i, (list, tuple, set)):
                    yield from flatten(i)
                else:
                    yield str(i).strip()

        parts = [p for p in flatten(args) if p]
        if titlecase:
            parts = [p.title() for p in parts]
        return sep.join(parts)
    
    def t_test_stats(self, df, group_col, value_col, alpha=0.05):
        """
        Compute independant t-tests between groups in `group_col`.
        Returns a list of dicts with (group1, group2, p_val, significant).
        """
        import itertools
        from scipy.stats import ttest_ind

        results = []
        groups = df[group_col].unique()
        for g1, g2 in itertools.combinations(groups, 2):
            vals1 = df[df[group_col] == g1][value_col].dropna()
            vals2 = df[df[group_col] == g2][value_col].dropna()
            if len(vals1) < 2 or len(vals2) < 2:
                continue
            t_stat, p_val = ttest_ind(vals1, vals2, equal_var=False)
            results.append({
                "group1": g1,
                "group2": g2,
                "p_val": p_val,
                "significant": p_val < alpha
            })
        return results
    
    def rgba_color(self, color, alpha=1.0):
        """
        Convert a matplotlib color name/hex/RGB to RGBA with custom alpha.
        """
        return mcolors.to_rgba(color, alpha=alpha)

    

