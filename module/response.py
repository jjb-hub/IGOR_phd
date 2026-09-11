from module.figure_common import *
from module.figure_base import Figure

@dataclass
class ResponseCharecterisation(Figure):
    """
    Response tables:
    - response_bin_df: every sliding POST window tested per cell/DV.
    - response_event_df: DV-specific response windows.
    - response_episode_df: overlapping DV events merged per cell.
    - response_cell_df: one row per cell for plotting latency/magnitude/duration.
    """
    filename: str = None
    n_minimum: float = field(kw_only = True, default = 3)
    pre_sweep_window: int = None # window before and after drug_in
    post_sweep_window: int = None

    diff_threshs: list[int] | int = field(kw_only=True) # ie 3mV RMP 0 AP_count
    diff_thresh: int = field(kw_only = True, default = 0) # ie 3mV difference required to consider it a response 

    dependant_vars: list[str] | str = field(kw_only=True)
    dependant_var: str = field(init=False)  # will be set in __post_init__

    p_thresh: float = field(kw_only = True, default = 0.05)
    dynamic_search: bool = field(kw_only=True, default=False)
    bin_width: int = field(kw_only=True, default=3)
    cache_response: bool = field(kw_only=True, default=False)
    valid_only: bool = field(kw_only=True, default=False)



    def __post_init__(self):
        self.dependant_vars = self.normalized_response_vars(self.dependant_vars)
        self.diff_threshs = self.normalized_diff_threshs(self.diff_threshs)
        self.requested_pre_sweep_window = self.pre_sweep_window
        self.requested_post_sweep_window = self.post_sweep_window
        pre_label = (
            f"pre_{self.requested_pre_sweep_window}"
            if self.requested_pre_sweep_window is not None
            else "pre_auto"
        )
        post_label = (
            f"post_{self.requested_post_sweep_window}"
            if self.requested_post_sweep_window is not None
            else "post_auto"
        )
        self.filename = self.build_name(
            "response",
            self.data_type,
            self.dependant_vars,
            f"thresh_{'-'.join(map(str, self.diff_threshs))}",
            self.selection_label_parts(for_filename=True),
            f"access_{self.threshold_access_change}",
            "valid_only" if self.valid_only else None,
            pre_label,
            post_label,
            "dynamic" if self.dynamic_search else None,
            f"bin_{self.bin_width}" if self.dynamic_search else None,
            sep="_",
        )
        self.slice = False if self.requested_post_sweep_window == "all" else True 
        
        super().__post_init__()
        self.validate_response_inputs()
        self.raw_time_df = self.agg_df.copy()
        self.raw_time_df = self.apply_response_valid_only(self.raw_time_df)
        self.pre_post_dfs = {}
        self.response_bin_df = pd.DataFrame()
        self.response_window_df = pd.DataFrame()
        self.response_event_df = self.aggregate_cell_responses()
        self.response_df = self.response_event_df  # temporary alias for older notebook/Application calls
        self.response_episode_df = self.build_response_episode_df()
        self.response_cell_df = self.build_response_cell_df()
        if self.cache_response:
            self.cache_response_tables()
        self.fig = self.plot_functional_response_pie()

    def normalized_response_vars(self, dependant_vars) -> list[str]:
        vars_list = (
            list(dependant_vars)
            if isinstance(dependant_vars, (list, tuple, np.ndarray, pd.Series))
            else [dependant_vars]
        )
        return [str(var).replace("sweep_", "", 1) for var in vars_list]

    def normalized_diff_threshs(self, diff_threshs) -> list[float]:
        return (
            list(diff_threshs)
            if isinstance(diff_threshs, (list, tuple, np.ndarray, pd.Series))
            else [diff_threshs]
        )

    def validate_response_inputs(self):
        if len(self.dependant_vars) != len(self.diff_threshs):
            raise ValueError(
                "ResponseCharecterisation requires one diff_thresh per dependant_var. "
                f"Got dependant_vars={self.dependant_vars}, diff_threshs={self.diff_threshs}."
            )

        missing_cols = [
            f"sweep_{dependant_var}"
            for dependant_var in self.dependant_vars
            if f"sweep_{dependant_var}" not in self.agg_df.columns
        ]
        if missing_cols:
            valid_sweep_dvs = [
                col.replace("sweep_", "", 1)
                for col in self.agg_df.columns
                if col.startswith("sweep_")
            ]
            raise ValueError(
                f"Invalid response dependant variable(s): {missing_cols}. "
                f"Valid sweep variables: {valid_sweep_dvs}"
            )

    def apply_response_valid_only(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.valid_only or df is None or df.empty:
            return df

        if "valid" not in df.columns:
            print("ResponseCharecterisation valid_only skipped: no valid column found.")
            return df

        before_cells = df["cell_id"].nunique() if "cell_id" in df.columns else len(df)
        filtered_df = df[df["valid"] != False].copy()
        after_cells = filtered_df["cell_id"].nunique() if "cell_id" in filtered_df.columns else len(filtered_df)
        dropped = before_cells - after_cells
        if dropped > 0:
            print(f"ResponseCharecterisation valid_only dropped {dropped} cell(s) marked valid=False.")
        return filtered_df

    def aggregate_cell_responses(self):
        dv_dfs = []
        bin_dfs = []
        window_rows = []
        for i, (dependant_var, diff_thresh) in enumerate(zip(self.dependant_vars, self.diff_threshs)):
            self.pre_sweep_window = self.requested_pre_sweep_window
            self.post_sweep_window = self.requested_post_sweep_window
            self.dependant_var = f"sweep_{dependant_var}"
            self.diff_thresh = diff_thresh

            data = self.filter_n_minimum(self.raw_time_df.copy())
            if data is None or data.empty:
                print(f"ResponseCharecterisation skipped {dependant_var}: no data after n_minimum filtering.")
                continue

            pre_post_df = self.build_pre_post_df(data, slice=self.slice, verbose=(i == 0))
            pre_post_df["base_dependant_var"] = dependant_var
            pre_post_df["diff_thresh"] = diff_thresh
            pre_post_df["p_thresh"] = self.p_thresh
            pre_post_df["dynamic_search"] = self.dynamic_search
            pre_post_df["bin_width"] = self.bin_width
            self.pre_post_dfs[dependant_var] = pre_post_df
            if pre_post_df.empty:
                print(f"ResponseCharecterisation skipped {dependant_var}: no cells with PRE and POST sweeps.")
                continue

            window_rows.append({
                "base_dependant_var": dependant_var,
                "requested_pre_sweep_window": self.requested_pre_sweep_window,
                "requested_post_sweep_window": self.requested_post_sweep_window,
                "pre_sweep_window_used_min": pre_post_df["pre_sweep_window_used"].min(),
                "pre_sweep_window_used_max": pre_post_df["pre_sweep_window_used"].max(),
                "post_sweep_window_used_min": pre_post_df["post_sweep_window_used"].min(),
                "post_sweep_window_used_max": pre_post_df["post_sweep_window_used"].max(),
                "n_cells": pre_post_df["cell_id"].nunique(),
                "dynamic_search": self.dynamic_search,
                "bin_width": self.bin_width,
            })

            dv_bin_df = self.get_response_bins(pre_post_df)
            if not dv_bin_df.empty:
                dv_bin_df["base_dependant_var"] = dependant_var
                dv_bin_df["diff_thresh"] = diff_thresh
                dv_bin_df["p_thresh"] = self.p_thresh
                dv_bin_df["dynamic_search"] = self.dynamic_search
                dv_bin_df["bin_width"] = self.bin_width
                dv_bin_df = self.add_threshold_normalized_delta(dv_bin_df)
                bin_dfs.append(dv_bin_df)

            dv_response_df = self.get_responses(pre_post_df)
            dv_response_df["base_dependant_var"] = dependant_var
            dv_response_df["diff_thresh"] = diff_thresh
            dv_response_df["p_thresh"] = self.p_thresh
            dv_response_df["dynamic_search"] = self.dynamic_search
            dv_response_df["bin_width"] = self.bin_width
            dv_response_df = self.add_threshold_normalized_delta(dv_response_df)
            dv_dfs.append(dv_response_df)

        self.response_bin_df = (
            pd.concat(bin_dfs, ignore_index=True)
            if bin_dfs
            else pd.DataFrame()
        )
        self.response_window_df = pd.DataFrame(window_rows)

        if not dv_dfs:
            return pd.DataFrame(columns=[
                "cell_id",
                "dependant_var",
                "base_dependant_var",
                "response",
                "latency",
                "start_s",
                "end_s",
                "diff_thresh",
            ])
        return pd.concat(dv_dfs, ignore_index=True)

    def add_threshold_normalized_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Magnitude score: abs(raw delta) divided by the response threshold."""
        if df is None or df.empty or "delta" not in df.columns:
            return df

        df = df.copy()
        deltas = pd.to_numeric(df["delta"], errors="coerce").abs()
        thresholds = pd.to_numeric(df.get("diff_thresh", np.nan), errors="coerce").abs()
        df["threshold_normalized_delta"] = np.where(
            thresholds > 0,
            deltas / thresholds,
            np.nan,
        )
        if "response" in df.columns:
            active_response = self.active_response_mask(df)
            df.loc[~active_response, "threshold_normalized_delta"] = np.nan
        return df

    def response_pre_post_df(self) -> pd.DataFrame:
        if not self.pre_post_dfs:
            return pd.DataFrame()
        return pd.concat(self.pre_post_dfs.values(), ignore_index=True)

    def response_table_bundle(self) -> dict:
        return {
            "bins": self.response_bin_df,
            "events": self.response_event_df,
            "episodes": self.response_episode_df,
            "cells": self.response_cell_df,
            "pre_post": self.response_pre_post_df(),
            "windows": self.response_window_df,
        }

    def cache_response_tables(self):
        """
        Cache this response-analysis run as one plot-specific bundle.

        Bundle tables:
        - bins: every checked POST bin before consecutive bins are grouped.
        - events: DV-specific response windows.
        - episodes: overlapping DV events merged within each cell.
        - cells: one row per cell for latency/magnitude/duration plots.
        - pre_post: the PRE/POST sweep windows used for each cell.
        - windows: requested and actual PRE/POST window sizes.

        Event/episode timing is seconds from drug_in when sweep_duration_s exists.
        """
        tables = self.response_table_bundle()
        self.cache(f"{self.filename}_response_bundle", tables)
        self.save_response_workbook(f"{self.filename}_response_bundle", tables)

    def active_response_mask(self, df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty or "response" not in df.columns:
            return pd.Series(False, index=df.index if df is not None else None)
        return df["response"].isin(["increase", "decrease", "biphasic"])

    def range_sweep_start(self, value):
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return value[0]
        return np.nan

    def range_sweep_end(self, value):
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return value[1]
        return np.nan

    def signed_peak_delta(self, deltas):
        values = pd.to_numeric(pd.Series(deltas), errors="coerce").dropna()
        if values.empty:
            return np.nan
        return float(values.iloc[np.argmax(np.abs(values.to_numpy()))])

    def response_label(self, value):
        if pd.isna(value):
            return "no response"
        value = str(value).strip().lower()
        return value if value else "no response"

    def collapse_response_labels(self, values, cell_level=False):
        labels = {self.response_label(value) for value in values}
        active = labels & {"increase", "decrease", "biphasic"}
        if not active:
            return "no response"
        if "biphasic" in active or {"increase", "decrease"}.issubset(active):
            return "biphasic"
        if "increase" in active:
            return "excitatory" if cell_level else "increase"
        if "decrease" in active:
            return "inhibitory" if cell_level else "decrease"
        return "no response"

    def event_sort_value(self, row):
        for col in ["start_s", "start_sweeps", "latency", "latency_sweeps"]:
            value = row.get(col, np.nan)
            if pd.notna(value):
                return float(value)
        return np.inf

    def events_overlap(self, episode, event):
        start_s = event.get("start_s", np.nan)
        if pd.notna(episode.get("end_s", np.nan)) and pd.notna(start_s):
            return float(start_s) <= float(episode["end_s"])

        start_sweeps = event.get("start_sweeps", np.nan)
        if pd.notna(episode.get("end_sweeps", np.nan)) and pd.notna(start_sweeps):
            return float(start_sweeps) <= float(episode["end_sweeps"])

        return False

    def build_response_episode_df(self) -> pd.DataFrame:
        """One row per merged response episode; overlapping DVs become one episode."""
        event_df = getattr(self, "response_event_df", pd.DataFrame())
        if event_df is None or event_df.empty:
            return pd.DataFrame()

        active_events = event_df[self.active_response_mask(event_df)].copy()
        if active_events.empty:
            return pd.DataFrame()

        exclude = {
            "PRE_sweeps",
            "POST_sweeps",
            "dependant_var",
            "base_dependant_var",
            "response",
            "baseline_mean",
            "event_mean",
            "delta",
            "mean_delta",
            "signed_peak_delta",
            "threshold_normalized_delta",
            "p_val",
            "latency",
            "start_s",
            "end_s",
            "duration_s",
            "latency_sweeps",
            "range_sweeps",
            "start_sweeps",
            "end_sweeps",
            "duration_sweeps",
        }

        episode_rows = []
        for cell_id, cell_events in active_events.groupby("cell_id", dropna=False):
            events = [
                row for _, row in cell_events.sort_values(
                    by=["start_s", "start_sweeps"],
                    na_position="last",
                ).iterrows()
            ]

            episodes = []
            for event in events:
                if not episodes or not self.events_overlap(episodes[-1], event):
                    episodes.append({
                        "events": [event],
                        "start_s": event.get("start_s", np.nan),
                        "end_s": event.get("end_s", np.nan),
                        "start_sweeps": event.get("start_sweeps", np.nan),
                        "end_sweeps": event.get("end_sweeps", np.nan),
                    })
                    continue

                episode = episodes[-1]
                episode["events"].append(event)
                for end_col in ["end_s", "end_sweeps"]:
                    current_end = episode.get(end_col, np.nan)
                    event_end = event.get(end_col, np.nan)
                    if pd.notna(event_end) and (pd.isna(current_end) or event_end > current_end):
                        episode[end_col] = event_end

            for episode_index, episode in enumerate(episodes, start=1):
                event_table = pd.DataFrame([event.to_dict() for event in episode["events"]])
                first = event_table.iloc[0]
                row = first.drop(labels=list(exclude), errors="ignore").to_dict()
                row["cell_id"] = cell_id
                row["episode_index"] = episode_index
                row["episode_dependant_vars"] = (
                    event_table["base_dependant_var"].dropna().astype(str).unique().tolist()
                    if "base_dependant_var" in event_table.columns
                    else []
                )
                row["episode_response"] = self.collapse_response_labels(event_table["response"])

                start_s = pd.to_numeric(event_table.get("start_s", pd.Series(dtype=float)), errors="coerce")
                end_s = pd.to_numeric(event_table.get("end_s", pd.Series(dtype=float)), errors="coerce")
                start_sweeps = pd.to_numeric(event_table.get("start_sweeps", pd.Series(dtype=float)), errors="coerce")
                end_sweeps = pd.to_numeric(event_table.get("end_sweeps", pd.Series(dtype=float)), errors="coerce")
                magnitude = pd.to_numeric(
                    event_table.get("threshold_normalized_delta", pd.Series(dtype=float)),
                    errors="coerce",
                )

                first_start_s = start_s.min() if start_s.notna().any() else np.nan
                last_end_s = end_s.max() if end_s.notna().any() else np.nan
                first_start_sweeps = start_sweeps.min() if start_sweeps.notna().any() else np.nan
                last_end_sweeps = end_sweeps.max() if end_sweeps.notna().any() else np.nan

                if magnitude.notna().any():
                    peak_idx = magnitude.idxmax()
                    magnitude_score = float(magnitude.loc[peak_idx])
                    magnitude_dv = event_table.loc[peak_idx].get("base_dependant_var", np.nan)
                    magnitude_delta = event_table.loc[peak_idx].get("delta", np.nan)
                else:
                    magnitude_score = np.nan
                    magnitude_dv = np.nan
                    magnitude_delta = np.nan

                row.update({
                    "latency_s": first_start_s,
                    "start_s": first_start_s,
                    "end_s": last_end_s,
                    "duration_s": (
                        float(last_end_s - first_start_s)
                        if pd.notna(first_start_s) and pd.notna(last_end_s)
                        else np.nan
                    ),
                    "latency_sweeps": first_start_sweeps,
                    "start_sweeps": first_start_sweeps,
                    "end_sweeps": last_end_sweeps,
                    "duration_sweeps": (
                        float(last_end_sweeps - first_start_sweeps)
                        if pd.notna(first_start_sweeps) and pd.notna(last_end_sweeps)
                        else np.nan
                    ),
                    "episode_magnitude": magnitude_score,
                    "episode_magnitude_dv": magnitude_dv,
                    "episode_magnitude_delta": magnitude_delta,
                    "n_response_events": int(len(event_table)),
                })
                episode_rows.append(row)

        return pd.DataFrame(episode_rows)

    def build_response_cell_df(self):
        """One row per cell; used by response pies and ResponseHistogram."""
        event_df = getattr(self, "response_event_df", pd.DataFrame())
        included_cells = (
            event_df["cell_id"].dropna().unique()
            if event_df is not None and not event_df.empty and "cell_id" in event_df.columns
            else []
        )
        if len(included_cells) == 0:
            print("ResponseCharecterisation found no response rows.")
            return pd.DataFrame(columns=list(self.cell_df.columns) + ["response", "responder"])

        cell_df = self.cell_df[self.cell_df["cell_id"].isin(included_cells)].copy()
        episodes = getattr(self, "response_episode_df", pd.DataFrame())
        active_events = event_df[self.active_response_mask(event_df)].copy()

        for col in [
            "response",
            "response_label",
            "response_latency_s",
            "first_response_duration_s",
            "first_response_magnitude",
            "first_response_dvs",
            "second_response_latency_s",
            "second_response_duration_s",
            "second_response_magnitude",
            "second_response_dvs",
            "response_episode_count",
            "total_response_duration_s",
            "total_response_magnitude",
            "peak_response_magnitude",
            "washout_time_s",
        ]:
            cell_df[col] = np.nan

        cell_df["response"] = "no response"
        cell_df["response_label"] = "no response"
        cell_df["responder"] = False
        cell_df["response_episode_count"] = 0
        cell_df["total_response_duration_s"] = 0.0
        cell_df["total_response_magnitude"] = 0.0
        cell_df["peak_response_magnitude"] = 0.0

        if episodes is None or episodes.empty:
            return cell_df

        for cell_id, episode_df in episodes.groupby("cell_id", dropna=False):
            episode_df = episode_df.sort_values(["start_s", "start_sweeps"], na_position="last")
            mask = cell_df["cell_id"] == cell_id
            if not mask.any():
                continue

            cell_event_labels = (
                active_events.loc[active_events["cell_id"] == cell_id, "response"]
                if not active_events.empty
                else pd.Series(dtype=object)
            )
            response_label = self.collapse_response_labels(cell_event_labels, cell_level=True)
            magnitudes = pd.to_numeric(episode_df["episode_magnitude"], errors="coerce")
            durations = pd.to_numeric(episode_df["duration_s"], errors="coerce")
            end_s = pd.to_numeric(episode_df["end_s"], errors="coerce")
            recording_end_s = pd.to_numeric(
                episode_df.get("recording_end_s", pd.Series(dtype=float)),
                errors="coerce",
            )

            cell_df.loc[mask, "response"] = response_label
            cell_df.loc[mask, "response_label"] = response_label
            cell_df.loc[mask, "responder"] = True
            cell_df.loc[mask, "response_episode_count"] = int(len(episode_df))
            cell_df.loc[mask, "total_response_duration_s"] = (
                float(durations.sum()) if durations.notna().any() else np.nan
            )
            cell_df.loc[mask, "total_response_magnitude"] = (
                float(magnitudes.sum()) if magnitudes.notna().any() else np.nan
            )
            cell_df.loc[mask, "peak_response_magnitude"] = (
                float(magnitudes.max()) if magnitudes.notna().any() else np.nan
            )

            if end_s.notna().any() and recording_end_s.notna().any():
                washout = float(recording_end_s.max() - end_s.max())
                if washout > 0:
                    cell_df.loc[mask, "washout_time_s"] = washout

            first_episode = episode_df.iloc[0]
            cell_df.loc[mask, "response_latency_s"] = first_episode.get("latency_s", np.nan)
            cell_df.loc[mask, "first_response_duration_s"] = first_episode.get("duration_s", np.nan)
            cell_df.loc[mask, "first_response_magnitude"] = first_episode.get("episode_magnitude", np.nan)
            cell_df.loc[mask, "first_response_dvs"] = ", ".join(map(str, first_episode.get("episode_dependant_vars", [])))

            if len(episode_df) > 1:
                second_episode = episode_df.iloc[1]
                cell_df.loc[mask, "second_response_latency_s"] = second_episode.get("latency_s", np.nan)
                cell_df.loc[mask, "second_response_duration_s"] = second_episode.get("duration_s", np.nan)
                cell_df.loc[mask, "second_response_magnitude"] = second_episode.get("episode_magnitude", np.nan)
                cell_df.loc[mask, "second_response_dvs"] = ", ".join(map(str, second_episode.get("episode_dependant_vars", [])))

        return cell_df



    def plot_functional_response_pie(self):
        """
        Plots cell-level response labels from self.response_cell_df.
        """
        if self.response_cell_df.empty:
            print("No cells available for response pie plot.")
            return None

        response_series = self.response_cell_df['response'].fillna('no response').str.lower()
        counts = response_series.value_counts()
        total_n = len(response_series)

        # Define colors
        colors = {
            'excitatory': 'salmon',
            'inhibitory': 'deepskyblue',
            'biphasic': 'mediumorchid',
            'mixed': 'mediumorchid',
            'no response': 'whitesmoke'
        }
        pie_colors = [colors.get(label, 'gray') for label in counts.index]

        # Print cell IDs for each category
        for label in ['excitatory', 'inhibitory', 'biphasic', 'no response']:
            matching_cells = self.response_cell_df.loc[
                response_series == label, 'cell_id'
            ].tolist()
            print(f"{label} cells ({len(matching_cells)}): {matching_cells}")

        def format_autopct(pct):
            count = int(round(pct * total_n / 100.0))
            return f'{pct:.1f}%\n({count})' #/{total_n}

        fig, ax = plt.subplots(figsize=(6, 6))
        counts.plot.pie(
            autopct=format_autopct,
            startangle=90,
            ylabel='',
            textprops={'fontsize': 12},
            colors=pie_colors,
            ax=ax
        )

        selection_title = self.build_name(
            *self.selection_label_parts(for_filename=False),
            sep=" | ",
        )
        title = (
            f"{selection_title}\nresponse (n={total_n})"
            if selection_title
            else f"{self.project} response (n={total_n})"
        )
        ax.set_title(title, fontsize=14)

        fig.text(0.5, 0.06, f"Significant response > {self.dependant_vars} for {self.diff_threshs}", 
                ha='center', fontsize=10, style='italic')

        fig.text(0.5, 0.02, 
                f"PRE sweep window: {self.pre_sweep_window}   |   POST sweep window: {None if self.dynamic_search else self.post_sweep_window}",
                ha='center', fontsize=10, style='italic')

        plt.tight_layout()
        plt.show()
        self.save_plot(fig, self.filename + '_functional_pie')
        return fig


        
