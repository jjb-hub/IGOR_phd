from module.figure_common import *
from module.figure_base import Figure

@dataclass
class AggregateApplication(Figure):
    filename: str = None
    sweep_in_s: float = field(kw_only = True, default = 20)
    dependant_var: str = field(kw_only=True) 
    bin_size: float = field(kw_only = True, default = 3) #sweeps to pool default 3 3x20sec - 1min
    n_minimum: float = field(kw_only = True, default = 3)
    normalise: bool = field(kw_only = True, default = False)


    def __post_init__(self):
        self.filename = f"{self.dependant_var}_{self.bin_size}" 
        super().__post_init__()
        self.check_valid_dependant_var()
        self.data = self.filter_n_minimum(self.agg_df)
        self.timepoints = ['PRE', 'APP', 'WASH']
        self.colors = {'PRE': 'lightgrey', 'APP': 'black', 'WASH': 'grey'} #black preset will be changed in plot
        # self.binned_data = self.process_data(self.data)
        # self.plot()                       #OLD CODE segregated data
        self.continuous_plot()



    def continuous_plot(self):
        column_to_plot = f'sweep_{self.dependant_var}'
        subset_df = self.data[[column_to_plot, 'cell_id', 'time']]

        fig, ax = plt.subplots(figsize=(10, 5))
        cell_counts = self.cell_df[self.cell_df['cell_id'].isin(subset_df['cell_id'])].groupby('treatment')['cell_id'].nunique().to_dict()
        legend_handles = {}

        for cell_id, cell_data in subset_df.groupby('cell_id'):
            sweeps = {
                row['time']: row[column_to_plot]
                for _, row in cell_data.iterrows()
                if isinstance(row[column_to_plot], list)
            }
            # --- PRE binning ---
            pre_vals = sweeps.get('PRE', [])[-10:]
            remainder = len(pre_vals) % self.bin_size
            if remainder:
                pre_vals = pre_vals[remainder:]
            pre_bins = [
                np.mean(pre_vals[i:i + self.bin_size])
                for i in range(0, len(pre_vals), self.bin_size)
            ]
            pre_times = ['PRE'] * len(pre_bins)
            # --- APP + WASH binning ---
            app_vals = sweeps.get('APP', [])
            wash_vals = sweeps.get('WASH', [])
            aw_vals = app_vals + wash_vals
            aw_times = ['APP'] * len(app_vals) + ['WASH'] * len(wash_vals)
            aw_bins, aw_bin_times = [], []
            for i in range(0, len(aw_vals), self.bin_size):
                chunk = aw_vals[i:i + self.bin_size]
                time_chunk = aw_times[i:i + self.bin_size]
                if not chunk:
                    continue
                aw_bins.append(np.mean(chunk))
                aw_bin_times.append('APP' if 'APP' in time_chunk else 'WASH')

            # Combine all bins
            binned_values = pre_bins + aw_bins
            binned_times = pre_times + aw_bin_times
            cell_ids = [cell_id] * len(binned_values)
            x_vals = list(range(len(binned_values)))
            plot_df = pd.DataFrame({
                column_to_plot: binned_values,
                'time': binned_times,
                'cell_id': cell_ids, 
                'x': x_vals
            })
            
            #normalisation
            y_label = f"{unit_dict[self.dependant_var]}"
            if self.normalise:
                y_label = f"{unit_dict[self.dependant_var]} as % of baseline"
                pre_avg = np.mean(pre_bins)  # Average of PRE bins
                plot_df[column_to_plot] = (plot_df[column_to_plot] / pre_avg) * 100  # Normalize as percentage of PRE

            drug = self.cell_df[self.cell_df['cell_id'] == cell_id]['treatment'].iloc[0]
            color_map = {
                'PRE': 'lightgrey',
                'APP': color_dict.get(drug, 'k'),
                'WASH': color_dict.get(drug, 'k')
            }
            alpha_map = {
                'PRE': 1.0,
                'APP': 1.0,
                'WASH': 0.5
            }
            # Assign colors and alpha values based on time condition
            plot_df['color'] = plot_df['time'].map(color_map)
            plot_df['alpha'] = plot_df['time'].map(alpha_map)

            # Plotting continuous line with changing colors and alpha values
            for i in range(1, len(plot_df)):
                ax.plot(
                    plot_df['x'].iloc[i-1:i+1], plot_df[column_to_plot].iloc[i-1:i+1],
                    color=plot_df['color'].iloc[i],
                    alpha=plot_df['alpha'].iloc[i],
                )

            if drug not in legend_handles:
                legend_handles[drug] = ax.plot([], [], color=color_dict.get(drug, 'k'), label=f'{drug} (n={cell_counts.get(drug, 0)})')

        ax.legend()
        selection_title = self.build_name(
            *self.selection_label_parts(for_filename=False),
            sep=" | ",
        )
        title = (
            f'{selection_title} {self.dependant_var} Applications'
            if selection_title
            else f'{self.dependant_var} Applications'
        )
        ax.set_title(title)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(y_label)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.show()

    
       

    # def process_data(self, df):
    #     '''
    #     Creates a df to plot with columns cell_id time data then loops rows to bin data +++++++++++ does % baselin if RMP or input R
    #     '''
    #     column_to_bin = f'sweep_{self.dependant_var}'
    #     binned_rows = []
    #     for _, row in df.iterrows():
    #         data = row[column_to_bin]
    #         if not isinstance(data, list) or len(data) == 0:
    #             print(f"{row['cell_id']} for {row['time']} is empty or invalid.")
    #             continue

    #         if row['time'] == 'PRE' and len(data) < 10:
    #             print(f"Skipping {row['cell_id']} due to insufficient PRE values ({len(data)}<10).")
    #             continue
    #         if row['time'] == 'PRE':
    #             data = data[-10:]  

    #         binned = [
    #             np.mean(data[i:i + self.bin_size])
    #             for i in range(0, len(data), self.bin_size)
    #             if len(data[i:i + self.bin_size]) == self.bin_size
    #         ]
    #         binned_rows.append({
    #             'cell_id': row['cell_id'],
    #             'time': row['time'],
    #             'binned_values': binned
    #         })

    #     binned_df = pd.DataFrame(binned_rows)

    #     if self.dependant_var in ['RMP', 'inputR'] : #and self.normalise == True
    #         normalized_rows = []
    #         for cell_id, group in binned_df.groupby('cell_id'):
    #             pre_row = group[group['time'] == 'PRE']
    #             if pre_row.empty:
    #                 print(f"No PRE data for {cell_id}, skipping normalization.")  
    #                 continue
    #             baseline_vals = pre_row.iloc[0]['binned_values']
    #             baseline_mean = np.mean(baseline_vals) 
    #             for _, row in group.iterrows():
    #                 norm_vals = [(val / baseline_mean) * 100 for val in row['binned_values']]  
    #                 normalized_rows.append({
    #                     'cell_id': row['cell_id'],
    #                     'time': row['time'],
    #                     'binned_values': norm_vals
    #                 })
    #         return pd.DataFrame(normalized_rows)
    #     else: 
    #         return binned_df

    # def plot(self):
    #     fig, ax = plt.subplots(figsize=(10, 5))
    #     time_per_bin_min = (self.sweep_in_s * self.bin_size) / 60
    #     padding = 0 * time_per_bin_min  # HARD CODE ADJUST
    #     section_widths = {}
    #     legend_handles = {}
    #     cell_counts = self.cell_df[self.cell_df['cell_id'].isin(self.binned_data['cell_id'])].groupby('treatment')['cell_id'].nunique().to_dict()


    #     for timepoint in self.timepoints:
    #         max_bins = self.binned_data[self.binned_data['time'] == timepoint]['binned_values'].apply(len).max()
    #         section_widths[timepoint] = max_bins * time_per_bin_min if max_bins else 0

    #     for cell_id, cell_data in self.binned_data.groupby('cell_id'):
    #         x_offset = 0
    #         prev_endpoint = None  

    #         for idx, timepoint in enumerate(self.timepoints):
    #             tp_data = cell_data[cell_data['time'] == timepoint]
    #             if tp_data.empty:
    #                 continue

    #             binned_vals = tp_data.iloc[0]['binned_values']
    #             n_bins = len(binned_vals)
    #             x_vals = np.arange(n_bins) * time_per_bin_min + x_offset
    #             y_vals = binned_vals

    #             if timepoint in ['APP', 'WASH']:
    #                 drug = self.cell_df[self.cell_df['cell_id'] == cell_id]['treatment'].iloc[0]
    #                 color = color_dict.get(drug, 'k')
    #                 alpha = 0.5 if timepoint == 'WASH' else 1.0
    #                 label = f"{drug} (n={cell_counts.get(drug, 0)})" if timepoint == 'APP' else None
    #             else:
    #                 color = self.colors[timepoint]
    #                 label = None
    #                 alpha = 1.0

    #             if label is not None and label not in legend_handles:
    #                 legend_handles[label] = ax.plot([], [], color=color, alpha=alpha, label=label)[0]
    #             ax.plot(x_vals, y_vals, color=color, alpha=alpha, linewidth=1)  

    #             # connector between last timepoint and current one
    #             if prev_endpoint is not None:  
    #                 connector_x = [prev_endpoint[0], x_vals[0]]  
    #                 connector_y = [prev_endpoint[1], y_vals[0]]  
    #                 ax.plot(connector_x, connector_y, color=color, linestyle=':', alpha=0.2, linewidth=0.5) 

    #             prev_endpoint = (x_vals[-1], y_vals[-1])  
    #             x_offset += section_widths[timepoint] + padding

    #     cumulative_offset = 0
    #     for timepoint in self.timepoints[:-1]:
    #         cumulative_offset += section_widths[timepoint] + padding
    #         ax.axvline(cumulative_offset, color='lightgrey', linestyle='--', linewidth=0.8)

    #     ax.set_title(f'{self.dependant_var} ')
    #     ax.set_xlabel("Time (min)")
    #     ax.set_ylabel(f"{unit_dict[self.dependant_var]} as % of baseline")
    #     ax.legend(fontsize='small', loc='upper right')
    #     ax.spines[['top', 'right']].set_visible(False)
    #     plt.tight_layout()
    #     plt.show()

       
@dataclass
class Application(Figure):

    '''Plot a single APP file from cell_id or list of.'''
    project: str = field(kw_only = True)
    data_type: str = field(kw_only=True, default='APP_IC')
    cell_id: str|list = field(kw_only = True, default = None) # optional pram for plotting specific cell/s application
    plot_all_APs: bool = field(kw_only=True, default=False)
    plot_AP_segments: bool = field(kw_only=True, default=False)
    valid_only: bool = field(kw_only=True, default=False)
    pre_window_sweeps: int = field(kw_only=True, default=None)
    show_response_annotations: bool = field(kw_only=True, default=True)
    response_df: object = field(kw_only=True, default=None, repr=False)
    response_cache_key: str = field(kw_only=True, default=None)


    def __post_init__(self):
        self.initialize_application_data()
        self._response_annotation_messages = set()
        self.response_annotation_df = self.load_response_annotation_df()
        if self.pre_window_sweeps is None:
            self.pre_window_sweeps = self.default_pre_window_sweeps_from_response()
        if self.cell_id == None:
            self.cell_id = self.default_application_cell_ids()
        self.fig = self.plot_applications()

    def initialize_application_data(self):
        """
        Lightweight setup for raw APP trace plotting.

        Application does not need the full aggregate dataframe or treatment-count
        cache built by DataSelection/Figure, so this keeps single-cell inspection
        fast while still using the same cached extractors and project metadata.
        """
        Cachable.__init__(self, cache_dir=f"{ROOT}/{self.project}/cache")
        self.location = f"{ROOT}/{self.project}"
        self.input_dir = self._checkFileSystem("input")
        self.output_dir = self._checkFileSystem("output")
        self.figure_output_dir = self._checkFileSystem("figures")

        self.project_obj = Project(self.project)
        self.load_extractor(self.data_type)
        self.cell_df = Ephys(self.project).df
        self.valid_files, self.valid_cell_ids = self.application_valid_files_and_cells()

    def application_valid_files_and_cells(self):
        df = getattr(self, f"{self.data_type}_df").copy()

        if self.valid_only and "valid" in df.columns:
            df = df[df["valid"] != False]

        for column_name, attribute in self.selection_filters().items():
            requested_values = self.filter_values(attribute)
            if column_name in df.columns:
                df = df[df[column_name].isin(requested_values)]
            elif column_name in self.cell_df.columns:
                selected_cells = self.cell_df[
                    self.cell_df[column_name].isin(requested_values)
                ]["cell_id"].dropna().unique()
                df = df[df["cell_id"].isin(selected_cells)]
            else:
                raise ValueError(
                    f"Cannot filter Application on '{column_name}'. "
                    f"Available filters are: {self.available_selection_filters(df)}"
                )

        self.application_df = df.copy()
        return (
            df["folder_file"].dropna().tolist() if "folder_file" in df.columns else [],
            df["cell_id"].dropna().unique().tolist() if "cell_id" in df.columns else [],
        )

    def default_application_cell_ids(self):
        """
        Prefer response_df cell_ids when a response annotation table is supplied.
        """
        response_df = getattr(self, "response_annotation_df", pd.DataFrame())
        if (
            response_df is not None
            and not response_df.empty
            and "cell_id" in response_df.columns
        ):
            cell_ids = response_df["cell_id"].dropna().unique().tolist()
            if cell_ids:
                print(
                    "Application cell_id not specified; plotting "
                    f"{len(cell_ids)} cell_id(s) from response_df."
                )
                return cell_ids

        return self.valid_cell_ids

    def default_pre_window_sweeps_from_response(self):
        response_df = getattr(self, "response_annotation_df", pd.DataFrame())
        if response_df is None or response_df.empty:
            return None

        for col in ["requested_pre_sweep_window", "pre_sweep_window_used"]:
            if col not in response_df.columns:
                continue
            windows = pd.to_numeric(response_df[col], errors="coerce").dropna().unique()
            if len(windows) == 1:
                window = int(windows[0])
                print(
                    "Application pre_window_sweeps not specified; using "
                    f"{window} from response_df."
                )
                return window
            if len(windows) > 1:
                print(
                    "Application pre_window_sweeps not specified and response_df "
                    f"contains multiple {col} values: {sorted(windows.tolist())}."
                )
                return None

        return None

    def load_response_annotation_df(self) -> pd.DataFrame:
        if not self.show_response_annotations:
            return pd.DataFrame()

        explicit_df = self.response_annotation_input_to_df(self.response_df)
        if explicit_df is not None:
            self.print_response_annotation_summary(explicit_df, "explicit response_df")
            return explicit_df

        if self.response_cache_key is not None:
            identifier = self.response_cache_key[:-4] if self.response_cache_key.endswith(".pkl") else self.response_cache_key
            try:
                cached = self.getCache(identifier)
            except FileNotFoundError:
                print(f"Application response annotations skipped: cache not found ({identifier}).")
                return pd.DataFrame()
            except Exception as exc:
                print(f"Application response annotations skipped: could not load {identifier}: {exc}")
                return pd.DataFrame()

            response_df = self.response_annotation_input_to_df(cached)
            if response_df is not None and not response_df.empty:
                self.print_response_annotation_summary(response_df, identifier)
                return response_df

        print(
            "Application response annotations skipped: pass response_df=response_result "
            "or response_df=response_result.response_df. Use response_cache_key only for an explicit saved bundle."
        )
        return pd.DataFrame()

    def response_annotation_input_to_df(self, response_input):
        if response_input is None:
            return None

        if isinstance(response_input, pd.DataFrame):
            return response_input.copy()

        if isinstance(response_input, dict):
            for key in ["events", "response_events_df", "response_df"]:
                value = response_input.get(key)
                if isinstance(value, pd.DataFrame):
                    return value.copy()
            return None

        if hasattr(response_input, "response_df"):
            value = getattr(response_input, "response_df")
            if isinstance(value, pd.DataFrame):
                return value.copy()

        return None

    def response_annotation_note_once(self, message: str):
        if message not in self._response_annotation_messages:
            print(f"Application response annotation note: {message}")
            self._response_annotation_messages.add(message)

    def print_response_annotation_summary(self, response_df: pd.DataFrame, source: str):
        if response_df is None or response_df.empty:
            print(f"Application response annotations loaded from {source}, but the table is empty.")
            return

        active_count = (
            response_df["response"].isin(["increase", "decrease", "biphasic"]).sum()
            if "response" in response_df.columns
            else 0
        )
        cell_count = response_df["cell_id"].nunique() if "cell_id" in response_df.columns else "unknown"
        print(
            f"Application response annotations loaded from {source}: "
            f"{active_count} active response rows across {cell_count} cells."
        )

    def list_or_empty(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Series):
            return value.tolist()
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []

    def plot_cached_AP_markers(self, ax, app_row, V_array, dt_s, seconds_per_sweep):
        ap_locs = self.list_or_empty(app_row.get("AP_locs", []))
        ap_sweeps = self.list_or_empty(app_row.get("AP_sweep_locs", []))
        if not ap_locs or not ap_sweeps:
            return

        ra_locs = self.list_or_empty(app_row.get("RA_locs", []))
        ra_sweeps = self.list_or_empty(app_row.get("RA_sweep_locs", []))
        ra_pairs = set(zip(map(int, ra_locs), map(int, ra_sweeps))) if ra_sweeps else set()
        ra_locs_only = set(map(int, ra_locs))

        counts = {"RA": 0, "somatic": 0}
        for loc, sweep in zip(ap_locs, ap_sweeps):
            try:
                loc = int(loc)
                sweep = int(sweep)
            except (TypeError, ValueError):
                continue
            if sweep < 0 or sweep >= V_array.shape[1] or loc < 0 or loc >= V_array.shape[0]:
                continue

            is_ra = (loc, sweep) in ra_pairs if ra_pairs else loc in ra_locs_only
            ap_type = "RA" if is_ra else "somatic"
            counts[ap_type] += 1
            color = "red" if is_ra else "blue"
            ax.scatter(
                sweep * seconds_per_sweep + loc * dt_s,
                V_array[loc, sweep],
                s=18 if is_ra else 12,
                color=color,
                alpha=0.75 if is_ra else 0.35,
                linewidths=0,
                zorder=4,
            )

        for ap_type, count in counts.items():
            if count > 0:
                label = f"{ap_type} APs (n={count})"
                color = "red" if ap_type == "RA" else "blue"
                alpha = 0.75 if ap_type == "RA" else 0.35
                ax.scatter([], [], s=18, color=color, alpha=alpha, label=label)

    def plot_AP_segments_from_trace(self, ax, cell_id, folder_file, V_array, I_array, command_array, sampling_rate_hz, dt_s, seconds_per_sweep):
        color_map = {'RA': 'red', 'somatic': 'blue'}
        alpha_map = {'RA': 0.6, 'somatic': 0.2}

        AP_df = self.folder_file_AP_df(
            cell_id,
            folder_file,
            V_array,
            I_array,
            command_array,
            sampling_rate=sampling_rate_hz,
        )
        if AP_df.empty:
            return

        for ap_type in ['RA', 'somatic']:
            color = color_map[ap_type]
            alpha = alpha_map[ap_type]
            ap_df = AP_df[AP_df['AP_type'] == ap_type]
            n_aps = len(ap_df)
            for upshoot_location, sweep, peak_location in ap_df[['upshoot_location', 'sweep', 'peak_location']].values:
                v_temp = np.array(V_array[:, sweep][upshoot_location:peak_location])
                time_temp = np.arange(len(v_temp)) * dt_s
                time_temp += seconds_per_sweep * sweep + upshoot_location * dt_s
                ax.plot(time_temp, v_temp, color=color, lw=2, alpha=alpha, label=None)
            if n_aps > 0:
                ax.plot([], [], color=color, lw=2, alpha=0.6, label=f'{ap_type} APs (n={n_aps})')

    def response_rows_for_folder_file(self, folder_file: str, cell_id: str) -> pd.DataFrame:
        response_df = getattr(self, "response_annotation_df", pd.DataFrame())
        if response_df is None or response_df.empty:
            return pd.DataFrame()
        if "cell_id" not in response_df.columns:
            self.response_annotation_note_once("response_df has no cell_id column; annotations skipped.")
            return pd.DataFrame()

        rows = response_df[response_df["cell_id"] == cell_id].copy()
        if rows.empty:
            self.response_annotation_note_once(f"{folder_file}: no response rows for cell_id {cell_id}.")
            return pd.DataFrame()
        if "response" not in rows.columns:
            self.response_annotation_note_once("response_df has no response column; annotations skipped.")
            return pd.DataFrame()

        rows = rows[rows["response"].isin(["increase", "decrease", "biphasic"])].copy()
        if rows.empty:
            self.response_annotation_note_once(f"{folder_file}: response_df has no active response for {cell_id}.")
            return pd.DataFrame()

        if "folder_file" not in rows.columns:
            return rows

        folder_values = rows["folder_file"].dropna().astype(str)
        if folder_values.empty:
            return rows

        exact_rows = rows[rows["folder_file"].astype(str) == str(folder_file)].copy()
        if not exact_rows.empty:
            return exact_rows

        plotted_files = (
            self.application_df.loc[self.application_df["cell_id"] == cell_id, "folder_file"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
            if "folder_file" in self.application_df.columns
            else []
        )
        if len(plotted_files) == 1:
            self.response_annotation_note_once(
                f"{folder_file}: no exact folder_file match in response_df; using cell-level response rows."
            )
            return rows

        self.response_annotation_note_once(
            f"{folder_file}: response rows found for {cell_id}, but folder_file did not match."
        )
        return pd.DataFrame()

    def annotate_response_windows(self, ax, folder_file, cell_id, drug_in, seconds_per_sweep, n_sweeps):
        rows = self.response_rows_for_folder_file(folder_file, cell_id)
        if rows.empty:
            return

        if self.pre_window_sweeps is not None and "pre_sweep_window_used" in rows.columns:
            used_windows = sorted(rows["pre_sweep_window_used"].dropna().unique().tolist())
            if used_windows and any(int(window) != int(self.pre_window_sweeps) for window in used_windows):
                message = (
                    f"{folder_file}: cached response pre_sweep_window_used={used_windows}, "
                    f"Application pre_window_sweeps={self.pre_window_sweeps}"
                )
                self.response_annotation_note_once(message)

        response_colors = {
            "increase": {
                "RMP_mV": "crimson",
                "AP_count": "darkorange",
                "RA_count": "orangered",
                "SAP_count": "goldenrod",
                "inputR_MOhm": "firebrick",
                "sEPSP_frequency_Hz": "tomato",
                "sEPSP_mean_amplitude_mV": "mediumvioletred",
            },
            "decrease": {
                "RMP_mV": "royalblue",
                "AP_count": "deepskyblue",
                "RA_count": "dodgerblue",
                "SAP_count": "steelblue",
                "inputR_MOhm": "navy",
                "sEPSP_frequency_Hz": "teal",
                "sEPSP_mean_amplitude_mV": "slateblue",
            },
            "biphasic": {
                "default": "mediumorchid",
            },
        }
        y0, y1 = ax.get_ylim()
        y_span = y1 - y0 if y1 != y0 else 1
        shown_labels = set()
        plotted = 0

        for idx, (_, row) in enumerate(rows.iterrows()):
            start_relative_s = row.get("start_s", np.nan)
            end_relative_s = row.get("end_s", np.nan)
            timing_note = ""

            if pd.isna(start_relative_s) or pd.isna(end_relative_s):
                range_sweeps = row.get("range_sweeps", None)
                if isinstance(range_sweeps, (tuple, list)) and len(range_sweeps) == 2:
                    start_relative_s = float(range_sweeps[0]) * seconds_per_sweep
                    end_relative_s = float(range_sweeps[1]) * seconds_per_sweep
                else:
                    max_relative_s = max(0, (n_sweeps - drug_in) * seconds_per_sweep)
                    if max_relative_s <= 0:
                        self.response_annotation_note_once(
                            f"{folder_file}: response annotation timing missing and no POST time available."
                        )
                        continue
                    start_relative_s = 0
                    end_relative_s = max_relative_s
                    timing_note = "timing unavailable"

            max_relative_s = max(0, (n_sweeps - drug_in) * seconds_per_sweep)
            start_relative_s = max(0, float(start_relative_s))
            end_relative_s = min(max_relative_s, float(end_relative_s))
            if end_relative_s <= start_relative_s:
                end_relative_s = start_relative_s + seconds_per_sweep

            start_s = (drug_in * seconds_per_sweep) + start_relative_s
            end_s = (drug_in * seconds_per_sweep) + end_relative_s
            base_dv = row.get("base_dependant_var", row.get("dependant_var", "response"))
            base_dv = str(base_dv).replace("sweep_", "")
            response = str(row.get("response", "response")).lower()
            color = response_colors.get(response, {}).get(
                base_dv,
                response_colors.get(response, {}).get("default", "mediumorchid"),
            )
            delta = row.get("mean_delta", row.get("delta", np.nan))
            p_val = row.get("p_val", np.nan)
            label = f"{base_dv}: {response}"
            text_parts = [label]
            if pd.notna(delta):
                text_parts.append(f"d={float(delta):.3g}")
            if pd.notna(p_val):
                text_parts.append(f"p={float(p_val):.3g}")
            if pd.notna(row.get("latency", np.nan)):
                text_parts.append(f"t={float(row['latency']):.0f}s")
            if timing_note:
                text_parts.append(timing_note)
            text_label = " | ".join(text_parts)
            legend_label = label if label not in shown_labels else None
            shown_labels.add(label)

            ax.axvspan(
                start_s,
                end_s,
                facecolor=color,
                edgecolor=color,
                alpha=0.24,
                linewidth=2.0,
                label=legend_label,
                zorder=1.5,
            )
            ax.axvline(start_s, color=color, linestyle="--", linewidth=1.3, alpha=0.9, zorder=3)
            ax.axvline(end_s, color=color, linestyle=":", linewidth=1.1, alpha=0.75, zorder=3)
            ax.text(
                start_s,
                y1 - (0.05 + 0.06 * (idx % 4)) * y_span,
                text_label,
                color=color,
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="top",
                bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.8, "pad": 2.0},
                zorder=5,
            )
            plotted += 1

        if plotted > 0:
            self.response_annotation_note_once(
                f"{folder_file}: plotted {plotted} response annotation(s)."
            )

    def app_row_is_invalid(self, app_row: pd.Series) -> bool:
        value = app_row.get("valid", None)
        return isinstance(value, (bool, np.bool_)) and not bool(value)

    def app_invalid_reason(self, app_row: pd.Series) -> str:
        reason = app_row.get("invalid_reason", None)
        if isinstance(reason, str) and reason.strip():
            return reason.strip()

        warnings_value = app_row.get("warnings", None)
        if isinstance(warnings_value, str):
            warnings_list = [warnings_value]
        elif isinstance(warnings_value, (list, tuple, np.ndarray, pd.Series)):
            warnings_list = [str(value) for value in warnings_value]
        else:
            warnings_list = []

        for warning in warnings_list:
            if "APP_IC invalid:" in warning:
                return warning.split("APP_IC invalid:", 1)[1].strip()

        return "valid=False"

    def annotate_app_validity(self, ax, app_row: pd.Series):
        if not self.app_row_is_invalid(app_row):
            return

        folder_file = app_row.get("folder_file", "missing")
        reason = self.app_invalid_reason(app_row)
        wrapped_reason = "\n".join(textwrap.wrap(reason, width=72)) if reason else "valid=False"
        message = f"INVALID APP_IC\n{wrapped_reason}"
        self.response_annotation_note_once(
            f"{folder_file}: APP_IC marked invalid | reason: {reason}"
        )
        ax.text(
            0.01,
            0.98,
            message,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="firebrick",
            fontsize=9,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "firebrick",
                "alpha": 0.88,
                "pad": 3.0,
            },
            zorder=10,
        )


    def plot_applications(self):
        cell_ids = [self.cell_id] if isinstance(self.cell_id, str) else self.cell_id #make list of string if a single string
        for cell_id in cell_ids:
            self.filename = f'{cell_id}_application'

            # Fetch folder_file for the specific cell_id
            cell_sub_df = self.application_df[self.application_df['cell_id'] == cell_id]
            if self.valid_only == True and "valid" in cell_sub_df.columns:
                cell_sub_df = cell_sub_df[cell_sub_df['valid'] != False]

            for _, app_row in cell_sub_df.iterrows():
                folder_file = app_row['folder_file']
                cell_id = app_row['cell_id']
                I_set = app_row.get('I_set', None)
                drug = app_row.get('treatment', None)
                drug_in = app_row.get('drug_in', np.nan)
                drug_out = app_row.get('drug_out', np.nan)
                self.fig_filename = f"{cell_id} {drug} Application"
                
                V_array, I_array, command_array, stim_array, V_list = self.project_obj.load_data(folder_file)
                sampling_rate_hz = self.project_obj.sampling_rate
                dt_s = 1 / sampling_rate_hz

                seconds_per_sweep = len(V_array[:,0]) * dt_s # multiplying this by drug_in/out gives the sweep edge in seconds
                
                x_V = np.arange(V_array.shape[0] * V_array.shape[1]) * dt_s

                #build figure 
                fig = plt.figure(figsize = (12,9))
                ax1 = plt.subplot2grid((11, 8), (0, 0), rowspan = 8, colspan =11) #(nrows, ncols)
                ax2 = plt.subplot2grid((11, 8), (8, 0), rowspan = 2, colspan=11)

                #plot voltage / time
                n_sweeps = V_array.shape[1]  # Number of sweeps based on the second dimension of V_array
                cropped_array = V_array[:, :n_sweeps]  # Crop the array to match the number of sweeps
                continuous_plot = cropped_array.ravel(order='F')  # Flatten the array in column-major (Fortran) order
                ax1.plot(x_V, continuous_plot, c='k' if drug is None else color_dict.get(drug, 'k'), lw=1, alpha=0.8)  # Plot voltage

                # handle action potentials
                if self.plot_all_APs:
                    if self.plot_AP_segments:
                        self.plot_AP_segments_from_trace(
                            ax1,
                            cell_id,
                            folder_file,
                            V_array,
                            I_array,
                            command_array,
                            sampling_rate_hz,
                            dt_s,
                            seconds_per_sweep,
                        )
                    else:
                        self.plot_cached_AP_markers(
                            ax1,
                            app_row,
                            V_array,
                            dt_s,
                            seconds_per_sweep,
                        )

                if command_array is not None:
                    _, display_array, _ = select_protocol_array(
                        V_array,
                        command_array=command_array,
                        I_array=None,
                        clean_I_fallback=False,
                    )
                    display_ylabel = "Command (pA)"
                else:
                    display_array = I_array
                    display_ylabel = "Current (pA)"

                if display_array is not None:
                    display_plot = display_array.ravel(order='F')
                    x_I = np.arange(len(display_plot)) * dt_s
                    ax2.plot(x_I, display_plot, label = I_set, color=color_dict['I_display'] )
                    ax2.legend(loc='upper right')

                # SPINES
                ax1.spines['top'].set_visible(False) # 'top', 'right', 'bottom', 'left'
                ax1.spines['right'].set_visible(False)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)

                # DRUG APPLICATION BAR
                drug_in = 0 if pd.isna(drug_in) else int(drug_in)
                drug_out = V_array.shape[1] if pd.isna(drug_out) else int(drug_out)
                drug_start = max((drug_in * seconds_per_sweep) - seconds_per_sweep, 0)
                drug_end = drug_out * seconds_per_sweep
                ax1.axvspan(drug_start, drug_end, facecolor = "grey", alpha = 0.3) #drug bar shows start of drug_in sweep to end of drug_out sweep
                if self.pre_window_sweeps is not None:
                    pre_start = (drug_in - self.pre_window_sweeps) * seconds_per_sweep
                    pre_end = drug_in * seconds_per_sweep
                    ax1.axvspan(
                        pre_start,
                        pre_end,
                        facecolor="lightgrey",
                        alpha=0.4
                    )
                self.annotate_response_windows(
                    ax1,
                    folder_file,
                    cell_id,
                    drug_in,
                    seconds_per_sweep,
                    n_sweeps,
                )
                self.annotate_app_validity(ax1, app_row)
                handles, labels = ax1.get_legend_handles_labels()
                if handles:
                    ax1.legend(loc='upper right')
                
                #LABELS / TITLES
                ax1.set_xlabel( "Time (s)", fontsize = 12) #, fontsize = 15
                ax1.set_ylabel( "Membrane Potential (mV)", fontsize = 12) #, fontsize = 15
                ax2.set_xlabel( "Time (s)", fontsize = 10) #, fontsize = 15
                ax2.set_ylabel(display_ylabel if display_array is not None else "Protocol", fontsize = 10) #, fontsize = 15
                ax1.set_title(cell_id + ' '+ drug +' '+ " Application", fontsize = 16) # , fontsize = 25
                plt.tight_layout()
                plt.show()
                self.save_plot(fig, f"{cell_id}_APP")
                


