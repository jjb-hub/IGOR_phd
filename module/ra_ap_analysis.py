from module.figure_common import *
from module.figure_base import Figure

@dataclass
class RA_AP_analysis(Figure):
    '''
    AP analsysis for a single cell_id: 
    either by folder_file or pooled, data_type indicate the daa types to be analised for that cell_id (invalidated files removed)
    '''
    project: str = field(kw_only = True)
    cell_id: str = field(kw_only = True) # single cell ID ONLY 
    data_type: list = field(kw_only=True, default='APP_IC') # defaults to this all will be returned in the aggg_df
    data_types: list = field(kw_only=True, default='APP_IC') 
    pooled: bool = field(kw_only=True, default=False)

    color_map: dict = field(default_factory=lambda: {'RA': 'red', 'somatic': 'blue'}, init=False)
    forwards_window: int = field(default=50, init=False)
    backwards_window: int = field(default=70, init=False)
    voltage_max: float = field(default=60.0, init=False)
    voltage_min: float = field(default=-120.0, init=False)

    def __post_init__(self):

        super().__post_init__()
        self.agg_AP_df = self.build_aggregate_AP_df()
        self.plot_AP_traces() # runs all plotters contains pool logic

    def build_aggregate_AP_df (self):
        agg_AP_dfs = []
        #fetch valid folder_files for each data_type   #NO pAD hunter and no second application files!! #TODO
        agg_folder_files = []
        if 'APP_IC' in self.data_types:
            valid_folder_files = self.cell_df[self.cell_df['cell_id'] == self.cell_id]['APP_IC_folder_files'].iloc[0]
            if valid_folder_files is None: 
                 print(f"No valid APP files for cell_id {self.cell_id}, skipping.")
            else:
                agg_folder_files.append(valid_folder_files)

        if 'IF_IC' in self.data_types:
            valid_folder_files = self.cell_df[self.cell_df['cell_id'] == self.cell_id]['IF_IC_folder_files'].iloc[0]

            if valid_folder_files is None or pd.isna(valid_folder_files):
                 print(f"No valid FP files for cell_id {self.cell_id}, skipping.")
            else:
                agg_folder_files.extend(valid_folder_files)
                
        for folder_file in agg_folder_files: #TOD add drug presence to AP_df
            #fetch raw trace
            V_array, I_array, command_array, stim_array, V_list = self.project_obj.load_data(folder_file)
            sampling_rate_hz = self.project_obj.sampling_rate
            #build AP_df for folder_file
            AP_df = self.folder_file_AP_df(self.cell_id, folder_file, V_array, I_array, command_array, sampling_rate=sampling_rate_hz)
            if AP_df.empty:
                print(f'No APs detected in {folder_file}, skipping')
                continue
            else:
                agg_AP_dfs.append(AP_df) 

        agg_AP_df = pd.concat(agg_AP_dfs)
        return agg_AP_df

    def plot_AP_traces(self):
        '''handeling pooled logic'''
        if self.agg_AP_df is None:
            print(f"No APs have been detected for {self.cell_id}")
            return
        
        if self.pooled == True:
            self.filename = f'{self.cell_id}_mean_APs'
            self.plot_meanAPs(self.agg_AP_df)
            self.filename = f'{self.cell_id}_phaseplot_APs'
            self.plot_phaseplotAPs(self.agg_AP_df)
            self.filename = f'{self.cell_id}_hisogram_APs'
            self.plot_histogramAPs(self.agg_AP_df)
        else:
            for folder_file, subset_df in self.agg_AP_df.groupby('folder_file'):
                self.filename = f"{self.cell_id}_mean_APs_{folder_file.replace('/','_')}"
                self.plot_meanAPs(subset_df)
                self.filename = f"{self.cell_id}_phaseplot_APs_{folder_file.replace('/','_')}"
                self.plot_phaseplotAPs(subset_df)
                self.filename = f"{self.cell_id}_hisogram_APs_{folder_file.replace('/','_')}"
                self.plot_histogramAPs(subset_df)
        

    def plot_meanAPs(self, df):
        '''
        Takes AP_df
        Makes a figure and plots traces and means for raw trace and phase plot
        '''
        fig, ax = plt.subplots(figsize=(10, 6))
        traces_by_type = defaultdict(list)  

        #collect traces from folder_files
        for folder_file, AP_df_folder_file in df.groupby('folder_file'):
            V_array, I_array, command_array, stim_array, V_list = self.project_obj.load_data(folder_file)
            sampling_rate_hz = self.project_obj.sampling_rate
            for ap_type in AP_df_folder_file['AP_type'].unique():
                ap_indices = AP_df_folder_file[AP_df_folder_file['AP_type'] == ap_type][["upshoot_location", "sweep"]].values
                traces = []

                for upshoot_location, sweep_idx in ap_indices:
                    lower_bound = max(0, upshoot_location - self.backwards_window)
                    upper_bound = upshoot_location + self.forwards_window
                    if upper_bound <= V_array.shape[0]:
                        trace = V_array[lower_bound:upper_bound, sweep_idx]
                    else:
                        trace = V_array[lower_bound:, sweep_idx]
                    traces.append((trace, sampling_rate_hz))
                max_len = max([len(t[0]) for t in traces], default=0)
                valid_traces = [t for t in traces if len(t[0]) == max_len]
                if len(valid_traces) < len(traces): 
                    print(f"{len(traces) - len(valid_traces)} {ap_type} APs dropped due to short trace length.")
                traces_by_type[ap_type].extend(valid_traces) 
        #plot 
        for ap_type, traces in traces_by_type.items():
            if not traces:
                continue
            color = self.color_map[ap_type]
            for trace, sampling_rate_hz in traces:
                time_ms = (np.arange(0, len(trace)) * 1000) / sampling_rate_hz
                ax.plot(time_ms, trace, color=color, alpha=0.1, linewidth=0.9) # raw trace

            mean_trace = np.mean([trace for trace, _ in traces], axis=0)
            mean_sampling_rate_hz = float(np.nanmedian([rate for _, rate in traces]))
            time_ms = (np.arange(0, len(mean_trace)) * 1000) / mean_sampling_rate_hz
            ax.plot(time_ms, mean_trace, color=color, linewidth=1.3, label=f'{ap_type} mean (n={len(traces)})') # mean trace

        ax.set_ylabel('Membrane Potential (mV)')
        ax.set_xlabel('Time (ms)')
        ax.legend()
        ax.set_title(self.filename , fontsize=16)
        plt.tight_layout()
        plt.show()
        self.save_plot(fig, self.filename )
        return fig

                        
    def plot_phaseplotAPs(self, df):
        fig, ax = plt.subplots(figsize=(8, 6))
        ap_type_data = {}

        #collect traces
        for folder_file, AP_df_folder_file in df.groupby('folder_file'):
            # Load V_array and I_array for each folder_file
            V_array, I_array, command_array, stim_array, V_list = self.project_obj.load_data(folder_file)
            sampling_rate_hz = self.project_obj.sampling_rate

            for ap_type in AP_df_folder_file['AP_type'].unique():
                color = self.color_map.get(ap_type, 'gray')
                subset = AP_df_folder_file[AP_df_folder_file['AP_type'] == ap_type]
                traces = []

                for upshoot_location, sweep in subset[["upshoot_location", "sweep"]].values:
                    v_temp = V_array[upshoot_location: upshoot_location + self.forwards_window, sweep]
                    if len(v_temp) < 2:
                        continue
                    dv_temp = np.diff(v_temp) * sampling_rate_hz / 1000
                    if max(v_temp) <= self.voltage_max and min(v_temp) >= self.voltage_min:
                        ax.plot(v_temp[:-1], dv_temp, color=color, alpha=0.05)
                        traces.append((v_temp[:-1], dv_temp))

                if ap_type not in ap_type_data:
                    ap_type_data[ap_type] = {"color": color, "traces": []}
                ap_type_data[ap_type]["traces"].extend(traces)

        # Plot mean traces and add to legend
        legend_elements = []
        for ap_type, info in ap_type_data.items():
            color = info["color"]
            traces = info["traces"]

            # Interpolate to same x-axis length for mean
            min_len = min(len(x[0]) for x in traces)
            v_mat = np.array([t[0][:min_len] for t in traces])
            dv_mat = np.array([t[1][:min_len] for t in traces])
            v_mean = np.mean(v_mat, axis=0)
            dv_mean = np.mean(dv_mat, axis=0)

            ax.plot(v_mean, dv_mean, color=color, lw=1.3, label=f'{ap_type} mean (n={len(traces)})')
            legend_elements.append(Line2D([0], [0], color=color, lw=2, label=f'{ap_type} mean (n={len(traces)})'))

        ax.set_title(self.filename, fontsize=16)
        ax.set_xlabel("Membrane Potential (mV)")
        ax.set_ylabel("dV/dt (mV/ms)")
        ax.legend(handles=legend_elements)
        plt.tight_layout()
        plt.show()
        self.save_plot(fig, self.filename)
        return fig

    def plot_histogramAPs(self, df):
            fig, axs = plt.subplots(4, 2, figsize=(10, 10))
            legend_elements = []
            plot_labels = df["AP_type"].unique() 

            column_map = {
                'voltage_threshold': ('Voltage Thresholds', 'mV'),
                'peak_rise': ('AP Rise', 'V/s'),
                'peak_decay': ('AP Decay', 'V/s'),
                'peak_max_rise': ('AP Rise max', 'V/s'),
                'height': ('AP Heights', 'mV'),
                'peak_voltage': ('AP peak', 'mV'),
                'width': ('AP FWHM', 'ms'),
                'latency': ('Peak Latency', 'ms')
            }

            plot_columns = list(column_map.keys())
            for i, col in enumerate(plot_columns):
                row, col_pos = divmod(i, 2)
                # Combine both RA and somatic data for binning
                data_all = df[col].dropna()
                bins = np.histogram_bin_edges(data_all, bins=20)

                for label in plot_labels:
                    color = self.color_map[label]
                    data = df[df["AP_type"] == label][col].dropna()
                    n_aps = len(data)
                    axs[row, col_pos].hist(data, bins=bins, color=color,
                                        label=f'{label} (n={n_aps})', alpha=0.6) #histtype='step',

            # Labeling and titles
            for i, col in enumerate(plot_columns):
                row, col_pos = divmod(i, 2)
                title, xlabel = column_map[col]
                axs[row, col_pos].set_title(title)
                axs[row, col_pos].set_xlabel(xlabel)
                axs[row, col_pos].set_ylabel('AP count')
                axs[row, col_pos].legend()

            fig.tight_layout(h_pad=2.5, rect=[0, 0, 1, 0.95])
            plt.suptitle(self.filename, fontsize=14)
            plt.show()
            self.save_plot(fig, self.filename)
