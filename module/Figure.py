import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import ClassVar
from itertools import cycle
import seaborn as sns
from typing import Optional
from module.utils import subselectDf, saveFigure, getCache, isCached, cache, cache_excel, load_file #should become Cashable class
from module.constants import CACHE_DIR, color_dict
from module.Ephys import ephys
from module.action_potential_functions import ap_characteristics_extractor_main, normalise_array_length #should become ActionPotential class

@dataclass
class DataSelection:
    ''' 
    Attributes:
        - filename (str): defining the project and feature mapping ie RAW_df in Ephys
        - data_type (str): The data type (e.g., 'APP' or 'FP').
        - cell_type (str | list): The type of cell to filter on (optional) / can inout list 
        - treatment (str | list): The treatment to filter on, i.e. drug applied (optional).
        - cell_subtype (str | list): The subtype of cell to filter on (optional).
        - I_set (str | list): The I_set to filter on (optional).
        - threshold_access_change (float): The threshold for access change filtering (optional, default 30).
        '''
    
    filename: str  = field(kw_only=True)
    data_type: str = field(kw_only=True)
    cell_type: str | list = field(kw_only=True, default=None)
    cell_subtype: str | list  = field(kw_only=True, default=None)
    treatment: str | list  = field(kw_only=True, default=None)
    I_set: str | list  = field(kw_only=True, default=None)
    threshold_access_change: float = field(kw_only = True, default=30)


    def __post_init__(self):
        self.FP_df = getCache(self.filename, 'FP_df') if isCached(self.filename, 'FP_df') else ephys.generate_FP_df
        self.APP_df = getCache(self.filename, 'APP_df') if isCached(self.filename, 'APP_df') else ephys.generate_APP_df
        self.cell_df = getCache(self.filename, 'cell_df') if isCached(self.filename, 'cell_df') else ephys.generate_cell_df
        self.validate_inputs()
        self.valid_files, self.valid_cell_ids = self.get_valid_folder_files()
        self.agg_df = self.get_filtered_data()
        self.treatment_count_df = self.generate_treatment_count_df()

    def validate_inputs(self):
        if self.data_type not in ['FP', 'APP']:
            raise ValueError(f"Invalid data_type: {self.data_type}. Must be one of ['FP', 'APP'].")
        valid_df = self.cell_df[self.cell_df[f'{self.data_type}_valid'].notna()]
        def validate_attribute(attribute, column_name):
            if attribute is not None:
                valid_values = valid_df[column_name].unique()
                if isinstance(attribute, list):
                    invalid_values = [val for val in attribute if val not in valid_values]
                    if invalid_values:
                        raise ValueError(f"Invalid {column_name}(s): {invalid_values}. Must be within {list(valid_values)}.")
                elif attribute not in valid_values:
                    raise ValueError(f"Invalid {column_name}: {attribute}. Must be within {list(valid_values)}.")
        validate_attribute(self.cell_type, 'cell_type')
        validate_attribute(self.treatment, 'treatment')
        validate_attribute(self.cell_subtype, 'cell_subtype')
   
    def get_valid_folder_files(self):
        """
        Filters the cell_df based on the input parameters including threshold_access_change.
        Returns a list of valid folder_files and cell_ids.
        """
        valid_column = f'{self.data_type}_valid'
        if valid_column not in self.cell_df.columns:
            raise ValueError(f"{valid_column} column does not exist in cell_df.")
        
        filtered_cell_df = self.cell_df.copy()
        if self.cell_type is not None:
            filtered_cell_df = filtered_cell_df[filtered_cell_df['cell_type'].isin([self.cell_type] if isinstance(self.cell_type, str) else self.cell_type)]
        if self.treatment is not None:
            filtered_cell_df = filtered_cell_df[filtered_cell_df['treatment'].isin([self.treatment] if isinstance(self.treatment, str) else self.treatment)]
        if self.cell_subtype is not None:
            filtered_cell_df = filtered_cell_df[filtered_cell_df['cell_subtype'].isin([self.cell_subtype] if isinstance(self.cell_subtype, str) else self.cell_subtype)]
        if self.I_set is not None:
            filtered_cell_df = filtered_cell_df[filtered_cell_df['I_set'].isin([self.I_set] if isinstance(self.I_set, str) else self.I_set)]

        filtered_cell_df = filtered_cell_df[filtered_cell_df['access_change'].abs() <= self.threshold_access_change]
        valid_cell_ids = filtered_cell_df['cell_id'].tolist()
        valid_files = filtered_cell_df[valid_column].dropna().tolist()
        valid_files = [item for sublist in valid_files for item in sublist] if isinstance(valid_files[0], list) else valid_files

        return valid_files, valid_cell_ids

    def get_filtered_data(self):
        """
        Fetches the data_type _df and filters it and returns restructured aggergate df for stats and plotting (one row for each cell_id and time).
        """
        valid_files, valid_cell_ids = self.get_valid_folder_files()
        
        if self.data_type == 'APP':
            filtered_df = self.APP_df[self.APP_df['folder_file'].isin(valid_files)]
            #column names {dependant_vairable}_{time}
            timepoints = ['PRE', 'APP', 'WASH']
            columns = {
            'RMP': 'RMP',
            'inputR': 'inputR',
            'pADcount': 'pAD_count',
            'APcount': 'AP_count' }
            reshaped_data = []

            for timepoint in timepoints:
                time_specific_columns = {f'{var}_{timepoint}': name for var, name in columns.items() if f'{var}_{timepoint}' in filtered_df.columns}
                
                existing_columns = [col for col in time_specific_columns.keys() if col in filtered_df.columns]
                current_data = filtered_df[['cell_id'] + existing_columns].copy()
                current_data['time'] = timepoint
                current_data.rename(columns=time_specific_columns, inplace=True)
                # aggregate
                for col in ['RMP', 'inputR']:
                    if col in current_data.columns:
                        current_data[col] = current_data[col].apply(lambda x: np.nanmean(x) if isinstance(x, list) and len(x) > 0 else (np.nan if isinstance(x, list) else x))

                reshaped_data.append(current_data)
            agg_APP_df = pd.concat(reshaped_data, ignore_index=True)
            agg_APP_df = self.add_cell_mapping(agg_APP_df)

            return agg_APP_df

        elif self.data_type == 'FP':
            filtered_df = self.FP_df[self.FP_df['folder_file'].isin(valid_files)].copy()
            filtered_df['time'] = filtered_df['drug'].apply(lambda x: 'WASH' if x != 'PRE' else 'PRE')
            #columns to keep
            filtered_df = filtered_df[['cell_id', 'time', 
                        'AP_dvdt_max', 'AP_height', 'AP_latency', 'AP_peak_voltages',
                        'AP_slope', 'AP_width', 'FI_slope', 'max_firing', 
                        'rheobased_threshold', 'sag', 'tau_rc', 'voltage_threshold']]
            #aggregate 
            for col in ['AP_dvdt_max', 'AP_height', 'AP_latency', 'AP_peak_voltages',
                        'AP_slope', 'AP_width', 'sag', 'tau_rc', 'voltage_threshold']:
                filtered_df[col] = filtered_df[col].apply(lambda x: np.mean(x) if isinstance(x, list) else x)
            agg_FP_df = filtered_df.groupby(['cell_id', 'time']).agg({
            'AP_dvdt_max': 'mean',
            'AP_height': 'mean',
            'AP_latency': 'mean',
            'AP_peak_voltages': 'mean',
            'AP_slope': 'mean',
            'AP_width': 'mean',
            'FI_slope': 'mean',
            'max_firing': 'mean',
            'rheobased_threshold': 'mean',
            'sag': 'mean',
            'tau_rc': 'mean',
            'voltage_threshold': 'mean'
            }).reset_index()
            agg_FP_df = self.add_cell_mapping(agg_FP_df)
            return agg_FP_df
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

    def add_cell_mapping(self, df):
        '''
        Adds cell_ feature columns based off cell_id in cell_df.
        '''
        if 'cell_id' not in df.columns or 'cell_id' not in self.cell_df.columns:
            raise ValueError("Both DataFrames must have 'cell_id' column.")
        
        columns_to_map = ['cell_id', 'treatment', 'cell_type', 'cell_subtype', 'I_set']
        for column in columns_to_map:
            if column not in self.cell_df.columns:
                raise ValueError(f"Column '{column}' is missing from cell_df.")
    
        return df.merge(self.cell_df[columns_to_map].drop_duplicates(), on='cell_id', how='left')
    

    def generate_treatment_count_df(self) -> pd.DataFrame:
        '''
        Calculates the n for each treatment x cell_type given the threshold_access_change, saved as excel in cache.
        '''

        def is_valid_app(app_valid):
            return isinstance(app_valid, str) and len(app_valid) > 0
        def is_valid_fp(fp_valid):
            return isinstance(fp_valid, list) and all(isinstance(x, str) for x in fp_valid)
        def process_group(group_df):
            valid_fp = group_df[group_df['FP_valid'].apply(is_valid_fp)]
            valid_app = group_df[group_df['APP_valid'].apply(is_valid_app)]

            fp_count = valid_fp['cell_id'].nunique()
            app_count = valid_app['cell_id'].nunique()
            both_valid_count = group_df[
                group_df['FP_valid'].apply(is_valid_fp) & group_df['APP_valid'].apply(is_valid_app)
            ]['cell_id'].nunique()

            cell_id_fp = valid_fp['cell_id'].unique().tolist()
            cell_id_app = valid_app['cell_id'].unique().tolist()

            return pd.Series({
                'FP_valid_count': fp_count,
                'APP_valid_count': app_count,
                'both_valid_count': both_valid_count,
                'cell_id_FP': cell_id_fp,
                'cell_id_APP': cell_id_app
            })
        
        access_filtered_df = self.cell_df[self.cell_df['access_change'].abs() <= self.threshold_access_change]
        treatment_count_df = access_filtered_df.groupby(['treatment', 'cell_type']).apply(process_group).reset_index()
        cache(self.filename, f'treatment_count_df_{self.threshold_access_change}', treatment_count_df)
        cache_excel(self.filename, f'treatment_count_df_{self.threshold_access_change}', treatment_count_df)
        return treatment_count_df

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
        DataSelection.__post_init__(self)
        

    def save_figure(self, fig): #ADD load
        """
        Save the generated figure as SVG and PNG.
        """
        saveFigure(fig, self.fig_filename, 'figure')



@dataclass
class Histogram(Figure):
    dependant_var: str = field(kw_only=True)
    specify: str = field(kw_only = True, default = 'treatment') # specify marker to see subsets e.g. I_set or cell_id

    def __post_init__(self):
        Figure.__post_init__(self)
        self.check_valid_dependant_var()
        self.fig_filename = self.generate_fig_filename()
        self.order = [t for t in color_dict.keys() if t in self.agg_df['treatment'].unique()]
        self.hue_order = [t for t in ['PRE', 'APP', 'WASH'] if t in self.agg_df['time'].unique()]
        self.fig = self.plot_histogram()


    def check_valid_dependant_var(self):
            if self.dependant_var not in self.agg_df.columns:
                dvs = [col for col in self.agg_df.columns if col not in ['cell_id', 'time', 'treatment', 'cell_type', 'cell_subtype', 'I_set']]
                raise ValueError(f"Invalid dependant variable: {self.dependant_var}. Valid dv's : {dvs}")
    
    def generate_fig_filename(self) -> str:
        filename_parts = [f"{self.dependant_var}_{self.cell_type}"]
        if self.treatment:
            filename_parts.append(f"_{self.treatment}")
        if self.cell_subtype:
            filename_parts.append(f"_{self.cell_subtype}")
        if self.I_set:
            filename_parts.append(f"_{self.I_set}")
        return "_".join(filename_parts)
    
    def specify_markers(self, ax):
        """
        Adds a marker legend to the plot based on the `specify` attribute.
        """
        unique_values = self.agg_df[self.specify].unique()
        markers = cycle(['o', 's', '^', 'D', 'v', '<', '>'])  # Define markers to use
        legend_handles_labels = {}

        for i, value in enumerate(unique_values):
            marker = next(markers)
            subset_to_plot = self.agg_df[self.agg_df[self.specify] == value]
            
            sns.stripplot(
                x='treatment',
                y=self.dependant_var,
                hue='time',
                hue_order=['PRE', 'APP', 'WASH'],
                order=self.order ,
                data=subset_to_plot,
                palette='Set2',
                edgecolor="k",
                linewidth=1,
                linestyle="-",
                dodge=True,
                ax=ax, 
                legend=False,
                marker=marker,
            )

            # Add value to legend dictionary
            legend_handles_labels[value] = plt.Line2D(
                [0], [0], marker=marker, label=value, color='black'
            )

        return legend_handles_labels


    def plot_histogram(self):
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.barplot(
            x='treatment',
            y=self.dependant_var,
            hue='time',
            hue_order=self.hue_order,
            order=self.order ,
            data=self.agg_df,
            errorbar=('ci', 68),
            palette='Set2',
            edgecolor="k",
            ax=ax
        )
        sns.swarmplot(
            x='treatment',
            y=self.dependant_var,
            hue='time',
            hue_order=self.hue_order,
            order=self.order ,
            data=self.agg_df,
            palette='Set2',
            edgecolor="k",
            linewidth=0.5,
            ax=ax, 
            legend=False,
            marker="o",
            size=0.05, #small as will be plotted over by specify_markers
            alpha = 0.7,
            dodge=True,
        )
 
        legend_handles_labels = self.specify_markers(ax)
    
        current_handles, current_labels = ax.get_legend_handles_labels()
        combined_handles = current_handles + list(legend_handles_labels.values())
        combined_labels = current_labels + [handle.get_label() for handle in legend_handles_labels.values()]
        ax.legend(handles=combined_handles, labels=combined_labels, loc='best', title='Legend')


        counts = self.agg_df.groupby('treatment')['cell_id'].nunique()
        for tick, treatment in enumerate(self.order):
            count = counts.get(treatment, 0)
            ax.text(tick, -0.1, f'n={count}', ha='center', va='top', fontsize=12, color='black', transform=ax.get_xaxis_transform())

        # Customize plot labels and titles
        ax.set_ylabel(self.dependant_var, fontsize=14)
        ax.set_title(f'{self.cell_type} - {self.dependant_var}', fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        self.save_figure(fig)


@dataclass
class Application(Figure):
    cell_id: str|list = field(kw_only = True, default = None) # optional pram for plotting specific cell/s application

    def __post_init__(self):
        Figure.__post_init__(self)
        if self.cell_id == None:
            self.cell_id = self.valid_cell_ids
        self.fig = self.plot_applications()

    
    def build_AP_DF(self, folder_file, V_array, I_array) -> pd.DataFrame: #maybe belongs elsewhere 
        '''
        Builds df for single fplder_file  each row an action potential (AP) with columns for AP characteristics.
        Attributes:
            folder_file (str)  : name of unique file identifier
            V_array (np.ndarray) : 2D voltage array for folder_file
            I_array (np.ndarray) : 2D current array for folder_file

        '''
        V_array_adj, I_array_adj = normalise_array_length(V_array, I_array, columns_match=True)
        
        # Extract AP characteristics
        (peak_voltages_all, peak_latencies_all, v_thresholds_all, peak_slope_all,
         peak_dvdt_max_all, peak_locs_corr_all, upshoot_locs_all, peak_heights_all,
         peak_fw_all, peak_indices_all, sweep_indices_all) = ap_characteristics_extractor_main(folder_file, V_array, all_sweeps=True)
        
        # Early return if no APs found
        if np.all(np.isnan(peak_latencies_all)):
            print(f"No APs detected in voltage trace {folder_file}.")
            return pd.DataFrame(columns=['folder_file', 'peak_location', 'upshoot_location', 'voltage_threshold',
                                         'slope', 'latency', 'peak_voltage', 'height', 'width', 'sweep',
                                         'I_injected', 'AP_type'])

        # Create DataFrame of APs
        AP_df = pd.DataFrame({
            'folder_file': folder_file,
            'peak_location': peak_locs_corr_all,
            'upshoot_location': upshoot_locs_all,
            'voltage_threshold': v_thresholds_all,
            'slope': peak_slope_all,
            'latency': peak_latencies_all,
            'peak_voltage': peak_voltages_all,
            'height': peak_heights_all,
            'width': peak_fw_all,
            'sweep': sweep_indices_all,
            'I_injected': [I_array[loc, 0] for loc in peak_locs_corr_all],  # Sweep index is 0 as I_array is identical
            'AP_type': np.NaN  # Default
        })

        AP_df.loc[(AP_df['voltage_threshold'] < -65) & (AP_df['peak_voltage'] > 20), 'AP_type'] = 'RA'
        return AP_df
    
    def plot_applications(self):
        for cell_id in self.cell_id:

            # Fetch folder_file for the specific cell_id
            cell_sub_df = self.APP_df[self.APP_df['cell_id'] == cell_id]
            for folder_file, cell_id, I_set, drug, drug_in, drug_out, application_order, pAD_locs in cell_sub_df[['folder_file','cell_id', 'I_set', 'drug', 'drug_in', 'drug_out', 'application_order', 'pAD_locs']].values:
                self.fig_filename = f"{cell_id}_application{application_order}"
                V_array , I_array, V_list = load_file(folder_file)
                if I_array is None:
                    I_array = np.zeros((len(V_array), 1))

                # sampeling at 20KHz -->  time (s)
                seconds_per_sweep = len(V_array[:,0]) * 0.00005 # multiplying this  by drug_in/out will give you the point at the end of the sweep in seconds
                x_V = np.arange(len(V_list)) * 0.00005 
                # x_V = np.arange(V_array.shape[0] * V_array.shape[1]) * 0.00005 
                x_I = np.arange(len(I_array)) * 0.00005 

                #build figure 
                fig = plt.figure(figsize = (12,9))
                ax1 = plt.subplot2grid((11, 8), (0, 0), rowspan = 8, colspan =11) #(nrows, ncols)
                ax2 = plt.subplot2grid((11, 8), (8, 0), rowspan = 2, colspan=11)

                #plot voltage / time
                ax1.plot(x_V, V_list, c = 'k' if drug is None else color_dict.get(drug, 'k'), lw=1, alpha=0.8) 

                AP_df = self.build_AP_DF(folder_file, V_array, I_array)
                #label RA APs
                if pAD_locs and not AP_df.empty: 
                    AP_df = AP_df.dropna(subset=['AP_type'])
                    if not AP_df.empty:
                        color_cycle = cycle(['salmon', 'cornflowerblue', 'seagreen'])
                        color_map = {}
                        
                        for ap_type in AP_df['AP_type'].unique(): # loop AP_type 
                            if ap_type not in color_map:
                                color_map[ap_type] = next(color_cycle) # assign color 

                            ap_sub_df = AP_df[AP_df.AP_type == ap_type]
                            for upshoot_location, sweep, peak_location in ap_sub_df[['upshoot_location', 'sweep', 'peak_location']].values: # loop APs for AP_type
                                v_temp = np.array(V_array[ : , sweep] [upshoot_location : peak_location ]) #voltage trace of AP in sweep
                                time_temp = np.linspace(0, len(v_temp) * 0.00005, len(v_temp))  # time values in sweep
                                time_temp += seconds_per_sweep * sweep + upshoot_location * 0.00005 # time values in x_V
                                ax1.plot(time_temp, v_temp, color=color_map[ap_type], lw=2, alpha=0.5, label=ap_type)

                        # LEGENDS
                        handles, labels = ax1.get_legend_handles_labels()
                        unique_labels = list(color_map.keys())  # Unique AP_type labels based on color_map
                        ax1.legend(handles, unique_labels, loc='upper right')
                ax2.plot(x_I, I_array, label = I_set, color=color_dict['I_display'] )
                ax2.legend()

                # SPINES
                ax1.spines['top'].set_visible(False) # 'top', 'right', 'bottom', 'left'
                ax1.spines['right'].set_visible(False)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)

                # DRUG APPLICATION BAR
                ax1.axvspan((int((drug_in)* seconds_per_sweep) - seconds_per_sweep), (int(drug_out)* seconds_per_sweep), facecolor = "grey", alpha = 0.2) #drug bar shows start of drug_in sweep to end of drug_out sweep 
                
                #LABELS / TITLES
                ax1.set_xlabel( "Time (s)", fontsize = 12) #, fontsize = 15
                ax1.set_ylabel( "Membrane Potential (mV)", fontsize = 12) #, fontsize = 15
                ax2.set_xlabel( "Time (s)", fontsize = 10) #, fontsize = 15
                ax2.set_ylabel( "Current (pA)", fontsize = 10) #, fontsize = 15
                ax1.set_title(cell_id + ' '+ drug +' '+ " Application" + " (" + str(application_order) + ")", fontsize = 16) # , fontsize = 25
                plt.tight_layout()
                plt.show()
                self.save_figure(fig)
                