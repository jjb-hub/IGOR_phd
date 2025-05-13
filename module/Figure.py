import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import ClassVar
# from Cachable import Cachable
from itertools import cycle
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import os
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
from typing import Optional
# from module.utils import  subselectDf, saveFigure, getCache, isCached, cache, cache_excel #should become Cashable class
from module.constants import CACHE_DIR, color_dict, unit_dict
# from module.Ephys import Ephys, APP, FP, EphysData # I THINK THIS IS OLD?
from module.Ephys_Project import Ephys, APP, FP, Project
from module.Cachable import Cachable
from collections import defaultdict
#Readapting
from module.action_potential_functions import ap_characteristics_extractor_main, normalise_array_length #should become ActionPotential class
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D


# Root directory for projects #HACKY SHIT should have a project or filesystem class to prevent dupicate code
ROOT = f"{os.getcwd()}/PROJECTS"
if not os.path.exists(ROOT):
    os.mkdir(ROOT)

@dataclass
class DataSelection (Cachable): 
    ''' 
    Attributes:
        - project (str): defining the project and feature mapping ie RAW_df in Ephys
        - data_type (str): The data type (e.g., 'APP' or 'FP').
        - cell_type (str | list): The type of cell to filter on (optional) / can inout list 
        - treatment (str | list): The treatment to filter on, i.e. drug applied (optional).
        - cell_subtype (str | list): The subtype of cell to filter on (optional).
        - I_set (str | list): The I_set to filter on (optional).
        - threshold_access_change (float): The threshold for access change filtering (optional, default 30).
        '''
    
    project: str  = field(kw_only=True)
    data_type: str = field(kw_only=True)
    cell_type: str | list = field(kw_only=True, default=None)
    cell_subtype: str | list  = field(kw_only=True, default=None)
    treatment: str | list  = field(kw_only=True, default=None)
    I_set: str | list  = field(kw_only=True, default=None)
    threshold_access_change: float = field(kw_only = True, default=30)


    def __post_init__(self):

        super().__init__(cache_dir=f"{ROOT}/{self.project}/cache")#HACKY SHIT 
        self.location = f"{ROOT}/{self.project}"
        self.input_dir = self._checkFileSystem("input")
        self.output_dir = self._checkFileSystem("output")
        self.figure_output_dir = self._checkFileSystem("figures")

        self.FP_df = FP(self.project).df
        self.APP_df = APP(self.project).df
        #addd Hunter when ready
        self.cell_df = Ephys(self.project).df
        # super().__post_init__()
        self.validate_inputs()
        self.valid_files, self.valid_cell_ids = self.get_valid_folder_files()
        self.agg_df = self.get_filtered_data()
        self.treatment_count_df = self.generate_treatment_count_df()

    def validate_inputs(self):
        if self.data_type not in ['FP', 'APP']: #TODO add pAD_hunter
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
        Filters the cell_df based on the input parameters including threshold_access_change if not None.
        Returns a list of valid folder_files and cell_ids.
        """
        valid_column = f'{self.data_type}_valid'
        if valid_column not in self.cell_df.columns:
            raise ValueError(f"{valid_column} column does not exist in cell_df.")
        
        filtered_cell_df = self.cell_df.copy()
        #apply filters 
        if self.cell_type is not None:
            filtered_cell_df = filtered_cell_df[filtered_cell_df['cell_type'].isin([self.cell_type] if isinstance(self.cell_type, str) else self.cell_type)]
        if self.treatment is not None:
            filtered_cell_df = filtered_cell_df[filtered_cell_df['treatment'].isin([self.treatment] if isinstance(self.treatment, str) else self.treatment)]
        if self.cell_subtype is not None:
            filtered_cell_df = filtered_cell_df[filtered_cell_df['cell_subtype'].isin([self.cell_subtype] if isinstance(self.cell_subtype, str) else self.cell_subtype)]
        if self.I_set is not None:
            filtered_cell_df = filtered_cell_df[filtered_cell_df['I_set'].isin([self.I_set] if isinstance(self.I_set, str) else self.I_set)]
        if self.threshold_access_change is not None:
            filtered_cell_df = filtered_cell_df[filtered_cell_df['access_change'].abs() <= self.threshold_access_change]

        valid_cell_ids = filtered_cell_df['cell_id'].tolist()
        valid_files = filtered_cell_df[valid_column].dropna().tolist()
        if not valid_files:
            print("No valid files found for data selection.")
            return [],[]
        valid_files = [item for sublist in valid_files for item in sublist] if isinstance(valid_files[0], list) else valid_files

        return valid_files, valid_cell_ids

    def get_filtered_data(self):
        """
        Fetches the data_type _df and filters it and returns restructured aggergate df for stats and plotting (one row for each cell_id and time).
        """
        # valid_files, valid_cell_ids = self.get_valid_folder_files()
        
        if self.data_type == 'APP':
            filtered_df = self.APP_df[self.APP_df['folder_file'].isin(self.valid_files)]
            #column names {dependant_vairable}_{time}
            timepoints = ['PRE', 'APP', 'WASH']
            columns = {
            'RMP': 'RMP',
            'inputR': 'inputR',
            'RAcount': 'RA_count',
            'APcount': 'AP_count' }
            reshaped_data = []

            for timepoint in timepoints:
                time_specific_columns = {f'{var}_{timepoint}': name for var, name in columns.items() if f'{var}_{timepoint}' in filtered_df.columns}
                
                existing_columns = [col for col in time_specific_columns.keys() if col in filtered_df.columns]
                current_data = filtered_df[['cell_id'] + existing_columns].copy()
                current_data['time'] = timepoint
                current_data.rename(columns=time_specific_columns, inplace=True)
                # aggregate mean data
                for col in ['RMP', 'inputR']:
                    if col in current_data.columns:
                        current_data[f'sweep_{col}'] = current_data[col]
                        current_data[col] = current_data[col].apply(lambda x: np.nanmean(x) if isinstance(x, list) and len(x) > 0 else (np.nan if isinstance(x, list) else x))
                # APs per sweep for each timepoint (returns a zero-padded list)
                for sweep_col in ['AP_sweep_locs', 'RA_sweep_locs']:
                    if sweep_col in filtered_df.columns:
                        base = sweep_col.split('_')[0]  # 'AP' or 'RA'
                        colname = f'sweep_{base}_count'
                        current_data[colname] = filtered_df.apply(
                            lambda row: (
                                [] if not isinstance(row[sweep_col], list)
                                else [
                                    row[sweep_col].count(s) for s in range(
                                        len(row.get(f'inputR_{timepoint}', []))
                                    )
                                    if (
                                        (timepoint == 'PRE' and s < row['drug_in']) or
                                        (timepoint == 'APP' and row['drug_in'] <= s < row['drug_out']) or
                                        (timepoint == 'WASH' and s >= row['drug_out'])
                                    )
                                ]
                            ),
                            axis=1
                        )
                reshaped_data.append(current_data)
            agg_APP_df = pd.concat(reshaped_data, ignore_index=True)
            agg_APP_df = self.add_cell_mapping(agg_APP_df)

            return agg_APP_df
        

        elif self.data_type == 'FP':
            filtered_df = self.FP_df[self.FP_df['folder_file'].isin(self.valid_files)].copy()
            filtered_df['time'] = filtered_df['drug'].apply(lambda x: 'WASH' if x != 'PRE' else 'PRE')
            #columns to keep
            filtered_df = filtered_df[['cell_id', 'time', 'AP_decay_dvdt', 'AP_rise_dvdt',
                        'AP_dvdt_max', 'AP_height', 'AP_latency', 'AP_peak_voltages',
                        'AP_width', 'FI_slope', 'max_firing', 
                        'rheobased_threshold', 'sag', 'tau_rc', 'voltage_threshold']]
            #aggregate 
            for col in ['AP_dvdt_max', 'AP_height', 'AP_latency', 'AP_peak_voltages',
                        'AP_decay_dvdt', 'AP_rise_dvdt', 'AP_width', 'sag', 'tau_rc', 'voltage_threshold']:
                filtered_df[col] = filtered_df[col].apply(lambda x: np.mean(x) if isinstance(x, list) else x)
            agg_FP_df = filtered_df.groupby(['cell_id', 'time']).agg({
            'AP_dvdt_max': 'mean',
            'AP_height': 'mean',
            'AP_latency': 'mean',
            'AP_peak_voltages': 'mean',
            'AP_decay_dvdt': 'mean',
            'AP_rise_dvdt': 'mean',
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
        Adds cell feature columns based off cell_id in cell_df.
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
        if self.threshold_access_change is not None:
            access_filtered_df = self.cell_df[self.cell_df['access_change'].abs() <= self.threshold_access_change]
        else:
            access_filtered_df = self.cell_df

        treatment_count_df = access_filtered_df.groupby(['treatment', 'cell_type']).apply(process_group).reset_index()
        self.cache(f'treatment_count_df_{self.threshold_access_change}', treatment_count_df)
        self.save_excel( f'treatment_count_df_{self.threshold_access_change}', treatment_count_df)
        return treatment_count_df
        
    def fetch_data_type (self, folder_file):
        '''
        Finds relevant info for folder_file and returns truple of:  string lable , the row from the df
        '''
        if folder_file in self.FP_df['folder_file'].values:
            row = self.FP_df[self.FP_df['folder_file'] == folder_file]
            return 'FP', row
        elif folder_file in self.APP_df['folder_file'].values:
            row = self.APP_df[self.APP_df['folder_file'] == folder_file]
            return 'APP', row
        else:
            return 'Unknown', None

        
    def folder_file_AP_df(self, cell_id, folder_file,  V_array=None, I_array=None): # add drug column with PRE APP or WASH
        '''builds AP_df for single folder_file.'''
        if V_array is None or I_array is None:
            V_array , I_array, V_list = Project(self.project).IGOR_load(folder_file)
            if I_array is None:
                I_array = np.zeros((len(V_array), 1))
        
        V_array, I_array = normalise_array_length(V_array, I_array, columns_match=False, verbose=True)
        
        # Extract AP characteristics
        peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_rise_all  , peak_max_dvdt_all,  peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all   , peak_indices_all , sweep_indices_all , peak_decay_all = ap_characteristics_extractor_main(folder_file, V_array)
        
        # Early return if no APs found
        if np.all(np.isnan(peak_latencies_all)):
            print(f"No APs detected in voltage trace {folder_file}.")
            return pd.DataFrame(columns=['folder_file', 'peak_location', 'upshoot_location', 'voltage_threshold',
                                        'slope', 'latency', 'peak_voltage', 'height', 'width', 'sweep',
                                        'I_injected', 'AP_type'])
        
        file_data_type, row_info = self.fetch_data_type(folder_file)
        drug_used = row_info['drug'].iloc[0]

        if file_data_type == 'APP':
            drug_in =  row_info['drug_in'].iloc[0]
            drug_out = row_info['drug_out'].iloc[0]
            drug_labels = [ 'PRE' if _ < drug_in else 'APP' if drug_in <= _ <= drug_out else 'WASH' for _ in sweep_indices_all ]
            current_injected = [I_array[loc, 0] for loc in peak_locs_corr_all]

        if file_data_type == 'FP':
            drug_labels = [row_info['drug'].iloc[0] if row_info['drug'].iloc[0] == 'PRE' else 'WASH' for _ in sweep_indices_all]
            current_injected = [I_array[loc, sweep] for loc, sweep in zip(peak_locs_corr_all, sweep_indices_all)]


        AP_df = pd.DataFrame({
        'folder_file': folder_file,
        'cell_id': cell_id,
        'data_type': file_data_type,
        'cell_treatment': drug_used,
        'drug': drug_labels,
        'peak_location': peak_locs_corr_all,
        'upshoot_location': upshoot_locs_all,
        'voltage_threshold': v_thresholds_all,
        'peak_rise': peak_rise_all,
        'peak_decay': peak_decay_all,
        'peak_max_rise': peak_max_dvdt_all,
        'latency': peak_latencies_all,
        'peak_voltage': peak_voltages_all,
        'height': peak_heights_all,
        'width': peak_fw_all,
        'sweep': sweep_indices_all,
        'I_injected': current_injected,  # Sweep index is 0 as I_array is identical
        'AP_type': 'somatic'  
        })
        try:
            valid_FP_folder_files = self.cell_df[self.cell_df['cell_id']==cell_id]['FP_valid'].values[0][:2]
            mean_voltage_threshold = self.FP_df[self.FP_df['folder_file'].isin(valid_FP_folder_files)]['voltage_threshold'].explode().astype(float).mean()
            AP_df.loc[(AP_df['voltage_threshold'] < mean_voltage_threshold-20 ), 'AP_type'] = 'RA'
        except(IndexError, TypeError):
            print(f" Cell {cell_id} has no valid FP to assess voltage threshold, setting RMP<-60mV")
            AP_df.loc[(AP_df['voltage_threshold'] < -60) & (AP_df['peak_voltage'] > 20), 'AP_type'] = 'RA'

        if file_data_type == 'FP':
            AP_df_positive = AP_df[AP_df['I_injected'] > 0].iloc[:20] #HARD CODE keeping first 20 APs on + I step
            AP_df_non_positive = AP_df[AP_df['I_injected'] <= 0]
            AP_df = pd.concat([AP_df_non_positive, AP_df_positive]).sort_index().reset_index(drop=True)

        return AP_df


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
    
    def filter_n_minimum(self,df):
        df = df.dropna(subset=[self.dependant_var]).reset_index(drop=True)
        group_sizes = df.groupby(['treatment', 'time']).size()
        insufficient_groups = group_sizes[group_sizes < self.n_minimum]
        if not insufficient_groups.empty:
            print(f"Warning: The following groups have less than {self.n_minimum} samples and will be excluded:")
            print(insufficient_groups)
            df = df[~df[['treatment', 'time']].apply(tuple, axis=1).isin(insufficient_groups.index)].reset_index(drop=True)
        if df.empty:
            print("No groups meet the minimum sample size requirement. Statistical analysis will not be performed.")
            return None
        else:
            return df
    
    def check_valid_dependant_var(self):
            if self.dependant_var not in self.agg_df.columns:
                dvs = [col for col in self.agg_df.columns if col not in ['cell_id', 'time', 'treatment', 'cell_type', 'cell_subtype', 'I_set']]
                raise ValueError(f"Invalid dependant variable: {self.dependant_var}. Valid dv's : {dvs}")
    
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




@dataclass
class Histogram(Figure):
    filename: str = None
    dependant_var: str = field(kw_only=True)
    specify: str = field(kw_only = True, default = 'treatment') # specify marker to see subsets e.g. I_set or cell_id
    n_minimum: float = field(kw_only = True, default = 3)

    def __post_init__(self):
        self.filename = f"{self.dependant_var}_{self.specify}" 
        super().__post_init__()
        self.check_valid_dependant_var()
        self.data = self.filter_n_minimum(self.agg_df)
        self.stats = self.generate_statistics()
        self.order = [t for t in color_dict.keys() if t in self.data['treatment'].unique()]
        self.hue_order = [t for t in ['PRE', 'APP', 'WASH'] if t in self.data['time'].unique()]
        self.fig = self.plot_histogram()
        
    def generate_statistics(self):
        #mixed effects models --> Tukey
        model = mixedlm(f"{self.dependant_var} ~ time * treatment", self.data, groups=self.data["cell_id"])
        result = model.fit()
        interaction_pvalues = {term: result.pvalues[term] for term in result.pvalues.keys() if 'time' in term and 'treatment' in term}
        significant_interactions = {term: pval for term, pval in interaction_pvalues.items() if pval < 0.05}

        if significant_interactions:
            print(f"significant interaction/s: {significant_interactions}, performingm tukey post hoc.")
            tukey = pairwise_tukeyhsd(endog=self.data[self.dependant_var], groups=self.data['time'] + self.data['treatment'], alpha=0.05)
            significant_pairs = []
            for row in tukey.summary().data[1:] :
                reject = row[-1]  # The last column indicates whether the null hypothesis was rejected
                if reject == 'True':  #  if the comparison is significant
                    group1, group2 = row[0], row[1]
                    significant_pairs.append((group1, group2))
                    return significant_pairs
        else:
            print("No significant interaction found.")
            return None


    def specify_markers(self, df, ax):
        """
        Adds a marker legend to the plot based on the `specify` attribute.
        """
        unique_values = df[self.specify].unique()
        markers = cycle(['o', 's', '^', 'D', 'v', '<', '>'])  # Define markers to use
        legend_handles_labels = {}
        for i, value in enumerate(unique_values):
            marker = next(markers)
            subset_to_plot = df[df[self.specify] == value]
            sns.stripplot(
                x='treatment',
                y=self.dependant_var,
                hue='time',
                hue_order=['PRE', 'APP', 'WASH'],
                order=self.order ,
                data=subset_to_plot,
                palette={"PRE":"azure", "APP": "teal", "WASH":"cadetblue"},
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
        fig, ax = plt.subplots(figsize=(15, 10))
        df = self.data
        sns.barplot(
            x='treatment',
            y=self.dependant_var,
            hue='time',
            hue_order=self.hue_order,
            order=self.order ,
            data=df,
            errorbar = 'sd',
            palette={"PRE":"azure", "APP": "teal", "WASH":"cadetblue"},
            edgecolor="k",
            ax=ax
        )
        sns.swarmplot(
            x='treatment',
            y=self.dependant_var,
            hue='time',
            hue_order=self.hue_order,
            order=self.order ,
            data=df,
            palette={"PRE":"azure", "APP": "teal", "WASH":"cadetblue"},
            edgecolor="k",
            linewidth=0.5,
            ax=ax, 
            legend=False,
            marker="o",
            size=0.05, #small as will be plotted over by specify_markers
            alpha = 0.7,
            dodge=True,
        )
 
        legend_handles_labels = self.specify_markers(df, ax)
    
        current_handles, current_labels = ax.get_legend_handles_labels()
        combined_handles = current_handles + list(legend_handles_labels.values())
        combined_labels = current_labels + [handle.get_label() for handle in legend_handles_labels.values()]
        ax.legend(handles=combined_handles, labels=combined_labels, loc='best', title='Legend')


        counts = df.groupby('treatment')['cell_id'].nunique()
        for tick, treatment in enumerate(self.order):
            count = counts.get(treatment, 0)
            ax.text(tick, -0.1, f'n={count}', ha='center', va='top', fontsize=24, color='black', transform=ax.get_xaxis_transform())

        # Customize plot labels and titles
        ax.set_ylabel(unit_dict[self.dependant_var], fontsize=24)
        ax.set_xlabel('')
        ax.set_title(f'{self.cell_type} - {self.dependant_var}', fontsize=28)
        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
        plt.tight_layout()
        plt.show()
        
        # Save the figure
        self.save_plot(fig, self.filename)



@dataclass
class AggregateApplication(Figure):
    filename: str = None
    sweep_in_s: float = field(kw_only = True, default = 20)
    dependant_var: str = field(kw_only=True) #  'RMP', 'inputR', 'RA_count', 'AP_count'
    bin_size: float = field(kw_only = True, default = 3) #sweeps to pool default 3 3x20sec - 1min
    n_minimum: float = field(kw_only = True, default = 3)


    def __post_init__(self):
        self.filename = f"{self.dependant_var}_{self.bin_size}" 
        super().__post_init__()
        self.check_valid_dependant_var()
        self.data = self.filter_n_minimum(self.agg_df)
        
        self.timepoints = ['PRE', 'APP', 'WASH']
        self.colors = {'PRE': 'lightgrey', 'APP': 'black', 'WASH': 'grey'} #black preset will be changed in plot
        self.binned_data = self.process_data(self.data)
        self.plot()

    def process_data(self, df):
        '''
        Creates a df to plot with columns cell_id time data then loops rows to bin data +++++++++++ does % baselin if RMP or input R
        '''
        column_to_bin = f'sweep_{self.dependant_var}'
        binned_rows = []
        for _, row in df.iterrows():
            data = row[column_to_bin]
            if not isinstance(data, list) or len(data) == 0:
                print(f"{row['cell_id']} for {row['time']} is empty or invalid.")
                continue
            binned = [
                np.mean(data[i:i + self.bin_size])
                for i in range(0, len(data), self.bin_size)
                if len(data[i:i + self.bin_size]) == self.bin_size
            ]
            binned_rows.append({
                'cell_id': row['cell_id'],
                'time': row['time'],
                'binned_values': binned
            })

        binned_df = pd.DataFrame(binned_rows)

        if self.dependant_var in ['RMP', 'inputR']:
            normalized_rows = []
            for cell_id, group in binned_df.groupby('cell_id'):
                pre_row = group[group['time'] == 'PRE']
                if pre_row.empty:
                    print(f"No PRE data for {cell_id}, skipping normalization.")  
                    continue
                baseline_vals = pre_row.iloc[0]['binned_values']
                baseline_mean = np.mean(baseline_vals) 
                for _, row in group.iterrows():
                    norm_vals = [(val / baseline_mean) * 100 for val in row['binned_values']]  
                    normalized_rows.append({
                        'cell_id': row['cell_id'],
                        'time': row['time'],
                        'binned_values': norm_vals
                    })
        normalised_binned_df = pd.DataFrame(normalized_rows)
        return normalised_binned_df

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        time_per_bin_min = (self.sweep_in_s * self.bin_size) / 60
        padding = 0 * time_per_bin_min  # HARD CODE ADJUST
        section_widths = {}
        legend_handles = {}
        cell_counts = self.cell_df[self.cell_df['cell_id'].isin(self.binned_data['cell_id'])].groupby('treatment')['cell_id'].nunique().to_dict()


        for timepoint in self.timepoints:
            max_bins = self.binned_data[self.binned_data['time'] == timepoint]['binned_values'].apply(len).max()
            section_widths[timepoint] = max_bins * time_per_bin_min if max_bins else 0

        for cell_id, cell_data in self.binned_data.groupby('cell_id'):
            x_offset = 0
            prev_endpoint = None  

            for idx, timepoint in enumerate(self.timepoints):
                tp_data = cell_data[cell_data['time'] == timepoint]
                if tp_data.empty:
                    continue

                binned_vals = tp_data.iloc[0]['binned_values']
                n_bins = len(binned_vals)
                x_vals = np.arange(n_bins) * time_per_bin_min + x_offset
                y_vals = binned_vals

                if timepoint in ['APP', 'WASH']:
                    drug = self.cell_df[self.cell_df['cell_id'] == cell_id]['treatment'].iloc[0]
                    color = color_dict.get(drug, 'k')
                    alpha = 0.5 if timepoint == 'WASH' else 1.0
                    label = f"{drug} (n={cell_counts.get(drug, 0)})" if timepoint == 'APP' else None
                else:
                    color = self.colors[timepoint]
                    label = None
                    alpha = 1.0

                if label is not None and label not in legend_handles:
                    legend_handles[label] = ax.plot([], [], color=color, alpha=alpha, label=label)[0]
                ax.plot(x_vals, y_vals, color=color, alpha=alpha, linewidth=1)  

                # connector between last timepoint and current one
                if prev_endpoint is not None:  
                    connector_x = [prev_endpoint[0], x_vals[0]]  
                    connector_y = [prev_endpoint[1], y_vals[0]]  
                    ax.plot(connector_x, connector_y, color=color, linestyle=':', alpha=0.2, linewidth=0.5) 

                prev_endpoint = (x_vals[-1], y_vals[-1])  
                x_offset += section_widths[timepoint] + padding

        cumulative_offset = 0
        for timepoint in self.timepoints[:-1]:
            cumulative_offset += section_widths[timepoint] + padding
            ax.axvline(cumulative_offset, color='lightgrey', linestyle='--', linewidth=0.8)

        ax.set_title(f'{self.dependant_var} ')
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(f"{unit_dict[self.dependant_var]} as % of baseline")
        ax.legend(fontsize='small', loc='upper right')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        plt.show()


       
@dataclass
class Application(Figure):

    '''Plot a single APP file from cell_id or list of.'''
    project: str = field(kw_only = True)
    cell_id: str|list = field(kw_only = True, default = None) # optional pram for plotting specific cell/s application
    plot_all_APs: bool = field(kw_only=True, default=False)
    valid_only: bool = field(kw_only=True, default=False)


    def __post_init__(self):
        # self.filename = f"{self.dependant_var}_{self.specify}" # TODO handel better 
        super().__post_init__()
        if self.cell_id == None:
            self.cell_id = self.valid_cell_ids
        self.fig = self.plot_applications()

    
    def plot_applications(self):
        color_map = {'RA': 'red', 'somatic': 'blue'}
        alpha_map = {'RA': 0.6, 'somatic': 0.2}
        cell_ids = [self.cell_id] if isinstance(self.cell_id, str) else self.cell_id #make list of string if a single string
        for cell_id in cell_ids:
            self.filename = f'{cell_id}_application'

            # Fetch folder_file for the specific cell_id
            cell_sub_df = self.APP_df[self.APP_df['cell_id'] == cell_id]
            if self.valid_only == True:
                cell_sub_df = cell_sub_df[cell_sub_df['valid'] != False]

            for folder_file, cell_id, I_set, drug, drug_in, drug_out, application_order, RA_locs in cell_sub_df[['folder_file','cell_id', 'I_set', 'drug', 'drug_in', 'drug_out', 'application_order', 'RA_locs']].values:
                self.fig_filename = f"{cell_id}_application{application_order}"
                
                V_array , I_array, V_list = Project(self.project).IGOR_load(folder_file)
                if I_array is None:
                    I_array = np.zeros((len(V_array), 1))

                # sampeling at 20KHz -->  time (s)
                seconds_per_sweep = len(V_array[:,0]) * 0.00005 # multiplying this  by drug_in/out will give you the point at the end of the sweep in seconds
                # x_V = np.arange(len(V_list)) * 0.00005 #trying to rewmove list handeling 10_4_25
                
                x_V = np.arange(V_array.shape[0] * V_array.shape[1]) * 0.00005 
                x_I = np.arange(len(I_array)) * 0.00005 

                #build figure 
                fig = plt.figure(figsize = (12,9))
                ax1 = plt.subplot2grid((11, 8), (0, 0), rowspan = 8, colspan =11) #(nrows, ncols)
                ax2 = plt.subplot2grid((11, 8), (8, 0), rowspan = 2, colspan=11)

                #plot voltage / time
                n_sweeps = V_array.shape[1]  # Number of sweeps based on the second dimension of V_array
                cropped_array = V_array[:, :n_sweeps]  # Crop the array to match the number of sweeps
                continuous_plot = cropped_array.ravel(order='F')  # Flatten the array in column-major (Fortran) order
                ax1.plot(x_V, continuous_plot, c='k' if drug is None else color_dict.get(drug, 'k'), lw=1, alpha=0.8)  # Plot voltage

                #handle action potentials 
                AP_df = self.folder_file_AP_df(cell_id, folder_file, V_array, I_array)
                if self.plot_all_APs and not AP_df.empty:
                    for ap_type in ['RA', 'somatic']:
                        color = color_map[ap_type]
                        alpha = alpha_map[ap_type]
                        ap_df = AP_df[AP_df['AP_type'] == ap_type]
                        n_aps = len(ap_df)
                        for upshoot_location, sweep, peak_location in ap_df[['upshoot_location', 'sweep', 'peak_location']].values:
                            v_temp = np.array(V_array[:, sweep][upshoot_location:peak_location])
                            time_temp = np.linspace(0, len(v_temp) * 0.00005, len(v_temp))
                            time_temp += seconds_per_sweep * sweep + upshoot_location * 0.00005
                            ax1.plot(time_temp, v_temp, color=color, lw=2, alpha=alpha, label=None)
                        if n_aps > 0:
                            ax1.plot([], [], color=color, lw=2, alpha=0.6, label=f'{ap_type} (n={n_aps})')
                    ax1.legend(loc='upper right')

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
                self.save_plot(fig, f"{cell_id}_APP_{str(application_order)}")
                



@dataclass
class RA_AP_analysis(Figure):
    '''
    AP analsysis for a single cell_id: 
    either by folder_file or pooled, data_type indicate the daa types to be analised for that cell_id (invalidated files removed)
    '''
    project: str = field(kw_only = True)
    cell_id: str = field(kw_only = True) # single cell ID ONLY 
    data_type: list = field(kw_only=True, default='APP') # this may need to be corrected it is for the inheretence from Cashable/Dataselector/Figure
    data_types: list = field(kw_only=True, default='APP') 
    pooled: bool = field(kw_only=True, default=False)

    color_map: dict = field(default_factory=lambda: {'RA': 'red', 'somatic': 'blue'}, init=False)
    forwards_window: int = field(default=50, init=False)
    backwards_window: int = field(default=70, init=False)
    sampling_rate: float = field(default=2e4, init=False)
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
        if 'APP' in self.data_types:
            valid_folder_files = self.cell_df[self.cell_df['cell_id'] == self.cell_id]['APP_valid'].iloc[0]
            if valid_folder_files is None: 
                 print(f"No valid APP files for cell_id {self.cell_id}, skipping.")
            else:
                agg_folder_files.append(valid_folder_files)

        if 'FP' in self.data_types:
            valid_folder_files = self.cell_df[self.cell_df['cell_id'] == self.cell_id]['FP_valid'].iloc[0]

            if valid_folder_files is None:
                 print(f"No valid FP files for cell_id {self.cell_id}, skipping.")
            else:
                agg_folder_files.extend(valid_folder_files)
                
        for folder_file in agg_folder_files: #TOD add drug presence to AP_df
            #fetch raw trace
            V_array , I_array, V_list = Project(self.project).IGOR_load(folder_file)
            if I_array is None:
                I_array = np.zeros((len(V_array), 1))
            #build AP_df for folder_file
            AP_df = self.folder_file_AP_df(self.cell_id, folder_file, V_array, I_array)  
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
            V_array, I_array, _ = Project(self.project).IGOR_load(folder_file)
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
                    traces.append(trace)
                max_len = max([len(t) for t in traces], default=0)  
                valid_traces = [t for t in traces if len(t) == max_len]
                if len(valid_traces) < len(traces): 
                    print(f"{len(traces) - len(valid_traces)} {ap_type} APs dropped due to short trace length.")
                traces_by_type[ap_type].extend(valid_traces) 
        #plot 
        for ap_type, traces in traces_by_type.items():
            if not traces:
                continue
            color = self.color_map[ap_type]
            for trace in traces:
                time_ms = (np.arange(0, len(trace)) * 1000) / self.sampling_rate
                ax.plot(time_ms, trace, color=color, alpha=0.1, linewidth=0.9) # raw trace

            mean_trace = np.mean(traces, axis=0)
            time_ms = (np.arange(0, len(mean_trace)) * 1000) / self.sampling_rate
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
            V_array, I_array, _ = Project(self.project).IGOR_load(folder_file)

            for ap_type in AP_df_folder_file['AP_type'].unique():
                color = self.color_map.get(ap_type, 'gray')
                subset = AP_df_folder_file[AP_df_folder_file['AP_type'] == ap_type]
                traces = []

                for upshoot_location, sweep in subset[["upshoot_location", "sweep"]].values:
                    v_temp = V_array[upshoot_location: upshoot_location + self.forwards_window, sweep]
                    if len(v_temp) < 2:
                        continue
                    dv_temp = np.diff(v_temp)
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
        ax.set_ylabel("dV (mV)")
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



# @dataclass #old
# class RA_AP_analysis(Figure):

#     #TODO - mean trace plots for the phase plot, add all AP_parms to the histogram analysis

#     '''Plot a single APP file from cell_id or list of.'''
#     project: str = field(kw_only = True)
#     cell_id: str|list = field(kw_only = True, default = None) 
#     valid_only: bool = field(kw_only=True, default=False)
#     
#     color_map: dict = field(default_factory=lambda: {'RA': 'red', 'somatic': 'blue'}, init=False)
#     forwards_window: int = field(default=50, init=False)
#     backwards_window: int = field(default=70, init=False)
#     sampling_rate: float = field(default=2e4, init=False)
#     voltage_max: float = field(default=60.0, init=False)
#     voltage_min: float = field(default=-120.0, init=False)

#     def __post_init__(self):
#         # self.filename = f"{self.dependant_var}_{self.specify}" # TODO handel better 
#         super().__post_init__()

#         if self.cell_id == None:
#             self.cell_id = self.valid_cell_ids

#         self.fig = self.plot_meanAPs()
#         self.phase_fig = self.plot_phaseplotAPs()
#         self.hist_fig = self.plot_histogramAPs()


#     def plot_meanAPs(self):
#         for cell_id in self.cell_id:
#             self.filename = f'{cell_id}_mean_AP'

#             # Fetch folder_file for the specific cell_id
#             cell_sub_df = self.APP_df[self.APP_df['cell_id'] == cell_id] # Likethis will only have APP files .... could adapt for RA hunter too
#             if self.valid_only == True:
#                 cell_sub_df = cell_sub_df[cell_sub_df['valid'] != False]

#             for folder_file, cell_id, I_set, drug, drug_in, drug_out, application_order, RA_locs in cell_sub_df[['folder_file','cell_id', 'I_set', 'drug', 'drug_in', 'drug_out', 'application_order', 'RA_locs']].values:
#                 self.fig_filename = f"{cell_id}_application{application_order}"
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 V_array , I_array, V_list = Project(self.project).IGOR_load(folder_file)
#                 if I_array is None:
#                     I_array = np.zeros((len(V_array), 1))


#                 AP_df = self.folder_file_AP_df(cell_id, folder_file, V_array, I_array)
#                 spike_arrays = {'RA': [], 'somatic': []}

#                 for ap_type in AP_df['AP_type'].unique():
#                     color = self.color_map[ap_type]
#                     ap_indices = AP_df[AP_df['AP_type'] == ap_type][["upshoot_location", "sweep"]].values
#                     traces = []
#                     # Prepare spike data
#                     for idx in range(len(ap_indices)):
#                         upshoot_location = ap_indices[idx, 0]
#                         sweep_idx = ap_indices[idx, 1]
#                         lower_bound = max(0, upshoot_location - self.backwards_window)
#                         upper_bound = upshoot_location + self.forwards_window
#                         # spike_arrays[ap_type].append(V_array[lower_bound:upper_bound, ap_indices[idx, 1]]) #old was returning 0 when couldnt slice
#                         if upper_bound <= V_array.shape[0]:
#                             trace = V_array[lower_bound:upper_bound, sweep_idx]
#                         else:
#                             trace = V_array[lower_bound:, sweep_idx]
#                         traces.append(trace)

  
#                     max_len = max([len(t) for t in traces], default=0)  
#                     valid_traces = [t for t in traces if len(t) == max_len]  
#                     #dropped AP trace count due to length shorter
#                     if len(valid_traces) < len(traces): 
#                         print(f"{len(traces) - len(valid_traces)} {ap_type} APs dropped due to short trace length.")
#                     spike_arrays[ap_type] = valid_traces 
#                     # Plot individual traces for each AP type
#                     if spike_arrays[ap_type]:
#                         for trace in spike_arrays[ap_type]:
#                             time_ms = (np.arange(0, len(trace)) * 1000) / self.sampling_rate  # Convert to ms
#                             ax.plot(time_ms, trace, color=color, alpha=0.1, linewidth=0.9)
                            

#                         # Plot mean trace for this AP type
#                         mean_spike = np.mean(np.array(spike_arrays[ap_type]), axis=0)
#                         ax.plot(time_ms, mean_spike, color=color, label=f'{ap_type} mean voltage (n={len(spike_arrays[ap_type])})', linewidth=1.3, alpha=1)
#                         ax.set_ylabel('Membrane Potential (mV)')
#                         ax.set_xlabel('Time (ms)')
#                         ax.legend()
#                         ax.set_title(f"{cell_id} {drug} Application{application_order}_meanAPs", fontsize = 16) # , fontsize = 25
                
#             plt.tight_layout()
#             plt.show()
#             self.save_plot(fig, f"{cell_id}_meanAPs_application{application_order}")
#             return fig
        
#     def plot_phaseplotAPs(self):
#         for cell_id in self.cell_id:
#             self.filename = f'{cell_id}_phase_plot'

#             cell_sub_df = self.APP_df[self.APP_df['cell_id'] == cell_id]
#             if self.valid_only:
#                 cell_sub_df = cell_sub_df[cell_sub_df['valid'] != False]

#             for folder_file, cell_id, _, drug, _, _, application_order, _ in cell_sub_df[['folder_file','cell_id', 'I_set', 'drug', 'drug_in', 'drug_out', 'application_order', 'RA_locs']].values:
#                 self.fig_filename = f"{cell_id}_application{application_order}_phase"
#                 fig, ax = plt.subplots(figsize=(8, 6))

#                 V_array , I_array, _ = Project(self.project).IGOR_load(folder_file)

#                 AP_df = self.folder_file_AP_df(cell_id, folder_file, V_array, I_array)

#                 for ap_type in AP_df['AP_type'].unique():
#                     color = self.color_map.get(ap_type, 'gray')
#                     ap_indices = AP_df[AP_df['AP_type'] == ap_type][["upshoot_location", "sweep"]].values
#                     for idx in range(len(ap_indices)):
#                         upshoot_location = ap_indices[idx, 0]
#                         sweep = ap_indices[idx, 1]
#                         v_temp = V_array[upshoot_location: upshoot_location + self.forwards_window, sweep]
#                         if len(v_temp) < 2:  # skip if not enough points
#                             continue
#                         dv_temp = np.diff(v_temp)
#                         if max(v_temp) <= self.voltage_max and min(v_temp) >= self.voltage_min:
#                             ax.plot(v_temp[:-1], dv_temp, color=color, alpha=0.05)

#                 ax.set_title(f"{cell_id} {drug} Application ({application_order})")
#                 ax.set_xlabel("Membrane Potential (mV)")
#                 ax.set_ylabel("dV (mV)")
#                 legend_elements = []

#                 for ap in AP_df['AP_type'].unique():
#                     if ap in self.color_map:
#                         count = len(AP_df[AP_df['AP_type'] == ap])
#                         legend_elements.append(Line2D([0], [0], color=self.color_map[ap], lw=2,
#                                                     label=f'{ap} dV vs V (n={count})'))

#             ax.legend(handles=legend_elements)
#             plt.tight_layout()
#             plt.show()
#             self.save_plot(fig, f"{cell_id}_application{application_order}_phase")
#             return fig

#     def plot_histogramAPs(self):
#         for cell_id in self.cell_id:
#             self.filename = f'{cell_id}_histogram_AP'

#             cell_sub_df = self.APP_df[self.APP_df['cell_id'] == cell_id]
#             if self.valid_only:
#                 cell_sub_df = cell_sub_df[cell_sub_df['valid'] != False]

#             for folder_file, cell_id, _, drug, _, _, application_order, _ in cell_sub_df[['folder_file','cell_id', 'I_set', 'drug', 'drug_in', 'drug_out', 'application_order', 'RA_locs']].values:
#                 self.fig_filename = f"{cell_id}_application{application_order}_hist"

#                 V_array , I_array, _ = Project(self.project).IGOR_load(folder_file)
#                 AP_df = self.folder_file_AP_df(cell_id, folder_file, V_array, I_array)

#                 fig, axs = plt.subplots(4, 2, figsize=(10, 10))
#                 plot_labels = ['RA', 'somatic']

#                 column_map = {
#                     'voltage_threshold': ('Voltage Thresholds', 'mV'),
#                     'peak_rise': ('AP Rise', 'V/s'),
#                     'peak_decay': ('AP Decay', 'V/s'),
#                     'peak_max_rise': ('AP Rise max', 'V/s'),
#                     'height': ('AP Heights', 'mV'),
#                     'peak_voltage': ('AP peak', 'mV'),
#                     'width': ('AP FWHM', 'ms'),
#                     'latency': ('Peak Latency', 'ms')
#                 }

#                 plot_columns = list(column_map.keys())
#                 for i, col in enumerate(plot_columns):
#                     row, col_pos = divmod(i, 2)
#                     # Combine both RA and somatic data for binning
#                     data_all = AP_df[col].dropna()
#                     bins = np.histogram_bin_edges(data_all, bins=20)

#                     for label in plot_labels:
#                         color = self.color_map[label]
#                         data = AP_df[AP_df["AP_type"] == label][col].dropna()
#                         n_aps = len(data)
#                         axs[row, col_pos].hist(data, bins=bins, color=color,
#                                             label=f'{label} (n={n_aps})', alpha=0.6) #histtype='step',

#                 # Labeling and titles
#                 for i, col in enumerate(plot_columns):
#                     row, col_pos = divmod(i, 2)
#                     title, xlabel = column_map[col]
#                     axs[row, col_pos].set_title(title)
#                     axs[row, col_pos].set_xlabel(xlabel)
#                     axs[row, col_pos].set_ylabel('AP count')
#                     axs[row, col_pos].legend()

#                 fig.tight_layout(h_pad=2.5, rect=[0, 0, 1, 0.95])
#                 plt.suptitle(f'{folder_file}', fontsize=14)
#                 plt.show()
#                 self.save_plot(fig, f"{cell_id}_application{application_order}_hist")
