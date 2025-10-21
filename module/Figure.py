import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import ClassVar
import itertools
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
from module.Stats import Stats
import importlib



# Root directory for projects #HACKY SHIT should have a project or filesystem class to prevent dupicate code
ROOT = f"{os.getcwd()}/PROJECTS"
if not os.path.exists(ROOT):
    os.mkdir(ROOT)

@dataclass
class DataSelection (Cachable): 
    ''' 
    Dataselection for a single data_type based of the folder_files used in the cell_df.

    Attributes:
        - project (str): defining the project and feature mapping ie RAW_df in Ephys
        - data_type (str): The data type 
        - cell_type (str | list): The type of cell to filter on (optional) / can inout list 
        - treatment (str | list): The treatment to filter on, i.e. drug applied (optional).
        - cell_subtype (str | list): The subtype of cell to filter on (optional).
        - I_set (str | list): The I_set to filter on (optional).
        - threshold_access_change (float): The threshold for access change filtering (optional, default 30).
        '''
    
    project: str  = field(kw_only=True)
    project_obj: Project = field(init=False, repr=False)

    data_type: str = field(kw_only=True)
    cell_type: str | list = field(kw_only=True, default=None)
    cell_subtype: str | list  = field(kw_only=True, default=None)
    region: str | list  = field(kw_only=True, default=None)
    treatment: str | list  = field(kw_only=True, default=None)
    I_set: str | list  = field(kw_only=True, default=None)
    threshold_access_change: float = field(kw_only = True, default=30) # Rs % change threshold


    def __post_init__(self):

        super().__init__(cache_dir=f"{ROOT}/{self.project}/cache")#HACKY SHIT 
        self.location = f"{ROOT}/{self.project}"
        self.input_dir = self._checkFileSystem("input")
        self.output_dir = self._checkFileSystem("output")
        self.figure_output_dir = self._checkFileSystem("figures")

        self.project_obj = Project(self.project) #gives self.project_type to include time or not 
        self.load_extractor(self.data_type)
        self.cell_df = Ephys(self.project).df         

        self.validate_inputs() #except dv
        self.valid_files, self.valid_cell_ids = self.get_valid_folder_files()
        self.agg_df = self.biild_agg_df() #validates dv
        if  self.project_obj.project_type == "application":
            self.treatment_count_df = self.generate_treatment_count_df() # not generic enpough yet #TODO
        # elif self.project_obj.project_type == ""

    def load_extractor(self, data_type: str):
        """Dynamically load the extractor class from Ephys_Project by name (data_type)."""
        module = importlib.import_module("module.Ephys_Project")  
        try:
            cls = getattr(module, data_type)  
        except AttributeError:
            raise ValueError(f"No extractor class found for data_type '{data_type}' in Ephys_Project")
        
        df = cls(self.project).df
        setattr(self, f"{data_type}_df", df)  
        return df


    def validate_inputs(self):
        '''
        Checks for valid data_type and that the cell_type, region cell_subtype and treatment are withing the data_type.columns()
        '''
        data_types=['FP', 'APP', 'st_VC', 'ramp_IC', 'IV_VC', 'spont_IC', 'IF_IC' ]
        if self.data_type not in data_types: #complete list of data types
            raise ValueError(f"Invalid data_type: {self.data_type}. Must be one of {data_types}.")
        
        # if self.data_type in ["FP", "APP"]: #TODO file validator inbuild fo project_type = application only needs to be cleaned
        valid_df = self.cell_df[self.cell_df[f'{self.data_type}_folder_files'].notna()] 
        # else:
        #     valid_df = self.cell_df

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
        validate_attribute(self.region, 'region')
   
    def get_valid_folder_files(self):
        """
        Filters the cell_df based on the input parameters including threshold_access_change if not None.
        Returns a list of valid folder_files and cell_ids.
        """
        valid_column = f'{self.data_type}_folder_files'
        if valid_column not in self.cell_df.columns:
            raise ValueError(f"{valid_column} column does not exist in cell_df.")
        
        filtered_cell_df = self.cell_df.copy()

        #apply filters if set
        if self.cell_type is not None:
            filtered_cell_df = filtered_cell_df[filtered_cell_df['cell_type'].isin([self.cell_type] if isinstance(self.cell_type, str) else self.cell_type)]
        if self.treatment is not None:
            filtered_cell_df = filtered_cell_df[filtered_cell_df['treatment'].isin([self.treatment] if isinstance(self.treatment, str) else self.treatment)]
        if self.cell_subtype is not None:
            filtered_cell_df = filtered_cell_df[filtered_cell_df['cell_subtype'].isin([self.cell_subtype] if isinstance(self.cell_subtype, str) else self.cell_subtype)]
        if self.I_set is not None:
            filtered_cell_df = filtered_cell_df[filtered_cell_df['I_set'].isin([self.I_set] if isinstance(self.I_set, str) else self.I_set)]
        if self.threshold_access_change is not None:
            filtered_cell_df = filtered_cell_df[filtered_cell_df['Rs_pct_change'].abs() <= self.threshold_access_change]
        if self.region is not None:
            filtered_cell_df = filtered_cell_df[filtered_cell_df['region'].isin([self.region] if isinstance(self.region, str) else self.region)]

        valid_cell_ids = filtered_cell_df['cell_id'].tolist()
        valid_files = filtered_cell_df[valid_column].dropna().tolist()
        if not valid_files:
            print("No valid files found for data selection.")
            return [],[]
        valid_files = [item for sublist in valid_files for item in sublist] if isinstance(valid_files[0], list) else valid_files

        return valid_files, valid_cell_ids
    
    def biild_agg_df(self):
        """
        Filters self.{data_type}_df for foler_files in cell_df["f{data_type}_folder_files"] and restructures it to a long format for plotting.
        
        Returns:
          agg_df aggregate df for data_type for stats and plotting (one row per cell_id and time).

        """
        data_type_df = getattr(self, f"{self.data_type}_df")

        
        independant_vairables = ['cell_id', 'folder_file', 'treatment', 'cell_type', 'cell_subtype', 'I_set', 'region', 'error', 'traceback'] #region and I_set are project specific this need to be generalised

        # Determine which dependent variables exist for this data_type
        valid_dvs_for_data_type = [col for col in data_type_df.columns if col not in independant_vairables]

        # Filter only valid folder_files
        filtered_df = data_type_df[data_type_df['folder_file'].isin(self.valid_files)].copy()
        
        if  self.project_obj.project_type == "intrinsic_properties":
            # Intrinsic properties: no time, just keep cell_id, folder_file, and dependent variables
            cols_to_keep = ['cell_id', 'folder_file'] + valid_dvs_for_data_type
            filtered_df = filtered_df[cols_to_keep]

            agg_df = self.add_cell_mapping(filtered_df)
            return agg_df

        
        # time required --> self.project_obj.project_type == "application": generalise #TODO
        elif self.data_type == 'APP':
            filtered_df = self.APP_df[self.APP_df['folder_file'].isin(self.valid_files)].copy()
            timepoints = ['PRE', 'APP', 'WASH']
            sweep_vars = ['AP_count', 'RA_count', 'SAP_count', 'RMP'] #, 'inputR']
            reshaped_data = []

            for timepoint in timepoints:
                current_data = pd.DataFrame()
                current_data['cell_id'] = filtered_df['cell_id']
                current_data['folder_file'] = filtered_df['folder_file']
                current_data['time'] = timepoint

                for base in sweep_vars:
                    sweep_col = f'sweep_{base}'
                    if sweep_col not in filtered_df.columns:
                        continue

                    # Filter out invalid types (not list/array)
                    mask_invalid = filtered_df[sweep_col].apply(lambda x: not isinstance(x, (list, np.ndarray)))
                    dropped_cells = filtered_df.loc[mask_invalid, 'cell_id'].unique()
                    if len(dropped_cells) > 0:
                        print(f"Dropping cells with invalid {sweep_col}: {dropped_cells}")
                    valid_df = filtered_df.loc[~mask_invalid].copy()

                    # Align to time window
                    def extract_time_segment(row):
                        sweep_data = row[sweep_col]
                        if timepoint == 'PRE':
                            return sweep_data[:int(row.get('drug_in', 0))]
                        elif timepoint == 'APP':
                            return sweep_data[int(row.get('drug_in', 0)):int(row.get('drug_out', len(sweep_data)))]
                        elif timepoint == 'WASH':
                            return sweep_data[int(row.get('drug_out', len(sweep_data))):]
                        return []

                    # Apply extraction + mean aggregation
                    extracted = valid_df.apply(extract_time_segment, axis=1)
                    current_data[sweep_col] = extracted
                    current_data[base] = extracted.apply(lambda x: np.nanmean(x) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.nan)

                reshaped_data.append(current_data)

            agg_APP_df = pd.concat(reshaped_data, ignore_index=True)
            agg_APP_df = self.add_cell_mapping(agg_APP_df, additional_cols=['I_set'])
            return agg_APP_df
        

        elif self.data_type == 'FP':
            filtered_df = self.FP_df[self.FP_df['folder_file'].isin(self.valid_files)].copy()
            filtered_df['time'] = filtered_df['treatment'].apply(lambda x: 'WASH' if x != 'PRE' else 'PRE')
            #columns to keep
            filtered_df = filtered_df[['cell_id', 'time', 'AP_decay_dvdt', 'AP_rise_dvdt',
                        'AP_dvdt_max', 'AP_height', 'AP_latency', 'AP_peak_voltages',
                        'AP_width', 'FI_slope', 'max_firing', 
                        'rheobased_threshold', 'sag', 'voltage_threshold']]
            #aggregate 
            for col in ['AP_dvdt_max', 'AP_height', 'AP_latency', 'AP_peak_voltages',
                        'AP_decay_dvdt', 'AP_rise_dvdt', 'AP_width', 'sag', 'voltage_threshold']:
                filtered_df[col] = filtered_df[col].apply(lambda x: np.mean(x) if isinstance(x, list) else x)
            agg_FP_df = filtered_df.groupby(['cell_id', 'time']).agg({
            'AP_dvdt_max': 'mean',
            'AP_height': 'mean',
            'AP_latency': 'mean',
            'AP_peak_voltages': 'mean',  #averaging catch now in histogram if it helps ?
            'AP_decay_dvdt': 'mean',
            'AP_rise_dvdt': 'mean',
            'AP_width': 'mean',
            'FI_slope': 'mean',
            'max_firing': 'mean',
            'rheobased_threshold': 'mean',
            'sag': 'mean',
            'voltage_threshold': 'mean'
            }).reset_index()
            agg_FP_df = self.add_cell_mapping(agg_FP_df, additional_cols=['I_set'])
            return agg_FP_df
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

    def add_cell_mapping(self, df, additional_cols: list = None):
        """
        Adds cell feature columns based off cell_id in cell_df.

        Parameters:
            df (pd.DataFrame): DataFrame to merge with cell_df features.
            additional_cols (list, optional): List of extra columns to include from cell_df.

        Returns:
            pd.DataFrame: Merged DataFrame with added features.
        """
        if 'cell_id' not in df.columns or 'cell_id' not in self.cell_df.columns:
            raise ValueError("Both DataFrames must have 'cell_id' column.")

        # Always include these base columns
        columns_to_map = ['cell_id', 'treatment', 'cell_type', 'cell_subtype', 'region', 'sex']

        # Add any extra columns specified by the caller
        if additional_cols:
            for col in additional_cols:
                if col in self.cell_df.columns and col not in columns_to_map:
                    columns_to_map.append(col)

        # Ensure all columns exist in cell_df
        for column in columns_to_map:
            if column not in self.cell_df.columns:
                raise ValueError(f"Column '{column}' is missing from cell_df.")

        return df.merge(self.cell_df[columns_to_map].drop_duplicates(), on='cell_id', how='left')

    


    def generate_treatment_count_df(self) -> pd.DataFrame:
        '''
        Calculates the n for each treatment x cell_type given the threshold_access_change, saved as excel in cache.
        TODO: 
        '''

        def is_valid_app(app_valid):
            return isinstance(app_valid, str) and len(app_valid) > 0
        def is_valid_fp(fp_valid):
            return isinstance(fp_valid, list) and all(isinstance(x, str) for x in fp_valid)
        def process_group(group_df):
            valid_fp = group_df[group_df['FP_folder_files'].apply(is_valid_fp)]
            valid_app = group_df[group_df['APP_folder_files'].apply(is_valid_app)]

            fp_count = valid_fp['cell_id'].nunique()
            app_count = valid_app['cell_id'].nunique()
            both_valid_count = group_df[
                group_df['FP_folder_files'].apply(is_valid_fp) & group_df['APP_folder_files'].apply(is_valid_app)
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
            access_filtered_df = self.cell_df[self.cell_df['Rs_pct_change'].abs() <= self.threshold_access_change]
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

        
    def folder_file_AP_df(self, cell_id, folder_file,  V_array=None, I_array=None): 
        '''builds AP_df for single folder_file.'''
        if V_array is None or I_array is None:
            V_array , I_array, V_list = Project(self.project).load_data(folder_file)
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
        drug_used = row_info['treatment'].iloc[0]

        if file_data_type == 'APP':
            drug_in =  row_info['drug_in'].iloc[0]
            drug_out = row_info['drug_out'].iloc[0]
            drug_labels = [ 'PRE' if _ < drug_in else 'APP' if drug_in <= _ <= drug_out else 'WASH' for _ in sweep_indices_all ]
            current_injected = [I_array[loc, 0] for loc in peak_locs_corr_all]

        if file_data_type == 'FP':
            drug_labels = [row_info['treatment'].iloc[0] if row_info['treatment'].iloc[0] == 'PRE' else 'WASH' for _ in sweep_indices_all]
            current_injected = [I_array[loc, sweep] for loc, sweep in zip(peak_locs_corr_all, sweep_indices_all)]


        AP_df = pd.DataFrame({
        'folder_file': folder_file,
        'cell_id': cell_id,
        'data_type': file_data_type,
        'cell_treatment': drug_used,
        'treatment': drug_labels,
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
            FP_cell_id_PRE = self.FP_df[(self.FP_df['cell_id'] == cell_id) & (self.FP_df['treatment'] == 'PRE')]
            cell_threshold = (FP_cell_id_PRE['voltage_threshold'].apply(lambda x: sum(x) / len(x) if isinstance(x, list) else x)).mean()

            # valid_FP_folder_files = self.cell_df[self.cell_df['cell_id']==cell_id]['FP_valid'].values[0][:2]
            # mean_voltage_threshold = self.FP_df[self.FP_df['folder_file'].isin(valid_FP_folder_files)]['voltage_threshold'].explode().astype(float).mean()

            AP_df.loc[(AP_df['voltage_threshold'] < cell_threshold-20 ), 'AP_type'] = 'RA'
        except(IndexError, TypeError):
            print(f" Cell {cell_id} has no valid FP to assess voltage threshold, setting RMP<-65mV")
            AP_df.loc[(AP_df['voltage_threshold'] < -65) , 'AP_type'] = 'RA'

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

    def filter_n_minimum(self, df):
        """
        Filters a dataframe to enforce a minimum number of samples per group.

        If 'time' column exists, groups by ['treatment', 'time'].
        Otherwise, groups only by ['treatment'].
        """
        df = df.dropna(subset=[self.dependant_var]).reset_index(drop=True)

        group_cols = ['treatment']
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

    # def filter_n_minimum(self,df): #old 18Sept2025
    #     df = df.dropna(subset=[self.dependant_var]).reset_index(drop=True)
    #     group_sizes = df.groupby(['treatment', 'time']).size()
    #     insufficient_groups = group_sizes[group_sizes < self.n_minimum]
    #     if not insufficient_groups.empty:
    #         print(f"Warning: The following groups have less than {self.n_minimum} samples and will be excluded:")
    #         print(insufficient_groups)
    #         df = df[~df[['treatment', 'time']].apply(tuple, axis=1).isin(insufficient_groups.index)].reset_index(drop=True)
    #     if df.empty:
    #         print("No groups meet the minimum sample size requirement. Statistical analysis will not be performed.")
    #         return None
    #     else:
    #         return df
    
    def check_valid_dependant_var(self):
        if self.dependant_var not in self.agg_df.columns:
            dvs = [col for col in self.agg_df.columns if col not in ['cell_id', 'time', 'treatment', 'cell_type', 'cell_subtype', 'I_set']]#HARD CODE
            raise ValueError(f"Invalid dependant variable: {self.dependant_var}. Valid dv's : {dvs}")
            
    def get_pre_post_sweep_windows(self,
        df: pd.DataFrame,
        dependant_var: str,
        pre_sweep_window: int | None = None,
        post_sweep_window: int | None = None,
        verbose: bool = True
        ):
        """
        Determine pre and post sweep windows and filter df to ensure each cell has sufficient sweeps.
        If pre or post sweep window is not provided, the minimum available for all cells is used.
        A cell is kept only if it has >= pre_sweep_window PRE sweeps and >= post_sweep_window (APP + WASH) sweeps.
        """
        filtered_df = df.copy()

        # Calculate default pre_sweep_window
        if pre_sweep_window is None:
            pre_sweep_window = (
                filtered_df.loc[filtered_df["time"] == "PRE", dependant_var]
                .map(len)
                .groupby(filtered_df["cell_id"])
                .min()
                .min()
            )
            if verbose:
                print(f"pre_sweep_window set to {pre_sweep_window}")

        #  cells with enough PRE data
        pre_counts = (
            filtered_df.loc[filtered_df["time"] == "PRE"]
            .groupby("cell_id")[dependant_var]
            .apply(lambda x: x.map(len).sum())
        )
        sufficient_pre_cells = pre_counts[pre_counts >= pre_sweep_window].index.tolist()

        # Calculate default post_sweep_window
        if post_sweep_window is None:
            def get_total_post_sweeps(group):
                return group.loc[group["time"].isin(["APP", "WASH"]), dependant_var].map(len).sum()

            post_counts = filtered_df.groupby("cell_id").apply(get_total_post_sweeps)
            post_sweep_window = post_counts.min()
            if verbose:
                print(f"post_sweep_window default calculated: {post_sweep_window}")

        #  cells with enough POST data 
        def has_enough_post(group):
            return group.loc[group["time"].isin(["APP", "WASH"]), dependant_var].map(len).sum()

        post_counts = filtered_df.groupby("cell_id").apply(has_enough_post)
        sufficient_post_cells = post_counts[post_counts >= post_sweep_window].index.tolist()

        # Intersect both criteria 
        valid_cells = sorted(set(sufficient_pre_cells) & set(sufficient_post_cells))

        # Verbose output for dropped cells 
        all_cells = filtered_df["cell_id"].unique().tolist()
        dropped_cells = sorted(set(all_cells) - set(valid_cells))
        if verbose and dropped_cells:
            print(f"Dropping {len(dropped_cells)} cells due to insufficient PRE or POST sweeps: {dropped_cells}")

        filtered_df = filtered_df[filtered_df["cell_id"].isin(valid_cells)]

        return filtered_df, pre_sweep_window, post_sweep_window

    def build_pre_post_df(self, df: Optional[pd.DataFrame] = None, slice: bool = True) -> pd.DataFrame:
        '''
        Builds a DataFrame with PRE and POST sweeps for each cell_id.

        Parameters:
            df (pd.DataFrame, optional): DataFrame to process. If None, uses self.data.
            slice (bool, optional): If True, slices to uniform pre/post sweep window sizes. 
                                    If False, uses full APP+WASH as POST, but still slices PRE.

        Returns: 
            pd.DataFrame with columns: cell_id, PRE_sweeps, POST_sweeps
        '''
        if df is None:
            df = self.data

        if slice:
            df, self.pre_sweep_window, self.post_sweep_window = self.get_pre_post_sweep_windows(
                df, 
                dependant_var=self.dependant_var, 
                pre_sweep_window=self.pre_sweep_window, 
                post_sweep_window=self.post_sweep_window
            )
        else:
            _, self.pre_sweep_window, self.post_sweep_window = self.get_pre_post_sweep_windows( #return but dont use filtered_df
                df, 
                dependant_var=self.dependant_var, 
                pre_sweep_window=self.pre_sweep_window, 
                post_sweep_window=self.post_sweep_window
            )

        rows = []
        for cell_id, sub_df in df.groupby('cell_id'):
            try:
                pre_vals = sub_df[sub_df['time'] == 'PRE'][self.dependant_var].values[0]
                app_vals = sub_df[sub_df['time'] == 'APP'][self.dependant_var].values[0]
                wash_vals = sub_df[sub_df['time'] == 'WASH'][self.dependant_var].values[0]
            except IndexError:
                print(f"Skipping cell {cell_id} due to missing timepoints.")
                continue

            pre = pre_vals[-self.pre_sweep_window:] if self.pre_sweep_window is not None else pre_vals

            if slice:
                post = app_vals[:self.post_sweep_window]
                if len(post) < self.post_sweep_window:
                    post = np.concatenate([post, wash_vals[:self.post_sweep_window - len(post)]])
            else:
                post = np.concatenate([app_vals, wash_vals])

            if len(pre) == 0 or len(post) == 0:
                print(f"Skipping cell {cell_id} due to empty PRE or POST sweeps.")
                continue

            rows.append({
                'cell_id': cell_id,
                'dependant_var': self.dependant_var,
                'PRE_sweeps': pre,
                'POST_sweeps': post,
            })

        return pd.DataFrame(rows)



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
            end_sweep = group[-1]['range_sweeps'][1] + 1

            # Deltas
            deltas = [g['delta'] for g in group]
            mean_delta = np.mean(deltas)
            max_delta = np.max(deltas)

            # Handle p-values
            p_vals = [g['p_val'] for g in group if g['p_val'] is not None]
            if p_vals:
                p_val = np.min(p_vals)
                # Latency is first bin with p ≤ threshold
                latency_sweep = next(
                    (g['range_sweeps'][0] for g in group if g['p_val'] is not None and g['p_val'] <= self.p_thresh),
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
                'mean_delta': mean_delta,
                'max_delta': max_delta,
                'p_val': p_val,
                'latency_sweeps': latency_sweep,
                'range_sweeps': (start_sweep, end_sweep)
            })

        return grouped




    def get_responses(self, df) -> pd.DataFrame:  #take in pre_post_bins filtered or not but df with just cell_id pre post
        result_rows = []

        for _, row in df.iterrows():
            pre = row['PRE_sweeps']
            post = row['POST_sweeps']
            cell_id = row['cell_id']

            if self.dynamic_search:
                bin_results = self.analyze_post_bins(pre, post, cell_id) #sliding window analysis, t-test and filter on diff_threshold
                grouped = self.group_consecutive_responses(bin_results)
                if grouped:
                    result_rows.extend(grouped)
                else:
                    # print(f"No significant bins found for {cell_id}")
                    result_rows.append({
                        'cell_id': cell_id,
                        'dependant_var': self.dependant_var,
                        'PRE_sweeps': pre,
                        'POST_sweeps': post, 
                        'response': 'no response',
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
                        'mean_diff': mean_diff,
                        'p_val': np.nan
                    }
                else:
                    result = Stats(
                        p_thresh=self.p_thresh,
                        diff_thresh=self.diff_thresh
                    ).welchs_t_test(pre, post)

                    result_rows.append({
                        'cell_id': cell_id,
                        'dependant_var': self.dependant_var,
                        'PRE_sweeps': pre,
                        'POST_sweeps': post,
                        'response': result['response'],
                        'delta': result['mean_diff'] if self.dependant_var != 'sweep_inputR' else result['percent_diff'],
                        'p_val': result['p_val'],
                        'latency_sweeps': 0,
                        'range_sweeps': (0, len(post))
                    })

        return pd.DataFrame(result_rows)
    
    def get_unit(self): 
        if self.dependant_var == 'sweep_inputR':
            return '%'
        elif self.dependant_var == 'sweep_RMP':
            return 'mV'
        else:
            return 'Hz'
        
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
                    'mean_diff': mean_diff,
                    'p_val': None
                }

            else:
                result = Stats(
                    p_thresh=self.p_thresh,
                    diff_thresh=self.diff_thresh,
                    percentage_threshold= False if self.dependant_var != 'sweep_inputR' else True
                ).welchs_t_test(pre, bin_post)

            results.append({
                'cell_id': cell_id,
                'PRE_sweeps': pre,
                'POST_sweeps': post,
                'response': result['response'],
                'delta':result['mean_diff'] if self.dependant_var != 'sweep_inputR' else result['percent_diff'],
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
   

@dataclass
class ResponseCharecterisation(Figure):
    '''Extends cell_df with response charecterisation of multiple dependant vairables (dependant_vars) and corresponding thresholds (diff_thresholds)
    '''
    filename: str = None
    n_minimum: float = field(kw_only = True, default = 3)
    pre_sweep_window: int = None # window before and after drug_in
    post_sweep_window: int = None 

    diff_threshs: list[int] = field(kw_only=True) # ie 3mV RMP 0 AP_count
    diff_thresh: int = field(kw_only = True, default = 0) # ie 3mV difference required to consider it a response 

    dependant_vars: list[str] = field(kw_only=True)
    dependant_var: str = field(init=False)  # will be set in __post_init__

    p_thresh: float = field(kw_only = True, default = 0.05)
    dynamic_search: bool = field(kw_only=True, default=False)
    bin_width: int = field(kw_only=True, default=3)



    def __post_init__(self):
        self.filename = f"{self.dependant_vars}_{self.cell_type}_{self.treatment}_response"
        self.slice = False if self.dynamic_search else True 
        
        super().__post_init__()
        self.response_df = self.aggregate_cell_responses() #long format of significant bins for each dependant variable
        self.response_cell_df = self.build_responder_cell_df() # added columns to cell_df for cell_ids in response_df
        self.plot_functional_response_pie()

    def aggregate_cell_responses(self):
        dv_dfs = []
        for dependant_var, diff_thresh in zip(self.dependant_vars, self.diff_threshs):
            self.dependant_var = f"sweep_{dependant_var}"
            self.diff_thresh = diff_thresh

            data = self.filter_n_minimum(self.agg_df)
            pre_post_df = self.build_pre_post_df(data, slice=self.slice)
            dv_response_df = self.get_responses(pre_post_df)
            dv_dfs.append(dv_response_df)

        return pd.concat(dv_dfs, ignore_index=True)

    def build_responder_cell_df(self):
        """
        Returns a subset of self.cell_df containing only cell_ids present in self.response_df,
        with one response column and one latency column per dependant variable,
        plus a boolean 'responder' column and a global 'response' summary column.
        """
        included_cells = self.response_df['cell_id'].unique()
        temp_cell_df = self.cell_df[self.cell_df['cell_id'].isin(included_cells)].copy()

        valid_responses = {'increase', 'decrease', 'biphasic'}

        for dv in self.dependant_vars:
            full_var_name = f"sweep_{dv}"
            sub_df = self.response_df[self.response_df['dependant_var'] == full_var_name].copy()

            # Lowercase all responses
            sub_df['response'] = sub_df['response'].str.lower()

            # Fill response per cell_id
            response_series = sub_df.groupby('cell_id')['response'].apply(
                lambda resps: (
                    'biphasic' if {'increase', 'decrease'}.issubset(set(resps.dropna()))
                    else resps.dropna().unique()[0] if len(resps.dropna().unique()) >= 1
                    else 'no response'
                )
            ).reindex(temp_cell_df['cell_id'])

            temp_cell_df[dv] = response_series.values

            # Latency (min latency_sweeps for valid responses)
            latency_series = sub_df[~sub_df['response'].isin(['no response', np.nan])].groupby('cell_id')['latency_sweeps'].min()
            latency_series = latency_series.reindex(temp_cell_df['cell_id'])
            temp_cell_df[f"{dv}_latency"] = latency_series.values

        # Responder: True if any DV is not 'no response' or nan
        def is_responder(row):
            return any(val in valid_responses for val in row[self.dependant_vars])
        
        temp_cell_df['responder'] = temp_cell_df.apply(is_responder, axis=1)

        def categorize_response_multiple_dvs(row, dvs=('RMP', 'AP_count')):
            values = {row[dv].lower().strip() for dv in dvs}
            if 'increase' in values and 'decrease' in values:
                return 'mixed'
            if 'biphasic' in values:
                return 'mixed'
            if values <= {'increase', 'no response'}:
                return 'excitatory'
            if values <= {'decrease', 'no response'}:
                return 'inhibitory'
            return 'no response'

        temp_cell_df['response'] = temp_cell_df.apply(
            lambda row: categorize_response_multiple_dvs(row) if row['responder'] else np.nan, axis=1
        )
        return temp_cell_df



    def plot_functional_response_pie(self):
        """
        Plots a pie chart showing the distribution of functional response types
        ('excitatory', 'inhibitory', 'mixed', 'no response') from self.response_cell_df
        """
        response_series = self.response_cell_df['response'].fillna('no response').str.lower()
        counts = response_series.value_counts()
        total_n = len(response_series)

        # Define colors
        colors = {
            'excitatory': 'salmon',
            'inhibitory': 'deepskyblue',
            'mixed': 'mediumorchid',
            'no response': 'whitesmoke'
        }
        pie_colors = [colors.get(label, 'gray') for label in counts.index]

        # Print cell IDs for each category
        for label in ['excitatory', 'inhibitory', 'mixed', 'no response']:
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

        ax.set_title(f"{self.cell_type} response to {self.treatment} \n(n={total_n})", fontsize=14)

        fig.text(0.5, 0.06, f"Significant response > {self.dependant_vars} for {self.diff_threshs}", 
                ha='center', fontsize=10, style='italic')

        fig.text(0.5, 0.02, 
                f"PRE sweep window: {self.pre_sweep_window}   |   POST sweep window: {None if self.dynamic_search else self.post_sweep_window}",
                ha='center', fontsize=10, style='italic')

        plt.tight_layout()
        plt.show()
        self.save_plot(fig, self.filename + '_functional_pie')



    
@dataclass
class ApplicationResponse(Figure):
    filename: str = None
    response_var: str = field(kw_only=True)
    n_minimum: float = field(kw_only = True, default = 3)
    pre_sweep_window: int = None # window before and after drug_in
    post_sweep_window: int = None 
    diff_thresh: int = field(kw_only = True, default = 0) # ie 3mV difference required to consider it a response 

    dependant_var: str = field(init=False)  # will be set in __post_init__
    p_thresh: float = field(kw_only = True, default = 0.05)
    dynamic_search: bool = field(kw_only=True, default=False)
    bin_width: int = field(kw_only=True, default=3)

    def __post_init__(self):
        self.dependant_var = f"sweep_{self.response_var}"
        self.filename = f"{self.dependant_var}_{self.cell_type}_{self.treatment}_response"
        self.slice = True if self.dynamic_search else False 
        
        super().__post_init__()
        self.check_valid_dependant_var()
        self.data = self.filter_n_minimum(self.agg_df)
        self.pre_post_df = self.build_pre_post_df(slice=self.slice) #update hist after TODO SLICE PRE but not post! when slice=False
        self.response_df = self.get_responses(self.pre_post_df)
        self.fig = self.plot_pie()
    
    def plot_pie(self):
        def collapse_responses(responses):
            """
            Collapse multiple responses for a single dependant vairable per cell_id to a single string: biphasic, increase, decrease or no response.
            """
            cleaned = responses.dropna().astype(str).str.strip().unique()
            cleaned = [resp for resp in cleaned if resp.lower() != 'no response']

            if {'increase', 'decrease'}.issubset(set(cleaned)):
                return 'biphasic'
            elif len(cleaned) == 1:
                return cleaned[0]
            elif len(cleaned) > 1:
                return cleaned[-1]  # Use last non-'no response' value
            else:
                return 'no response'

        # self.response_df['response'] = self.response_df['response'].astype(str).str.strip() #MOVE LATER


        #Collapse to one row per cell_id
        cell_responses = self.response_df.groupby('cell_id')['response'].apply(collapse_responses).reset_index()

        # cell_responses = self.response_df.groupby('cell_id')['response'].apply(
        #     lambda responses: (
        #         'biphasic' if {'increase', 'decrease'}.issubset(set(responses.dropna()))
        #         else responses.dropna().unique()[0] if len(responses.dropna().unique()) == 1
        #         else responses.dropna().unique()[0] if len(responses.dropna().unique()) > 0
        #         else 'no response'
        #     )
        # ).reset_index()

        counts = cell_responses['response'].value_counts()
        total_n = len(cell_responses)

        colors = {
            'increase': 'salmon',
            'decrease': 'deepskyblue',
            'biphasic': 'mediumorchid',
            'no response': 'whitesmoke'
        }
        pie_colors = [colors.get(label, 'gray') for label in counts.index]

        # Print cell IDs by response label
        for label in ['increase', 'decrease', 'biphasic', 'no response']:
            matching_cells = cell_responses[cell_responses['response'] == label]['cell_id'].tolist()
            print(f"{label} cells ({len(matching_cells)}): {matching_cells}")

        def format_autopct(pct):
            count = int(round(pct * total_n / 100.0))
            return f'{pct:.1f}%\n({count}/{total_n})'

        fig, ax = plt.subplots(figsize=(6, 6))
        counts.plot.pie(
            autopct=format_autopct,
            startangle=90,
            ylabel='',
            textprops={'fontsize': 12},
            colors=pie_colors,
            ax=ax
        )

        ax.set_title(f"{self.response_var} response to {self.treatment} in {self.cell_type}\n(n={total_n})", fontsize=14)

        
        unit = self.get_unit()
        fig.text(0.5, 0.06, f"Significant response > {self.diff_thresh} {unit}", 
                ha='center', fontsize=10, style='italic')
        fig.text(0.5, 0.02, f"PRE sweep window: {self.pre_sweep_window}   |   POST sweep window: {None if self.dynamic_search else self.post_sweep_window}",
                ha='center', fontsize=10, style='italic')

        plt.tight_layout()
        plt.show()
        self.save_plot(fig, self.filename)


        


@dataclass
class Histogram(Figure):
    '''
    Generic histogram class for plotting histograms of a specified dependant variable across treatments (and timepoints if project == application).
    '''
    filename: str = None
    dependant_var: str = field(kw_only=True)
    specify: str = field(kw_only = True, default = 'treatment') # specify marker to see subsets e.g. I_set or cell_id
    n_minimum: float = field(kw_only = True, default = 3)

    #for project_type ==  application 
    pre_sweep_window: int = None # window before and after drug_in
    post_sweep_window: int = None 

    def __post_init__(self):
        self.filename = f"{self.dependant_var}_{self.specify}" 
        super().__post_init__()
        self.check_valid_dependant_var()
        self.data = self.filter_n_minimum(self.agg_df)

        # If dependant_var contains lists or arrays, average them to a single numeric value
        self.data[self.dependant_var] = self.data[self.dependant_var].apply(
            lambda x: np.mean(x) if isinstance(x, (list, np.ndarray, pd.Series)) else x
        )
        if self.data_type == "APP":
            self.data, pre_sweep_window, post_sweep_window = self.get_pre_post_sweep_windows(self.data, dependant_var=self.dependant_var, pre_sweep_window=self.pre_sweep_window, post_sweep_window=self.post_sweep_window)
        self.order = [t for t in color_dict.keys() if t in self.data['treatment'].unique()]

        if  self.project_obj.project_type == "application":
            # self.stats = self.generate_statistics() #DEPRICATED AND NOT GENERIC #TODO
            self.hue_order = [t for t in ['PRE', 'APP', 'WASH'] if t in self.data['time'].unique()] #not generic #TODO
        
        self.fig = self.plot_histogram()
        
    # def generate_statistics(self): DEPRICATED AND NOT GENERIC
    #     #mixed effects models --> Tukey
    #     model = mixedlm(f"{self.dependant_var} ~ time * treatment", self.data, groups=self.data["cell_id"])
    #     result = model.fit()
    #     interaction_pvalues = {term: result.pvalues[term] for term in result.pvalues.keys() if 'time' in term and 'treatment' in term}
    #     significant_interactions = {term: pval for term, pval in interaction_pvalues.items() if pval < 0.05}

    #     if significant_interactions:
    #         print(f"significant interaction/s: {significant_interactions}, performingm tukey post hoc.")
    #         tukey = pairwise_tukeyhsd(endog=self.data[self.dependant_var], groups=self.data['time'] + self.data['treatment'], alpha=0.05)
    #         significant_pairs = []
    #         for row in tukey.summary().data[1:] :
    #             reject = row[-1]  # The last column indicates whether the null hypothesis was rejected
    #             if reject == 'True':  #  if the comparison is significant
    #                 group1, group2 = row[0], row[1]
    #                 significant_pairs.append((group1, group2))
    #                 return significant_pairs
    #     else:
    #         print("No significant interaction found.")
    #         return None


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
                hue='time' if self.project_obj.project_type == "application" else 'treatment',
                hue_order=self.hue_order if hasattr(self, 'hue_order') else None,
                order=self.order ,
                data=subset_to_plot,
                palette=color_dict,
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
            hue='time' if self.project_obj.project_type == "application" else 'treatment',
            hue_order=self.hue_order if hasattr(self, 'hue_order') else None,
            order=self.order ,
            data=df,
            errorbar = 'sd',
            palette=color_dict,
            edgecolor="k",
            ax=ax
        )
        sns.swarmplot(
            x='treatment',
            y=self.dependant_var,
            hue='time' if self.project_obj.project_type == "application" else 'treatment',
            hue_order=self.hue_order if hasattr(self, 'hue_order') else None,
            order=self.order ,
            data=df,
            palette=color_dict,
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


        counts = df.groupby('treatment')['cell_id'].nunique() #count unique cells per treatment #TODO add mouse count 
        for tick, treatment in enumerate(self.order):
            count = counts.get(treatment, 0)
            ax.text(tick, -0.1, f'n={count}', ha='center', va='top', fontsize=24, color='black', transform=ax.get_xaxis_transform())

        # Customize plot labels and titles
        ax.set_ylabel(unit_dict[self.dependant_var], fontsize=24)
        ax.set_xlabel('')
        ax.set_title(f'{self.cell_type if self.cell_type is not None else ""} {unit_dict[self.dependant_var]}', fontsize=28)
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
        ax.set_title(f'{self.cell_type} {self.dependant_var} Applications')
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
    cell_id: str|list = field(kw_only = True, default = None) # optional pram for plotting specific cell/s application
    plot_all_APs: bool = field(kw_only=True, default=False)
    valid_only: bool = field(kw_only=True, default=False)
    pre_window_sweeps: int = field(kw_only=True, default=None)


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

            for folder_file, cell_id, I_set, drug, drug_in, drug_out, application_order, RA_locs in cell_sub_df[['folder_file','cell_id', 'I_set', 'treatment', 'drug_in', 'drug_out', 'application_order', 'RA_locs']].values:
                self.fig_filename = f"{cell_id}_application{application_order}"
                
                V_array , I_array, V_list = Project(self.project).load_data(folder_file)
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
                ax1.axvspan((int((drug_in)* seconds_per_sweep) - seconds_per_sweep), (int(drug_out)* seconds_per_sweep), facecolor = "grey", alpha = 0.3) #drug bar shows start of drug_in sweep to end of drug_out sweep 
                if self.pre_window_sweeps is not None:
                    pre_start = (int(drug_in) - self.pre_window_sweeps) * seconds_per_sweep
                    pre_end = int(drug_in) * seconds_per_sweep
                    ax1.axvspan(
                        pre_start,
                        pre_end,
                        facecolor="lightgrey",
                        alpha=0.4
                    )
                
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
            valid_folder_files = self.cell_df[self.cell_df['cell_id'] == self.cell_id]['APP_folder_files'].iloc[0]
            if valid_folder_files is None: 
                 print(f"No valid APP files for cell_id {self.cell_id}, skipping.")
            else:
                agg_folder_files.append(valid_folder_files)

        if 'FP' in self.data_types:
            valid_folder_files = self.cell_df[self.cell_df['cell_id'] == self.cell_id]['FP_folder_files'].iloc[0]

            if valid_folder_files is None:
                 print(f"No valid FP files for cell_id {self.cell_id}, skipping.")
            else:
                agg_folder_files.extend(valid_folder_files)
                
        for folder_file in agg_folder_files: #TOD add drug presence to AP_df
            #fetch raw trace
            V_array , I_array, V_list = Project(self.project).load_data(folder_file)
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
            V_array, I_array, _ = Project(self.project).load_data(folder_file)
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
            V_array, I_array, _ = Project(self.project).load_data(folder_file)

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


