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
from scipy.stats import ttest_ind
import seaborn as sns
from typing import Optional
# from module.utils import  subselectDf, saveFigure, getCache, isCached, cache, cache_excel #should become Cashable class
from module.constants import CACHE_DIR, color_dict, unit_dict
# from module.Ephys import Ephys, APP, FP, EphysData # I THINK THIS IS OLD?
from module.Ephys_Project import Ephys, APP_IC, Project, IF_IC
from module.Cachable import Cachable
from collections import defaultdict
#Readapting
from module.action_potential_functions import ap_characteristics_extractor_main, normalise_array_length #should become ActionPotential class
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
    behaviour: str | list  = field(kw_only=True, default=None)
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
        self.agg_df = self.build_agg_df() #validates dv
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
        data_types=['APP_IC', 'st_VC', 'ramp_IC', 'IV_VC', 'spont_IC', 'IF_IC', 'PPR_VC' ]
        if self.data_type not in data_types: #complete list of data types
            raise ValueError(f"Invalid data_type: {self.data_type}. Must be one of {data_types}.")
        
        #handel subgroup prefix
        folder_file_cols = [c for c in self.cell_df.columns if c.endswith(f"{self.data_type}_folder_files")]
        valid_df = self.cell_df[self.cell_df[folder_file_cols].notna().any(axis=1)]

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
        # valid_column = f'{self.data_type}_folder_files'
        # if valid_column not in self.cell_df.columns:
        #     raise ValueError(f"{valid_column} column does not exist in cell_df.")
        
        # find all relevant folder_files columns for this data_type
        folder_file_cols = [c for c in self.cell_df.columns if c.endswith(f"{self.data_type}_folder_files")]
        if not folder_file_cols:
            raise ValueError(f"No columns found for {self.data_type}_folder_files in cell_df.")

        
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
        if self.behaviour is not None:
            filtered_cell_df = filtered_cell_df[filtered_cell_df['behaviour'].isin([self.behaviour] if isinstance(self.behaviour, str) else self.behaviour)]
       
        valid_cell_ids = filtered_cell_df['cell_id'].tolist()
        # valid_files = filtered_cell_df[folder_file_cols].dropna().tolist()
        valid_files = [
            f
            for row in filtered_cell_df[folder_file_cols].dropna().values.tolist()
            for cell in row
            for f in (cell if isinstance(cell, list) else [cell])
        ]

        if not valid_files:
            print("No valid files found for data selection.")
            return [],[]
        # valid_files = [item for sublist in valid_files for item in sublist] if isinstance(valid_files[0], list) else valid_files

        return valid_files, valid_cell_ids
    
    def reduce_series(s):
            vals = s.dropna()
            if len(vals) == 0:
                return np.nan
            # flatten nested lists
            if any(isinstance(v, list) for v in vals):
                out = []
                for v in vals:
                    if isinstance(v, list):
                        out.extend(v)
                    else:
                        out.append(v)
                return out
            uniq = vals.unique()
            # collapse if identical
            if len(uniq) == 1:
                return uniq[0]
            # otherwise preserve structure
            return vals.tolist()
    
    def build_agg_df(self):
        """
        Filters self.{data_type}_df for foler_files in cell_df["f{data_type}_folder_files"] and restructures it to a long format for plotting.
        
        Returns:
          agg_df aggregate df for data_type for stats and plotting (one row per cell_id and time).

        """
        data_type_df = getattr(self, f"{self.data_type}_df")

        
        independant_vairables = ['cell_id', 'folder_file', 'treatment', 'cell_type', 'cell_subtype', 'I_set', 'region', 'error', 'traceback', 'sex', 'behaviour', 'subject_id'] # I_set are project specific this need to be generalised

        # Determine which dependent variables exist for this data_type
        valid_dvs_for_data_type = [col for col in data_type_df.columns if col not in independant_vairables]

        # Filter only valid folder_files
        filtered_df = data_type_df[data_type_df['folder_file'].isin(self.valid_files)].copy()

        # DATA TYPE SPECIFIC AGGREGATION #
        if self.data_type == "PPR_VC": # combine all sweeps per cell_id + ISI + treatment
            rows = []
            for keys, sub in filtered_df.groupby(["cell_id", "treatment", "ISI_ms"]):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                row = dict(zip(["cell_id", "treatment", "ISI_ms"], keys))
                for col in sub.columns:
                    if col in row or col in {"error", "traceback"}:
                        continue
                    row[col] = DataSelection.reduce_series(sub[col])
                row["folder_files"] = sub["folder_file"].tolist()
                if "PPR" in row and isinstance(row["PPR"], list): # seperate raw values and average first 8
                    row["PPR_raw"] = row["PPR"]
                    # row["PPR"] = row["PPR"][:8] if len(row["PPR"]) >= 8 else np.nan # HARD CODE
                    row["PPR"] = np.nanmean(row["PPR"][:8]) if len(row["PPR"]) >= 6 else np.nan
                rows.append(row)
            filtered_df = pd.DataFrame(rows)

        # ACCOUNTING FOR TWO PROJECT TYPES #
        if  self.project_obj.project_type == "intrinsic_properties":
            # Intrinsic properties: no time, just keep cell_id, folder_file, and dependent variables
            cols_to_keep = ['cell_id', 'folder_file'] + valid_dvs_for_data_type
            filtered_df = filtered_df[cols_to_keep]

            agg_df = self.add_cell_mapping(filtered_df)
            return agg_df

        elif self.project_obj.project_type == "application": # all data_type files other than APP_IC will be PRE and POST

            # should become a generic for loop for each data_type PRE and POST 
            if self.data_type == 'IF_IC': 
                filtered_df = self.IF_IC_df[self.IF_IC_df['folder_file'].isin(self.valid_files)].copy()
                filtered_df['time'] = filtered_df['treatment'].apply(lambda x: 'WASH' if x != 'PRE' else 'PRE')


                #aggregate appropriate cols ie not "I_steps_pA" , "AP_frequencies_Hz" / 'V_step_steady_mV', 'I_steady_pA'
                for col in ['AP_max_rise_mV_ms', 'AP_height_mV', 'AP_latency_ms', 'AP_peaks_mV',
                            'AP_decay_mV_ms', 'AP_rise_mV_ms', 'AP_width_ms', '%_sag', 'IF_voltage_threshold_mV']:
                    filtered_df[col] = filtered_df[col].apply(lambda x: np.mean(x) if isinstance(x, list) else x)
                agg_IF_IC_df = filtered_df.groupby(['cell_id', 'time']).agg({
                'AP_max_rise_mV_ms': 'mean',
                'AP_height_mV': 'mean',
                'AP_latency_ms': 'mean',
                'AP_peaks_mV': 'mean',  #averaging catch now in histogram if it helps ?
                'AP_decay_mV_ms': 'mean',
                'AP_rise_mV_ms': 'mean',
                'AP_width_ms': 'mean',
                'IF_slope': 'mean',
                'max_firing_Hz': 'mean',
                'IF_rheobase_pA': 'mean',
                '%_sag': 'mean',
                'IF_voltage_threshold_mV': 'mean'
                }).reset_index()
                agg_IF_IC_df = self.add_cell_mapping(agg_IF_IC_df, additional_cols=['I_set'])
                return agg_IF_IC_df

            elif self.data_type == 'APP_IC': #
                filtered_df = self.APP_IC_df[self.APP_IC_df['folder_file'].isin(self.valid_files)].copy()
                timepoints = ['PRE', 'APP', 'WASH']
                sweep_vars = ['AP_count', 'RA_count', 'SAP_count', 'RMP_mV', 'inputR_MOhm'] 
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
                        # if len(dropped_cells) > 0:
                        #     print(f"Cells with invalid {sweep_col}: {dropped_cells} for {timepoint} set to NaN.")
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

                agg_APP_IC_df = pd.concat(reshaped_data, ignore_index=True)
                agg_APP_IC_df = self.add_cell_mapping(agg_APP_IC_df, additional_cols=['I_set'])
                return agg_APP_IC_df
            
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
        columns_to_map = ['cell_id', 'treatment', 'cell_type', 'cell_subtype', 'region', 'sex', 'subject_id', 'behaviour']

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
            valid_fp = group_df[group_df['IF_IC_folder_files'].apply(is_valid_fp)]
            valid_app = group_df[group_df['APP_IC_folder_files'].apply(is_valid_app)]

            fp_count = valid_fp['cell_id'].nunique()
            app_count = valid_app['cell_id'].nunique()
            both_valid_count = group_df[
                group_df['IF_IC_folder_files'].apply(is_valid_fp) & group_df['APP_IC_folder_files'].apply(is_valid_app)
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
        data_type_df = getattr(self, f"{self.data_type}_df")
        if folder_file in self.IF_IC_df['folder_file'].values:
            row = self.IF_IC_df[self.IF_IC_df['folder_file'] == folder_file]
            return 'IF_IC', row
        elif folder_file in self.APP_IC_df['folder_file'].values:
            row = self.APP_IC_df[self.APP_IC_df['folder_file'] == folder_file]
            return 'APP_IC', row
        else:
            return 'Unknown', None

        
    def folder_file_AP_df(self, cell_id, folder_file,  V_array=None, I_array=None): 
        '''builds action potential pd.DataFrame 'AP_df' for single folder_file.'''
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
        
        # file_data_type, row_info = self.fetch_data_type(folder_file) #check it always wors
        file_data_type = self.data_type
        row_info = self.project_obj.feature_df[self.project_obj.feature_df['cell_id'] == cell_id]

        drug_used = row_info['treatment'].iloc[0]

        if file_data_type == 'APP_IC':
            drug_in =  row_info['drug_in'].iloc[0]
            drug_out = row_info['drug_out'].iloc[0]
            drug_labels = [ 'PRE' if _ < drug_in else 'APP' if drug_in <= _ <= drug_out else 'WASH' for _ in sweep_indices_all ]
            current_injected = [I_array[loc, 0] for loc in peak_locs_corr_all]

        if file_data_type == 'IF_IC': 
            drug_labels = [row_info['treatment'].iloc[0] if row_info['treatment'].iloc[0] == 'PRE' else 'WASH' for _ in sweep_indices_all]
            current_injected = [I_array[loc, sweep] for loc, sweep in zip(peak_locs_corr_all, sweep_indices_all)]

        else:
            print(f"JASMINE GENERALISE THIS FOR ALL DATA TYEPS :) !")
            #TODO generalise for all data_types also

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
        try: # should try ramp_IC for ramp_voltage_threshold_mV first #TODO
            IF_IC_local = IF_IC(self.project_obj.project).df
            FP_cell_id_PRE = IF_IC_local[(IF_IC_local['cell_id'] == cell_id) & (IF_IC_local['treatment'] == 'PRE')]
            cell_threshold = (FP_cell_id_PRE['IF_voltage_threshold_mV'].apply(lambda x: sum(x) / len(x) if isinstance(x, list) else x)).mean()

            # valid_IF_IC_folder_files = self.cell_df[self.cell_df['cell_id']==cell_id]['FP_valid'].values[0][:2]
            # mean_voltage_threshold = IF_IC_local[IF_IC_local['folder_file'].isin(valid_IF_IC_folder_files)]['voltage_threshold'].explode().astype(float).mean()

            AP_df.loc[(AP_df['voltage_threshold'] < cell_threshold-20 ), 'AP_type'] = 'RA'
        except(IndexError, TypeError):
            print(f" Cell {cell_id} has no valid FP to assess voltage threshold, setting RMP<-65mV")
            AP_df.loc[(AP_df['voltage_threshold'] < -65) , 'AP_type'] = 'RA'

        if file_data_type == 'FP':
            AP_df_positive = AP_df[AP_df['I_injected'] > 0].iloc[:20] #HARD CODE keeping first 20 APs on + I step
            AP_df_non_positive = AP_df[AP_df['I_injected'] <= 0]
            AP_df = pd.concat([AP_df_non_positive, AP_df_positive]).sort_index().reset_index(drop=True)

        return AP_df

class MixedLMStatsMixin:
    def clean_mixedlm_df(self, df, group_col, value_col):
        needed = [group_col, value_col, "subject_id"]
        missing = [col for col in needed if col not in df.columns]
        if missing:
            raise ValueError(f"MixedLM requires missing columns: {missing}")

        model_df = df.dropna(subset=needed).copy()
        model_df[value_col] = pd.to_numeric(model_df[value_col], errors="coerce")
        model_df = model_df.dropna(subset=[value_col])

        if model_df["subject_id"].nunique() < 2:
            raise ValueError("MixedLM needs at least 2 animals in subject_id.")

        if model_df[group_col].nunique() < 2:
            raise ValueError(f"MixedLM needs at least 2 groups in {group_col}.")

        return model_df

    def fixed_effect_row(self, mixedlm_result, group_col, group_value):
        design_info = mixedlm_result.model.data.design_info
        new_df = pd.DataFrame({group_col: [group_value]})
        row = build_design_matrices([design_info], new_df)[0]
        return np.asarray(row)[0]

    def p_to_star(self, p_val):
        if p_val < 0.001:
            return "***"
        if p_val < 0.01:
            return "**"
        if p_val < 0.05:
            return "*"
        return "ns"

    def mixedlm_pairwise_stats(
        self,
        df,
        group_col=None,
        value_col=None,
        alpha=None,
        group_order=None,
        p_adjust="holm",
    ):
        if alpha is None:
            alpha = getattr(self, "alpha", 0.05)

        if group_col is None:
            group_col = self.stats_group_col

        if value_col is None:
            value_col = self.dependant_var

        model_df = self.clean_mixedlm_df(df, group_col, value_col)

        if group_order is None:
            group_order = list(model_df[group_col].dropna().unique())

        available_groups = set(model_df[group_col].dropna().unique())
        group_order = [g for g in group_order if g in available_groups]

        if len(group_order) < 2:
            raise ValueError(f"Need at least 2 valid groups for pairwise MixedLM: {group_order}")

        reference = group_order[0]
        formula = f"{value_col} ~ C({group_col}, Treatment(reference='{reference}'))"

        posthoc_model = mixedlm(
            formula,
            data=model_df,
            groups=model_df["subject_id"],
        ).fit(reml=True, method="powell")

        raw_results = []

        for g1, g2 in itertools.combinations(group_order, 2):
            row1 = self.fixed_effect_row(posthoc_model, group_col, g1)
            row2 = self.fixed_effect_row(posthoc_model, group_col, g2)

            contrast = np.asarray(row1 - row2, dtype=float)[None, :]
            test = posthoc_model.t_test(contrast)

            raw_results.append({
                "group1": g1,
                "group2": g2,
                "p_uncorrected": float(np.ravel(test.pvalue)[0]),
                "effect": float(np.ravel(test.effect)[0]),
            })

        reject, pvals_adj, _, _ = multipletests(
            [res["p_uncorrected"] for res in raw_results],
            alpha=alpha,
            method=p_adjust,
        )

        results = []
        for res, p_adj, is_sig in zip(raw_results, pvals_adj, reject):
            results.append({
                "group1": res["group1"],
                "group2": res["group2"],
                "p_val": float(p_adj),
                "p_uncorrected": res["p_uncorrected"],
                "effect": res["effect"],
                "significant": bool(is_sig),
            })
        
        # print("\n" + "=" * 60)
        # print(f"POSTHOC MIXEDLM - group_col = {group_col}, p_adjust = {p_adjust}")
        # print("=" * 60)
        # for res in results:
        #     sig = "SIGNIFICANT" if res["significant"] else "ns"
        #     print(
        #         f"{res['group1']:20s} vs {res['group2']:20s} | "
        #         f"effect = {res['effect']:.4g} | "
        #         f"p_unc = {res['p_uncorrected']:.4g} | "
        #         f"p_adj = {res['p_val']:.4g} | {sig}"
        #     )
        # print("=" * 60 + "\n")

        self.posthoc_results = results
        self.posthoc_mixedlm_result = posthoc_model

        return results
    
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
            group_cols = ['treatment'] # some classes sont have compare like ApplicationResponse

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
            dvs = [col for col in self.agg_df.columns if col not in ['cell_id', 'time', 'treatment', 'cell_type', 'cell_subtype', 'behaviour', 'I_set', 'subject_id']]#TODO centralise independant vairables
            raise ValueError(f"Invalid dependant variable: {self.dependant_var}. Valid dv's : {dvs}")
            
    def get_pre_post_sweep_windows(self,
        df: pd.DataFrame,
        dependant_var: str, #should have sweep_{dv}
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
        sweep_dv = f"sweep_{self.dependant_var}" 
        
        if df is None:
            df = self.data

        if slice:
            df, self.pre_sweep_window, self.post_sweep_window = self.get_pre_post_sweep_windows(
                df, 
                dependant_var=sweep_dv, 
                pre_sweep_window=self.pre_sweep_window, 
                post_sweep_window=self.post_sweep_window
            )
        else:
            _, self.pre_sweep_window, self.post_sweep_window = self.get_pre_post_sweep_windows( #return but dont use filtered_df
                df, 
                dependant_var=sweep_dv, 
                pre_sweep_window=self.pre_sweep_window, 
                post_sweep_window=self.post_sweep_window
            )

        rows = []
        for cell_id, sub_df in df.groupby('cell_id'):
            try:
                pre_vals = sub_df[sub_df['time'] == 'PRE'][sweep_dv].values[0]
                app_vals = sub_df[sub_df['time'] == 'APP'][sweep_dv].values[0]
                wash_vals = sub_df[sub_df['time'] == 'WASH'][sweep_dv].values[0]
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
                        'delta': result['mean_diff'] if self.dependant_var != 'sweep_inputR_MOhm' else result['percent_diff'],
                        'p_val': result['p_val'],
                        'latency_sweeps': 0,
                        'range_sweeps': (0, len(post))
                    })

        return pd.DataFrame(result_rows)
    
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
        if self.dependant_var == 'sweep_inputR_mOhm':
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
                    percentage_threshold= False if self.dependant_var != 'sweep_inputR_MOhm' else True
                ).welchs_t_test(pre, bin_post)

            results.append({
                'cell_id': cell_id,
                'PRE_sweeps': pre,
                'POST_sweeps': post,
                'response': result['response'],
                'delta':result['mean_diff'] if self.dependant_var != 'sweep_inputR_MOhm' else result['percent_diff'],
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

        def categorize_response_multiple_dvs(row, dvs=('RMP_mV', 'AP_count')):
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
    dependant_var: str = field(kw_only=True)
    n_minimum: float = field(kw_only = True, default = 3)
    pre_sweep_window: int = None # window before and after drug_in
    post_sweep_window: int = None 
    diff_thresh: int = field(kw_only = True, default = 0) # ie 3mV difference required to consider it a response 

    p_thresh: float = field(kw_only = True, default = 0.05)
    dynamic_search: bool = field(kw_only=True, default=False)
    bin_width: int = field(kw_only=True, default=3)

    def __post_init__(self):
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

        ax.set_title(f"{self.dependant_var} response to {self.treatment} in {self.cell_type}\n(n={total_n})", fontsize=14)

        
        unit = self.get_unit()
        fig.text(0.5, 0.06, f"Significant response > {self.diff_thresh} {unit}", 
                ha='center', fontsize=10, style='italic')
        fig.text(0.5, 0.02, f"PRE sweep window: {self.pre_sweep_window}   |   POST sweep window: {None if self.dynamic_search else self.post_sweep_window}",
                ha='center', fontsize=10, style='italic')

        plt.tight_layout()
        plt.show()
        self.save_plot(fig, self.filename)


        
@dataclass
class IF_curve(MixedLMStatsMixin, Figure):
    '''
    Plotting IF_IC data between compare which defaults to treatment. 
    Dependant vairables will be I_steps_pA AND AP_frequencies_Hz, not an input pram.
    '''
    filename: str = None
    alpha: float = 0.05
    plot_params: dict = field(default_factory=dict)
    first_factor: str = field(kw_only = True, default = 'treatment') 

    I_range_pA: str = field(kw_only = True, default = 'all_cells')
    
    specify: str = field(kw_only = True, default = 'treatment') # specify marker to see subsets e.g. I_set or cell_id if set to None single cells will not be plotted 
    n_minimum: float = field(kw_only = True, default = 3)
    show_values: bool = field(kw_only=True, default=False)
    dependant_var: str = 'AP_frequencies_Hz' 
    significant_only: bool = field(kw_only=True, default=True)

    def __post_init__(self):
        if self.data_type != 'IF_IC':
            print(f"IF curve is only possible with data_type IF_IC. ")
            return
        
        self.filename = self.build_IF_filename()

        super().__post_init__()
        self.data = self.filter_n_minimum(self.agg_df)
        self.df_long =  self.preprocess_IF_data(n_min=self.n_minimum, I_range_pA=self.I_range_pA)
        self.fig = self.plot_IF_curve()
        self.cell_fig = self.plot_cell_id_curves()

    
    def preprocess_IF_data(self, n_min=None, I_range_pA=None):
        """
        Build a long-format DataFrame for IF plotting, with optional binning,
        filtering for minimum number of cells per group, and optional I-step range.
        
        Parameters
        ----------
        n_min : int
            Minimum number of cells per compare group to include a bin.
        bin_size : int
            Size of I-step binning in pA.
        I_range_pA : None, tuple, or 'all_cells'
            - None: include all bins
            - (min_I, max_I): include only bins within this current range
            - 'all_cells': include only bins where all cells are represented
        """

        df = self.data.copy()
        keep_cols = ['cell_id', 'subject_id', self.first_factor, 'I_steps_pA', 'AP_frequencies_Hz']        
        if self.specify is not None and self.specify != self.first_factor:
            keep_cols.append(self.specify)
        df = df[keep_cols]

        df = df[df['I_steps_pA'].str.len() == df['AP_frequencies_Hz'].str.len()]
        df = df.explode(['I_steps_pA', 'AP_frequencies_Hz'])

        # Convert to numeric
        df['I_steps_pA'] = pd.to_numeric(df['I_steps_pA'], errors='coerce')
        df['AP_frequencies_Hz'] = pd.to_numeric(df['AP_frequencies_Hz'], errors='coerce')

        # Create binned current steps
        df['I_step_bin'] =  df['I_steps_pA']

        # Filter bins with fewer than n_min cells per compare group
        counts = df.groupby(['I_step_bin', self.first_factor])['cell_id'].nunique().reset_index(name='n_cells')
        valid_bins = counts.groupby('I_step_bin').filter(lambda g: (g['n_cells'] >= n_min).all())['I_step_bin'].unique()
        df = df[df['I_step_bin'].isin(valid_bins)]

        # filter by I_range_pA
        if I_range_pA is not None:
            if I_range_pA == 'all_cells':   #  only bins where all cells are represented
                cells_per_bin = df.groupby('I_step_bin')['cell_id'].nunique()
                max_cells = df['cell_id'].nunique()
                valid_bins = cells_per_bin[cells_per_bin == max_cells].index
                df = df[df['I_step_bin'].isin(valid_bins)]
            elif isinstance(I_range_pA, (tuple, list)) and len(I_range_pA) == 2: # set I step range
                df = df[(df['I_step_bin'] >= I_range_pA[0]) & (df['I_step_bin'] <= I_range_pA[1])]
            else:
                raise ValueError("I_range_pA must be None, a tuple/list (min,max), or 'all_cells'")
        return df
    
    def plot_IF_curve(self):
        df = self.df_long.copy()
        agg = self.aggregate_IF_data(df)

        fig, ax = self.draw_IF_curve(df)
        legend_handles_labels = self.draw_IF_points(ax, df)

        bin_stats = self.run_IF_bin_stats(df)
        self.annotate_IF_bin_stats(ax, df, bin_stats)

        self.finalize_IF_curve(ax, fig, agg, legend_handles_labels)

        return fig
    
    def aggregate_IF_data(self, df):
        return (
            df.groupby(["I_step_bin", self.first_factor])[self.dependant_var]
            .agg(mean="mean", sd="std", n="count")
            .reset_index()
        )

    def draw_IF_curve(self, df):
        """
        Draw the main mean IF curve.

        This creates the figure/axis and plots the group-level IF curve with SE error.
        Optional individual points are handled separately by draw_IF_points().
        """
        fig, ax = plt.subplots(
            figsize=(
                self.get_plot_param("figwidth", 12),
                self.get_plot_param("figheight", 8),
            )
        )

        sns.lineplot(
            data=df,
            x="I_step_bin",
            y=self.dependant_var,
            hue=self.first_factor,
            errorbar=self.get_plot_param("errorbar", "se"),
            ax=ax,
            palette=color_dict,
            linewidth=self.get_plot_param("line_width", 2.5),
            marker=self.get_plot_param("line_marker", "o"),
            markersize=self.get_plot_param("line_markersize", 6),
        )

        return fig, ax
    
    def draw_IF_points(self, ax, df):
        """
        Optionally draw individual cell/value points on top of the mean IF curve.

        Points are grouped by self.specify using different marker shapes, while
        edge color follows self.first_factor.
        """
        legend_handles_labels = {}

        if not self.show_values:
            return legend_handles_labels

        if self.specify is None:
            return legend_handles_labels

        if self.specify not in df.columns:
            return legend_handles_labels

        marker_df = df.copy()
        marker_df[self.specify] = marker_df[self.specify].fillna("none")

        markers = cycle(["o", "s", "^", "D", "v", "<", ">"])

        for spec_value in marker_df[self.specify].unique():
            marker = next(markers)
            sub_df = marker_df[marker_df[self.specify] == spec_value].copy()

            for comp in sub_df[self.first_factor].unique():
                comp_df = sub_df[sub_df[self.first_factor] == comp]

                ax.scatter(
                    comp_df["I_step_bin"],
                    comp_df[self.dependant_var],
                    marker=marker,
                    s=self.get_plot_param("raw_point_size", 10),
                    facecolors="none",
                    edgecolors=color_dict.get(comp, "k"),
                    linewidths=self.get_plot_param("raw_point_linewidth", 1.2),
                    alpha=self.get_plot_param("raw_point_alpha", 0.9),
                    zorder=10,
                )

            legend_handles_labels[spec_value] = plt.Line2D(
                [0],
                [0],
                marker=marker,
                label=str(spec_value),
                color="black",
                linestyle="",
                markersize=self.get_plot_param("raw_marker_legend_size", 6),
            )

        return legend_handles_labels

    def finalize_IF_curve(self, ax, fig, agg, legend_handles_labels):
        """
        Apply legend, labels, title, layout, and save the main IF curve figure.
        """
        n_mapping = agg.groupby(self.first_factor)["n"].max().to_dict()

        handles, labels = ax.get_legend_handles_labels()

        new_labels = [
            f"{label} (n={n_mapping[label]})"
            if label in n_mapping
            else label
            for label in labels
        ]

        scatter_handles = list(legend_handles_labels.values())
        scatter_labels = [handle.get_label() for handle in scatter_handles]

        combined_handles = handles + scatter_handles
        combined_labels = new_labels + scatter_labels

        ax.legend(
            combined_handles,
            combined_labels,
            loc=self.get_plot_param("legend_loc", "best"),
            title="Legend",
            fontsize=self.get_plot_param("legend_fontsize", None),
        )

        ax.set_xlabel(
            "Current injection (pA)",
            fontsize=self.get_plot_param("xlabel_fontsize", 16),
        )
        ax.set_ylabel(
            "Firing frequency (Hz)",
            fontsize=self.get_plot_param("ylabel_fontsize", 16),
        )

        ax.set_title(
            self.build_IF_title(),
            fontsize=self.get_plot_param("title_fontsize", 18),
        )
        sns.despine(ax=ax)
        plt.tight_layout()
        plt.show()

        self.save_plot(fig, self.filename)

    def plot_cell_id_curves(self):
        """
        Plot each cell_id's I–F curve with a unique color, separated by self.compare (e.g., treatment group).
        Each compare group is shown in its own subplot for easier visual inspection.
        """
        df = self.df_long.copy()

        if self.first_factor not in df.columns:
            raise ValueError(f"'{self.first_factor}' not found in DataFrame columns: {df.columns.tolist()}")

        compare_groups = df[self.first_factor].unique()
        n_groups = len(compare_groups)

        fig, axes = plt.subplots(
            n_groups, 1,
            figsize=(12, 6 * n_groups),
            sharex=True,
            sharey=True
        )

        if n_groups == 1:
            axes = [axes]  # ensure iterable if single axis

        for ax, comp in zip(axes, compare_groups):
            sub_df = df[df[self.first_factor] == comp].copy()

            # Reset palette per group so colors don't repeat across groups
            cell_ids = sub_df["cell_id"].unique()
            palette = sns.color_palette("husl", len(cell_ids))

            for color, cell_id in zip(palette, cell_ids):
                cell_df = sub_df[sub_df["cell_id"] == cell_id]
                ax.plot(
                    cell_df["I_step_bin"],
                    cell_df[self.dependant_var],
                    color=color,
                    linewidth=1.5,
                    alpha=0.9,
                    label=str(cell_id)
                )

            # Legend (always includes n= count)
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=color, lw=2, label=str(cid))
                for color, cid in zip(palette, cell_ids)
            ]
            leg = ax.legend(
                handles=legend_elements,
                title=f"{comp} (n={len(cell_ids)})",
                fontsize=8,
                ncol=2 if len(cell_ids) > 20 else 1,
                frameon=True,
                loc='best'
            )
            leg.get_frame().set_alpha(0.8)

            # Labels (no subplot titles)
            ax.set_xlabel("Current injection (pA)", fontsize=12)
            ax.set_ylabel("Firing frequency (Hz)", fontsize=12)
            sns.despine(ax=ax)

        fig.suptitle("Per-cell I–F curves by group", fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
        self.save_plot(fig, self.build_IF_filename(suffix="cell_traces"))
        return fig

    def IF_bin_is_testable(self, df_bin):
        """
        Decide whether one I_step_bin has enough information for MixedLM.

        Skip bins where:
        - fewer than 2 compare groups exist
        - fewer than 2 animals exist
        - the dependent variable has no variation, e.g. all AP frequencies are 0

        If some values are 0 and some are not, the bin is still testable.
        """
        values = pd.to_numeric(df_bin[self.dependant_var], errors="coerce").dropna()

        if df_bin[self.first_factor].nunique() < 2:
            return False

        if df_bin["subject_id"].nunique() < 2:
            return False

        if values.nunique() < 2:
            return False

        if np.isclose(values.var(ddof=0), 0):
            return False

        return True
    
    def run_IF_bin_stats(self, df):
        """
        Run mixed-model pairwise stats separately at each current step.

        Skips untestable bins, especially early IF steps where every cell is 0 Hz.
        """
        all_results = {}
        skipped_bins = []

        for i_step, df_bin in df.groupby("I_step_bin"):
            if not self.IF_bin_is_testable(df_bin):
                skipped_bins.append(i_step)
                continue

            try:
                results = self.mixedlm_pairwise_stats(
                    df_bin,
                    group_col=self.first_factor,
                    value_col=self.dependant_var,
                    alpha=self.alpha,
                )

                results = [
                    res for res in results
                    if np.isfinite(res["p_val"])
                ]

                if results:
                    all_results[i_step] = results
                else:
                    skipped_bins.append(i_step)

            except Exception as e:
                skipped_bins.append(i_step)
                print(f"[IF_curve stats skipped] I_step_bin={i_step}: {e}")

        if skipped_bins:
            print(f"[IF_curve stats skipped bins] {skipped_bins}")

        self.IF_bin_stats = all_results
        self.IF_skipped_bins = skipped_bins

        return all_results

    def annotate_IF_bin_stats(self, ax, df, bin_stats):
        """
        Add significance annotation above each current step.

        Places labels close to the data using the current axis range, not the full
        data range. This keeps stars from floating too high above the curve.
        """
        if not bin_stats:
            return

        y0, y1 = ax.get_ylim()
        axis_range = y1 - y0
        offset = axis_range * self.get_plot_param("stats_offset_frac", 0.015)

        used_labels = []

        for i_step, results in bin_stats.items():
            visible = [
                res for res in results
                if res["significant"] or not self.significant_only
            ]

            if not visible:
                continue

            df_bin = df[df["I_step_bin"] == i_step]
            # y = df_bin[self.dependant_var].max() + offset # based off the highest value
            y_base = (
                df_bin
                .groupby(self.first_factor)[self.dependant_var]
                .mean()
                .max()
            )
            y = y_base + offset # based off the mean

            best_p = min(res["p_val"] for res in visible)
            label = self.p_to_star(best_p)

            if not any(res["significant"] for res in visible):
                label = f"p={best_p:.2f}"

            ax.text(
                i_step,
                y,
                label,
                ha="center",
                va="bottom",
                fontsize=self.get_plot_param("stats_fontsize", 10),
                color="black",
            )

            used_labels.append(y)

            for res in visible:
                print(
                    f"I={i_step}: {res['group1']} vs {res['group2']} "
                    f"p={res['p_val']:.4g}"
                )

        if used_labels:
            current_top = ax.get_ylim()[1]
            needed_top = max(used_labels) + axis_range * 0.04
            if needed_top > current_top:
                ax.set_ylim(top=needed_top)

    def IF_optional_params(self, for_filename=False):
        """
        Optional IF-curve parameters to show in filenames/titles.
        """
        params = {
            "I_range_pA": self.I_range_pA,
            "n_minimum": self.n_minimum,
        }

        labels = self.optional_param_labels(params, for_filename=for_filename)

        if self.show_values:
            labels.append("show_values" if for_filename else "show values")

        if self.specify is not None:
            labels.append(
                f"markers_{self.specify}" if for_filename else f"markers = {self.specify}"
            )

        return labels

    def build_IF_filename(self, suffix=None):
        """
        Build saved filename for IF curve figures.
        """
        parts = [
            "IF_curve",
            self.data_type,
            self.first_factor,
            self.region,
            self.cell_type,
        ]

        parts.extend(self.IF_optional_params(for_filename=True))

        if suffix is not None:
            parts.append(suffix)

        return self.sanitize_filename(
            self.build_name(*parts, sep="_")
        )

    def build_IF_title(self):
        """
        Build visible IF curve title.
        """
        parts = [
            "IF curve",
            self.region,
            self.cell_type,
        ]

        parts.extend(self.IF_optional_params(for_filename=False))

        return self.build_name(*parts, sep=" ")




@dataclass
class Histogram(MixedLMStatsMixin, Figure):
    '''
    Generic histogram class for plotting histograms of a specified dependant variable across treatments (and timepoints if project == application).
    '''
    filename: str = None
    dependant_var: str = field(kw_only=True)
    first_factor: str = field(kw_only=True, default='treatment') #bars to compare on x-axis
    second_factor: str | None = None
    plot_params: dict = field(default_factory=dict)
    alpha: float = 0.05 # p value threshold
    specify: str = field(kw_only = True, default = 'treatment') # specify marker to see subsets e.g. I_set or cell_id
    n_minimum: float = field(kw_only = True, default = 3)
    significant_only: bool = field(kw_only=True, default=True)

    pre_sweep_window: int = None # window before and after drug_in
    post_sweep_window: int = None 
    subgroup_key: str = field(kw_only=True, default=None) # if specified, will plot separate histograms for each subgroup in this column
    I_steps_pA: int = None  # only for plotting IF_IC AP_frequencies_Hz
    ISI_ms: int = None # only for plotting PPR_VC PPR


    def __post_init__(self):
        self.filename = self.build_histogram_filename()
        super().__post_init__()
        self.check_valid_dependant_var()
        self.data = self.filter_n_minimum(self.agg_df) # TODO NOW here there is a col RMP_mV averaged dont know why or what it is / and there is the sweep_RMP_mV CHECK WHATS HAPPENING

            # REDUNDANT?
            # If dependant_var contains lists or arrays, average them to a single numeric value
            # if self.data[self.dependant_var].apply(lambda x: isinstance(x, (list, np.ndarray, pd.Series))).any(): #phasing this out
            #     print(f"[Histogram DEBUG] collapsing lists in {self.dependant_var}")
            # self.data[self.dependant_var] = self.data[self.dependant_var].apply(
            #     lambda x: np.mean(x) if isinstance(x, (list, np.ndarray, pd.Series)) else x
            # )


        if self.data_type == "APP_IC":
            self.data, pre_sweep_window, post_sweep_window = self.get_pre_post_sweep_windows(self.data, dependant_var=f"sweep_{self.dependant_var}", pre_sweep_window=self.pre_sweep_window, post_sweep_window=self.post_sweep_window)
      
        if  self.project_obj.project_type == "application":
            self.hue_order = [t for t in ['PRE', 'APP', 'WASH'] if t in self.data['time'].unique()] #not generic #TODO
        
        self.fig = self.plot_histogram() 
        if self.subgroup_key is not None and self.subgroup_key in self.data.columns: # REDUNDANT?
            for subgroup in self.data[self.subgroup_key].unique():
                subgroup_data = self.data[self.data[self.subgroup_key] == subgroup]
                self.fig = self.plot_histogram(subgroup_data, subgroup_name=subgroup)


    def plot_histogram(self, df=None, subgroup_name=None):
        plot_df = self.prepare_histogram_df(df)
        self.plot_df = plot_df.copy()
        self.filename = self.build_histogram_filename(subgroup_name=subgroup_name)

        self.configure_plot_groups(plot_df)

        fig, ax = self.draw_histogram(plot_df)

        stats_results = self.run_histogram_stats(plot_df)
        self.annotate_stats(ax, plot_df, stats_results)

        self.finalize_histogram(ax, fig, subgroup_name=subgroup_name)

        return fig
    
    def configure_plot_groups(self, df):
        """
        Configure x/hue/stat grouping for plotting and stats.

        Sets:
        - self.order
        - self.hue_order
        - self.x_axis
        - self.hue
        - self.stats_group_col
        - self.stats_group_order
        """
        self.order = [
            t for t in color_dict.keys()
            if t in df[self.first_factor].unique()
        ]

        if self.second_factor is None:
            self.hue_order = None
            self.x_axis = self.first_factor
            self.hue = (
                "time"
                if self.project_obj.project_type == "application"
                else self.first_factor
            )
            self.stats_group_col = self.first_factor
            self.stats_group_order = self.order

        else:
            self.hue_order = [
                t for t in color_dict.keys()
                if t in df[self.second_factor].unique()
            ]
            self.x_axis = self.first_factor
            self.hue = self.second_factor
            self.stats_group_col = "plot_group"
            self.stats_group_order = [
                f"{a}_{b}"
                for a in self.order
                for b in self.hue_order
                if f"{a}_{b}" in set(df["plot_group"])
            ]

    def draw_histogram(self, df):
        """
        Draw bars, hidden base swarm, marker overlay, legend, and n labels.
        """
        fig, ax = plt.subplots(
            figsize=(
                self.get_plot_param("figwidth", 15),
                self.get_plot_param("figheight", 10),
            )
        )

        sns.barplot(
            x=self.x_axis,
            y=self.dependant_var,
            hue=self.hue,
            hue_order=self.hue_order,
            order=self.order,
            data=df,
            errorbar=self.get_plot_param("errorbar", "se"),
            palette=color_dict,
            edgecolor=self.get_plot_param("bar_edgecolor", "k"),
            ax=ax,
            alpha=self.get_plot_param("bar_alpha", 1.0),
        )

        sns.swarmplot(
            x=self.x_axis,
            y=self.dependant_var,
            hue=self.hue,
            hue_order=self.hue_order,
            order=self.order,
            data=df,
            palette=color_dict,
            edgecolor="k",
            linewidth=0.5,
            ax=ax,
            legend=False,
            marker="o",
            size=0.05,
            alpha=0.7,
            dodge=True,
        )

        legend_handles_labels = self.specify_markers(df, ax, self.x_axis, self.hue)

        current_handles, current_labels = ax.get_legend_handles_labels()
        combined_handles = current_handles + list(legend_handles_labels.values())
        combined_labels = current_labels + [
            handle.get_label()
            for handle in legend_handles_labels.values()
        ]

        ax.legend(
            handles=combined_handles,
            labels=combined_labels,
            loc=self.get_plot_param("legend_loc", "best"),
            title="Legend",
        )

        self.add_sample_size_labels(ax, df)

        return fig, ax
    

    def specify_markers(self, df, ax, x_axis, hue):
        """
        Adds marker overlay based on self.specify, including missing/None values.

        Default behavior:
        - marker color follows color_dict
        - marker shape follows self.specify

        Optional plot_params:
        - marker_by_specify: False makes all markers circles and hides marker legend
        - marker_facecolor: "none", "white", etc.
        - marker_facealpha: alpha for marker facecolor
        - marker_edgecolor: fixed edge color fallback
        - marker_jitter: stripplot jitter
        """
        marker_df = df.copy()
        marker_col = self.specify

        if marker_col is None:
            return {}

        if marker_col not in marker_df.columns:
            return {}

        marker_df[marker_col] = marker_df[marker_col].fillna("none")

        unique_values = marker_df[marker_col].unique()

        if self.get_plot_param("marker_by_specify", True):
            markers = cycle(["o", "D", "s", "^", "v", "<", ">"])
        else:
            markers = cycle(["o"])

        legend_handles_labels = {}

        for value in unique_values:
            marker = next(markers)
            subset_to_plot = marker_df[marker_df[marker_col] == value]

            before_collections = len(ax.collections)

            sns.stripplot(
                x=x_axis,
                y=self.dependant_var,
                hue=hue,
                hue_order=self.hue_order,
                order=self.order,
                data=subset_to_plot,
                palette=self.get_plot_param("marker_color", color_dict),
                edgecolor=self.get_plot_param("marker_edgecolor", "k"),
                linewidth=self.get_plot_param("marker_linewidth", 1),
                linestyle="-",
                dodge=True,
                jitter=self.get_plot_param("marker_jitter", 0.15),
                ax=ax,
                legend=False,
                marker=marker,
                size=self.get_plot_param("markersize", 7),
                alpha=self.get_plot_param("marker_alpha", 1.0),
            )

            new_collections = ax.collections[before_collections:]

            for collection in new_collections:
                edgecolors = collection.get_facecolors()
                collection.set_edgecolors(edgecolors)

                marker_facecolor = self.get_plot_param("marker_facecolor", None)

                if marker_facecolor == "none":
                    collection.set_facecolors("none")
                elif marker_facecolor is not None:
                    face_color = self.rgba_color(
                        marker_facecolor,
                        self.get_plot_param("marker_facealpha", 1.0),
                    )
                    n_points = len(collection.get_offsets())
                    facecolors = np.tile(face_color, (n_points, 1))
                    collection.set_facecolors(facecolors)
                    collection.set_alpha(None)

            if self.get_plot_param("marker_by_specify", True):
                legend_facecolor = self.get_plot_param("marker_facecolor", None)

                if legend_facecolor == "none":
                    markerfacecolor = "none"
                elif legend_facecolor is not None:
                    markerfacecolor = self.rgba_color(
                        legend_facecolor,
                        self.get_plot_param("marker_facealpha", 1.0),
                    )
                else:
                    markerfacecolor = "white"

                legend_handles_labels[value] = plt.Line2D(
                    [0],
                    [0],
                    marker=marker,
                    label=value,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor="black",
                    color="black",
                    linestyle="None",
                )

        return legend_handles_labels
        
    def add_sample_size_labels(self, ax, df):
        """
        Add cell/animal counts under each first_factor x tick.
        """
        cell_counts = df.groupby(self.first_factor)["cell_id"].nunique()
        animal_counts = (
            df.groupby(self.first_factor)["subject_id"].nunique()
            if "subject_id" in df.columns
            else None
        )

        for tick, treatment in enumerate(self.order):
            n_cells = cell_counts.get(treatment, 0)
            n_animals = animal_counts.get(treatment, 0) if animal_counts is not None else None

            if n_animals is not None:
                text_label = f"n (cells) = {n_cells}\n n (animals) = {n_animals}"
            else:
                text_label = f"n = {n_cells}"

            ax.text(
                tick,
                -0.1,
                text_label,
                ha="center",
                va="top",
                fontsize=22,
                color="black",
                transform=ax.get_xaxis_transform(),
                linespacing=1.2,
            )
    
    def run_histogram_stats(self, df):
        """
        Run the appropriate mixed-model stats and return pairwise results.

        One-factor:
            pairwise MixedLM over first_factor

        Two-factor:
            two-way MixedLM for main effects/interaction
            plus pairwise MixedLM over plot_group
        """
        if self.second_factor is None:
            return self.mixedlm_pairwise_stats(
                df,
                group_col=self.stats_group_col,
                value_col=self.dependant_var,
                group_order=self.stats_group_order,
            )

        print("\n=== TWO-WAY MIXED MODEL ===")
        self.two_way_mixed_model(df)

        return self.mixedlm_pairwise_stats(
            df,
            group_col=self.stats_group_col,
            value_col=self.dependant_var,
            group_order=self.stats_group_order,
        )

    def get_selector_for_dependant_var(self):
        """
        Return the selector column/value needed for special dependent variables.

        Examples:
        - IF_IC AP_frequencies_Hz needs I_steps_pA
        - PPR_VC PPR needs ISI_ms

        Returns:
            (selector_col, selector_value) or (None, None)
        """
        selector_map = {
            ("IF_IC", "AP_frequencies_Hz"): ("I_steps_pA", self.I_steps_pA),
            ("PPR_VC", "PPR"): ("ISI_ms", self.ISI_ms),
        }

        return selector_map.get(
            (self.data_type, self.dependant_var),
            (None, None)
        )

    def is_list_like_value(self, value):
        """
        True for row values that store multiple measurements.
        """
        return isinstance(value, (list, np.ndarray, pd.Series))

    def apply_selector_filter(self, df, selector_col, selector_value):
        """
        Filter/extract rows for selector-specific variables.

        Handles two cases:

        1. selector_col is scalar per row:
            keep rows where df[selector_col] == selector_value

        2. selector_col is list-like per row:
            find selector_value inside that list and extract the matching item from
            self.dependant_var.

        Returns a copy of df.
        """
        if selector_col is None:
            return df.copy()

        if selector_value is None:
            raise ValueError(
                f"{self.dependant_var} requires {selector_col}. "
                f"Please pass {selector_col}=..."
            )

        if selector_col not in df.columns:
            raise ValueError(f"Selector column {selector_col} not found in dataframe.")

        if self.dependant_var not in df.columns:
            raise ValueError(f"Dependent variable {self.dependant_var} not found in dataframe.")

        df = df.copy()

        selector_is_list = df[selector_col].apply(self.is_list_like_value).any()

        if not selector_is_list:
            df = df[df[selector_col] == selector_value].copy()
            return df

        def extract_selected_value(row):
            selectors = row[selector_col]
            values = row[self.dependant_var]

            if not self.is_list_like_value(selectors):
                return pd.Series({
                    self.dependant_var: np.nan,
                    selector_col: np.nan,
                })

            if not self.is_list_like_value(values):
                return pd.Series({
                    self.dependant_var: np.nan,
                    selector_col: np.nan,
                })

            selectors = list(selectors)
            values = list(values)

            try:
                idx = selectors.index(selector_value)
            except ValueError:
                return pd.Series({
                    self.dependant_var: np.nan,
                    selector_col: np.nan,
                })

            if idx >= len(values):
                return pd.Series({
                    self.dependant_var: np.nan,
                    selector_col: np.nan,
                })

            return pd.Series({
                self.dependant_var: values[idx],
                selector_col: selector_value,
            })

        df[[self.dependant_var, selector_col]] = df.apply(extract_selected_value, axis=1)
        df = df.dropna(subset=[self.dependant_var, selector_col]).copy()

        return df

    def collapse_unselected_lists(self, df):
        """
        Collapse list-like dependent variable values only when no selector is needed.

        This preserves your old behavior for variables where a list should simply
        become its mean, but avoids averaging variables like AP_frequencies_Hz before
        selecting I_steps_pA.
        """
        df = df.copy()

        has_lists = df[self.dependant_var].apply(self.is_list_like_value).any()
        if has_lists:
            print(f"[Histogram DEBUG] collapsing lists in {self.dependant_var}")
            df[self.dependant_var] = df[self.dependant_var].apply(
                lambda x: np.mean(x) if self.is_list_like_value(x) else x
            )

        return df

    def prepare_histogram_df(self, df=None):
        """
        Build the dataframe used for plotting and stats.

        self.data remains the base data.
        This method returns a transformed plot_df with:
        - missing factor rows removed
        - selector-specific variables extracted/filtered
        - remaining list-like dependent values collapsed
        - plot_group added for two-factor designs
        """
        if df is None:
            plot_df = self.data.copy()
        else:
            plot_df = df.copy()

        group_cols = [self.first_factor]
        if self.second_factor is not None:
            group_cols.append(self.second_factor)

        plot_df = plot_df.dropna(subset=group_cols).copy()

        selector_col, selector_value = self.get_selector_for_dependant_var()
        plot_df = self.apply_selector_filter(plot_df, selector_col, selector_value)

        if selector_col is None:
            plot_df = self.collapse_unselected_lists(plot_df)

        plot_df[self.dependant_var] = pd.to_numeric(
            plot_df[self.dependant_var],
            errors="coerce"
        )
        plot_df = plot_df.dropna(subset=[self.dependant_var]).copy()

        if self.second_factor is not None:
            plot_df["plot_group"] = (
                plot_df[self.first_factor].astype(str) + "_" +
                plot_df[self.second_factor].astype(str)
            )

        return plot_df

    def two_way_mixed_model(self, df):
        """
        Fit the main two-factor mixed-effects model.

        Model:
            dependent_variable ~ first_factor * second_factor + (1 | subject_id)

        This tests main effects and interaction while accounting for multiple cells
        from the same animal.
        """
        model_df = self.clean_mixedlm_df(
            df,
            group_col=self.stats_group_col,
            value_col=self.dependant_var
        )

        combo_table = pd.crosstab(model_df[self.first_factor], model_df[self.second_factor])
        has_empty_combinations = (combo_table == 0).any().any()

        if has_empty_combinations:
            print(
                "\nWARNING: Some first_factor x second_factor combinations are missing. "
                "Cannot fit a full two-way interaction model. "
                "Fitting combined plot_group model instead.\n"
            )

            formula = f"{self.dependant_var} ~ C({self.stats_group_col})"

        else:
            formula = (
                f"{self.dependant_var} ~ "
                f"C({self.first_factor}, Treatment(reference='{self.order[0]}')) * "
                f"C({self.second_factor}, Treatment(reference='{self.hue_order[0]}'))"
            )

        self.mixedlm_result = mixedlm(
            formula,
            data=model_df,
            groups=model_df["subject_id"],
        ).fit(reml=True, method="powell")   
        
        pvals = self.mixedlm_result.pvalues
        p_first = [pvals[k] for k in pvals.index if f"C({self.first_factor}" in k and ":" not in k]
        p_second = [pvals[k] for k in pvals.index if f"C({self.second_factor}" in k and ":" not in k]
        p_interaction = [pvals[k] for k in pvals.index if ":" in k]
        self.two_way_pvals = {
            self.first_factor: p_first[0] if p_first else np.nan,
            self.second_factor: p_second[0] if p_second else np.nan,
            "interaction": p_interaction[0] if p_interaction else np.nan,
        }
        for label, p_val in self.two_way_pvals.items():
            print(f"{label:15s}: p = {p_val:.4g}")
        print("=" * 35)

        return self.mixedlm_result

    def add_two_way_stats_box(self, ax):
        """
        Add main-effect/intervention p-values from the two-way MixedLM to the plot.
        """
        if self.second_factor is None:
            return

        if not hasattr(self, "two_way_pvals"):
            return

        lines = ["MixedLM"]
        for label, p_val in self.two_way_pvals.items():
            if np.isnan(p_val):
                lines.append(f"{label}: p = NA")
            else:
                lines.append(f"{label}: p = {p_val:.3g}")

        ax.text(
            0.98,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=self.get_plot_param("stats_box_fontsize", 14),           
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.8,
            },
        )
    
    def histogram_optional_labels(self, subgroup_name=None, for_filename=False):
        """
        Return optional parameter labels for filenames/titles.

        Includes only parameters that are actually being used.
        """
        parts = []

        selector = self.selector_label(sep="_" if for_filename else " = ")
        if selector is not None:
            parts.append(selector)

        if subgroup_name is not None:
            if for_filename:
                parts.append(f"subgroup_{subgroup_name}")
            else:
                parts.append(str(subgroup_name))

        if self.pre_sweep_window is not None:
            if for_filename:
                parts.append(f"pre_sweep_window_{self.pre_sweep_window}")
            else:
                parts.append(f"pre sweep window = {self.pre_sweep_window}")

        if self.post_sweep_window is not None:
            if for_filename:
                parts.append(f"post_sweep_window_{self.post_sweep_window}")
            else:
                parts.append(f"post sweep window = {self.post_sweep_window}")

        return parts

    def selector_label(self, sep=" = "):
        """
        Return a readable selector label for variables that need one.

        Examples:
            I_steps_pA = 100
            ISI_ms = 50
        """
        selector_col, selector_value = self.get_selector_for_dependant_var()

        if selector_col is None:
            return None

        return f"{selector_col}{sep}{selector_value}"

    def build_histogram_filename(self, subgroup_name=None):
        """
        Build the saved plot filename from the current histogram parameters.
        """
        return self.build_name(
            *self.histogram_name_parts(subgroup_name=subgroup_name, for_filename=True),
            sep="_"
        )

    def histogram_name_parts(self, subgroup_name=None, for_filename=False):
        """
        Build consistent filename parts for histogram outputs.

        Filenames include factors because they help distinguish saved plots.
        Titles are handled separately by build_histogram_title().
        """
        factor_label = self.first_factor
        if self.second_factor is not None:
            factor_label = f"{self.first_factor}_by_{self.second_factor}"

        parts = [
            self.dependant_var,
            self.data_type,
            self.region,
            self.cell_type,
            factor_label,
        ]

        parts.extend(
            self.histogram_optional_labels(
                subgroup_name=subgroup_name,
                for_filename=for_filename
            )
        )

        if self.specify is not None:
            parts.append(f"markers_{self.specify}")

        return [p for p in parts if p is not None]
    
    def build_histogram_title(self, subgroup_name=None):
        """
        Build the visible plot title.

        Keep the title clean: no factor labels, only biological context and optional
        parameters like I_steps_pA, ISI_ms, sweep windows, or subgroup.
        """
        y_label = unit_dict.get(self.dependant_var, self.dependant_var)

        parts = [
            y_label,
            self.region,
            self.cell_type,
        ]

        parts.extend(
            self.histogram_optional_labels(
                subgroup_name=subgroup_name,
                for_filename=False
            )
        )

        return self.build_name(*parts, sep=" ")

    def group_x_position(self, df, group):
        """
        Return x position for a stats group.

        One-factor:
            group is a first_factor value.

        Two-factor:
            group is plot_group, and the x position is the dodged bar center.
        """
        if self.second_factor is None:
            return self.order.index(group)

        row = df[df[self.stats_group_col] == group].iloc[0]
        first_value = row[self.first_factor]
        second_value = row[self.second_factor]

        x_index = self.order.index(first_value)
        hue_index = self.hue_order.index(second_value)

        n_hue = len(self.hue_order)
        total_width = 0.8
        hue_width = total_width / n_hue

        return x_index - total_width / 2 + hue_width * (hue_index + 0.5)

    def annotate_stats(self, ax, df, stats_results, alpha=None):
        """
        Annotate pairwise stats on the histogram.

        Works for:
        - one-factor bars
        - two-factor dodged bars using plot_group
        """
        if not stats_results:
            return
        if alpha is None:
            alpha = self.alpha

        y_min = df[self.dependant_var].min()
        y_max = df[self.dependant_var].max()
        y_range = y_max - y_min

        if y_range == 0:
            y_range = abs(y_max) * 0.1 if y_max != 0 else 1

        visible_results = []
        for res in stats_results:
            if getattr(self, "significant_only", True) and not res["significant"]:
                continue
            visible_results.append(res)

        if not visible_results:
            return

        base_y = y_max + y_range * 0.08
        step_y = y_range * 0.08
        tick_y = y_range * 0.02

        for i, res in enumerate(visible_results):
            group1 = res["group1"]
            group2 = res["group2"]

            try:
                x1 = self.group_x_position(df, group1)
                x2 = self.group_x_position(df, group2)
            except (ValueError, IndexError):
                print(f"[annotate_stats] could not place {group1} vs {group2}")
                continue

            y = base_y + i * step_y

            ax.plot(
                [x1, x1, x2, x2],
                [y, y + tick_y, y + tick_y, y],
                lw=1.5,
                color="black",
            )

            label = self.p_to_star(res["p_val"])
            if not res["significant"]:
                label = f"p={res['p_val']:.3f}"

            ax.text(
                (x1 + x2) / 2,
                y + tick_y,
                label,
                ha="center",
                va="bottom",
                fontsize=16,
                color="black",
            )

            print(f"{group1} vs {group2}: p={res['p_val']:.4f}")

        ax.set_ylim(top=base_y + len(visible_results) * step_y + y_range * 0.12)

    def finalize_histogram(self, ax, fig, subgroup_name=None):
        """
        Apply final labels/layout and save the figure.
        """
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_ylabel(unit_dict[self.dependant_var], fontsize=24)
        ax.set_xlabel("")
        ax.set_title(
            self.build_histogram_title(subgroup_name=subgroup_name),
            fontsize=28,
        )
        ax.tick_params(axis="x", labelsize=24)
        ax.tick_params(axis="y", labelsize=24)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.add_two_way_stats_box(ax)
        plt.tight_layout()
        plt.show()
        self.save_plot(fig, self.filename)

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
            cell_sub_df = self.APP_IC_df[self.APP_IC_df['cell_id'] == cell_id]
            if self.valid_only == True:
                cell_sub_df = cell_sub_df[cell_sub_df['valid'] != False]

            for folder_file, cell_id, I_set, drug, drug_in, drug_out, RA_locs in cell_sub_df[['folder_file','cell_id', 'I_set', 'treatment', 'drug_in', 'drug_out', 'RA_locs']].values:
                self.fig_filename = f"{cell_id} {drug} Application"
                
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
                ax1.set_title(cell_id + ' '+ drug +' '+ " Application", fontsize = 16) # , fontsize = 25
                plt.tight_layout()
                plt.show()
                self.save_plot(fig, f"{cell_id}_APP")
                



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


