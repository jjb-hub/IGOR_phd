import os
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from module.Cachable import Cachable
import traceback
from tqdm import tqdm
from IPython.display import display
from itertools import combinations
import igor2 as igor
import numpy as np
import matplotlib.pyplot as plt
from module.action_potential_functions import calculate_max_firing, ap_characteristics_extractor_main, extract_FI_slope_and_rheobased_threshold, extract_FI_x_y, sag_current_analyser, tau_analyser, mean_inputR_APP_calculator, mean_RMP_APP_calculator

tqdm.pandas()

# Root directory for projects
ROOT = f"{os.getcwd()}/PROJECTS"
if not os.path.exists(ROOT):
    os.mkdir(ROOT)


@dataclass
class Project(Cachable):
    '''
    Class for handeling file system for a project. 
    '''
    project: str
    input_dir: str = field(init=False)
    output_dir: str = field(init=False)
    figure_output_dir: str = field(init=False)

    def __post_init__(self):
        super().__init__(cache_dir=f"{ROOT}/{self.project}/cache")
        self.location = f"{ROOT}/{self.project}"
        self.input_dir = self._checkFileSystem("input")
        self.output_dir = self._checkFileSystem("output")
        self.figure_output_dir = self._checkFileSystem("figures")
        self.feature_df = self.load_xlsx('features')

    def load_xlsx(self, filename: str):
        """Loads data from cache or an Excel file."""
        if self.isCached(filename):
            return self.getCache(filename)
        else:
            filepath = os.path.join(self.input_dir, f"{filename}.xlsx")
            if os.path.exists(filepath):
                
                if filename =='features':
                    df = pd.read_excel(filepath, converters={'drug_in':int, 'drug_out':int}) #eventualy replace with validator of feature df
                    df['cell_subtype'].fillna(np.nan, inplace=True)
                else:
                    df = pd.read_excel(filepath)

                self.cache(filename, df)
                return df
            raise FileNotFoundError(f"Excel file {filename} not found in {self.input_dir}")
            

    def IGOR_load(self, folder_file):
        path_V, path_I = self.make_path(folder_file)
        V_list, V_array = self.igor_exporter(path_V)
        I_list, I_array = None, None
        try:
            I_list, I_array = self.igor_exporter(path_I)
        except FileNotFoundError:
            I_array = None
        return V_array, I_array, V_list

    def make_path(self, folder_file): 
        """Generates file paths for voltage and current data."""
        if not isinstance(folder_file, str) or pd.isna(folder_file):
            raise ValueError(f"Invalid folder_file: {folder_file}")
        extension_V = "Soma.ibw"  # Voltage data file extension
        extension_I = "Soma_outwave.ibw"  # Current data file extension

        path_V = os.path.join(self.input_dir, 'PatchData',  folder_file + extension_V)
        path_I = os.path.join(self.input_dir, 'PatchData', folder_file + extension_I)
        return path_V, path_I

    def igor_exporter(self, path):
        """Loads and processes .ibw files using igor binarywave."""
        igor_file = igor.binarywave.load(path)
        wave = igor_file["wave"]["wData"]
        igor_df = pd.DataFrame(wave)
        V_array_2d = igor_df.to_numpy()
        point_list = V_array_2d.ravel(order='F') 
        return point_list, V_array_2d
    

    def inspect_IGOR_file(self, folder_file, stacked=False, n_sweeps=None):
        '''
        Plots any waveform based off folder_file.
        Stacked will plot each column on top of each other, defaults to False.
        '''
        feature_df = self.load_xlsx('features')
        V_array , I_array, V_list = self.IGOR_load(folder_file)
        display(feature_df[feature_df['folder_file'] == folder_file])  # Show file info
        self.quick_line_plot(V_array, f'Voltage trace for {folder_file}', 'Voltage (mV)', n_sweeps=n_sweeps, stacked=stacked )
        try:
            self.quick_line_plot(I_array, f'Current (I) trace for {folder_file}', 'Current (pA)', n_sweeps=n_sweeps,  stacked=stacked) #TODO add if check shape hwen no I 
        except FileNotFoundError:
            print(f'No I file found for {folder_file}')

    def quick_line_plot(self, plot_array, plottitle, y_label,  n_sweeps=None, stacked=False):
        '''
        Plots line plot for given array without adding a legend for stacked plots.
        
        Parameters:
            plot_array (numpy.ndarray): 2D array to plot, where each column is a sweep.
            plottitle (str): Title for the plot.
            stacked (bool): If True, plots each sweep stacked. If False, concatenates sweeps.
        '''
        plt.figure()
        num_sweeps = plot_array.shape[1]
        if n_sweeps is None or n_sweeps > num_sweeps:
            n_sweeps = num_sweeps 
        
        if stacked:
            for i in range(n_sweeps):
                plt.plot(plot_array[:, i])  # Plot each sweep
        else:
            # Concatenate sweeps for continuous plotting
            cropped_array = plot_array[:, :n_sweeps] 
            continuous_plot = cropped_array.ravel(order='F')  # Flatten array in column-major order
            plt.plot(continuous_plot)  # Plot continuous
        
        plt.title(plottitle)
        plt.xlabel('Time in ms')
        plt.ylabel(y_label)
        plt.show()




@dataclass
class EphysData (Project):

    '''Generic data_type extractor, child classes process the data nd make aggregate dfs'''
    
    # project: str #name of the excel_filename project_filename in notebook
    initial_columns: list = None #defined by child classes
    sampling_rate: float = 2e4
    data_type: str = None #defined by child class
    filename: str = None # defined by child class


    def __post_init__(self):
        super().__post_init__()
        if  self.isCached(self.filename): 
            self.df = self.getCache(self.filename)
        else:
            self.df = self.generate()
        

   
    def generate(self):
        ''' generic generator for dfs'''
        df = self.feature_df[self.feature_df['data_type'] == self.data_type][self.initial_columns] 

        df = df.progress_apply(lambda row: self._handle_extraction(row, self.process), axis=1) # log errors
        # df = df.progress_apply(lambda row: self._debug_extraction(row, self.process), axis=1) # raise errors
        additional_columns = [col for col in df.columns if col not in self.initial_columns]
        df = df[self.initial_columns + additional_columns]
        # cache(self.project, self.filename, df)
        self.cache(self.filename, df)
        return df
    
    def process(self):
        raise NotImplementedError
    
    def _debug_extraction(self, row: pd.Series, process_function) -> pd.Series:
        '''Direct processing without error catching — use during debugging.'''
        row = row.copy()
        row = process_function(row)  # Let any exception raise naturally
        row['error'] = 'ran'
        row['traceback'] = None
        return row
    
    def _handle_extraction(self, row: pd.Series, process_function) -> pd.Series:
        '''Error handeling to add rows 'error' and 'ran' to each df made. '''
        row = row.copy()
        error_msg = None
        error_traceback = None

        def log_error(msg, tb):
            nonlocal error_msg
            nonlocal error_traceback
            error_msg = msg
            error_traceback = tb

        try:
            row = process_function(row)
        except Exception as e:
            error_type = type(e).__name__
            error_tb = traceback.format_exc()
            lines = error_tb.split('\n')
            relevant_tb = [lines[idx - 1].strip() for idx, line in enumerate(lines) if f'{error_type}: {str(e)}' in line]

            log_error(f'{error_type}: {str(e)}', relevant_tb)
            error_traceback = relevant_tb
        
        if error_msg:
            row['error'] = error_msg
            row['traceback'] = error_traceback
            print(f'{row.cell_id} error message logged: {error_msg}')
            print(f'{row.cell_id} traceback: {error_traceback}')
        else:
            row['error'] = 'ran'
            row['traceback'] = None
        return row
    

    

@dataclass
class FP(EphysData):
    
    filename: str = "FP_df"
    data_type: str = 'FP'
    
  
    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'I_set', 'drug', 'replication_no', 'application_order', 'R_series', 'cell_type', 'cell_subtype']
        super().__post_init__()
    
    def process(self, row: pd.Series) -> pd.Series:
        """Processing logic specific to FP data type. Could also handle FP_APP data if sufficient to analise."""
        V_array , I_array, V_list = self.IGOR_load(row['folder_file'])

        row["max_firing"] = calculate_max_firing(V_array)
        peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_rise_all  , peak_max_dvdt_all,  peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all   , peak_indices_all , sweep_indices_all , peak_decay_all = ap_characteristics_extractor_main(row['folder_file'], V_array)        
        
        if len(peak_voltages_all)==0: #returns is no APs are detected
            return row
        
        step_current_values, ap_counts, V_rest, off_step_peak_locs, ap_frequencies_Hz = extract_FI_x_y(row['folder_file'], V_array, I_array, peak_locs_corr_all, sweep_indices_all)
        FI_slope, rheobase_threshold = extract_FI_slope_and_rheobased_threshold(row['folder_file'], step_current_values, ap_counts)

        row["rheobased_threshold"] = rheobase_threshold
        row["FI_slope"] = FI_slope

        row['AP_peak_voltages'] = peak_voltages_all[:10]
        row["voltage_threshold"] = v_thresholds_all[:10]
        row["AP_height"] = peak_heights_all[:10]
        row["AP_width"] = peak_fw_all[:10]
        row["AP_rise_dvdt"] = peak_rise_all[:10]
        row["AP_decay_dvdt"] = peak_decay_all[:10]
        row["AP_latency"] = peak_latencies_all[:10]
        row["AP_dvdt_max"] = peak_max_dvdt_all[:10]

        row["tau_rc"] = tau_analyser(row['folder_file'], V_array, I_array, step_current_values, ap_counts)
        row["sag"] = sag_current_analyser(row['folder_file'], V_array, I_array, step_current_values, ap_counts)

         #fetch FP data for this cell and use the average threshold to define the RA 
        try:
            cell_threshold = np.mean(row['voltage_threshold'])
        except:
            cell_threshold = -45 #so when you -20 is 65 for cells without FP

        RA_condition = lambda peak_voltage, threshold: threshold <= (cell_threshold - 20) and peak_voltage > 0 #HARD CODE was -65 for all , now based on cell Threshold 

        if any(RA_condition(peak_voltage, threshold) for peak_voltage, threshold in zip(peak_voltages_all, v_thresholds_all)):
            row['RA'] = True
            row['RA_locs'] = [peak_locs_corr_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if threshold <= -65 and peak_voltage > 20]
            row['RA_per_min'] = len(row['RA_locs']) / V_array.shape[0] * V_array.shape[1] / self.sampling_rate / 60 #RA/minute

        # FP FILE VALIDATOR
        if np.mean(np.array(peak_voltages_all[:10])[~np.isnan(peak_voltages_all[:10])]) < 15: #mean of first 11 AP peaks is less than 15mV the file is marked invalid
            row['valid'] = False 

        return row
    
@dataclass
class APP(EphysData):
    
    filename: str = "APP_df"
    data_type: str = 'APP'

    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'I_set', 'drug', 'drug_in', 'drug_out', 'replication_no', 'application_order', 'cell_type', 'cell_subtype']
        super().__post_init__()

    def process(self, row: pd.Series) -> pd.Series:
        """Generate APP_df from scratch, 
        Processing logic specific to APP data type."""
        V_array , I_array, V_list = self.IGOR_load(row['folder_file'])

        
        if I_array is not None and (I_array[:, 0] != 0).any():
            input_R_PRE, input_R_APP, input_R_WASH = mean_inputR_APP_calculator(V_array, I_array, row.drug_in, row.drug_out)
            row['inputR_PRE'] = input_R_PRE
            row['inputR_APP'] = input_R_APP
            row['inputR_WASH'] = input_R_WASH
            pass_I_array = I_array
        else:
            row['inputR_PRE'] = []
            row['inputR_APP'] = []
            row['inputR_WASH'] = []
            pass_I_array = None

        mean_RMP_PRE, mean_RMP_APP, mean_RMP_WASH = mean_RMP_APP_calculator(V_array, row.drug_in, row.drug_out, I_array=pass_I_array)
        row['RMP_PRE'] = mean_RMP_PRE[2:]
        row['RMP_APP'] = mean_RMP_APP
        row['RMP_WASH'] = mean_RMP_WASH

        peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_rise_all  , peak_max_dvdt_all,  peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all   , peak_indices_all , sweep_indices_all , peak_decay_all = ap_characteristics_extractor_main(row.folder_file, V_array)

        #fetch FP data for this cell and use the average threshold to define the RA 
        FP_df = self.getCache("FP_df")
        try:
            FP_cell_id_PRE = FP_df[(FP_df['cell_id'] == row['cell_id']) & (FP_df['drug'] == 'PRE')]
            cell_threshold = (FP_cell_id_PRE['voltage_threshold'].apply(lambda x: sum(x) / len(x) if isinstance(x, list) else x)).mean()
        except:
            cell_threshold = -45 #so when you -20 is 65 for cells without FP

        RA_condition = lambda peak_voltage, threshold: threshold <= (cell_threshold - 20) and peak_voltage > 0 #HARD CODE was -65 for all , now based on cell Threshold 

        if any(RA_condition(peak_voltage, threshold) for peak_voltage, threshold in zip(peak_voltages_all, v_thresholds_all)):
            row['RA'] = True
            row['RA_locs'] = [peak_locs_corr_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if threshold <= -65 and peak_voltage > 20]
            row['RA_per_min'] = len(row['RA_locs']) / V_array.shape[0] * V_array.shape[1] / self.sampling_rate / 60 #RA/minute
            row['RAcount_PRE'] = len([peak_loc for peak_loc, sweep_index, peak_voltage, threshold in zip(peak_locs_corr_all, sweep_indices_all, peak_voltages_all, v_thresholds_all) if sweep_index < row['drug_in'] and RA_condition(peak_voltage, threshold)])
            row['RAcount_APP'] = len([peak_loc for peak_loc, sweep_index, peak_voltage, threshold in zip(peak_locs_corr_all, sweep_indices_all, peak_voltages_all, v_thresholds_all) if row['drug_in'] <= sweep_index <= row['drug_out'] and RA_condition(peak_voltage, threshold)])
            row['RAcount_WASH'] = len([peak_loc for peak_loc, sweep_index, peak_voltage, threshold in zip(peak_locs_corr_all, sweep_indices_all, peak_voltages_all, v_thresholds_all) if sweep_index > row['drug_out'] and RA_condition(peak_voltage, threshold)])
            #row['RA_per_min'] = len(row['RA_locs']) / trace_time_in_min
        else:
            row['RA_locs'] = []
            row['RAcount_PRE'] = 0
            row['RAcount_APP'] = 0
            row['RAcount_WASH'] = 0
            #row['RA_per_min'] = 0

        row['AP_locs'] = peak_locs_corr_all
        row['peak_voltages_all'] = peak_voltages_all

        if len(peak_locs_corr_all) > 0:
            row['APcount_PRE'] = len([peak_loc for peak_loc, sweep_index in zip(peak_locs_corr_all, sweep_indices_all) if sweep_index < row['drug_in']])
            row['APcount_APP'] = len([peak_loc for peak_loc, sweep_index in zip(peak_locs_corr_all, sweep_indices_all) if row['drug_out'] >= sweep_index >= row['drug_in']])
            row['APcount_WASH'] = len([peak_loc for peak_loc, sweep_index in zip(peak_locs_corr_all, sweep_indices_all) if sweep_index > row['drug_out']])
        else:
            row['AP_locs'] = []
            row['APcount_PRE'] = 0
            row['APcount_APP'] = 0
            row['APcount_WASH'] = 0

        # GENERIC functions
        def check_variability(values, Vairability_threshold=0.30): 
            """Check if variability of values exceeds the given threshold."""
            values = np.array(values)[~np.isnan(values)]
            if len(values) <= 1:
                return True
            min_val = np.min(values)
            max_val = np.max(values)
            # print(f" % var  {abs((max_val - min_val) / min_val)}")
            return abs((max_val - min_val) / min_val) <= Vairability_threshold
        
        def group_AP_bursts(peak_locs_corr_all, sweep_indices_all, peak_voltages_all, burst_window_seconds=0.5):
            """
            Groups APs into bursts based on the time difference between them.
            Condenses each burst into the maximum peak voltage and returns a list of these max values.
            - peak_locs_corr_all: AP peak locations within sweep
            - sweep_indices_all: sweep of each AP
            - peak_voltages_all: List of AP peak voltages 
            - burst_window_seconds: The time window (in seconds) to consider APs as part of the same burst. Default is 0.5 seconds.
            """            
            burst_window_samples = int(burst_window_seconds * self.sampling_rate)
            bursts = []
            current_burst = []
            # Iterate over each AP's peak location, voltage, and sweep index
            for i, (peak_loc, sweep_index) in enumerate(zip(peak_locs_corr_all, sweep_indices_all)):
                curr_time = (sweep_index * V_array.shape[0] + peak_loc) / self.sampling_rate
                if not current_burst: #first AP
                    current_burst.append((peak_loc, peak_voltages_all[i], curr_time))
                    continue
                prev_peak_loc, prev_voltage, prev_time = current_burst[-1]
                time_diff = curr_time - prev_time
                time_diff_samples = time_diff * self.sampling_rate
                if time_diff_samples <= burst_window_samples:
                    current_burst.append((peak_loc, peak_voltages_all[i], curr_time))
                else:
                    # Finalize the current burst and start a new one
                    bursts.append(max(voltage for _, voltage, _ in current_burst))
                    current_burst = [(peak_loc, peak_voltages_all[i], curr_time)]
            if current_burst:
                bursts.append(max(voltage for _, voltage, _ in current_burst))
            return bursts
        
        def unidirectional_trend(values, threshold=20):
            '''Check for a unidirectional trend that surpasses the threshold, if present returns False'''
            value_diff = np.diff(values)
            is_increasing = all(value_diff > 0)   # True if all differences are positive
            is_decreasing = all(value_diff < 0)   # True if all differences are negative
            total_change = abs(values[-1] - values[0])
            if (is_increasing or is_decreasing) and total_change >= threshold:
                return False  # data is not valid
            return True  
        
        # APP FILE INVALIDATORS 
        if check_variability([row['RMP_PRE']],Vairability_threshold=0.3)  == False: #assigns True if < vairability threshold
            row['valid'] = False 

        if len(peak_voltages_all)>0: # if APs 
            if np.mean(np.nanmean(peak_voltages_all)) < 15: #HARDCODE minimum 15 mV AP height to declare offset issues
                row['offset']= True

            peak_voltage_burst_max = group_AP_bursts(peak_locs_corr_all, sweep_indices_all, peak_voltages_all, burst_window_seconds=1)
            ap_burst_valid = unidirectional_trend(peak_voltage_burst_max, threshold=10)
            if ap_burst_valid == False:
                row['valid'] = False

        rmp_valid = unidirectional_trend([np.mean(mean_RMP_PRE), np.mean(mean_RMP_APP), np.mean(mean_RMP_WASH)], threshold=20) #assigns True if 
        if  rmp_valid == False:
            row['valid'] = False

        return row
        
class Hunter(EphysData):
    '''Handels data type Hunter currently just fetching the RA locations.'''
    filename: str = "RA_hunter_df"
    data_type: str = 'Hunter'

    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'drug', 'replication_no', 'application_order', 'cell_type', 'cell_subtype']
        super().__post_init__()

    def process(self, row: pd.Series) -> pd.Series:
        V_array , I_array, V_list = self.IGOR_load(row['folder_file'])


        peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_rise_all  , peak_max_dvdt_all,  peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all   , peak_indices_all , sweep_indices_all , peak_decay_all = ap_characteristics_extractor_main(row.folder_file, V_array)

        
        if any(threshold <= -65 and peak_voltage > 20 for peak_voltage, threshold in zip(peak_voltages_all, v_thresholds_all)):
            row['RA'] = True
            row['RA_locs'] = [peak_locs_corr_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if threshold <= -65 and peak_voltage > 20]
        return row



@dataclass
class Ephys(EphysData):
    ''' 
    Buiilding aggregate df with cell info based off extracted data from each data type: APP, FP and RA_hunter each with their own class
        feature_df: excel input mapping folder_files to features

        FP_df: extraction of firing property data (FP)
        APP_df: extraction of applications data (APP)
        RA_hunter_df: last unofficial data_type needs developing* #TODO

    Ephys class:
        cell_df: mapping of cells to features including change in access and FP_valid and APP_valid columns with valid folder_files
          '''
    filename: str = 'cell_df'
    sampling_rate: float = 2e4
    
    def __post_init__(self):
        
        self.FP_df = FP(self.project).df
        self.APP_df = APP(self.project).df
        # self.hunter_df = Hunter(self.project).df some issue with 
        super().__post_init__()
        
    
    def generate(self) -> pd.DataFrame:
        """
        Builds cell_df with each row a cell_id, access_change reported where possible and valid data is marked True in 'data_type' column i.e. "FP".
        """
        df = self.feature_df.copy()
        df['treatment'] = df.apply(lambda row: row['drug'] if row['application_order'] == 1 else np.nan, axis=1)  # make treatment column

        def check_unique(series, cell_id):
            unique_values = series.dropna().unique()
            if len (unique_values) == 0:
                return None
            if len(unique_values) == 1:
                return unique_values[0]
            else:
                raise ValueError(f"Non-unique values found for cell_id: {cell_id} with values: {unique_values}")

        def apply_check_unique(group):
            cell_id = group.name
            I_set_values = group.loc[(group['data_type'] == 'APP') & (group['replication_no'] == 1) & (group['application_order'] == 1), 'I_set' ].unique()
            I_set_value = I_set_values[0] if len(I_set_values) > 0 else np.nan
        
            aggregated_data = group.agg({
                'treatment': lambda series: check_unique(series, cell_id),
                'cell_type': lambda series: check_unique(series, cell_id),
                'cell_subtype': lambda series: check_unique(series, cell_id)
            })
            return pd.concat([aggregated_data, pd.Series({'I_set': I_set_value})])

        def calculate_percentage_diff(group):
            """
            Selects the two PRE and two non-PRE FP files with the most similar R_series values to compute access change. 
            If several have the same access chose the filder_files that have the least mising values."""
            cell_id = group.name
            #FIRING PROPERTY 
            cell_fp_df = self.FP_df[self.FP_df['cell_id'] == cell_id]
            pre_values = cell_fp_df[cell_fp_df['drug'] == 'PRE'][['R_series', 'folder_file']]
            non_pre_values = cell_fp_df[cell_fp_df['drug'] != 'PRE'][['R_series', 'folder_file']]
            
            # Extract R_series and folder_file
            pre_series = pre_values['R_series'].dropna().values
            non_pre_series = non_pre_values['R_series'].dropna().values
            
            # Check if there are enough values
            if len(pre_series) < 2 or len(non_pre_series) < 2:
                return pd.Series({'access_change': None, 'FP_valid': None})
            
            # Generate all combinations of two values
            pre_combinations = list(combinations(pre_series, 2))
            non_pre_combinations = list(combinations(non_pre_series, 2))
            
            min_diff = float('inf')
            best_pre_pair = None
            best_non_pre_pair = None
            
            # Calculate percentage difference for all combinations
            for pre_pair in pre_combinations:
                pre_mean = np.mean(pre_pair)
                for non_pre_pair in non_pre_combinations:
                    non_pre_mean = np.mean(non_pre_pair)
                    if pre_mean == 0:
                        continue
                    percentage_change = (non_pre_mean - pre_mean) / pre_mean * 100
                    
                    if percentage_change < min_diff:
                        min_diff = percentage_change
                        best_pre_pair = pre_pair
                        best_non_pre_pair = non_pre_pair
            
            if best_pre_pair is None or best_non_pre_pair is None:
                return pd.Series({'access_change': None, 'FP_valid': None})
            
            # folder_file filtered on access
            pre_folder_files = pre_values[pre_values['R_series'].isin(best_pre_pair)]['folder_file'].tolist() 
            non_pre_folder_files = non_pre_values[non_pre_values['R_series'].isin(best_non_pre_pair)]['folder_file'].tolist()

            # filter folder_files on extracted features and absence of RA
            if len(pre_folder_files) > 2 or len(non_pre_folder_files) > 2:
                FP_feature_cols = [
                    'AP_peak_voltages', 'AP_rise_dvdt', 'AP_width', 'FI_slope',
                    'max_firing', 'rheobased_threshold', 'sag', 'tau_rc', 'voltage_threshold', 'AP_decay_dvdt'
                ]
                
                pre_df = cell_fp_df[cell_fp_df['drug'] == 'PRE'].copy()
                non_pre_df = cell_fp_df[cell_fp_df['drug'] != 'PRE'].copy()
                
                # Only keep rows that match the selected best R_series
                pre_df = pre_df[pre_df['R_series'].isin(best_pre_pair)]
                non_pre_df = non_pre_df[non_pre_df['R_series'].isin(best_non_pre_pair)]
                
                # Count missing values in relevant columns
                pre_df['missing_count'] = pre_df[FP_feature_cols].isna().sum(axis=1)
                non_pre_df['missing_count'] = non_pre_df[FP_feature_cols].isna().sum(axis=1)

                # Sort and select top 2
                pre_folder_files = pre_df.sort_values(by='missing_count')['folder_file'].iloc[:2].tolist()
                non_folder_files = non_pre_df.sort_values(by='missing_count')['folder_file'].iloc[:2].tolist()
            else:
                # Safe fallback if only 1–2 values are returned, keep them directly
                pre_folder_files = pre_folder_files[:2]
                non_folder_files = non_pre_folder_files[:2]

            return pd.Series({'access_change': min_diff, 'FP_valid': pre_folder_files + non_folder_files})


                    
        cell_df = df.groupby('cell_id').apply(apply_check_unique).reset_index()
        diff_df = self.FP_df.groupby('cell_id').apply(calculate_percentage_diff).reset_index()
        cell_df = cell_df.merge(diff_df, on='cell_id', how='left')


        # APPLICATION FILES
        filtered_app_df = self.APP_df[ 
                                    # (self.APP_df['valid'] == True) & vaildators based on vairability - changing exclusion criteria 
                                    (self.APP_df['valid'] != False) & 
                                    (self.APP_df['application_order'] == 1) &
                                    (self.APP_df['replication_no'] == 1)]
        valid_files_dict = filtered_app_df.set_index('cell_id')['folder_file'].to_dict()
        cell_df['APP_valid'] = cell_df['cell_id'].map(valid_files_dict)

        # Check RA status in FP_df and APP_df
        fp_ra_df = self.FP_df[self.FP_df['RA'] == True][['cell_id', 'folder_file', 'RA_per_min']] #FP and APP dataframes where RA is True
        app_ra_df = self.APP_df[self.APP_df['RA'] == True][['cell_id', 'folder_file', 'RA_per_min']]
        combined_ra_df = pd.concat([fp_ra_df, app_ra_df])

        ra_folder_files = combined_ra_df.groupby('cell_id')['folder_file'].apply(list).to_dict()

        ra_avg_per_min = combined_ra_df.groupby('cell_id')['RA_per_min'].mean().to_dict() #average RA_per_min per cell_id

        cell_df['RA'] = cell_df['cell_id'].isin(ra_folder_files)
        cell_df['RA_folder_file'] = cell_df['cell_id'].map(ra_folder_files)
        cell_df['RA_per_min'] = cell_df['cell_id'].map(ra_avg_per_min)


        self.cache("cell_df", cell_df)
        self.save_excel("cell_df", cell_df)
        return cell_df
