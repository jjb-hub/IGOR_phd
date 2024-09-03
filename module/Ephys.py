import os
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import traceback
from tqdm import tqdm
from itertools import combinations
from module.utils import * 
from module.getters import getRawDf, calculate_max_firing, ap_characteristics_extractor_main, extract_FI_slope_and_rheobased_threshold, extract_FI_x_y, sag_current_analyser, tau_analyser, mean_inputR_APP_calculator, mean_RMP_APP_calculator

tqdm.pandas()

@dataclass
class Ephys:
    ''' 
    Organising relational dfs and extracting data form the input feature_df. 
    Attributes:
        raw_df: excel input mapping folder_files to features
        FP_df: extraction of firing property data (FP)
        APP_df: extraction of applications data (APP)
        cell_df: mapping of cells to features including change in access and FP_valid and APP_valid columns with valid folder_files
        pAD_hunter_df: last unofficial data_type needs developing* #TODO
          '''
    filename: str
    sampling_rate: float = 2e4

    def __post_init__(self):
        initiateFileSystem()
        self.raw_df = self._get_raw_df()
        self.FP_df = getCache(self.filename, 'FP_df') if isCached(self.filename, 'FP_df') else self.generate_FP_df()
        self.APP_df = getCache(self.filename, 'APP_df') if isCached(self.filename, 'APP_df') else self.generate_APP_df()
        self.cell_df = getCache(self.filename, 'cell_df') if isCached(self.filename, 'cell_df') else self.generate_cell_df()
        self.pAD_hunter_df = getCache(self.filename, 'pAD_hunter_df') if isCached( self.filename, 'pAD_hunter_df') else self.generate_pAD_hunter_df()

    def _get_raw_df(self) -> pd.DataFrame:
        """
        Build or Load the raw DataFrame from the Excel file.
        """
        return getRawDf(self.filename)
    
    #generators
    def generate_FP_df(self) -> pd.DataFrame:
        ''' Regenerates FP_df from scratch, 
            Subselects the FP data and extracts data from files, expanding FP_df columns.'''
        
        initial_columns = ['folder_file', 'cell_id', 'data_type', 'I_set', 'drug', 'replication_no', 'application_order', 'R_series', 'cell_type', 'cell_subtype']
        FP_df = self.raw_df[self.raw_df['data_type'] == 'FP'][initial_columns]

        FP_df = FP_df.progress_apply(lambda row: self._handle_extraction(row, self._process_FP_data), axis=1)
        additional_columns = [col for col in FP_df.columns if col not in initial_columns]
        FP_df = FP_df[initial_columns + additional_columns]
        self.cell_df = self.generate_cell_df() #regenerate as its related
        cache(self.filename, 'FP_df', FP_df)
        return FP_df


    def generate_APP_df(self) -> pd.DataFrame:
        """Regenerates APP_df from scratch."""
        initial_columns = ['folder_file', 'cell_id', 'data_type', 'I_set', 'drug', 'drug_in', 'drug_out', 'replication_no', 'application_order', 'cell_type', 'cell_subtype']
        APP_df = self.raw_df [self.raw_df ['data_type'] == 'APP'][initial_columns]

        APP_df = APP_df.progress_apply(lambda row: self._handle_extraction(row, self._process_APP_data), axis=1)
        additional_columns = [col for col in APP_df.columns if col not in initial_columns]
        APP_df = APP_df[initial_columns + additional_columns]
        self.cell_df = self.generate_cell_df() #regenerate as its related

        cache(self.filename, 'APP_df', APP_df)
        return APP_df
    


    def generate_pAD_hunter_df(self) -> pd.DataFrame:
        """Regenerates pAD_hunter_df from scratch."""
        initial_columns = ['folder_file', 'cell_id', 'data_type', 'drug', 'replication_no', 'application_order', 'cell_type', 'cell_subtype']
        pAD_hunter_df = self.raw_df [self.raw_df ['data_type'] == 'pAD_hunter'][initial_columns]

        pAD_hunter_df = pAD_hunter_df.apply(lambda row: self._handle_extraction(row, self._process_pAD_hunter_data), axis=1)
        additional_columns = [col for col in pAD_hunter_df.columns if col not in initial_columns]
        pAD_hunter_df = pAD_hunter_df[initial_columns + additional_columns]

        cache(self.filename, 'pAD_hunter_df', pAD_hunter_df)
        return pAD_hunter_df
    
    def generate_cell_df(self) -> pd.DataFrame:
        """
        Builds cell_df with each row a cell_id, access_change reported where possible and valid data is marked True in 'data_type' column i.e. "FP".
        """
        df = self.raw_df.copy()
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
            
            # Get folder files for the selected pairs
            pre_folder_files = pre_values[pre_values['R_series'].isin(best_pre_pair)]['folder_file'].tolist() 
            non_pre_folder_files = non_pre_values[non_pre_values['R_series'].isin(best_non_pre_pair)]['folder_file'].tolist()
            folder_files = pre_folder_files[0:2] + non_pre_folder_files[0:2]
            
            return pd.Series({'access_change': min_diff, 'FP_valid': folder_files})
        
        cell_df = df.groupby('cell_id').apply(apply_check_unique).reset_index()
        diff_df = self.FP_df.groupby('cell_id').apply(calculate_percentage_diff).reset_index()
        cell_df = cell_df.merge(diff_df, on='cell_id', how='left')

        # APPLICATION FILES
        filtered_app_df = self.APP_df[ (self.APP_df['valid'] == True) &
                                    (self.APP_df['application_order'] == 1) &
                                    (self.APP_df['replication_no'] == 1)]
        valid_files_dict = filtered_app_df.set_index('cell_id')['folder_file'].to_dict()
        cell_df['APP_valid'] = cell_df['cell_id'].map(valid_files_dict)

        cache(self.filename, 'cell_df', cell_df) # CREATE HIGHER FUNCTION TO FETCH 
        return cell_df

    # data extractors
    def _process_FP_data(self, row: pd.Series) -> pd.Series:
        """Processing logic specific to FP data type. Could also handle FP_APP data if sufficient to analise."""
        V_array , I_array, V_list = load_file(row['folder_file'])

        row["max_firing"] = calculate_max_firing(V_array)
        (peak_voltages_all, peak_latencies_all, v_thresholds_all,
         peak_slope_all, AP_max_dvdt_all, peak_locs_corr_all,
         upshoot_locs_all, peak_heights_all, peak_fw_all,
         sweep_indices, sweep_indices_all) = ap_characteristics_extractor_main(
            row['folder_file'], V_array)
        
        if any(threshold <= -65 and peak_voltage > 20 for peak_voltage, threshold in zip(peak_voltages_all, v_thresholds_all)):
            row['pAD'] = True
            row['pAD_locs'] = [peak_locs_corr_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if threshold <= -65 and peak_voltage > 20]

        step_current_values, ap_counts, V_rest, off_step_peak_locs, ap_frequencies_Hz = extract_FI_x_y(
            row['folder_file'], V_array, I_array, peak_locs_corr_all, sweep_indices_all)
        FI_slope, rheobase_threshold = extract_FI_slope_and_rheobased_threshold(
            row['folder_file'], step_current_values, ap_counts)
        row["rheobased_threshold"] = rheobase_threshold
        row["FI_slope"] = FI_slope

        row['AP_peak_voltages'] = peak_voltages_all[:10]
        row["voltage_threshold"] = v_thresholds_all[:10]
        row["AP_height"] = peak_heights_all[:10]
        row["AP_width"] = peak_fw_all[:10]
        row["AP_slope"] = peak_slope_all[:10]
        row["AP_latency"] = peak_latencies_all[:10]
        row["AP_dvdt_max"] = AP_max_dvdt_all[:10]

        row["tau_rc"] = tau_analyser(row['folder_file'], V_array, I_array, step_current_values, ap_counts)
        row["sag"] = sag_current_analyser(row['folder_file'], V_array, I_array, step_current_values, ap_counts)

        return row
    
    def _process_APP_data(self, row: pd.Series) -> pd.Series:
        """Generate APP_df from scratch, 
        Processing logic specific to APP data type."""
        V_array , I_array, V_list = load_file(row['folder_file'])

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
        row['RMP_PRE'] = mean_RMP_PRE[1:]
        row['RMP_APP'] = mean_RMP_APP
        row['RMP_WASH'] = mean_RMP_WASH

     
        (peak_voltages_all, peak_latencies_all  , v_thresholds_all,
        peak_slope_all  ,AP_max_dvdt_all,  peak_locs_corr_all, 
        upshoot_locs_all  , peak_heights_all  , peak_fw_all,
        peak_indices_all , sweep_indices_all) = ap_characteristics_extractor_main(row.folder_file, V_array)
        
        pAD_condition = lambda peak_voltage, threshold: threshold <= -65 and peak_voltage > 20
        if any(pAD_condition(peak_voltage, threshold) for peak_voltage, threshold in zip(peak_voltages_all, v_thresholds_all)):
            row['pAD'] = True
            row['pAD_locs'] = [peak_locs_corr_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if threshold <= -65 and peak_voltage > 20]
            row['pADcount_PRE'] = len([peak_loc for peak_loc, sweep_index, peak_voltage, threshold in zip(peak_locs_corr_all, sweep_indices_all, peak_voltages_all, v_thresholds_all) if sweep_index < row['drug_in'] and pAD_condition(peak_voltage, threshold)])
            row['pADcount_APP'] = len([peak_loc for peak_loc, sweep_index, peak_voltage, threshold in zip(peak_locs_corr_all, sweep_indices_all, peak_voltages_all, v_thresholds_all) if row['drug_in'] <= sweep_index <= row['drug_out'] and pAD_condition(peak_voltage, threshold)])
            row['pADcount_WASH'] = len([peak_loc for peak_loc, sweep_index, peak_voltage, threshold in zip(peak_locs_corr_all, sweep_indices_all, peak_voltages_all, v_thresholds_all) if sweep_index > row['drug_out'] and pAD_condition(peak_voltage, threshold)])
        else:
            row['pAD_locs'] = []
            row['pADcount_PRE'] = 0
            row['pADcount_APP'] = 0
            row['pADcount_WASH'] = 0

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

        #APP validators
        if len(peak_voltages_all)>0:
            if np.mean(np.array(peak_voltages_all)[~np.isnan(peak_voltages_all)]) < 30: #HARDCODE minimum 30 mV AP height to declare offset issues
                row['offset']= True
            else:
                peak_voltage_burst_max = group_AP_bursts(peak_locs_corr_all, sweep_indices_all, peak_voltages_all, burst_window_seconds=1)

                if (check_variability(peak_voltage_burst_max, Vairability_threshold=0.5) == False or
                    check_variability([row['RMP_PRE']]) == False or
                    check_variability([row['inputR_PRE']], Vairability_threshold=1) == False):
                    print('check')

                row['valid'] = (
                    check_variability(peak_voltage_burst_max, Vairability_threshold=0.5) and
                    check_variability([row['RMP_PRE']]) and
                    check_variability([row['inputR_PRE']], Vairability_threshold=1) #HARD CODE vaitability threshold 
                    )
        else:
            row['valid'] = (
                check_variability([row['RMP_PRE']]) and
                check_variability([row['inputR_PRE']], Vairability_threshold=1)
                )

 
        return row

    def _process_pAD_hunter_data(self, row: pd.Series) -> pd.Series:
        V_array , I_array, V_list = load_file(row['folder_file'])

        (peak_voltages_all, peak_latencies_all, v_thresholds_all,
        peak_slope_all, AP_max_dvdt_all, peak_locs_corr_all,
        upshoot_locs_all, peak_heights_all, peak_fw_all,
        sweep_indices, sweep_indices_all) = ap_characteristics_extractor_main(row.folder_file, V_array, all_sweeps=True)
        
        if any(threshold <= -65 and peak_voltage > 20 for peak_voltage, threshold in zip(peak_voltages_all, v_thresholds_all)):
            row['pAD'] = True
            row['pAD_locs'] = [peak_locs_corr_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if threshold <= -65 and peak_voltage > 20]
        return row

    #error handeling 
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
    


