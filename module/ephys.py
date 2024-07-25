import os
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import traceback
from tqdm import tqdm
from module.utils import * 
from module.getters import * 
tqdm.pandas()

@dataclass
class ephys:
    ' A class for accessing and generating data form the input feature_df. '
    filename: str
    raw_df: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.raw_df = self._get_raw_df()
        # self.cell_df = self._get_cell_df()
        self.FP_df = getCache(self.filename, 'FP_df') if isCached(self.filename, 'FP_df') else None
        self.APP_df = getCache(self.filename, 'APP_df') if isCached(self.filename, 'APP_df') else None

    def _get_raw_df(self) -> pd.DataFrame:
        """
        Build or Load the raw DataFrame from the Excel file.
        """
        return getRawDf(self.filename)
    
    #generators
    def generate_FP_df(self) -> pd.DataFrame:
        ''' Regenerates FP_df from scratch, 
            Subselects the FP data and extracts data from files, expanding FP_df columns.'''
        
        initial_columns = ['folder_file', 'cell_id', 'data_type', 'drug', 'replication_no', 'application_order', 'R_series', 'cell_type', 'cell_subtype']
        FP_df = self.raw_df[self.raw_df['data_type'] == 'FP'][initial_columns]

        FP_df = FP_df.progress_apply(lambda row: self._handle_extraction(row, self._process_FP_data), axis=1)
        additional_columns = [col for col in FP_df.columns if col not in initial_columns]
        FP_df = FP_df[initial_columns + additional_columns]

        cache(self.filename, 'FP_df', FP_df)
        return FP_df
    
    def generate_APP_df(self) -> pd.DataFrame:
        """Regenerates APP_df from scratch."""
        initial_columns = ['folder_file', 'cell_id', 'data_type', 'drug', 'drug_in', 'drug_out', 'replication_no', 'application_order', 'cell_type', 'cell_subtype']
        APP_df = self.raw_df [self.raw_df ['data_type'] == 'APP'][initial_columns]

        APP_df = APP_df.progress_apply(lambda row: self._handle_extraction(row, self._process_APP_data), axis=1)
        additional_columns = [col for col in APP_df.columns if col not in initial_columns]
        APP_df = APP_df[initial_columns + additional_columns]

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
        Builds cell_df with each row a cell_id and valid data is marked True in the columns 'data_type'.
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
            return group.agg({
                'treatment': lambda series: check_unique(series, cell_id),
                'cell_type': lambda series: check_unique(series, cell_id),
                'cell_subtype': lambda series: check_unique(series, cell_id)
            })

        # Group by 'cell_id' and apply the function
        cell_df = df.groupby('cell_id').apply(apply_check_unique).reset_index()
        
        #TO DO  populate cell_df


        return cell_df

    # data extractors
    def _process_FP_data(self, row: pd.Series) -> pd.Series:
        """Processing logic specific to FP data type. Could also handle FP_APP data if sufficient to analise."""
        V_array, I_array = load_file(row['folder_file'])

        row["max_firing"] = calculate_max_firing(V_array)
        (peak_voltages_all, peak_latencies_all, v_thresholds_all,
         peak_slope_all, AP_max_dvdt_all, peak_locs_corr_all,
         upshoot_locs_all, peak_heights_all, peak_fw_all,
         sweep_indices, sweep_indices_all) = ap_characteristics_extractor_main(
            row['folder_file'], V_array, all_sweeps=True)
        
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
        V_array, I_array = load_file(row['folder_file'])

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
        row['RMP_PRE'] = mean_RMP_PRE
        row['RMP_APP'] = mean_RMP_APP
        row['RMP_WASH'] = mean_RMP_WASH

        (peak_voltages_all, peak_latencies_all, v_thresholds_all,
            peak_slope_all, AP_max_dvdt_all, peak_locs_corr_all,
            upshoot_locs_all, peak_heights_all, peak_fw_all,
            sweep_indices, sweep_indices_all) = ap_characteristics_extractor_main(
                row.folder_file, V_array, all_sweeps=True)
        
        pAD_condition = lambda peak_voltage, threshold: threshold <= -65 and peak_voltage > 20
        if any(pAD_condition(peak_voltage, threshold) for peak_voltage, threshold in zip(peak_voltages_all, v_thresholds_all)):
            row['pAD'] = True
            row['pAD_locs'] = [peak_locs_corr_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if threshold <= -65 and peak_voltage > 20]
            row['PRE_pAD_locs'] = [peak_loc for peak_loc, sweep_index, peak_voltage, threshold in zip(peak_locs_corr_all, sweep_indices, peak_voltages_all, v_thresholds_all) if sweep_index < row['drug_in'] and pAD_condition(peak_voltage, threshold)]
            row['APP_pAD_locs'] = [peak_loc for peak_loc, sweep_index, peak_voltage, threshold in zip(peak_locs_corr_all, sweep_indices, peak_voltages_all, v_thresholds_all) if row['drug_in'] <= sweep_index <= row['drug_out'] and pAD_condition(peak_voltage, threshold)]
            row['WASH_pAD_locs'] = [peak_loc for peak_loc, sweep_index, peak_voltage, threshold in zip(peak_locs_corr_all, sweep_indices, peak_voltages_all, v_thresholds_all) if sweep_index > row['drug_out'] and pAD_condition(peak_voltage, threshold)]
        else:
            row['pAD_locs'] = []
        row['AP_locs'] = peak_locs_corr_all
        if len(peak_locs_corr_all) > 0:
            row['PRE_AP_locs'] = [peak_loc for peak_loc, sweep_index in zip(peak_locs_corr_all, sweep_indices) if sweep_index < row['drug_in']]
            row['APP_AP_locs'] = [peak_loc for peak_loc, sweep_index in zip(peak_locs_corr_all, sweep_indices) if row['drug_out'] >= sweep_index >= row['drug_in']]
            row['WASH_AP_locs'] = [peak_loc for peak_loc, sweep_index in zip(peak_locs_corr_all, sweep_indices) if sweep_index > row['drug_out']]
        return row

    def _process_pAD_hunter_data(self, row: pd.Series) -> pd.Series:
        V_array, I_array = load_file(row['folder_file'])

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
    


