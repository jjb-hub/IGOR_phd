import os
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from pyabf import ABF 
from module.Cachable import Cachable
import traceback
from scipy.optimize import curve_fit
from tqdm import tqdm
from IPython.display import display
from itertools import combinations
import igor2 as igor
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from module.action_potential_functions import calculate_max_firing, sweep_mean_RMP_calculator, sweep_mean_inputR_calculator, ap_characteristics_extractor_main, extract_FI_x_y, sag_current_analyser, mean_RMP_APP_calculator, spike_remover_nan, peak_finder,correct_I_offset_IF, denoise_steps, FI_slope_and_rheobase
from scipy.stats import ttest_ind
from module.Stats import Stats
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
    accepted_extensions = ['.ibw', '.abf']
    project_type: str = None # user can pass either 'application' or 'intrinsic_properties'

    def __post_init__(self):
        super().__init__(cache_dir=f"{ROOT}/{self.project}/cache")
        self.location = f"{ROOT}/{self.project}"
        self.input_dir = self._checkFileSystem("input")
        self.output_dir = self._checkFileSystem("output")
        self.figure_output_dir = self._checkFileSystem("figures")
        self.feature_df = self.load_feature_xlsx('features')
        self.check_project_type()
            

    def check_project_type(self):
        if self.project_type is None:
            if "data_type" in self.feature_df.columns:
                unique_types = set(self.feature_df["data_type"].dropna().unique())
                if "APP_IC" in unique_types:
                    self.project_type = "application"
                elif unique_types & {"st_VC", "ramp_IC", "IV_VC", "spont_IC", "IF_IC"}:
                    self.project_type = "intrinsic_properties"
                else:
                    raise ValueError("Unrecognized data_type values in features.xlsx.")
            else:
                raise ValueError("features.xlsx must contain 'data_type' column.")
        # print(f"Project type set to: {self.project_type}")

    def _get_extension(self, folder_file: str) -> str:
        """
        Parameteres:
            folder_file: str - identifier of unique file e.g.  folder1/file_no_extension
        Returns:
            str: Either '.ibw' or '.abf'
        Raises:
            FileNotFoundError: If no accepted extensions are found in the folder.
        """
        if '/' in folder_file:
            folder, _ = folder_file.split('/', 1)
            base_dir = os.path.join(self.input_dir, 'PatchData', folder)
        else:
            base_dir = os.path.join(self.input_dir, 'PatchData')
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Folder not found: {base_dir}")
        files = os.listdir(base_dir)
        for ext in self.accepted_extensions:
            if any(f.endswith(ext) for f in files):
                return ext
        raise FileNotFoundError(f"No known file extensions {self.accepted_extensions} found in {base_dir}")
    
    def load_data(self, folder_file: str):
        """
        Parameters:
            folder_file (str): The folder_file identifier (can include a subfolder).
        Returns:
            tuple: (V_array, I_array, V_list)
        """
        extension = self._get_extension(folder_file)
        if extension == '.ibw':
            return self.IGOR_load(folder_file)
        elif extension == '.abf':
            return self.ABF_load(folder_file)
        else:
            raise ValueError(f"Unsupported extension type: {extension}")
    
    def load_feature_xlsx(self, filename: str):
        """Loads data from cache or an Excel file."""
        if self.isCached(filename):
            return self.getCache(filename)
        
        filepath = os.path.join(self.input_dir, f"{filename}.xlsx")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Excel file {filename} not found in {self.input_dir}")

        required_columns = ['folder_file', 'cell_id', 'data_type', 'treatment']
        converters = {'drug_in': int, 'drug_out': int}
        df = pd.read_excel(filepath, converters=converters)
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required column(s) in features.xlsx: {missing_cols}")

        for col in df.columns: # detect 1/0 True/False columns as boolian
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                # Only convert non-null values to boolean
                df[col] = df[col].where(df[col].isna(), df[col].astype(bool))
                print(f"[INFO] Column '{col}' inferred as boolean (True/False).")

        # if 'cell_subtype' in df.columns:
        #     df['cell_subtype'].fillna(np.nan, inplace=True)

        self.cache(filename, df)
        return df

            
    def ABF_load(self, folder_file: str):
        """
        Loads data from .abf (Axon) files using pyabf.

        Returns:
            V_array: 2D numpy array (time x sweeps) of voltage
            I_array: Currently None (or could be second channel if needed)
            V_list: 1D flattened array (column-major sweep order)
        """
        
        path = os.path.join(self.input_dir, 'PatchData', folder_file + '.abf')
        if not os.path.exists(path):
            raise FileNotFoundError(f"ABF file not found: {path}")

        abf = ABF(path)
        sampling_rate_hz = abf.dataRate

        # Identify voltage and current channels by unit
        unit_map = {i: unit for i, unit in enumerate(abf.adcUnits)}
        voltage_ch = next((i for i, unit in unit_map.items() if 'V' in unit.upper()), None)
        current_ch = next((i for i, unit in unit_map.items() if 'A' in unit.upper()), None)

        if voltage_ch is None or current_ch is None:
            raise ValueError(f"Couldn't identify voltage/current channels from units: {abf.adcUnits}")

        # Channel names (optional, for debug/info)
        # voltage_name = abf.adcNames[voltage_ch]
        # current_name = abf.adcNames[current_ch]

        num_sweeps = abf.sweepCount
        num_points = abf.sweepPointCount

        V_array = np.zeros((num_points, num_sweeps))
        I_array = np.zeros((num_points, num_sweeps))

        for i in range(num_sweeps):
            abf.setSweep(i, channel=voltage_ch)
            V_array[:, i] = abf.sweepY
            abf.setSweep(i, channel=current_ch)
            I_array[:, i] = abf.sweepY

        V_list = V_array.ravel(order='F')  # Column-major, like IGOR

        return V_array, I_array, V_list
            

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
    

    def inspect_folder_file(self, folder_file, stacked=False, n_sweeps=None):
        '''
        Plots any waveform based off folder_file.
        Stacked will plot each column on top of each other, defaults to False.
        '''
        feature_df = self.load_feature_xlsx('features')
        display(feature_df[feature_df['folder_file'] == folder_file])  # Show file info

        V_array , I_array, V_list = self.load_data(folder_file)
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
        plt.xlabel('Time (no samples)')
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

        # df = df.progress_apply(lambda row: self._handle_extraction(row, self.process), axis=1) # log errors
        df = df.progress_apply(lambda row: self._debug_extraction(row, self.process), axis=1) # raise errors
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
class st_VC(EphysData):
    filename: str = "st_VC_df"
    data_type: str = 'st_VC'
    
    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'treatment', 'region']
        super().__post_init__()
    
    def process(self, row: pd.Series) -> pd.Series:
        """Extract Rs, Rm, Cm, tau from each voltage step in the st_VC protocol.""" 
        V_array, I_array, _ = self.load_data(row['folder_file'])
        V = V_array[:, 0]  # mV
        I = I_array[:, 0]  # pA
        dt = 1 / self.sampling_rate
        t = np.arange(len(I)) * dt

        # Detect voltage steps 
        dV = np.diff(V)
        step_indices = np.where(np.abs(dV) > 0.5)[0]  # threshold in mV
        if len(step_indices) < 1:
            raise ValueError("No voltage steps detected.")

        #  time between steps to define analysis window
        step_durations = np.diff(step_indices) * dt
        avg_step_duration = np.median(step_durations)
        window_post = int(min(0.02, 0.5 * avg_step_duration) / dt)
        window_pre = int(min(0.005, 0.2 * avg_step_duration) / dt)
        steady_window = int(0.01 / dt)

        # voltage step size 
        unique_Vs, counts = np.unique(np.round(V, 1), return_counts=True)
        if len(unique_Vs) < 2:
            raise ValueError("Not enough voltage levels to define delta_V.")
        sorted_Vs = unique_Vs[np.argsort(-counts)]
        V_baseline_mode, V_step_mode = sorted_Vs[:2]
        global_delta_V = abs(V_step_mode - V_baseline_mode)

        Rs_list, Rm_list, tau_list, Cm_list = [], [], [], []

        for idx in step_indices:
            start = max(0, idx + 1)
            end = min(len(t), idx + 1 + window_post)
            baseline = max(0, idx - window_pre)
            baseline_I = np.mean(I[baseline:start])

            if end - start < 5:
                continue  # too short to analyze

            local_delta_V = V[start] - V[baseline] #less accurate than global assuming global is consistent
            if np.abs(local_delta_V) < 1e-3:
                continue  # ignore tiny steps

            I_step = I[start:end]
            t_step = t[start:end] - t[start]

            # get dI and steady state I
            I_peak = np.max(I_step) if local_delta_V > 0 else np.min(I_step)
            I_deflection = abs(I_peak - baseline_I)
            I_steady = np.mean(I_step[-steady_window:])

            try:
                Rs = global_delta_V * 1e-3 / (I_deflection * 1e-12)
                Rm = global_delta_V * 1e-3 / (abs(I_steady - baseline_I) * 1e-12)
            except ZeroDivisionError:
                continue

            # Tau: exponential decay fit
            def exp_decay(t, A, tau, C):
                return A * np.exp(-t / tau) + C

            try:
                popt, _ = curve_fit(exp_decay, t_step, I_step, p0=[I_deflection, 0.01, I_steady])
                tau = popt[1]
            except Exception:
                tau = np.nan

            Cm = tau / Rm if Rm != 0 else np.nan

            # BOUNDS CHECK with debug
            if not (0 < Rs < 1e9):
                print(f"Rs out of bounds: {Rs:.2e} Ω ({Rs/1e6:.2f} MΩ) | cell: {row.cell_id} | step: {idx}")
                Rs = np.nan
            if not (1e6 < Rm < 1e9):
                print(f"Rm out of bounds: {Rm:.2e} Ω ({Rm/1e6:.2f} MΩ) | cell: {row.cell_id} | step: {idx}")
                Rm = np.nan
            if not (0 < Cm < 500e-12):
                print(f"Cm out of bounds: {Cm:.2e} F ({Cm*1e12:.2f} pF) | cell: {row.cell_id} | step: {idx}")
                Cm = np.nan
            if not (0 < tau < 1):
                print(f"tau out of bounds: {tau:.2e} s ({tau*1e3:.2f} ms) | cell: {row.cell_id} | step: {idx}")
                tau = np.nan

            # # # plot to check
            # plt.figure()
            # plt.plot(t_step * 1e3, I_step, label="I_step")
            # if not np.isnan(tau):
            #     plt.plot(t_step * 1e3, exp_decay(t_step, *popt), 'r--', label=f"fit τ = {popt[1]*1e3:.2f} ms")
            # plt.axhline(I_peak, color='purple', linestyle=':', label=f"I_peak (amp = {I_deflection:.1f} pA)")
            # plt.axhline(I_steady, color='green', linestyle='--', label="I_steady")
            # plt.xlabel("Time (ms)")
            # plt.ylabel("Current (pA)")
            # plt.legend()
            # plt.title(f"{row.cell_id} step {idx}")
            # plt.show()

            Rs_list.append(Rs / 1e6)      # to MOhm
            Rm_list.append(Rm / 1e6)
            tau_list.append(tau * 1e3 if not np.isnan(tau) else np.nan)  # ms
            Cm_list.append(Cm * 1e12 if not np.isnan(Cm) else np.nan)    # pF

        # Final averaged values
        row['Rs_MOhm'] = np.nanmean(Rs_list) #Ra same
        row['Rm_MOhm'] = np.nanmean(Rm_list)
        row['tau_ms'] = np.nanmean(tau_list)
        row['Cm_pF'] = np.nanmean(Cm_list)

        return row


@dataclass
class ramp_IC(EphysData):
    filename: str = "ramp_IC_df"
    data_type: str = 'ramp_IC'
    
    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'treatment', 'region']
        super().__post_init__()
    
    def process(self, row: pd.Series) -> pd.Series:
        """Extract rheobase (pA), voltage_threshold (mV) and AP_charecteristics of the first AP."""  #HERE TO CHECK AND WORK FOR HFD
        V_array, I_array, _ = self.load_data(row['folder_file'])
        dt = 1 / self.sampling_rate
        t = np.arange(len(I_array)) * dt

        peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_rise_all  , peak_max_dvdt_all,  peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all   , peak_indices_all , sweep_indices_all , peak_decay_all = ap_characteristics_extractor_main(row['folder_file'], V_array)        
        
        rheobase_list = []
        holding_I_list = []
        threshold_list = []
        height_list = []
        rise_list = []
        decay_list = []
        fwhm_list = []
        sweep_RMP_mV = []

        num_sweeps = V_array.shape[1]

        for sweep_idx in range(num_sweeps):
            V_sweep = V_array[:, sweep_idx]
            I_sweep = I_array[:, sweep_idx]

            # Run AP detection on this sweep only
            (
                peak_voltages_all, peak_latencies_all, v_thresholds_all, peak_rise_all,
                peak_max_dvdt_all, peak_locs_corr_all, upshoot_locs_all, peak_heights_all,
                peak_fw_all, peak_indices_all, sweep_indices_all, peak_decay_all
            ) =  ap_characteristics_extractor_main(row['folder_file'], V_sweep)

            if peak_voltages_all is None or len(peak_voltages_all) == 0: 
                continue  # No AP detected

            firt_AP_peak_loc = peak_locs_corr_all[0]

            try:
                dI = np.diff(I_sweep) 
                ramp_end_idx = np.argmax(np.abs(dI)) 
                offset =  5 * round(np.mean(I_sweep[ramp_end_idx+50:]) / 5)     #np.mean(I_sweep[ramp_end_idx+50:])  
                rheobase = I_sweep[firt_AP_peak_loc]  # pA
                rmp = np.mean(V_sweep[ramp_end_idx+50:])
            except IndexError:
                continue  # Skip corrupted index
            
            sweep_RMP_mV.append(rmp)
            rheobase_list.append(rheobase)
            holding_I_list.append(offset)
            threshold_list.append(v_thresholds_all[0])
            height_list.append(peak_heights_all[0])
            rise_list.append(peak_rise_all[0])
            decay_list.append(peak_decay_all[0])
            fwhm_list.append(peak_fw_all[0])


        if any((lst is None or len(lst) == 0) for lst in [rheobase_list, holding_I_list, threshold_list, height_list, rise_list, decay_list, fwhm_list]):
            print(f"⚠️ File {row['folder_file']} has empty lists, inspect data.")


        row['holding_I'] = np.nanmean(holding_I_list)
        row['RMP_mV'] = np.nanmean(sweep_RMP_mV)
        row['ramp_rheobase_pA'] = np.nanmean(rheobase_list)
        row['ramp_voltage_threshold_mV'] = np.nanmean(threshold_list)
        row['AP_height_mV'] = np.nanmean(height_list)
        row['AP_rise_mV_ms'] = np.nanmean(rise_list)
        row['AP_decay_mV_ms'] = np.nanmean(decay_list)
        row['AP_width_ms'] = np.nanmean(fwhm_list)

       

        return row


@dataclass
class IV_VC(EphysData):
    filename: str = "IV_VC_df"
    data_type: str = 'IV_VC'
    
    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'treatment', 'region']
        super().__post_init__()
    
    def process(self, row: pd.Series) -> pd.Series:
        """Extract steady-state current (I_steady) for each voltage step (V_inj).""" 
        V_array, I_array, _ = self.load_data(row['folder_file'])
        dt = 1 / self.sampling_rate
        t = np.arange(len(I_array)) * dt

        n_sweeps = V_array.shape[1]
        I_steady_list = []
        V_inj_list = []

        for sweep in range(n_sweeps):
            V = V_array[:, sweep]
            I = I_array[:, sweep]

            #detect step start and finish of V step using dvdt
            dV = np.gradient(V)
            thresh = np.std(dV) * 3
            step_start_candidates = np.where(np.abs(dV) > thresh)[0] 

            if len(step_start_candidates) < 2:
                continue  # skip if no clear step
            start = step_start_candidates[0]
            end = step_start_candidates[-1] 
            step_len = end-start
            steady_start = int(start + 0.75 * step_len)

            V_steady = np.mean(V[steady_start:end])
            I_clean_step = spike_remover_nan(I[steady_start:end], threshold_sd=0.5) # remove APs / spikes
            I_steady = np.nanmean(I_clean_step)

            V_inj_list.append(V_steady)
            I_steady_list.append(I_steady)

        row['V_steps_mV'] = V_inj_list
        row['I_step_steady_mV'] = I_steady_list
        row['RMP_mV'] = np.mean(V[end+500:])
        row['holding_I']=np.mean(I[end+500:])
        return row


@dataclass
class spont_IC(EphysData):    #TODO BUILD EXCLUSION - traces with high vairability of baseline and remove APs
    filename: str = "spont_IC_df"
    data_type: str = 'spont_IC'
    amplitude_threshold: float = 0.9 # mV
    rise_time_range: tuple = (0.5e-3, 5e-3) # 0.5 - 5 ms
    decay_time_range: tuple = (2e-3, 20e-3) # 2 - 20 ms
    
    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'treatment', 'region']
        super().__post_init__()
    
    def process(self, row: pd.Series) -> pd.Series:
        """
        Extract AP, sEPSP and sIPSP frequency.
        Designed for a gap free recording. 
        """  
        V_array, I_array, _ = self.load_data(row['folder_file'])
        dt = 1 / self.sampling_rate
        t = np.arange(len(I_array)) * dt

        # Baseline correction
        baseline = np.median(V_array)
        trace = (V_array - baseline).flatten()

        #  EPSPs
        peaks, rise_times, amplitudes, frequency = peak_finder(
            trace,
            height=self.amplitude_threshold,
            smoothing_kernel = 10,
            prominence=(self.amplitude_threshold / 2, None), 
            rise_time_range =  (0.2e-3, 10e-3),
            width=None, #(self.decay_time_range[0] / dt, self.decay_time_range[1] / dt),
            dt=dt,
            distance=None, #int(self.rise_time_range[0] / dt),
            polarity='positive'  # avoid clustering
        )  

        # # PLOT TO CHECK 
        # plt.figure(figsize=(12, 4))
        # plt.plot(trace, label='Voltage trace', color='black', linewidth=0.5)
        # plt.plot(peaks, trace[peaks], 'r.', label='sEPSPs', markersize=10)
        # plt.xlabel('Time (samples)')
        # plt.ylabel('Voltage (mV)')
        # plt.title(f"Detected sEPSPs in {row['folder_file']}")
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        row['sEPSP_frequency_Hz'] = frequency
        row['sEPSP_rise_times_ms'] = rise_times #check units
        row['sEPSP_amplitudes_mV'] = amplitudes #check units
        return row



@dataclass
class IF_IC(EphysData):    
    filename: str = "IF_IC_df"
    data_type: str = 'IF_IC'

    
    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'treatment', 'region', 'cell_subtype', 'cell_type', 'R_series'] # R_series is redundant for pCLAMP data #TODO
        
        super().__post_init__()
    
    def process(self, row: pd.Series) -> pd.Series:
        """
        Processes current-clamp step protocols (I–F curves).

        Takes: 
            Voltage (V) and current (I) traces from a single cell recording.

        Returns:
            The input row (pd.Series) with extracted properties added. Columns added include:
                '%_sag'
                'IF_rheobase_pA', 'IF_slope'
                'valid_APs'
                'AP_peaks_mV', 'IF_voltage_threshold_mV', 'AP_height_mV', 
                'AP_width_ms', 'AP_rise_mV_ms', 'AP_decay_mV_ms', 
                'AP_latency_ms', 'AP_max_rise_mV_ms'
                'I_steps_pA', 'AP_frequencies_Hz', 'max_firing_Hz'
                'off_step_peak_locs'
                'holding_I', 'RMP_mV'
        """
        V_array, I_array, _ = self.load_data(row['folder_file'])
        dt = 1 / self.sampling_rate
        t = np.arange(len(I_array)) * dt

        peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_rise_all  , peak_max_dvdt_all,  peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all   , peak_indices_all , sweep_indices_all , peak_decay_all = ap_characteristics_extractor_main(row['folder_file'], V_array)  
        if len(peak_voltages_all)==0: #returns is no APs are detected
            return row
    
        
        I_array_offset, offset = correct_I_offset_IF(I_array) 
        I_array_adj_clean = denoise_steps(I_array_offset)
        I_steps_pA , AP_frequencies_Hz, V_rest , off_step_peak_locs = extract_FI_x_y(row['folder_file'], V_array, I_array_adj_clean, self.sampling_rate)
        FI_slope, rheobase_threshold, valid_APs = FI_slope_and_rheobase(row['folder_file'], I_steps_pA, AP_frequencies_Hz)

        row["%_sag"] = sag_current_analyser(row['folder_file'], V_array, I_array_adj_clean, I_steps_pA, AP_frequencies_Hz)
        row["IF_rheobase_pA"] = rheobase_threshold
        row["IF_slope"] = FI_slope
        row['valid_APs'] = valid_APs

        row['AP_peaks_mV'] = peak_voltages_all[:10] #turn around on APs
        row["IF_voltage_threshold_mV"] = v_thresholds_all[:10] #mV at upshoot
        row["AP_height_mV"] = peak_heights_all[:10]
        row["AP_width_ms"] = peak_fw_all[:10]
        row["AP_rise_mV_ms"] = peak_rise_all[:10]
        row["AP_decay_mV_ms"] = peak_decay_all[:10]
        row["AP_latency_ms"] = peak_latencies_all[:10]
        row["AP_max_rise_mV_ms"] = peak_max_dvdt_all[:10]

        row['I_steps_pA'] = I_steps_pA
        row['AP_frequencies_Hz'] = AP_frequencies_Hz
        row['max_firing_Hz'] = calculate_max_firing(V_array)

        row['off_step_peak_locs']=off_step_peak_locs
        row["RMP_mV"]=V_rest
        row['holding_I'] = offset 

        # retro axonal action potential detection RA APs
        try:
            cell_threshold = np.mean(row['IF_voltage_threshold_mV'])
        except:
            cell_threshold = -45 #so when you -20 is 65 for cells without FP
        RA_condition = lambda peak_voltage, threshold: threshold <= (cell_threshold - 20) and peak_voltage > 0 
        if any(RA_condition(peak_voltage, threshold) for peak_voltage, threshold in zip(peak_voltages_all, v_thresholds_all)):
            row['RA'] = True
            row['RA_locs'] = [peak_locs_corr_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if threshold <= -65 and peak_voltage > 20]
            row['RA_per_min'] = len(row['RA_locs']) / V_array.shape[0] * V_array.shape[1] / self.sampling_rate / 60 #RA/minute

        # FILE VALIDATOR 
        if np.mean(np.array(peak_voltages_all[:10])[~np.isnan(peak_voltages_all[:10])]) < 15: #mean of first 11 AP peaks is less than 15mV the file is marked invalid
            row['valid'] = False 

        return row



@dataclass
class IV_IC(EphysData):    
    filename: str = "IV_IC_df"
    data_type: str = 'IV_IC'

    
    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'treatment']
        super().__post_init__()
    
    def process(self, row: pd.Series) -> pd.Series:
        """
        Designed for a hyperpolarising step protocol of a single step per sweep. 
        Return:
            sag: ratio of sag current (* 100 => %)
            V_steady: steady state V during I step
            I_injected: I injection of step
            RMP:  restimg membrane potential (off step - check for holding current)
        
        """  
        V_array, I_array, _ = self.load_data(row['folder_file'])
        dt = 1 / self.sampling_rate
        t = np.arange(len(I_array)) * dt

        
        I_array_offset, offset = correct_I_offset_IF(I_array) # pCLAMP data with holding_I attached to steps | not IGOR data
        I_array_adj_clean = denoise_steps(I_array_offset)

        step_current_values, AP_frequencies_Hz, V_rest_FI, off_step_peak_locs = extract_FI_x_y(row['folder_file'], V_array, I_array_adj_clean, self.sampling_rate)
        if any(x < 0 for x in step_current_values):
            sag_ratio, asym_current, step_current, V_rest_sag =sag_current_analyser(row['folder_file'], V_array, I_array_adj_clean, step_current_values, AP_frequencies_Hz)
        else:
            print (f"No negative I steps for {row['folder_file']}, unable to calculate sag")
            sag_ratio, asym_current, step_current, V_rest_sag = np.nan, np.nan, np.nan, np.nan

        row['%_sag']=sag_ratio
        row['V_step_steady_mV']=asym_current
        row['I_steps_pA']=step_current
        row['RMP_mV']= V_rest_sag
        row['holding_I']=offset

        return row

@dataclass
class APP_IC(EphysData):
    
    filename: str = "APP_IC_df"
    data_type: str = 'APP_IC'

    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'I_set', 'treatment', 'drug_in', 'drug_out', 'cell_type', 'cell_subtype']
        super().__post_init__()

    def process(self, row: pd.Series) -> pd.Series:
        """Generate APP_IC_df from scratch, 
        Processing logic specific to APP data type."""
        V_array , I_array, V_list = self.load_data(row['folder_file'])

        if I_array is not None and (I_array[:, 0] != 0).any():
            row['sweep_inputR_MOhm']=sweep_mean_inputR_calculator(V_array, I_array)
            # input_R_PRE, input_R_APP, input_R_WASH = mean_inputR_APP_calculator(V_array, I_array, row.drug_in, row.drug_out)
            # row['inputR_PRE'] = input_R_PRE
            # row['inputR_APP'] = input_R_APP
            # row['inputR_WASH'] = input_R_WASH
            pass_I_array = I_array
        else:
            row['sweep_inputR_MOhm']= np.nan
            # row['inputR_PRE'] = []
            # row['inputR_APP'] = []
            # row['inputR_WASH'] = []
            pass_I_array = None


        #TODO REMOVE AFTER CHECHING USE #UPDATE HIST
        # mean_RMP_PRE, mean_RMP_APP, mean_RMP_WASH = mean_RMP_APP_calculator(V_array, row.drug_in, row.drug_out, I_array=pass_I_array) #mean per sweep
        # row['RMP_PRE'] = mean_RMP_PRE
        # row['RMP_APP'] = mean_RMP_APP
        # row['RMP_WASH'] = mean_RMP_WASH

        row['sweep_RMP_mV']=sweep_mean_RMP_calculator(V_array, I_array=pass_I_array)

        peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_rise_all  , peak_max_dvdt_all,  peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all   , peak_indices_all , sweep_indices_all , peak_decay_all = ap_characteristics_extractor_main(row.folder_file, V_array)

        
        # sweep_AP_count
        if len(v_thresholds_all)>0:
            if all (x > row.drug_in for x in sweep_indices_all):
                row['induced_APs'] = True #unused TODO
            APs_per_sweep = np.zeros(V_array.shape[1], dtype=int)
            unique, counts = np.unique(sweep_indices_all, return_counts=True)
            APs_per_sweep[unique] = counts
            row['sweep_AP_count']=APs_per_sweep 

        else:
            row['sweep_AP_count']=np.zeros(V_array.shape[1], dtype=int)
            
        IF_IC_df = self.getCache("IF_IC_df")
        try:
            FP_cell_id_PRE = IF_IC_df[(IF_IC_df['cell_id'] == row['cell_id']) & (IF_IC_df['treatment'] == 'PRE')]
            cell_threshold = (FP_cell_id_PRE['voltage_threshold'].apply(lambda x: sum(x) / len(x) if isinstance(x, list) else x)).mean()
        except:
            cell_threshold = -45 #so when you -20 is 65 for cells without FP

        RA_condition = lambda peak_voltage, threshold: threshold <= (cell_threshold - 20) and peak_voltage > 0

        if any(RA_condition(peak_voltage, threshold) for peak_voltage, threshold in zip(peak_voltages_all, v_thresholds_all)):
            row['RA'] = True
            row['RA_locs'] = [peak_locs_corr_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if RA_condition(peak_voltage, threshold)]
            row['RA_sweep_locs'] = [sweep_indices_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if RA_condition(peak_voltage, threshold)]
            row['RA_per_min'] = len(row['RA_locs']) / V_array.shape[0] * V_array.shape[1] / self.sampling_rate / 60 

            # sweep_RA_count
            RA_sweep_locs = row['RA_sweep_locs']
            RA_per_sweep = np.zeros(V_array.shape[1], dtype=int)
            unique, counts = np.unique(RA_sweep_locs, return_counts=True)
            RA_per_sweep[unique] = counts
            row['sweep_RA_count'] = RA_per_sweep
            row['sweep_SAP_count'] = row['sweep_AP_count'] - row['sweep_RA_count']

        else:
            row['RA_locs'] = []
            row['sweep_RA_count'] = np.zeros(V_array.shape[1], dtype=int)
            row['sweep_SAP_count'] = row['sweep_AP_count']

        row['AP_locs'] = peak_locs_corr_all
        row['AP_sweep_locs'] = sweep_indices_all
        row['peak_voltages_all'] = peak_voltages_all

        # GENERIC functions
        def check_variability(values, vairability_threshold=0.30): 
            """Check if variability of values exceeds the given threshold."""
            values = np.array(values)[~np.isnan(values)]
            if len(values) <= 1:
                return True
            min_val = np.min(values)
            max_val = np.max(values)
            # print(f" % var  {abs((max_val - min_val) / min_val)}")
            return abs((max_val - min_val) / min_val) <= vairability_threshold
        
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
        baseline = row['sweep_RMP_mV'][1:int(row.get('drug_in', 0))]
        if check_variability(baseline, vairability_threshold=0.3)  == False: #assigns True if < vairability threshold
            row['valid'] = False 

        if len(peak_voltages_all)>0: # if APs 
            if np.mean(np.nanmean(peak_voltages_all)) < 15: #HARDCODE minimum 15 mV AP height 
                row['valid']= False

            peak_voltage_burst_max = group_AP_bursts(peak_locs_corr_all, sweep_indices_all, peak_voltages_all, burst_window_seconds=1)
            ap_burst_valid = unidirectional_trend(peak_voltage_burst_max, threshold=10)
            if ap_burst_valid == False:
                row['valid'] = False

        rmp_valid = unidirectional_trend(row['sweep_RMP_mV'], threshold=20) #assigns True if # REFACTOR as not used in plotter
        if  rmp_valid == False:
            row['valid'] = False

        else:
            row['valid'] = None #could be True
        return row
        
class Hunter(EphysData):
    '''Handels data type Hunter currently just fetching the RA locations.'''
    filename: str = "RA_hunter_df"
    data_type: str = 'Hunter'

    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'treatment', 'cell_type', 'cell_subtype']
        super().__post_init__()

    def process(self, row: pd.Series) -> pd.Series:
        V_array , I_array, V_list = self.load_data(row['folder_file'])


        peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_rise_all  , peak_max_dvdt_all,  peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all   , peak_indices_all , sweep_indices_all , peak_decay_all = ap_characteristics_extractor_main(row.folder_file, V_array)

        
        if any(threshold <= -65 and peak_voltage > 20 for peak_voltage, threshold in zip(peak_voltages_all, v_thresholds_all)):
            row['RA'] = True
            row['RA_locs'] = [peak_locs_corr_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if threshold <= -65 and peak_voltage > 20]
        return row



@dataclass
class Ephys(EphysData):
    ''' 
    Buiilding aggregate df with cell info based off extracted data from each data type in either : 
        application  ['APP_IC', 'IF_IC']         or       intrinsic_properties ['st_VC', 'ramp_IC', 'IV_VC', 'spont_IC', 'IF_IC' ] +AMPA?NMDA + PPR to come #TODO
        
        feature_df: excel input mapping folder_files to features
        
        ~ application = multiple timepoints                         
        IF_IC_df: extraction of firing property data (IF_IC) 
        APP_IC_df: extraction of applications data (APP_IC) 
        
        ~ intrinsic_properties = one timepoint
        st_VC: .. ect 


    Generates:
        cell_df: mapping of cells to features including change in access and IF_IC_valid and APP_IC_valid columns with valid folder_files
          '''
    filename: str = 'cell_df'
    sampling_rate: float = 2e4
    
    def __post_init__(self):
        Project.__post_init__(self) # initates project only to get self.project_type

        if self.project_type == 'application': #TODO change to loop for data types in project
            self.IF_IC_df = IF_IC(self.project).df 
            self.APP_IC_df = APP_IC(self.project).df

        elif self.project_type == 'intrinsic_properties':
            self.st_VC_df = st_VC(self.project).df
            self.IV_VC_df = IV_VC(self.project).df
            self.ramp_IC_df = ramp_IC(self.project).df
            self.IF_IC_df = IF_IC(self.project).df
            self.spont_IC_df = spont_IC(self.project).df
            
        super().__post_init__() # initiales all parent calsses including EphysData which will run generate()
        
    
    def generate(self) -> pd.DataFrame:
        """
        Builds cell_df with each row a cell_id, Rs_pct_change, keeps a record of the folder_files used in column f"{data_type}_folder_files".
        """
        if self.project_type == 'application':
            return self.generate_application_cell_df()
        elif self.project_type == 'intrinsic_properties':
            return self.generate_intrinsic_cell_df()

    def generate_intrinsic_cell_df(self):
        df = self.feature_df.copy()
        cell_wise_columns = ['cell_type', 'cell_subtype', 'p_age', 'treatment', 'region', 'sex', 'subject_id'] 
        cell_df = (df.groupby('cell_id')
                    .apply(lambda g: self.apply_check_unique(g, unique_cols=cell_wise_columns))
                    .reset_index()
                )
        
        # adds Rs_MOhm change from st_VC_df (track folder_files used)
        rs_changes = []
        for cell_id, group in self.st_VC_df.groupby("cell_id"):
            group = group.sort_values(by="folder_file")
            rs_values = group["Rs_MOhm"].values
            folder_files = group["folder_file"].tolist()

            if len(group)<2:
                print(f"Warning: cell_id {cell_id} has < 2 st_VC enteries, unable to calculate access change.")
                rs_changes.append((cell_id, np.nan, np.nan, []))
                continue

            # if len(group) != 2:
            #     print(f"Warning: cell_id {cell_id} has {len(group)} st_VC entries (expected 2)") 

            first_val, last_val = rs_values[0], rs_values[-1]
            rs_values = group["Rs_MOhm"].values
            abs_change = abs(last_val- first_val)
            pct_change = ((last_val- first_val) / first_val) * 100 if first_val != 0 else np.nan
            used_folder_files = [folder_files[0], folder_files[-1]]

            rs_changes.append((cell_id, abs_change, pct_change, used_folder_files))

        rs_df = pd.DataFrame(
            rs_changes, 
            columns=["cell_id", "Rs_abs_change", "Rs_pct_change", self.folder_files_col("st_VC")]
        )
        cell_df = cell_df.merge(rs_df, on="cell_id", how="left")
    

        # df , columns to reduce, data_type, average
        reductions = [
            (self.IF_IC_df, ["I_steps_pA", "AP_frequencies_Hz"], "IF_IC", False),
            (self.ramp_IC_df, ["ramp_rheobase_pA", "ramp_voltage_threshold_mV", "AP_height_mV",
                            "AP_rise_mV_ms", "AP_decay_mV_ms", "AP_width_ms"], "ramp_IC", True),
            (self.IV_VC_df, ["I_step_steady_mV", "V_steps_mV"], "IV_VC", False)
            #(self.spont_IC_df, ["sEPSP_frequency_Hz", "sEPSP_rise_times_ms", "sEPSP_amplitudes_mV"], "spont_IC", True), #currently not used
        ]

        for df_src, cols, data_type, avg in reductions:
            reduced = self.reduce_cellwise(df_src, cols, average=avg) # columns per cell
            cell_df = cell_df.merge(reduced, on="cell_id", how="left")

            # select best folder_file per cell based on RMP and holding current
            folder_col = self.folder_files_col(data_type)
            df_best = df_src.copy()
            df_best['rmp_score'] = -abs(df_best['RMP_mV'] + 70)  # closer to -70 is higher
            df_best['holding_score'] = -df_best['holding_I']      # smaller holding_I is higher
            df_best['total_score'] = df_best[['rmp_score', 'holding_score']].mean(axis=1)
            best_files = (
                df_best.sort_values(['cell_id', 'total_score'], ascending=[True, False])
                .groupby('cell_id')
                .first()
                .reset_index()
            )

            cell_df = cell_df.drop(columns=[folder_col], errors='ignore')
            cell_df = cell_df.merge(
                best_files[['cell_id', 'folder_file']],
                on='cell_id',
                how='left'
            ).rename(columns={'folder_file': folder_col})
        
        self.cache("cell_df", cell_df)
        self.save_excel("cell_df", cell_df)
        return cell_df


    def generate_application_cell_df(self):
        df = self.feature_df.copy()
        cell_wise_columns = ['cell_type', 'cell_subtype', 'axon_presence', 'axon_um', 'sex', 'region', 'subject_id'] # treatment and I_set added later based off APP_IC
        
        def _extract_APP_attributes(group):
            # There is exactly one APP_IC row per cell
            app_row = group[group['data_type'] == 'APP_IC'].iloc[0]
            return pd.Series({
                'I_set': app_row['I_set'],
                'treatment': app_row['treatment']
            })

        
        def calculate_percentage_diff(group):  
            #there are two ways access could be handeled 
            # 1 there is a column R_series for given folder files or 2 there are st_VC folder_files
            # 2 there is no column R_series and the change should be calculated from first to last st_VC recording 
            """
            USING R_series column for foler_files (data_type == 'IF_IC')
            
            Selects the two PRE and two non-PRE FP files with the most similar R_series values to compute access change. 
            If several have the same access chose the filder_files that have the least mising values."""
            cell_id = group.name

            #FIRING PROPERTY 
            cell_fp_df = self.IF_IC_df[self.IF_IC_df['cell_id'] == cell_id]
            pre_values = cell_fp_df[cell_fp_df['treatment'] == 'PRE'][['R_series', 'folder_file']]
            non_pre_values = cell_fp_df[cell_fp_df['treatment'] != 'PRE'][['R_series', 'folder_file']]
            
            # Extract R_series and folder_file
            pre_series = pre_values['R_series'].dropna().values
            non_pre_series = non_pre_values['R_series'].dropna().values
            
            # Check if there are enough values
            if len(pre_series) < 2 or len(non_pre_series) < 2:
                return pd.Series({'Rs_pct_change': None, self.folder_files_col("IF_IC"): None}) # mayher here files without pairs or not used should be dropped?
            
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
                return pd.Series({
                    'Rs_pct_change': None,
                    self.folder_files_col("IF_IC"): None
                })

            # folder_file filtered on access
            pre_folder_files = pre_values[pre_values['R_series'].isin(best_pre_pair)]['folder_file'].tolist() 
            non_pre_folder_files = non_pre_values[non_pre_values['R_series'].isin(best_non_pre_pair)]['folder_file'].tolist()

            # filter folder_files on extracted features and absence of RA
            if len(pre_folder_files) > 2 or len(non_pre_folder_files) > 2:
                IF_IC_feature_cols = [
                    'AP_peaks_mV', 'AP_rise_mV_ms', 'AP_width_ms', 'IF_slope',
                    'max_firing_Hz', 'IF_rheobase_pA', '%_sag',  'IF_voltage_threshold_mV', 'AP_decay_mV_ms'
                ]
                
                pre_df = cell_fp_df[cell_fp_df['treatment'] == 'PRE'].copy()
                non_pre_df = cell_fp_df[cell_fp_df['treatment'] != 'PRE'].copy()
                
                # Only keep rows that match the selected best R_series
                pre_df = pre_df[pre_df['R_series'].isin(best_pre_pair)]
                non_pre_df = non_pre_df[non_pre_df['R_series'].isin(best_non_pre_pair)]
                
                # Count missing values in relevant columns
                pre_df['missing_count'] = pre_df[IF_IC_feature_cols].isna().sum(axis=1)
                non_pre_df['missing_count'] = non_pre_df[IF_IC_feature_cols].isna().sum(axis=1)

                # Sort and select top 2
                pre_folder_files = pre_df.sort_values(by='missing_count')['folder_file'].iloc[:2].tolist()
                non_pre_folder_files = non_pre_df.sort_values(by='missing_count')['folder_file'].iloc[:2].tolist()
            else:
                # Safe fallback if only 1–2 values are returned, keep them directly
                pre_folder_files = pre_folder_files[:2]
                non_pre_folder_files = non_pre_folder_files[:2]

                return pd.Series({
                    'Rs_pct_change': min_diff,
                    self.folder_files_col("IF_IC"): pre_folder_files + non_pre_folder_files
                })

        #initalise off feature df
        cell_df = (
            df.groupby('cell_id')
            .apply(lambda g: self.apply_check_unique(g, unique_cols=cell_wise_columns, extra_logic=_extract_APP_attributes))
            .reset_index())
        
        diff_df = ( self.IF_IC_df 
                   .groupby('cell_id', group_keys=False)
                   .apply(calculate_percentage_diff))

        # diff_df = self.IF_IC_df.groupby('cell_id').apply(calculate_percentage_diff).reset_index()
        cell_df = cell_df.merge(diff_df, on='cell_id', how='left')


        # APPLICATION FILES
        filtered_app_df = self.APP_IC_df[ (self.APP_IC_df['valid'] != False)]
        valid_files_dict = filtered_app_df.set_index('cell_id')['folder_file'].to_dict()
        cell_df[self.folder_files_col("APP_IC")] = cell_df['cell_id'].map(valid_files_dict)


        # Check RA status in IF_IC_df and APP_IC_df
        fp_ra_df = self.IF_IC_df[self.IF_IC_df['RA'] == True][['cell_id', 'folder_file', 'RA_per_min']] #FP and APP dataframes where RA is True
        app_ra_df = self.APP_IC_df[self.APP_IC_df['RA'] == True][['cell_id', 'folder_file', 'RA_per_min']]
        combined_ra_df = pd.concat([fp_ra_df, app_ra_df])

        ra_folder_files = combined_ra_df.groupby('cell_id')['folder_file'].apply(list).to_dict()

        ra_avg_per_min = combined_ra_df.groupby('cell_id')['RA_per_min'].mean().to_dict() #average RA_per_min per cell_id

        cell_df['RA'] = cell_df['cell_id'].isin(ra_folder_files)
        cell_df['RA_folder_file'] = cell_df['cell_id'].map(ra_folder_files)
        cell_df['RA_per_min'] = cell_df['cell_id'].map(ra_avg_per_min)

        self.cache("cell_df", cell_df)
        self.save_excel("cell_df", cell_df)
        return cell_df
    

    def apply_check_unique(self, group: pd.DataFrame, unique_cols: list, extra_logic=None):
        cell_id = group.name

        def check_unique(series, cell_id):
            unique_values = series.dropna().unique()
            if len(unique_values) == 0:
                return None
            if len(unique_values) == 1:
                return unique_values[0]
            else:
                raise ValueError(f"Non-unique values found for cell_id: {cell_id} with values: {unique_values}")

        aggregated_data = group.agg({
            col: lambda series: check_unique(series, cell_id) for col in unique_cols
        })

        if extra_logic is not None:
            aggregated_data = pd.concat([aggregated_data, extra_logic(group)])

        return aggregated_data
    
    def reduce_cellwise(self, df: pd.DataFrame, cols: list, average: bool = False) -> pd.DataFrame:
        """
        Reduce a df to one row per cell_id for selected columns.

        Args:
            df: DataFrame with a 'cell_id' column.
            cols: List of columns to reduce.
            average: If True, average values per cell_id. If False, take the first value.

        Returns:
            DataFrame with columns: ['cell_id'] + cols
        """
        if average:
            reduced = df.groupby("cell_id")[cols].mean().reset_index()
        else:
            reduced = df.groupby("cell_id")[cols].first().reset_index()
        return reduced

    def folder_files_col(self, data_type: str) -> str:
        """Return standardized column name for valid folder files of a given data_type."""
        return f"{data_type}_folder_files"


