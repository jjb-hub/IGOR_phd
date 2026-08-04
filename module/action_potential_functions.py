
#change name to something more appropriate later 

import warnings
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter1d, median_filter
import scipy.signal as sg
from scipy import stats
from collections import namedtuple
from scipy.optimize import curve_fit
import sys

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from sklearn.mixture import GaussianMixture
from scipy.interpolate import interp1d

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler 
from scipy.stats import linregress
# from module.plotters import plot_ap_window

from scipy.signal import find_peaks


# GENERIC HANDELING 


def spike_remover_nan(array, threshold_sd=2): #not an elegant spike remover see mask_ap_regions #TODO replace in code
    """
    Identifies spikes (> threshold_sd * std from the mean of entire array) and replaces them with np.nan.
    
    Parameters:
        array (np.ndarray): 1D or 2D array of same condition (not ideal for large steps)
        threshold_sd (float): Number of standard deviations to define a spike.

    Returns:
        np.ndarray: Array with spikes replaced by np.nan (same shape).
    """
    array_cleaned = array.copy()

    if array.ndim == 1:
        mean = np.nanmean(array_cleaned)
        std = np.nanstd(array_cleaned)
        spike_mask = np.abs(array_cleaned - mean) > threshold_sd * std
        array_cleaned[spike_mask] = np.nan

    elif array.ndim == 2:
        for col in range(array.shape[1]):
            mean = np.nanmean(array_cleaned[:, col])
            std = np.nanstd(array_cleaned[:, col])
            spike_mask = np.abs(array_cleaned[:, col] - mean) > threshold_sd * std
            array_cleaned[spike_mask, col] = np.nan

    else:
        raise ValueError("Input array must be 1D or 2D.")

    return array_cleaned

def mask_ap_regions(trace, peak_locs, dt, threshold_sd=2):
    mask = np.ones(len(trace), dtype=bool)
    baseline = np.nanmedian(trace)
    noise_sd = np.nanstd(trace)
    tol = threshold_sd * noise_sd
    stable = int(2e-3 / dt)
    i = 0
    n = len(peak_locs)
    while i < n:
        p = peak_locs[i]
        start = p
        end = p
        while start > 0 and np.abs(trace[start] - baseline) > tol:
            start -= 1
        while end < len(trace):
            if np.all(np.abs(trace[end:end+stable] - baseline) < tol):
                break
            end += 1
            if end - p > int(100e-3 / dt):
                break
        mask[start:end] = False
        while i < n and peak_locs[i] <= end:
            i += 1
    return mask

def plot_ap_window(
    folder_file,
    v_array,
    peak_location,
    upshoot_location,
    input_sampling_rate,
    voltage_threshold=None,
    latency=None,
    rise_dvdt=None,
    max_dvdt=None,
    max_dvdt_location=None
):
    """
    Plot the action potential (AP) window around the upshoot and peak.

    Required:
        folder_file (str): Identifier for the trace (used in title).
        v_array (np.ndarray): Voltage trace.
        peak_location (int): Index of AP peak.
        upshoot_location (int): Index of AP upshoot.
        input_sampling_rate (float): Sampling rate in Hz.

    Optional:
        threshold_voltage (float): Voltage at upshoot.
        latency (float): Half-latency of the AP (in ms).
        rise_dvdt (float): Average slope between 20–80% AP height (mV/ms).
        max_dvdt (float): Max dV/dt value (mV/ms).
        max_dvdt_location (int): Index of max dV/dt.
    """
    time_points = np.arange(len(v_array)) / input_sampling_rate * 1000

    # Window definition
    window_half = int(latency * (input_sampling_rate / 1000)) if latency else int(10 * (input_sampling_rate / 1000))
    window_start = max(0, upshoot_location - window_half)
    window_end = min(len(v_array), peak_location + window_half)

    v_window = v_array[window_start:window_end]
    window_time_points = time_points[window_start:window_end]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(window_time_points, v_window, label='Raw Voltage Trace', color='blue')

    ax.axvline(time_points[upshoot_location], color='red', linestyle=':', label='Upshoot', linewidth=2)
    ax.axvline(time_points[peak_location], color='orange', linestyle='-', label='AP Peak', linewidth=2)

    #Optional voltage threshold 
    if voltage_threshold is not None:
        plt.axhline(voltage_threshold, color='red', linestyle=':', label=f"threshold in mV: {voltage_threshold}", linewidth=2)

    # Optional slope line
    if rise_dvdt is not None:
        # Calculate AP height from raw voltage trace
        ap_height = v_array[peak_location] - v_array[upshoot_location]
        V_80 = v_array[peak_location] - 0.2 * ap_height
        V_20 = v_array[peak_location] - 0.8 * ap_height

        # Extract voltage and time for the rise segment
        v_rise_segment = v_array[upshoot_location:peak_location]
        t_rise_segment = (np.arange(upshoot_location, peak_location) / input_sampling_rate) * 1000  # ms

        # Mask voltages between V_20 and V_80
        rise_mask = (v_rise_segment >= V_20) & (v_rise_segment <= V_80)
        v_20_80 = v_rise_segment[rise_mask]
        t_20_80 = t_rise_segment[rise_mask]

        if len(t_20_80) > 1:
            t_start, t_end = t_20_80[0], t_20_80[-1]
            V_start = v_20_80[0]
            V_end = V_start + rise_dvdt * (t_end - t_start)

            ax.plot([t_start, t_end], [V_start, V_end],
                    label=f'rising dv/dt: {rise_dvdt:.2f} mV/ms', color='green', linewidth=2)

    # Optional max dV/dt marker
    if max_dvdt is not None and max_dvdt_location is not None:
        # Only plot if max_dvdt_location is within the plotted window
        if window_start <= max_dvdt_location < window_end:
            ax.axvline(time_points[max_dvdt_location], color='limegreen', linestyle=':',
                       label=f'Max dV/dt: {max_dvdt:.2f} mV/ms', linewidth=2)
            ax.annotate(f'{max_dvdt:.2f} mV/ms',
                        xy=(time_points[max_dvdt_location], max_dvdt),
                        xytext=(10, 3),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.5'))
            
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title(f'AP Characteristics {folder_file}')
    ax.legend()
    ax.grid(True)
    plt.show()


########## BASE

def sigmoid_fit_0_0_trim (x,y, zeros_to_keep = 3):
    '''
    Set equal number of 0 values at start of x and y. 
 
    input:
        x (list): I values for FI curve
        y (list): Firing frequency in Hz?
        zeros_to_keep (int): Number of 0 to be kept. Defult = 3.

    return:
        x_cut (list): trimmed x 
        y_cut (list): trimmed y 

    '''
    number_of_0_at_start = np.where(np.diff(y)>0) [0][0] +1
    number_zeros_keept = min(zeros_to_keep, number_of_0_at_start)
    
    y_cut = y[number_of_0_at_start - number_zeros_keept:]
    x_cut = x[number_of_0_at_start - number_zeros_keept:]
    
    return x_cut, y_cut


def trim_after_AP_dropoff(x,y):
    '''
    trim x and y to ther depolarisation block (when APs max, dropoff after) to allow for a sigmoid function to be fitted

    Parameters
    ----------
    x : list - I values for FI curve 
    y : list - firing frequency in Hz

    Returns
    -------
    x_cut : list - trimed data 
    y_cut : list - trimed data 

    '''
    diff = np.diff (y) #diference between consecutive no in list
    cutoff = np.argmax(diff <0)+1 #index of first reduction  in APs +1 for indexing
    if cutoff < len(y)/2: #if APs reduce in the first half  of the steps applied search the second half  of APs to find true cutooff for depol block
        half_ish = round(len(y)/2) #if uneven no of sweeps cant split by 0.5
        diff_half = np.diff (y[half_ish:])
        cutoff = np.argmax(diff_half <0)
    
    if np.all(diff >= 0):
        cutoff = len(x)
    
    x_cut = x[:cutoff]
    y_cut = y[:cutoff]   
    return x_cut, y_cut  
                

def sigmoid(x, L ,x0, k):
    ''' 
    Sigmoid function: y = L / (1 + exp(-k*(x-x0))
    '''
    y = L / (1 + np.exp(-k*(x-x0))) # +b #b is the y intercept and has been removed as b = 0 for FI curves
    return (y)

def fit_sigmoid(xdata, ydata, maxfev = 5000, visualise = False):
    """
    Fits a sigmoid curve to the given data using non-linear least squares. The sigmoid function is defined as y = L / (1 + exp(-k*(x-x0))).
    input:
        xdata (array-like): The x-coordinates of the data points.
        ydata (array-like): The y-coordinates of the data points.
        maxfev (int, optional): The maximum number of function evaluations (default is 5000).
        visualise (bool, optional): If True, the function will plot the data points and the fitted curve (default is False).

    output:
        xfit (numpy.ndarray): The x-values of the fitted sigmoid curve.
        yfit (numpy.ndarray): The y-values of the fitted sigmoid curve.
        popt (list): The optimized parameters for the sigmoid curve. Format: [L, x0, k], where:
                    L is the curve's maximum value,
                    x0 is the x-value of the sigmoid's midpoint,
                    k is the steepness of the curve.
    """ 
    p0 = [max(ydata), np.median(xdata),0] # mandatory initial guess
    popt, pcov = curve_fit(sigmoid, xdata, ydata,p0, method='dogbox', maxfev = maxfev) #popt =  [L ,x0, k] 
    # RuntimeWarning: overflow encountered in exp - 
    # y = L / (1 + np.exp(-k*(x-x0))) 
    xfit = np.linspace(min(xdata), max(xdata), 1000)#generate points x and y for function best fit
    yfit = sigmoid(xfit, *popt)
    if visualise == True :
        plt.figure()
        plt.plot(xdata, ydata, 'o', label='data')
        plt.plot(xfit,yfit, label='fit')
        plt.legend(loc='best')
    return xfit, yfit , popt 

def steady_state_value(V_sweep, I_sweep, step_current_val=None, avg_window=0.5):
    """
    Calculates the steady state value of a voltage trace during a protocol step.

    Parameters:
    - V_sweep (array-like): Voltage trace for a single sweep.
    - I_sweep (array-like): Protocol/command trace corresponding to the voltage sweep.
    - step_current_val (float, optional): The command step value. If None, it is derived from I_sweep.
    - avg_window (float, optional): Fraction of the step current duration used for averaging. Default is 0.5 (50%).

    Returns:
    - asym_current (float): The steady state value of the voltage trace.
    - hyper (bool): True if the step current is hyperpolarizing; False otherwise.
    - first_current_point (int): Index of the start of the step current injection.
    - last_current_point (int): Index of the end of the step current injection.

    The function calculates the steady state value ('asym_current') by averaging the voltage trace over a window at the end of the current injection step.
    """

    V_sweep = np.asarray(V_sweep, dtype=float).flatten()
    I_sweep = np.asarray(I_sweep, dtype=float).flatten()

    # Check for empty inputs
    if len(V_sweep) == 0 or len(I_sweep) == 0:
        return np.nan, False, None, None

    step_indices, detected_step_value, _ = _step_indices_from_command_trace(I_sweep)
    if step_indices is None:
        # Legacy fallback for already-baseline-corrected square steps.
        if step_current_val is None:
            non_zero_I = I_sweep[I_sweep != 0]
            if len(non_zero_I) > 0:
                step_current_val = np.unique(non_zero_I)[0]
            else:
                return np.nan, False, None, None
        hyper = step_current_val < 0
        step_indices = np.where(I_sweep == (np.min(I_sweep) if hyper else np.max(I_sweep)))[0]
    elif step_current_val is None:
        step_current_val = detected_step_value

    # Determine if the current step is hyperpolarizing
    hyper = step_current_val < 0
    
    # Check if there are no current points found
    if len(step_indices) <= 1:
        print(f"No I step detected.")
        return np.nan, hyper, None, None

    # Calculate the first and last points of the current injection
    first_current_point = step_indices[0]
    last_current_point = step_indices[-1]

    # Calculate the duration for averaging, ensuring it does not exceed array bounds
    current_avg_duration = int(avg_window * (last_current_point - first_current_point))
    if last_current_point - current_avg_duration < 0:
        return np.nan, hyper, first_current_point, last_current_point

    # Calculate the steady state value by averaging over the determined window
    asym_current = np.mean(V_sweep[last_current_point - current_avg_duration:last_current_point])
    return asym_current, hyper, first_current_point, last_current_point


def calculate_max_firing(voltage_array, input_sampling_rate=2e4): 
    """
    Calculates the maximum firing rate (Hz) of action potentials in a series of voltage traces.

    Args:
        voltage_array (np.ndarray): A 2D array of voltage traces, where each column represents a different sweep.
        input_sampling_rate (float, optional): The sampling rate of the voltage traces (default: 10000 Hz).

    Returns:
        max_firing (float): The maximum firing rate in Hz, based on the sweep with the highest number of action potentials.
    """
    num_aps_all  = np.array(num_ap_finder(voltage_array))           # get num aps from each sweep
    index_max = np.where(num_aps_all == max(num_aps_all)) [0][0]    # get trace number with max aps 
    sampling_rate = input_sampling_rate
    _ , peak_locs , _ , _   =  ap_finder(voltage_array[:, index_max ] , smoothing_kernel = 10)
    
    return np.nanmean(sampling_rate / np.diff(peak_locs)) 


#SAG LOGIC


def sag_current_analyser(folder_file, V_array, protocol_array, step_current_values, AP_frequency_Hz, avg_window=0.5, visualise=False):
    """
    Calculate sag from voltage traces under the first hyperpolarizing protocol step without action potentials.
    
    Parameters:
    - V_array (2D array): 2D array containing voltage recordings for different current steps.
    - protocol_array (2D array): command/protocol trace used for step timing.
    - step_current_values (list): List of injected current values for each sweep.
    - AP_frequency_Hz (list): List of action potential frequency Hz for each sweep.
    - avg_window (float): Fraction of the step current duration used for averaging.
    
    Returns:
    - List containing [sag ratio, step_V_steady, I_step, RMP].
    """
    if protocol_array is None:
        return [np.nan, np.nan, np.nan, np.nan]

    V_array, protocol_array = protocol_array_to_match_V(V_array, protocol_array)
    if protocol_array is None:
        return [np.nan, np.nan, np.nan, np.nan]

    for sweep_index, ap_frequency in enumerate(AP_frequency_Hz):
        if sweep_index >= V_array.shape[1] or sweep_index >= protocol_array.shape[1]:
            continue
        if ap_frequency == 0 and step_current_values[sweep_index] < 0:  # Check for no APs and negative current
            V_sweep = V_array[:, sweep_index]
            protocol_sweep = protocol_array[:, sweep_index]
            step_current = step_current_values[sweep_index]


            # Calculate steady state value using the steady_state_value function
            step_V_steady, hyper, first_current_point, last_current_point = steady_state_value(V_sweep, protocol_sweep, step_current, avg_window)
            if first_current_point is None or last_current_point is None:
                continue
            
            # Calculate RMP before current injection
            RMP = np.mean(V_sweep[:first_current_point])
            
            # Calculat minimum on I step 
            sorted_voltages = np.sort(V_sweep[first_current_point:last_current_point])
            num_points = int(len(sorted_voltages) * 0.1)  # Take the lowest 10% of points
            if num_points < 1:
                continue
            min_sag_voltage = np.mean(sorted_voltages[:num_points])  # Average them to get a robust minimum
        
            # Calculate sag ratio
            sag_ratio = (step_V_steady - min_sag_voltage) / (RMP - min_sag_voltage)

            if  0<= sag_ratio <=0.45: #HARD CODE
                return [sag_ratio, step_V_steady, step_current, RMP]
            else:
                print (f"Sag calculated {sag_ratio} is outside physiological bound 0% to 45%.")
                return np.nan
                plot_sag(folder_file, V_sweep[first_current_point:last_current_point], np.arange(first_current_point, last_current_point) / 1000, RMP, step_V_steady, min_sag_voltage, sag_ratio)
        
    # print("No sweep found with negative current injecttion and without action potentials, unable to calculate sag.")
    return [np.nan, np.nan, np.nan, np.nan]


def plot_sag(folder_file, voltage_trace, time_trace, RMP, steady_state_voltage, min_sag_voltage, sag_ratio):
    """
    Plot the sag with the minimum sag voltage and sag ratio annotated.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time_trace, voltage_trace, label='Voltage Trace', color='blue')
    plt.axhline(y=min_sag_voltage, color='red', linestyle='--', label='Min Sag Voltage')
    plt.axhline(y=steady_state_voltage, color='green', linestyle='--', label='Steady State Voltage')
    plt.axhline(y=RMP, color='orange', linestyle='--', label='RMP')
    plt.title(f"{folder_file} Sag Analysis (Sag Ratio: {sag_ratio:.2f})")
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    plt.legend()
    plt.show()


########## ACTION POTENTIAL / SPIKE DETECTION

def ap_finder(voltage_trace, smoothing_kernel = 10):
    '''
    Lowest level AP detector. Detects action potentials in a 1D voltage trace using peak detection.
    
    Input:
        voltage_trace (numpy.ndarray): A 1D array representing a voltage trace.
        smoothing_kernel (int): Size of the kernel for Gaussian smoothing (default 10).
    
    Output:
        v_smooth (numpy.ndarray): Smoothed voltage trace.
        peak_locs (numpy.ndarray): Indices of detected peaks (action potentials).
        peak_info (dict): Scipy dictionary of peak info.
        num_peaks (int): Number of detected peaks.
    '''
    v_smooth = gaussian_filter1d(voltage_trace , smoothing_kernel)
    peak_locs , peak_info = sg.find_peaks(v_smooth, height = 10 + np.average(v_smooth), distance = 2, 
                                    prominence = [20,150], width = 1, rel_height= 1)
    
    num_peaks = len(peak_locs)
    
    return  v_smooth, peak_locs , peak_info , num_peaks 


def num_ap_finder(voltage_array): #not so sure why we nee dthis fun maybe DJ explain
    '''
    Counts the number of action potentials in each sweep (column) of a voltage array.

    input:
        voltage_array (np.ndarray): A 2D array of voltage traces, where each column represents a sweep.

    output:
        num_aps_all (list): The number of action potentials in each sweep.
    '''
    num_aps_all = []
    for idx in range(voltage_array.shape[-1]): 

        _, _, _,  num_peaks  = ap_finder(voltage_array[:, idx])
        num_aps_all += [num_peaks]
    return num_aps_all




########## ACTION POTENTIAL CHARECTERISTICS ie RETROAXONAL / ANTIDROMIC


def peak_finder(   # TODO remove rise time logic here its not modular 
    voltage_trace: np.ndarray,
    raw_trace=None, # optional trace to pull values from after detection
    smoothing_kernel: int = 5,
    height: float = None,
    prominence: tuple = None,
    distance: int = None,
    width: float = None,
    polarity: str = 'positive',  # 'positive' or 'negative'
    rise_time_range: tuple = (0.2e-3, 10e-3),  # (min, max) in seconds default 0.2-10 ms
    dt: float = 1e-4,  # Sampling interval in seconds default is for 10kHz
    backward_window_s: float = 0.025, #window from beack backwards to find upshoot default 25 ms
    correction_window_s: int = 0.001  # to refine peak/upshoot location from smoothed trace to raw default 1ms
    ):
    """
    Detect peaks in voltage_trace and estimate amplitudes from raw_trace if provided.

    Returns:
    - peaks: np.ndarray of peak indices
    - amplitudes: list of approximate peak amplitudes
    - frequency: float, peak count / total recording time (Hz)
    """
    if raw_trace is None:
        raw_trace = voltage_trace.copy()
    # ensure 1D array
    if voltage_trace.ndim == 2:    
        voltage_trace = voltage_trace.flatten(order='F')
    if raw_trace.ndim == 2:    
        raw_trace = raw_trace.flatten(order='F')
    # invert if negative
    signal = -voltage_trace if polarity == 'negative' else voltage_trace.copy()     
    raw_signal = -raw_trace if polarity == 'negative' else raw_trace.copy()

    v_smooth = gaussian_filter1d(signal, smoothing_kernel)
    peak_locs, _ = find_peaks(v_smooth, height=height, prominence=prominence, distance=distance, width=width)

    correction_window = int(correction_window_s / dt)
    backward_window = int(backward_window_s / dt)

    valid_peak_locs = []
    aprox_amplitudes = []
    for peak in peak_locs:
        start_idx = max(0, peak - backward_window) # upshoot detection doesnt work for EPSPs in the same way 
        correction_start = max(0, peak - correction_window)
        correction_end = min(len(raw_signal), peak + correction_window + 1)
        if correction_start >= correction_end:
            continue
        peak_val = np.nanmax(raw_signal[correction_start:correction_end])

        baseline_start = max(0, start_idx-(4*backward_window))
        baseline_window = raw_signal[baseline_start:start_idx]
        if baseline_window.size == 0 or np.all(np.isnan(baseline_window)):
            baseline_window = raw_signal[max(0, peak - backward_window):peak]
        baseline = np.nanmedian(baseline_window)

        valid_peak_locs.append(peak)
        aprox_amplitudes.append(peak_val - baseline)

        # # DEBUG PLOT
        # fig, ax = plt.subplots(figsize=(12,4))
        # x = np.arange(len(raw_signal)) * dt * 1000
        # ax.plot(x, raw_signal, color='lightgray', linewidth=0.6)
        # z_pre = int(0.2 / dt)
        # z_post = int(0.2 / dt)
        # z_start = max(0, peak - z_pre)
        # z_end = min(len(raw_signal), peak + z_post)
        # x_zoom = np.arange(z_start, z_end) * dt * 1000
        # ax.plot(x_zoom, raw_signal[z_start:z_end], color='black', linewidth=1.2)
        # ax.axhline(baseline, color='gray', linestyle='--', linewidth=0.8)
        # ax.vlines(
        #     peak * dt * 1000,
        #     baseline,
        #     raw_signal[peak],
        #     color='red',
        #     linewidth=1
        # )
        # ax.plot(peak * dt * 1000, raw_signal[peak], 'ro')
        # ax.set_title(f'Peak {peak} | Amp ~ {amp:.2f}')
        # ax.set_xlabel('Time (ms)')
        # ax.set_ylabel('mV')
        # plt.tight_layout()
        # plt.show()

    total_time = len(voltage_trace) * dt
    frequency = len(valid_peak_locs) / total_time if total_time > 0 else 0
    return np.asarray(valid_peak_locs, dtype=int), aprox_amplitudes, frequency


def _flatten_trace_time_first(trace):
    """Flatten a voltage trace with sweeps appended in recording order."""
    trace = np.asarray(trace, dtype=float)
    if trace.ndim == 2:
        return trace.ravel(order='F')
    return trace.flatten()


def _baseline_trace(trace, dt, method='rolling_median', window_s=0.1):
    """
    Estimate slow baseline drift.

    baseline_method options:
        rolling_median: original behavior; local and conservative, but slow.
        median_filter: scipy median filter; similar intent, usually faster.
        coarse_median: median per time bin, interpolated; fastest/smoothest.
        global_median: one median for the whole trace; no drift correction.
    """
    trace = np.asarray(trace, dtype=float).copy()
    if trace.size == 0:
        return trace

    fill_value = np.nanmedian(trace)
    if not np.isfinite(fill_value):
        return np.full(trace.shape, np.nan)
    trace[np.isnan(trace)] = fill_value

    window = max(1, int(window_s / dt))

    if method == 'rolling_median':
        return pd.Series(trace).rolling(window, center=True, min_periods=1).median().to_numpy()

    if method == 'median_filter':
        if window % 2 == 0:
            window += 1
        return median_filter(trace, size=window, mode='nearest')

    if method == 'coarse_median':
        starts = np.arange(0, len(trace), window)
        centers = []
        values = []
        for start in starts:
            end = min(len(trace), start + window)
            centers.append((start + end - 1) / 2)
            values.append(np.nanmedian(trace[start:end]))
        if len(values) == 1:
            return np.full(trace.shape, values[0])
        return np.interp(np.arange(len(trace)), centers, values, left=values[0], right=values[-1])

    if method == 'global_median':
        return np.full(trace.shape, fill_value)

    raise ValueError(
        "baseline_method must be one of: rolling_median, median_filter, coarse_median, global_median"
    )


def _measure_event_local_amplitudes(
    raw_trace,
    detection_trace,
    peak_locs,
    valid_mask,
    dt,
    polarity='positive',
    peak_window_s=0.001,
    onset_search_window_s=0.080,
    onset_threshold_mV=None,
    onset_threshold_fraction=0.2,
    onset_noise_multiplier=1.0,
    noise_sd=None,
    amplitude_threshold=None,
    onset_stable_window_s=0.001,
    upshoot_baseline_window_s=0.001,
):
    """
    Measure event amplitudes from raw voltage using the local EPSP upshoot.

    Peak locations come from the processed trace, but upshoot/takeoff is refined
    on the raw trace. The preferred upshoot is the local pre-rise point before a
    sustained raw upward deflection. If that is not found, a short local-minimum
    fallback is used. For stacked EPSPs, the search starts after the previous
    peak so amplitudes stay incremental.
    """
    raw_trace = np.asarray(raw_trace, dtype=float).flatten()
    detection_trace = np.asarray(detection_trace, dtype=float).flatten()
    valid_mask = np.asarray(valid_mask, dtype=bool).flatten()
    peak_locs = np.asarray(peak_locs, dtype=int)

    peak_half_window = max(1, int(peak_window_s / dt))
    onset_search_samples = max(2, int(onset_search_window_s / dt))
    onset_stable_samples = max(1, int(onset_stable_window_s / dt))
    upshoot_half_window = max(0, int(upshoot_baseline_window_s / dt / 2))

    raw_signal = -raw_trace if polarity == 'negative' else raw_trace

    def is_finite_scalar(value):
        try:
            return np.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    if onset_threshold_mV is None:
        candidates = []
        if is_finite_scalar(amplitude_threshold):
            candidates.append(abs(amplitude_threshold) * onset_threshold_fraction)
        if is_finite_scalar(noise_sd):
            candidates.append(abs(noise_sd) * onset_noise_multiplier)
        onset_threshold_mV = max(candidates) if candidates else 0

    amplitudes = []
    peak_values = []
    baselines = []
    raw_peak_locs = []
    baseline_windows = []
    onset_search_windows = []
    onset_locs = []
    baseline_status = []
    baseline_sample_locs = []
    previous_event_interval_ms = []
    next_event_interval_ms = []

    def local_upshoot_value(upshoot_loc):
        start = max(0, int(upshoot_loc) - upshoot_half_window)
        end = min(len(raw_trace), int(upshoot_loc) + upshoot_half_window + 1)
        locs = np.arange(start, end)
        locs = locs[valid_mask[locs] & np.isfinite(raw_trace[locs])]
        if locs.size == 0:
            return np.nan, np.array([], dtype=int), (start, end)
        return np.nanmedian(raw_trace[locs]), locs, (start, end)

    def smooth_raw_segment(segment):
        segment = np.asarray(segment, dtype=float).copy()
        finite_segment = np.isfinite(segment)
        if not np.any(finite_segment):
            return None
        segment[~finite_segment] = np.nanmedian(segment[finite_segment])
        return gaussian_filter1d(segment, max(1, int(0.0005 / dt)))

    def sustained_raw_rise_onset(smoothed_segment, candidate_offsets, peak_offset, event_excursion):
        if (
            smoothed_segment is None
            or candidate_offsets.size == 0
            or peak_offset <= 1
            or not np.isfinite(event_excursion)
            or event_excursion <= 0
        ):
            return None

        d_raw = np.diff(smoothed_segment)
        if d_raw.size == 0:
            return None
        finite_d = np.isfinite(d_raw)
        if not np.any(finite_d):
            return None

        d_center = np.nanmedian(d_raw[finite_d])
        d_noise = np.nanmedian(np.abs(d_raw[finite_d] - d_center)) / 0.6745
        if not np.isfinite(d_noise):
            d_noise = 0

        rise_window = max(3, int(0.001 / dt))
        d_mean = np.convolve(d_raw, np.ones(rise_window) / rise_window, mode='same')
        mean_event_slope = event_excursion / max(1, int(peak_offset))
        slope_threshold = max(d_noise * 0.25, mean_event_slope * 0.20, event_excursion * 0.0002)
        search_end_offset = min(max(1, int(peak_offset)), len(d_mean))
        rising = d_mean[:search_end_offset] > slope_threshold

        min_rise_samples = max(3, int(0.0006 / dt))
        gap_limit = max(1, int(0.0004 / dt))
        segments = []
        in_segment = False
        segment_start = None
        last_positive = None
        positive_count = 0
        gap_count = 0

        for idx, is_rising in enumerate(rising):
            if is_rising:
                if not in_segment:
                    in_segment = True
                    segment_start = idx
                    positive_count = 0
                last_positive = idx
                positive_count += 1
                gap_count = 0
            elif in_segment:
                gap_count += 1
                if gap_count > gap_limit:
                    if positive_count >= min_rise_samples:
                        segments.append((segment_start, last_positive))
                    in_segment = False
                    segment_start = None
                    last_positive = None
                    positive_count = 0
                    gap_count = 0

        if in_segment and positive_count >= min_rise_samples:
            segments.append((segment_start, last_positive))

        if not segments:
            return None

        max_gap_to_peak = max(int(0.005 / dt), min(int(0.025 / dt), int(peak_offset) // 2))
        usable_segments = [
            segment for segment in segments
            if int(peak_offset) - segment[1] <= max_gap_to_peak
        ]
        if not usable_segments:
            return None

        segment_start, _ = usable_segments[-1]
        pre_window = max(int(0.005 / dt), rise_window)
        pre_start = max(0, segment_start - pre_window)
        pre_stop = min(int(peak_offset), segment_start + 1)
        pre_offsets = candidate_offsets[
            (candidate_offsets >= pre_start)
            & (candidate_offsets < pre_stop)
        ]
        if pre_offsets.size == 0:
            return int(segment_start)

        pre_values = smoothed_segment[pre_offsets]
        return int(pre_offsets[np.nanargmin(pre_values)])

    for event_idx, peak in enumerate(peak_locs):
        previous_peak = peak_locs[event_idx - 1] if event_idx > 0 else None
        next_peak = peak_locs[event_idx + 1] if event_idx < len(peak_locs) - 1 else None
        previous_event_interval_ms.append(
            ((peak - previous_peak) * dt * 1000) if previous_peak is not None else np.nan
        )
        next_event_interval_ms.append(
            ((next_peak - peak) * dt * 1000) if next_peak is not None else np.nan
        )

        peak_start = max(0, peak - peak_half_window)
        peak_end = min(len(raw_trace), peak + peak_half_window + 1)
        peak_window = raw_trace[peak_start:peak_end]
        finite_peak_mask = np.isfinite(peak_window)
        if np.any(finite_peak_mask):
            finite_peak_values = peak_window[finite_peak_mask]
            finite_peak_offsets = np.where(finite_peak_mask)[0]
            if polarity == 'negative':
                local_idx = int(finite_peak_offsets[np.nanargmin(finite_peak_values)])
            else:
                local_idx = int(finite_peak_offsets[np.nanargmax(finite_peak_values)])
            raw_peak_loc = peak_start + local_idx
            peak_value = raw_trace[raw_peak_loc]
        else:
            raw_peak_loc = peak
            peak_value = np.nan

        search_start = max(0, raw_peak_loc - onset_search_samples)
        if previous_peak is not None:
            search_start = max(search_start, int(previous_peak) + 1)
        search_end = max(search_start, raw_peak_loc)

        onset_search_windows.append((search_start, search_end))

        search_mask = valid_mask[search_start:search_end] & np.isfinite(raw_signal[search_start:search_end])
        candidate_locs = np.arange(search_start, search_end)
        candidate_locs = candidate_locs[search_mask]
        onset_loc = None
        status = 'raw_rise_upshoot'

        if candidate_locs.size > 0:
            raw_segment = smooth_raw_segment(raw_signal[search_start:search_end])
            candidate_offsets = candidate_locs - search_start
            candidate_values = raw_segment[candidate_offsets] if raw_segment is not None else np.asarray([])
            peak_signal_value = (
                raw_signal[raw_peak_loc]
                if 0 <= int(raw_peak_loc) < len(raw_signal)
                else np.nan
            )
            local_reference = np.nanpercentile(candidate_values, 20) if candidate_values.size > 0 else np.nan
            event_excursion = peak_signal_value - local_reference
            onset_offset = sustained_raw_rise_onset(
                raw_segment,
                candidate_offsets,
                raw_peak_loc - search_start,
                event_excursion,
            )

            if onset_offset is not None:
                onset_loc = int(search_start + onset_offset)

            if onset_loc is None:
                fallback_window = max(int(0.020 / dt), onset_stable_samples)
                fallback_start = max(search_start, raw_peak_loc - fallback_window)
                fallback_locs = candidate_locs[candidate_locs >= fallback_start]
                if fallback_locs.size == 0:
                    fallback_locs = candidate_locs
                fallback_offsets = fallback_locs - search_start
                fallback_values = raw_segment[fallback_offsets] if raw_segment is not None else raw_signal[fallback_locs]
                fallback_idx = int(np.nanargmin(fallback_values))
                onset_loc = int(fallback_locs[fallback_idx])
                local_min_at_edge = onset_loc == int(candidate_locs[0])
                status = 'search_edge_upshoot' if local_min_at_edge else 'local_min_upshoot'
            elif onset_loc == int(candidate_locs[0]):
                status = 'search_edge_upshoot'
        else:
            onset_loc = int(peak)
            status = 'insufficient_upshoot'

        baseline, baseline_locs, baseline_window = local_upshoot_value(onset_loc)
        if not np.isfinite(baseline):
            status = 'insufficient_upshoot'

        baseline_windows.append(baseline_window)
        onset_locs.append(onset_loc)
        baseline_sample_locs.append(baseline_locs)

        if np.isfinite(peak_value) and np.isfinite(baseline):
            if polarity == 'negative':
                amplitude = baseline - peak_value
                invalid_peak_upshoot = peak_value >= baseline
            else:
                amplitude = peak_value - baseline
                invalid_peak_upshoot = peak_value <= baseline
            if invalid_peak_upshoot:
                amplitude = np.nan
                status = 'invalid_peak_upshoot'
        else:
            amplitude = np.nan

        raw_peak_locs.append(raw_peak_loc)
        peak_values.append(peak_value)
        baselines.append(baseline)
        amplitudes.append(amplitude)
        baseline_status.append(status)

    return {
        'amplitudes_mV': np.asarray(amplitudes, dtype=float),
        'peak_values_mV': np.asarray(peak_values, dtype=float),
        'baselines_mV': np.asarray(baselines, dtype=float),
        'upshoot_values_mV': np.asarray(baselines, dtype=float),
        'raw_peak_locs': np.asarray(raw_peak_locs, dtype=int),
        'onset_locs': np.asarray(onset_locs, dtype=int),
        'upshoot_locs': np.asarray(onset_locs, dtype=int),
        'baseline_windows': baseline_windows,
        'onset_search_windows': onset_search_windows,
        'upshoot_windows': baseline_windows,
        'baseline_sample_locs': baseline_sample_locs,
        'upshoot_sample_locs': baseline_sample_locs,
        'baseline_status': baseline_status,
        'upshoot_status': baseline_status,
        'onset_threshold_mV': onset_threshold_mV,
        'previous_event_interval_ms': np.asarray(previous_event_interval_ms, dtype=float),
        'next_event_interval_ms': np.asarray(next_event_interval_ms, dtype=float),
    }


def EPSP_detector(
    voltage_trace,
    sampling_rate=2e4,
    folder_file=None,
    amplitude_threshold=None,
    noise_multiplier=4,
    min_amplitude_threshold=0.1,
    max_amplitude_threshold=1.0,
    baseline_method='rolling_median',
    baseline_window_s=0.1,
    mask_aps=True,
    valid_mask=None,
    smoothing_kernel=10,
    prominence_fraction=0.5,
    rise_time_range=(0.2e-3, 10e-3),
    distance=None,
    polarity='positive',
    backward_window_s=0.02,
    peak_window_s=0.001,
    upshoot_baseline_window_s=0.001,
    onset_search_window_s=0.080,
    onset_threshold_mV=None,
    onset_threshold_fraction=0.2,
    onset_noise_multiplier=1.0,
    onset_stable_window_s=0.001,
    debug_local_baseline_plot=False,
    debug_event_count=12,
    debug_plot=False,
    print_warnings=False,
):
    """
    Detect positive synaptic events from an IC voltage trace.

    The detector is protocol-agnostic. It detects on the continuous
    baseline-corrected trace. AP and valid masks are applied after detection to
    reject candidate peaks and calculate the valid-time frequency denominator.

    Parameters:
        voltage_trace:
            1D trace or 2D time x sweeps voltage array in mV. 2D arrays are
            flattened in recording order. For APP_IC, call per sweep or pass a
            valid_mask from off-step command periods.
        sampling_rate:
            Sampling rate in Hz.
        folder_file:
            Optional label used only in warning/debug prints.
        amplitude_threshold:
            Fixed threshold in mV after baseline correction. If provided, this
            value is used directly and min/max threshold bounds are ignored.
        noise_multiplier:
            Used when amplitude_threshold is None. noise_sd is estimated as
            MAD / 0.6745, a robust noise estimate commonly used because rare
            events affect it less than standard deviation.
        min_amplitude_threshold:
            Lower bound for automatic thresholds. Default: 0.1 mV.
        max_amplitude_threshold:
            Upper bound for automatic thresholds. Default: 1.0 mV. Set to None
            to disable capping. A warning is returned/printed when the automatic
            threshold is floored or capped.
        baseline_method:
            Drift correction method passed to _baseline_trace.
        baseline_window_s:
            Window used by baseline_method.
        mask_aps:
            If True, AP regions are detected and candidate EPSPs inside those
            regions are rejected after EPSP detection.
        valid_mask:
            Optional boolean mask with the same shape as voltage_trace after
            flattening. True means usable. False samples are not counted in
            frequency and candidate EPSP peaks there are rejected after
            detection. For APP_IC/IF_IC, build this from off-step command
            periods outside this generic detector.
        smoothing_kernel:
            Gaussian smoothing kernel passed to peak_finder for event detection.
        prominence_fraction:
            Prominence threshold as a fraction of amplitude_threshold.
        rise_time_range:
            Reserved for EPSP kinetic filtering. It is passed through to
            peak_finder for API stability, but the current peak_finder does not
            enforce rise-time filtering yet.
        distance:
            Optional minimum event spacing in samples, passed to scipy peaks.
        polarity:
            'positive' for EPSPs, 'negative' for IPSP-like downward events.
        backward_window_s:
            Window used by peak_finder to estimate local pre-event baseline.
        peak_window_s:
            Raw peak search window around each accepted detection point.
        upshoot_baseline_window_s:
            Small raw-trace window around the detected EPSP upshoot. Amplitude
            is measured from this local voltage, so stacked EPSPs are measured
            incrementally from their own takeoff.
        onset_search_window_s:
            Backward window used to find EPSP upshoot/takeoff before the peak.
            If the selected upshoot is at the left edge of this window, the
            event is labelled search_edge_upshoot as a QC hint.
        debug_local_baseline_plot:
            If True, plot zoomed raw traces showing local upshoot windows,
            upshoot samples, and upshoot status.
        debug_plot:
            If True, plot accepted/rejected peaks and raw amplitudes.
        print_warnings:
            If True, print threshold/QC warnings while extracting. Warnings are
            returned in the output dict either way.

    Returns:
        dict with frequency_Hz, raw local amplitudes_mV, peak_locs, RMP_mV,
        baseline_drift_mV, amplitude_threshold_mV, noise_sd_mV, valid_time_s,
        upshoot_values_mV, peak_values_mV, onset/upshoot locations,
        upshoot status, ap_peak_locs, warnings, and debug intermediate traces.
    """
    dt = 1 / sampling_rate

    def empty_result(warning, frequency=np.nan, valid_time=0, ap_peak_locs=None):
        warnings_out = [warning] if warning else []
        return {
            'frequency_Hz': frequency,
            'amplitudes_mV': np.asarray([], dtype=float),
            'peak_locs': np.asarray([], dtype=int),
            'peak_values_mV': np.asarray([], dtype=float),
            'local_baselines_mV': np.asarray([], dtype=float),
            'upshoot_values_mV': np.asarray([], dtype=float),
            'raw_peak_locs': np.asarray([], dtype=int),
            'onset_locs': np.asarray([], dtype=int),
            'upshoot_locs': np.asarray([], dtype=int),
            'baseline_windows': [],
            'onset_search_windows': [],
            'upshoot_windows': [],
            'baseline_sample_locs': [],
            'upshoot_sample_locs': [],
            'baseline_status': [],
            'upshoot_status': [],
            'onset_threshold_mV': np.nan,
            'previous_event_interval_ms': np.asarray([], dtype=float),
            'next_event_interval_ms': np.asarray([], dtype=float),
            'RMP_mV': np.nan,
            'baseline_drift_mV': np.nan,
            'amplitude_threshold_mV': amplitude_threshold,
            'noise_sd_mV': np.nan,
            'valid_time_s': valid_time,
            'ap_peak_locs': (
                np.asarray(ap_peak_locs, dtype=int)
                if ap_peak_locs is not None
                else np.asarray([], dtype=int)
            ),
            'warnings': warnings_out,
            'debug': {},
        }

    V_raw = _flatten_trace_time_first(voltage_trace)
    if V_raw.size == 0 or np.all(np.isnan(V_raw)):
        return empty_result('empty voltage trace')

    finite_mask = np.isfinite(V_raw)
    if valid_mask is None:
        valid_samples_mask = np.ones(V_raw.shape, dtype=bool)
    else:
        valid_samples_mask = _flatten_trace_time_first(valid_mask).astype(bool)
        if valid_samples_mask.shape != V_raw.shape:
            raise ValueError("valid_mask must match voltage_trace after flattening")
    valid_samples_mask = valid_samples_mask & finite_mask

    if not np.any(valid_samples_mask):
        return empty_result('no usable EPSP samples')

    ap_peak_locs = np.array([], dtype=int)
    ap_mask = np.ones(V_raw.shape, dtype=bool)
    if mask_aps:
        V_for_ap = V_raw.copy()
        V_for_ap[~finite_mask] = np.nanmedian(V_raw[finite_mask])
        _, ap_peak_locs, _, _ = ap_finder(V_for_ap)
        ap_mask = mask_ap_regions(V_for_ap, ap_peak_locs, dt)
    accepted_samples_mask = valid_samples_mask & ap_mask

    if not np.any(accepted_samples_mask):
        return empty_result(
            'all EPSP samples masked by AP/valid mask',
            frequency=0,
            valid_time=0,
            ap_peak_locs=ap_peak_locs,
        )

    # Keep the detection trace continuous; masks are applied to candidate peaks below.
    V_for_detection = V_raw.copy()
    V_for_detection[~finite_mask] = np.nanmedian(V_raw[finite_mask])
    V_valid_for_baseline = V_for_detection.copy()
    drifting_baseline = _baseline_trace(
        V_valid_for_baseline,
        dt,
        method=baseline_method,
        window_s=baseline_window_s,
    )
    baseline_drift = np.nanmax(drifting_baseline) - np.nanmin(drifting_baseline)

    V_analysis_trace = V_for_detection - drifting_baseline
    V_variability_trace = V_analysis_trace.copy()
    V_variability_trace[~finite_mask] = np.nan

    global_baseline = np.nanmedian(np.where(accepted_samples_mask, V_raw, np.nan))

    median_val = np.nanmedian(V_variability_trace)
    mad = np.nanmedian(np.abs(V_variability_trace - median_val))
    noise_sd = mad / 0.6745

    warnings_list = []
    if amplitude_threshold is None:
        amplitude_threshold = noise_multiplier * noise_sd
        if not np.isfinite(amplitude_threshold):
            warning = (
                f"sEPSP threshold set to {min_amplitude_threshold:.2f} mV "
                "because noise threshold was non-finite"
            )
            warnings_list.append(warning)
            if print_warnings and folder_file is not None:
                print(f"[WARNING] {warning} | folder_file: {folder_file}")
            amplitude_threshold = min_amplitude_threshold
        if amplitude_threshold < min_amplitude_threshold:
            original_threshold = amplitude_threshold
            amplitude_threshold = min_amplitude_threshold
            warning = (
                f"sEPSP threshold raised to {min_amplitude_threshold:.2f} mV "
                f"(noise threshold was {original_threshold:.2f} mV)"
            )
            warnings_list.append(warning)
            if print_warnings and folder_file is not None:
                print(f"[WARNING] {warning} | folder_file: {folder_file}")
        if (
            max_amplitude_threshold is not None
            and amplitude_threshold > max_amplitude_threshold
        ):
            original_threshold = amplitude_threshold
            amplitude_threshold = max_amplitude_threshold
            warning = (
                f"sEPSP threshold capped at {max_amplitude_threshold:.2f} mV "
                f"(noise threshold was {original_threshold:.2f} mV)"
            )
            warnings_list.append(warning)
            if print_warnings and folder_file is not None:
                print(f"[WARNING] {warning} | folder_file: {folder_file}")

    peak_locs_all, _, _ = peak_finder(
        V_analysis_trace,
        raw_trace=V_for_detection,
        height=amplitude_threshold,
        smoothing_kernel=smoothing_kernel,
        prominence=(amplitude_threshold * prominence_fraction, None),
        rise_time_range=rise_time_range,
        width=None,
        dt=dt,
        distance=distance,
        polarity=polarity,
        backward_window_s=backward_window_s,
    )

    peak_locs_all = np.asarray(peak_locs_all, dtype=int)
    accepted_peak_mask = accepted_samples_mask[peak_locs_all] if peak_locs_all.size > 0 else np.array([], dtype=bool)
    epsp_peak_locs = peak_locs_all[accepted_peak_mask]
    local_amplitude = _measure_event_local_amplitudes(
        V_raw,
        V_analysis_trace - median_val,
        epsp_peak_locs,
        accepted_samples_mask,
        dt,
        polarity=polarity,
        peak_window_s=peak_window_s,
        upshoot_baseline_window_s=upshoot_baseline_window_s,
        onset_search_window_s=onset_search_window_s,
        onset_threshold_mV=onset_threshold_mV,
        onset_threshold_fraction=onset_threshold_fraction,
        onset_noise_multiplier=onset_noise_multiplier,
        noise_sd=noise_sd,
        amplitude_threshold=amplitude_threshold,
        onset_stable_window_s=onset_stable_window_s,
    )
    epsp_amplitudes = local_amplitude['amplitudes_mV']
    baseline_fail_count = int(np.sum(~np.isfinite(local_amplitude['baselines_mV'])))
    if baseline_fail_count > 0 and epsp_peak_locs.size > 0:
        warnings_list.append(
            f"sEPSP local upshoot unavailable for {baseline_fail_count}/{len(epsp_peak_locs)} events"
        )
    search_edge_count = int(np.sum(np.asarray(local_amplitude['upshoot_status']) == 'search_edge_upshoot'))
    if search_edge_count > 0 and epsp_peak_locs.size > 0:
        warnings_list.append(
            f"sEPSP upshoot at search edge for {search_edge_count}/{len(epsp_peak_locs)} events; "
            "check slow or closely stacked events"
        )
    invalid_peak_count = int(np.sum(np.asarray(local_amplitude['upshoot_status']) == 'invalid_peak_upshoot'))
    if invalid_peak_count > 0 and epsp_peak_locs.size > 0:
        warnings_list.append(
            f"sEPSP peak not above upshoot for {invalid_peak_count}/{len(epsp_peak_locs)} events"
        )
    valid_samples = np.sum(accepted_samples_mask)
    valid_time = valid_samples * dt
    epsp_frequency = len(epsp_peak_locs) / valid_time if valid_time > 0 else 0

    if debug_plot:
        fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        time_s = np.arange(len(V_raw)) * dt

        # Raw trace and raw local amplitude measurements.
        ax[0].plot(time_s, V_raw, color='lightgray', linewidth=0.5, label='raw')
        ax[0].plot(time_s, np.where(accepted_samples_mask, V_raw, np.nan), color='black', linewidth=0.6, label='usable raw')
        ax[0].axhline(global_baseline, color='gray', linestyle='--', linewidth=0.8, label='global baseline')
        for p in ap_peak_locs:
            ax[0].axvline(p * dt, color='red', linewidth=0.6, alpha=0.25)
        rejected_peak_locs = peak_locs_all[~accepted_peak_mask] if peak_locs_all.size > 0 else []
        if len(rejected_peak_locs) > 0:
            ax[0].plot(time_s[rejected_peak_locs], V_raw[rejected_peak_locs], 'x', color='orange', markersize=3, label='rejected')
        raw_peak_locs = local_amplitude['raw_peak_locs']
        if raw_peak_locs.size > 0:
            status_colors = {
                'raw_rise_upshoot': 'green',
                'local_min_upshoot': 'royalblue',
                'search_edge_upshoot': 'darkorange',
                'invalid_peak_upshoot': 'magenta',
                'insufficient_upshoot': 'crimson',
            }
            status_labels = {
                'raw_rise_upshoot': 'accepted: raw-rise upshoot',
                'local_min_upshoot': 'accepted: local-min upshoot',
                'search_edge_upshoot': 'accepted: search-edge upshoot',
                'invalid_peak_upshoot': 'invalid: peak <= upshoot',
                'insufficient_upshoot': 'accepted: no upshoot',
            }
            statuses = np.asarray(local_amplitude['baseline_status'])
            for status, color in status_colors.items():
                status_mask = statuses == status
                if np.any(status_mask):
                    ax[0].plot(
                        time_s[raw_peak_locs[status_mask]],
                        local_amplitude['peak_values_mV'][status_mask],
                        'o',
                        color=color,
                        markersize=3,
                        label=status_labels[status],
                    )
            ax[0].plot(time_s[local_amplitude['onset_locs']], V_raw[local_amplitude['onset_locs']], '|', color='purple', markersize=8, label='onset estimate')
            for raw_peak_loc, baseline, peak_value in zip(
                raw_peak_locs,
                local_amplitude['baselines_mV'],
                local_amplitude['peak_values_mV']
            ):
                if np.isfinite(baseline) and np.isfinite(peak_value):
                    ax[0].vlines(raw_peak_loc * dt, baseline, peak_value, color='green', linewidth=2.4, alpha=0.95, zorder=8)
        ax[0].set_title(
            f"{folder_file or ''} raw EPSP amplitudes | "
            f"threshold={amplitude_threshold:.2f} mV | valid_time={valid_time:.2f} s"
        )
        ax[0].set_ylabel("Voltage (mV)")
        ax[0].legend(loc='best')

        # Processed detection space with mask/rejection context.
        masked_detection_trace = np.where(~accepted_samples_mask, V_analysis_trace, np.nan)
        ax[1].plot(time_s, V_analysis_trace, color='blue', linewidth=0.6, label='processed trace')
        ax[1].plot(time_s, masked_detection_trace, color='tomato', linewidth=0.8, alpha=0.7, label='masked sections')
        ax[1].axhline(amplitude_threshold, color='gray', linestyle='--', linewidth=0.8, label='threshold')
        ax[1].axhline(0, color='gray', linestyle=':', linewidth=0.6)
        if peak_locs_all.size > 0:
            ax[1].plot(time_s[peak_locs_all], V_analysis_trace[peak_locs_all], '.', color='orange', markersize=3, label='candidate peaks')
        if epsp_peak_locs.size > 0:
            ax[1].plot(time_s[epsp_peak_locs], V_analysis_trace[epsp_peak_locs], 'go', markersize=3, label='accepted peaks')
            ax[1].plot(time_s[local_amplitude['onset_locs']], V_analysis_trace[local_amplitude['onset_locs']], '|', color='purple', markersize=8, label='onset estimate')
        ax[1].set_title("Processed detection trace: peaks only, amplitudes from raw trace")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Baseline-corrected mV")
        ax[1].legend(loc='best')
        fig.tight_layout()
        plt.show()

    if debug_local_baseline_plot and epsp_peak_locs.size > 0:
        n_events = min(int(debug_event_count), len(epsp_peak_locs))
        fig, ax = plt.subplots(n_events, 1, figsize=(8, max(2, 1.8 * n_events)), sharex=False)
        if n_events == 1:
            ax = [ax]
        zoom_pre = max(int(0.040 / dt), int(min(onset_search_window_s + 0.010, 0.250) / dt))
        zoom_post = int(0.030 / dt)
        for idx in range(n_events):
            p = int(local_amplitude['raw_peak_locs'][idx])
            z_start = max(0, p - zoom_pre)
            z_end = min(len(V_raw), p + zoom_post)
            x_ms = (np.arange(z_start, z_end) - p) * dt * 1000
            ax[idx].plot(x_ms, V_raw[z_start:z_end], color='black', linewidth=0.8)
            base_start, base_end = local_amplitude['baseline_windows'][idx]
            base_start = max(base_start, z_start)
            base_end = min(base_end, z_end)
            search_start, search_end = local_amplitude['onset_search_windows'][idx]
            search_start = max(search_start, z_start)
            search_end = min(search_end, z_end)
            if search_end > search_start:
                ax[idx].axvspan(
                    (search_start - p) * dt * 1000,
                    (search_end - p) * dt * 1000,
                    color='lightsteelblue',
                    alpha=0.14,
                    label='upshoot search' if idx == 0 else None,
                )
            if base_end > base_start:
                ax[idx].axvspan((base_start - p) * dt * 1000, (base_end - p) * dt * 1000, color='gray', alpha=0.2)
            baseline_locs = local_amplitude['baseline_sample_locs'][idx]
            baseline_locs = baseline_locs[(baseline_locs >= z_start) & (baseline_locs < z_end)]
            if baseline_locs.size > 0:
                ax[idx].plot(
                    (baseline_locs - p) * dt * 1000,
                    V_raw[baseline_locs],
                    '.',
                    color='slategray',
                    markersize=2,
                    label='upshoot samples',
                )
            baseline = local_amplitude['baselines_mV'][idx]
            peak_value = local_amplitude['peak_values_mV'][idx]
            onset_loc = local_amplitude['onset_locs'][idx]
            if z_start <= onset_loc < z_end:
                ax[idx].axvline((onset_loc - p) * dt * 1000, color='purple', linestyle=':', linewidth=1)
            if np.isfinite(baseline):
                ax[idx].axhline(baseline, color='gray', linestyle='--', linewidth=0.8)
            if np.isfinite(peak_value):
                ax[idx].plot(0, peak_value, 'go', markersize=4)
            if np.isfinite(baseline) and np.isfinite(peak_value):
                ax[idx].vlines(0, baseline, peak_value, color='green', linewidth=1.8)
            ax[idx].set_ylabel("mV")
            ax[idx].set_title(
                f"event {idx + 1}: amp={local_amplitude['amplitudes_mV'][idx]:.2f} mV | "
                f"{local_amplitude['baseline_status'][idx]} | "
                f"prev={local_amplitude['previous_event_interval_ms'][idx]:.1f} ms | "
                f"next={local_amplitude['next_event_interval_ms'][idx]:.1f} ms"
            )
            if idx == 0:
                ax[idx].legend(loc='best')
        ax[-1].set_xlabel("Time from raw peak (ms)")
        plt.tight_layout()
        plt.show()

    return {
        'frequency_Hz': epsp_frequency,
        'amplitudes_mV': epsp_amplitudes,
        'peak_locs': epsp_peak_locs,
        'peak_values_mV': local_amplitude['peak_values_mV'],
        'local_baselines_mV': local_amplitude['baselines_mV'],
        'upshoot_values_mV': local_amplitude['upshoot_values_mV'],
        'raw_peak_locs': local_amplitude['raw_peak_locs'],
        'onset_locs': local_amplitude['onset_locs'],
        'upshoot_locs': local_amplitude['upshoot_locs'],
        'baseline_windows': local_amplitude['baseline_windows'],
        'onset_search_windows': local_amplitude['onset_search_windows'],
        'upshoot_windows': local_amplitude['upshoot_windows'],
        'baseline_sample_locs': local_amplitude['baseline_sample_locs'],
        'upshoot_sample_locs': local_amplitude['upshoot_sample_locs'],
        'baseline_status': local_amplitude['baseline_status'],
        'upshoot_status': local_amplitude['upshoot_status'],
        'onset_threshold_mV': local_amplitude['onset_threshold_mV'],
        'previous_event_interval_ms': local_amplitude['previous_event_interval_ms'],
        'next_event_interval_ms': local_amplitude['next_event_interval_ms'],
        'RMP_mV': global_baseline,
        'baseline_drift_mV': baseline_drift,
        'amplitude_threshold_mV': amplitude_threshold,
        'noise_sd_mV': noise_sd,
        'valid_time_s': valid_time,
        'ap_peak_locs': ap_peak_locs,
        'warnings': warnings_list,
        'debug': {
            'V_raw': V_raw,
            'V_analysis_trace': V_analysis_trace,
            'V_variability_trace': V_variability_trace,
            'drifting_baseline': drifting_baseline,
            'usable_mask': accepted_samples_mask,
            'valid_mask': valid_samples_mask,
            'ap_mask': ap_mask,
            'baseline_method': baseline_method,
        },
    }
    





def ap_characteristics_extractor_subroutine_derivative(folder_file, df_V_arr, sweep_index,  sampling_rate = 2e4 ,  input_backwards_window = 6 , input_ap_forwards_window = 10, input_smoothing_kernel=10, force_upshoot_detection = False):
    '''
    Extracts detailed characteristics of action potentials (APs) from voltage data within a specified sweep.

    Input:
        df_V_arr (2d array): A pandas DataFrame containing voltage data from electrophysiological recordings. cols == sweeps
        sweep_index (int): The index of the sweep in the DataFrame from which AP characteristics are to be extracted.
        sampling_rate (float, optional): The sampling rate of the data in Hz. Defaults to 20000 Hz.
        input_smoothing_kernel (float, optional): The size of the smoothing kernel to apply to the voltage data. Defaults to 10.
        input_backwards_window (int, optional): The window size used for searching backward from a peak to find the AP upshoot. Defaults to 100ms.
        input_ap_forwards_window (int, optional): The window size in ms to correct peak location after smoothing. Defaults to 3ms.

    Returns:
        AP_peak_voltages (list): 
        valid_AP_locations (list): AP peak locations in sweep (of psysilogically validated APs).
        AP_upshoot_locations_lis (list): Locations of AP upshoots in sweep.
        AP_voltage_thresholds_list (list): Voltage values at each AP upshoot (mV).
        AP_heights_list (list): Differences in voltage between each AP's peak and its threshold (mV).
        AP_latencies_list (list): Time between the AP upshoot and peak (ms).
        AP_rise_dvdt_list (list): Slope of rising phase (mv/ms) - 2/10 to 8/10 of AP height. 
        AP_fwhm_list (list): Full width at half maximum (FWHM) of each AP, if <0 set as peak_latency (ms).
        AP_max_dvdt_list (list): max dv/dt - from 5ms before upshoot to peak (mv/ms).
        AP_decay_dvdt_list (list): Slope of decay phase (mv/ms) - 2/10 to 8/10 of AP height. 

    '''    
    ap_backwards_window = int((input_backwards_window / 1000) * sampling_rate)
    ap_forwards_window = int((input_ap_forwards_window / 1000) * sampling_rate)

    sec_to_ms        = 1e3 
    min_ap_peak_voltage =  -10 # peak voltage cutoff -10mV HARD CODE 
    ap_width_min = 0.1 # ms
    ap_width_max = 4 # ms
    smoothing_kernel = input_smoothing_kernel
    smoothing_correction_window = int(0.001 * sampling_rate) #1ms to correct peak location from smoothed to raw trace

    # initialise output lists
    AP_peak_voltages              = []            #  voltage at peak of AP
    AP_locations_list            = []            #  corrected location of all peaks detected
    valid_AP_locations           = []            #  peak location of ONLY valid APs
    AP_upshoot_locations_list    = []            #  location of upshoots  
    AP_voltage_thresholds_list   = []            #  voltage threshold (mV) - voltage at upshoot
    AP_heights_list              = []            #  peak heights (mV)  - voltage diff from peak to upshoot 
    AP_latencies_list            = []            #  latency (ms) -  peak and threshold points
    AP_fwhm_list                 = []            #  width of peak at half maximum (FWHM) (ms)
    AP_rise_dvdt_list            = []            #  slope (mv/ms) - 2/10 to 8/10 of AP_height_rising
    AP_decay_dvdt_list           = []            #  slope (mv/ms) - 2/10 to 8/10 of AP_height_decaying
    AP_max_dvdt_list             = []            #  max dv/dt - from 5ms before upshoot to peak


    V_array = df_V_arr[:,sweep_index] #slice V_array to sweep
    v_smooth, peak_locs , peak_info , num_peaks  = ap_finder(V_array, smoothing_kernel=smoothing_kernel) #get smoothed peak_locs


    if len(peak_locs) == 0 :
        # print("No peaks found in sweep.")
        return  [] ,   [] ,  []  , [] ,  [] ,  [] , [] , [], [], []
    
    # PEAK LOCATION correction from smoothed trace
    for peak_idx in range(len(peak_locs)):

        start_idx = max(0, peak_locs[peak_idx] - smoothing_correction_window)  # ensure window does not start before index 0
        end_idx = peak_locs[peak_idx] + smoothing_correction_window  # end index extends forward from peak
        window = V_array[start_idx:end_idx]  # voltage slice around peak
        v_max = np.max(window)  # max voltage in the slice
        
        #OLD doubble peaks in folder_file F2976/2025_08_27_0013.  31/5/25
        # peak_locs_shift = (peak_locs[peak_idx] - start_idx) - np.where(window == v_max)[0][0]  # shift from estimated peak to true peak
        # AP_locations_list += [peak_locs[peak_idx] - peak_locs_shift]  # corrected peak location appended


        #TODO test bellow to replave above
        window_max_indices = np.where(window == v_max)[0]
        peak_loc_in_window = peak_locs[peak_idx] - start_idx #  index closest to the smoothed peak position in the window
        closest_idx = window_max_indices[np.argmin(np.abs(window_max_indices - peak_loc_in_window))]
        peak_locs_shift = peak_loc_in_window - closest_idx
        AP_locations_list.append(peak_locs[peak_idx] - peak_locs_shift)



    #FILTER ON PEAK VOLTAGE > min_ap_peak_voltage
    ls = list(np.where(V_array[AP_locations_list] >= min_ap_peak_voltage)[0])
    AP_locations_list  = [AP_locations_list[ls_ ] for ls_ in ls ] # list of accurate AP peak locations

    # NO VALID APs FOUND 
    if len(AP_locations_list) == 0:
        # print(f"Detected peaks max voltage  < {min_ap_peak_voltage}.") #catches nois/ EPSPs mostly
        return [] ,   [] ,  []  , [] ,  [] ,  [] , [] , [], [], []

    # REDEFINE WINDOW ap_backwards_window if inter_spike_interval  < ap_backwards_window
    if len(AP_locations_list) >= 2 : 
        ap_backwards_window = int(min(ap_backwards_window ,  np.min(np.diff(AP_locations_list))))

    ########## LOOPING PEAKS  ##########
    for peak_location in AP_locations_list:

        # CHECK AP WINDOW and SLICE V_array for upshoot detection
        if ap_backwards_window <  peak_location: # peak is at least one ap_backwards_window into trace
            v_temp = V_array[peak_location - ap_backwards_window: peak_location ]
        else: 
            v_temp = V_array[0:peak_location]

        #CREATE PEAK FREE WINDOW
        peaks, _ = find_peaks(v_temp) 
        peaks_off_ap = peaks # peaks above bottem 30% of voltage removed (likely on AP)
        if len(peaks) > 0:
            # Filter out peaks+/-1 that are on the top 60% of the voltage values (i.e., likely on AP)
            peaks_off_ap = [peak for peak in peaks if np.mean([v_temp[peak-1],v_temp[peak+1]])  < np.percentile(v_temp, 40)] #HARD CODE 30 doent catch mini spike at I step onset FP
            if len(peaks_off_ap) > 0 :
                peak_free_ap_backwards_window = ap_backwards_window - peaks_off_ap[-1] # is there are peaks in bottom 30% of voltage set window from them to peak
            else:
                peak_free_ap_backwards_window = ap_backwards_window #peaks are on AP
            peak_free_v_temp = V_array[peak_location - peak_free_ap_backwards_window: peak_location ] #peak free trace with upshoot and peak

        # UPSHOOT LOCATION
        v_derivative_temp = np.heaviside( -np.diff(v_temp)+ np.exp(1), 0 ) #create binary derviitive (0 = negatice derivative = V decreasing / 1  = positive derivitive = voltage increasing) 
        x_                = np.diff(v_derivative_temp) #dv/dt in AP window
        upshoot_loc_array = np.where(x_  <  0)[0]   # indices where the second derivative is -ive (x_)  indicating a negative change in derivitive .˙. 

        if len(upshoot_loc_array) == 0 : # occures most for depolarisation block or unhealthy spikes
            
            # FORCE FIND (ogten too late on slope)
            if force_upshoot_detection == True:
                dvdt = np.diff(v_temp) * sampling_rate / 1000
                second_dvdt = -np.diff(dvdt) 
                upshoot_candidates = np.where(dvdt > 5.5)[0] # first point > 5 mV/ms (max dvdt used also but I dont like)
                min_2nd_dvdt_index = np.argmin(second_dvdt)
                if len(upshoot_candidates) > 0:
                    upshoot_location = upshoot_candidates[np.argmin(np.abs(upshoot_candidates - min_2nd_dvdt_index))]
                else:
                    upshoot_location = min_2nd_dvdt_index + 1
                print(f"New AP detection implimented. Check for doublle peaks or poor data {folder_file} sweep {sweep_index}, peak index {peak_location}, upshoot index {upshoot_location}.")

            else:
                # print(f"No upshoot found via derivitive {folder_file} sweep {sweep_index}, analising next peak.") 
                continue

        if len(upshoot_loc_array) == 1 : 
            upshoot_loc_in_window_bin  = upshoot_loc_array[0] 

        elif len(upshoot_loc_array)>1 and len(peaks) == 0 :
            upshoot_loc_in_window_bin  = upshoot_loc_array[0]

        elif len(upshoot_loc_array)  > 1 : #if several options use peak_free
                        peak_free_v_derivative_temp = np.heaviside( -np.diff(peak_free_v_temp)+ np.exp(1), 0 ) #create binary derviitive (0 = negatice derivative = V decreasing / 1  = positive derivitive = voltage increasing) 
                        peak_free_x_                = np.diff(peak_free_v_derivative_temp) #dv/dt in AP window
                        peak_free_upshoot_loc_array = np.where(peak_free_x_  <  0)[0]   # indices where the second derivative is -ive (x_)  indicating a negative change in derivitive .˙. 
                        ap_backwards_window = peak_free_ap_backwards_window #redefine backwards window
                        if len(peak_free_upshoot_loc_array) ==0 :
                            print ('fuk - upshoot detection gone wrong manualy inspect')
                            continue
                        upshoot_loc_in_window_bin  = peak_free_upshoot_loc_array[0]
                        upshoot_loc_array = peak_free_upshoot_loc_array

        # UPSHOOT LOCATION in V_array (instead of backward_window)
        upshoot_location  =   peak_location - ap_backwards_window + upshoot_loc_in_window_bin  

        time_diff_ms = (peak_location - upshoot_location) / sampling_rate * 1000  # Convert to ms
        peaks_on_slope, _ = find_peaks(V_array[upshoot_location:peak_location])
        
        if time_diff_ms < 0.2 or time_diff_ms > 5.0: #physiological range 0.2-5ms upshoot to peak
            # plot_ap_window(folder_file, V_array,peak_location, upshoot_location, sampling_rate)
            # print(f"{folder_file}: UPSHOOT to PEAK time exclusion {time_diff_ms} ms, sweep {sweep_index}  peak index {peak_location}, analising next peak.")
            continue
        if abs(V_array[upshoot_location]-V_array[peak_location]) < 20: #physiological range 20mV upshoot to peak and 20mV height
            # plot_ap_window(folder_file, V_array,peak_location, upshoot_location, sampling_rate)
            # print(f"{folder_file}: UPSHOOT to PEAK height exclusion {abs(V_array[upshoot_location]-V_array[peak_location])} mV, sweep {sweep_index}  peak index {peak_location}, analising next peak.")
            continue
        if len(peaks_on_slope)>0: #was using 1 for HFD need 0 for Psych
            # plot_ap_window(folder_file, V_array,peak_location, upshoot_location, sampling_rate)
            # print(f"PEAKS ON AP SLOPE {folder_file}: sweep {sweep_index}  peak index {peak_location}, analising next peak.")
            continue

        # VOLTAGE THRESHOLD
        voltage_threshold = V_array[upshoot_location]

        # LATENCY
        AP_latency = sec_to_ms * (peak_location - upshoot_location)  / sampling_rate

        # SLOPE 
        decay_dvdt, rise_dvdt, max_dvdt, max_dvdt_location = calculate_ap_slope_and_max_dvdt(V_array, upshoot_location, peak_location, voltage_threshold, AP_latency, sampling_rate)
        if rise_dvdt <= 0 or max_dvdt <= 0:
            print("Slope/derivative of AP is negative, setting to nan.")
            plot_ap_window(folder_file, V_array,peak_location, upshoot_location, sampling_rate, voltage_threshold=voltage_threshold, latency=AP_latency, rise_dvdt=rise_dvdt, max_dvdt=max_dvdt, max_dvdt_location=max_dvdt_location)
            rise_dvdt, max_dvdt, max_dvdt_location  = np.nan , np.nan, np.nan
        if decay_dvdt >= 0:
            decay_dvdt = np.nan

        #CHECK FOR BAD UPSHOOT DETECTION
        # print(f"Verifying upshoot: peak to upshoot / peak to max dvdt {np.diff([peak_location, upshoot_location])} / {np.diff([peak_location, max_dvdt_location])}, {(np.diff([peak_location, upshoot_location])) / (np.diff([peak_location, max_dvdt_location]))}")
        if not (1<=  ((np.diff([peak_location, upshoot_location])) / (np.diff([peak_location, max_dvdt_location]))) <= 3.5):
                # print(f'Upshoot uneasonably far from peak relative to max dv/dt. Recalculating...')
                if max_dvdt_location < upshoot_location : #ratio < 1
                    upshoot_location = max_dvdt_location # occures on very wide APs wher back window is insufficient - FP late sweeps
                if len(upshoot_loc_array)>1:
                    upshoot_location  =   peak_location - ap_backwards_window +  upshoot_loc_array[1] #occures for AP on I step - there sould be a second possible upshoot .˙. take upshoot_loc_array[1]
                
                #REDO SLOPE
                decay_dvdt, rise_dvdt, max_dvdt, max_dvdt_index = calculate_ap_slope_and_max_dvdt(V_array, upshoot_location, peak_location, voltage_threshold, AP_latency, sampling_rate)
                if rise_dvdt <= 0 or max_dvdt <= 0:
                    # print(f"Action potential average slope or max dv/dt is negative with new upshoot, setting to nan (sweep: {sweep_index}).")
                    rise_dvdt, max_dvdt, max_dvdt_location  = np.nan , np.nan, np.nan
                if decay_dvdt >= 0:
                    decay_dvdt = np.nan

                # REDO LATENCY and VOLTAGE THRESHOLD
                AP_latency   = sec_to_ms * (peak_location - upshoot_location)  / sampling_rate
                voltage_threshold = V_array[upshoot_location]

            
        # PEAK VOLTAGE
        AP_peak_voltage   = V_array[peak_location]        
        if AP_peak_voltage > 120 :
            # print(f"Artifact detected, index {peak_location}, {AP_peak_voltage:.2f} > 120mV, analising next peak.")
            continue

        # AP HEIGHT
        AP_height = V_array[peak_location]  - voltage_threshold
        if AP_height > 150 or AP_height < 10: #HARD CODE #TODO
            # print(f"Artifact detected, index {peak_location}, AP height of {AP_height:.2f} mV, analising next peak.")
            continue

        #WIDTH
        if len(AP_locations_list) >= 2:
            isi_values = np.diff(AP_locations_list)
            isi_values = isi_values[isi_values > 0]
            inter_spike_interval = int(np.min(isi_values)) if len(isi_values) > 0 else int(0.5 * sampling_rate)
        else:
            inter_spike_interval = int(0.5 * sampling_rate) #500ms if only 1 AP in trace

        fwhm_ms = calculate_fwhm(folder_file, V_array, peak_location, upshoot_location, sampling_rate, sec_to_ms, ap_width_min, ap_width_max, inter_spike_interval)
        if not np.isfinite(fwhm_ms) or fwhm_ms <= 0:
            fwhm_ms = np.nan
        elif fwhm_ms < ap_width_min or fwhm_ms > ap_width_max:
            # print(f"Calculated FWHM is {fwhm_ms:.2f}, outside of plausible limits ({ap_width_min} - {ap_width_max} ms).")
            height_to_width_ratio = AP_height/fwhm_ms
            if not 40 < height_to_width_ratio < 100: #HARD CODE #TODO
                # print(f"AP height/width ratio is {height_to_width_ratio:.2f}, outside plausable limmits (40 - 100), poor compensation, setting fwhm to nan.")
                fwhm_ms = np.nan
            # else: 
                # plot_fwhm(folder_file, V_array, upshoot_location, peak_location, sampling_rate, sec_to_ms, inter_spike_interval)
                # print(f"AP height to width ratio, {height_to_width_ratio}, inside relevant bounds 40-100, appending fwhm as {fwhm_ms}.")

        
        #APPEND VALUES TO LISTS
        AP_peak_voltages +=              [AP_peak_voltage]
        valid_AP_locations +=           [peak_location]
        AP_upshoot_locations_list +=    [upshoot_location]
        AP_voltage_thresholds_list +=   [voltage_threshold]
        AP_heights_list +=              [AP_height]
        AP_latencies_list +=            [AP_latency]
        AP_decay_dvdt_list +=           [decay_dvdt] 
        AP_rise_dvdt_list +=            [rise_dvdt]  # AP_slope_list old
        AP_max_dvdt_list +=             [max_dvdt]
        AP_fwhm_list +=                 [fwhm_ms]

    return AP_peak_voltages, valid_AP_locations , AP_upshoot_locations_list, AP_voltage_thresholds_list , AP_heights_list , AP_latencies_list , AP_rise_dvdt_list , AP_fwhm_list, AP_max_dvdt_list, AP_decay_dvdt_list

########## AP EXTRACTOR MODULES
def get_window_bounds(peak_location, upshoot_location, array_length, inter_spike_interval, isi_multiplier=3):
    """
    Define the window around the action potential using a conservative estimate based on the peak location.
    """
    # # Calculate average ISI to estimate the duration of an individual AP
    # average_isi = (peak_location - upshoot_location) * isi_multiplier
    # window_start = max(0, upshoot_location - average_isi)
    # window_end = min(array_length, peak_location + average_isi)
    window_start = max(0, upshoot_location - int(isi_multiplier * inter_spike_interval // 2))
    window_end   = min(array_length, peak_location + int(isi_multiplier * inter_spike_interval // 2))
    return window_start, window_end

def calculate_fwhm(folder_file, v_array, peak_location, upshoot_location, sampling_rate, sec_to_ms,  ap_width_min, ap_width_max,inter_spike_interval):
    """
    Calculate the full width at half maximum (FWHM) of an action potential (AP).

    Parameters:
    v_array (numpy.ndarray): The array containing voltage data for a single sweep.
    peak_location (int): The index in v_array corresponding to the peak of the AP.
    upshoot_location (int): The index in v_array corresponding to the upshoot of the AP.
    sampling_rate (float): The sampling rate at which the data was recorded (in Hz).
    sec_to_ms (float): Conversion factor from seconds to milliseconds.
    ap_width_min (float): Minimum width of AP to be considered biologically plausible (in ms).
    ap_width_max (float): Maximum width of AP to be considered biologically plausible (in ms).

    Returns:
    float: The FWHM of the AP (in ms).
    """
    # # Define the window size based on the latency period
    # latency_period = (peak_location - upshoot_location) / sampling_rate * sec_to_ms
    # # A conservative window size, set to cover the entire AP
    # window_size = 4 * latency_period  

    # window_start = max(0, upshoot_location - int(window_size / sec_to_ms * sampling_rate))
    # window_end = min(len(v_array), peak_location + int(window_size / sec_to_ms * sampling_rate))

    window_start, window_end = get_window_bounds(peak_location, upshoot_location, len(v_array), inter_spike_interval)


    v_window = v_array[window_start:window_end]

    # Adjust indices for the new window
    adjusted_upshoot_location = upshoot_location - window_start
    adjusted_peak_location = peak_location - window_start

    # Calculate half max voltage
    upshoot_voltage = v_window[adjusted_upshoot_location]
    AP_peak_voltage = v_window[adjusted_peak_location]
    half_max_voltage = upshoot_voltage + (AP_peak_voltage - upshoot_voltage) / 2

    # Find indices where the voltage crosses the half max value
    crossings = np.where(np.diff(np.sign(v_window - half_max_voltage)))[0] + window_start

    # Find the crossing after the upshoot (ascending phase)
    fwhm_start_candidates = crossings[crossings < peak_location]
    if fwhm_start_candidates.size > 0:
        fwhm_start = fwhm_start_candidates[-1]  # The last crossing before the peak
    else:
        warnings.warn("No crossing found before peak for FWHM calculation, setting to NaN.")
        plot_fwhm(folder_file, v_array, upshoot_location, peak_location, sampling_rate, sec_to_ms, inter_spike_interval)
        return np.nan

    # Find the crossing after the peak (descending phase)
    fwhm_end_candidates = crossings[crossings > peak_location]
    if fwhm_end_candidates.size > 0:
        fwhm_end = fwhm_end_candidates[0]  # The first crossing after the peak
    else:
        print("No crossing found after peak for FWHM calculation, setting to NaN.")
        # warnings.warn("No crossing found after peak for FWHM calculation, setting to NaN.")
        # plot_fwhm(folder_file, v_array, upshoot_location, peak_location, sampling_rate, sec_to_ms, inter_spike_interval)
        return np.nan
    
    # Calculate FWHM in ms
    fwhm_ms = (fwhm_end - fwhm_start) / sampling_rate * sec_to_ms

    return fwhm_ms

def plot_fwhm(folder_file, v_array, upshoot_location, peak_location, sampling_rate, sec_to_ms, inter_spike_interval):
    """
    Plot the action potential and the half-maximum level to visualize the FWHM calculation.

    Parameters:
    v_array (numpy.ndarray): The array containing voltage data for a single sweep.
    upshoot_location (int): The index in v_array corresponding to the upshoot of the AP.
    peak_location (int): The index in v_array corresponding to the peak of the AP.
    sampling_rate (float): The sampling rate at which the data was recorded (in Hz).
    sec_to_ms (float): Conversion factor from seconds to milliseconds.
    """

    window_start, window_end = get_window_bounds(peak_location, upshoot_location, len(v_array), inter_spike_interval)

    # Ensure the window is valid
    if window_end <= window_start:
        print("Invalid window for FWHM calculation.")
        return

    # Extract the relevant window for plotting
    time_array = np.arange(window_start, window_end) / sampling_rate * sec_to_ms
    v_window = v_array[window_start:window_end]

    # Adjust indices for the new window
    adjusted_upshoot_location = upshoot_location - window_start
    adjusted_peak_location = peak_location - window_start

    # Calculate the half-max voltage
    upshoot_voltage = v_window[adjusted_upshoot_location]
    AP_peak_voltage = v_window[adjusted_peak_location]
    half_max_voltage = upshoot_voltage + (AP_peak_voltage - upshoot_voltage) / 2

    # Plot the voltage trace
    plt.figure(figsize=(12, 7))
    plt.plot(time_array, v_window, label='Voltage Trace', color='blue')

    # Mark the upshoot and peak of the AP
    plt.axvline(time_array[adjusted_upshoot_location], color='orange', linestyle='--', label='Upshoot')
    plt.axvline(time_array[adjusted_peak_location], color='red', linestyle='--', label='Peak')

    # Draw a horizontal line at the half-max voltage
    plt.axhline(half_max_voltage, color='green', linestyle='--', label='Half-Max Voltage')

    # Set plot labels and title
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title(f'FWHM Visualization {folder_file}')
    plt.legend()
    plt.show()

def calculate_ap_slope_and_max_dvdt(V_array, upshoot_location, peak_location, voltage_threshold, AP_latency, sampling_rate):
    """

    Parameters:
    V_array (numpy.ndarray): The array containing voltage data for a single sweep.
    upshoot_location (int): The index in V_array corresponding to the AP upshoot.
    peak_location (int): The index in V_array corresponding to the AP peak.
    latency (float): The latency of the AP in milliseconds.  
    sampling_rate (float): The sampling rate at which the data was recorded (in Hz).

    Returns: 
        - slope (float): The slope from the linear fit between 1/10 to 9/10 AP latency (in mV/ms). CHANGE TO 0.2 and 0.8 of AP height becomes rise_dvdt
        - decay_dvdt (float): linear slope of the decay phase of the AP from 20-80% of the AP_height
        - max_dvdt (float): The maximum rate of voltage change (dV/dt) in that window (in mV/ms), in the rising phase.
        - max_dvdt_index (int): The index in V_array where the max dV/dt occurs.
    """
    # voltage at 20* and 80% of the AP_height
    ap_height = V_array[peak_location] - V_array[upshoot_location]
    V_80 = V_array[peak_location] - 0.2 * ap_height
    V_20 = V_array[peak_location] - 0.8 * ap_height


    # rising phase
    V_array_rise = V_array[upshoot_location:peak_location]
    
    rise_20_80_mask = (V_array_rise <= V_80) & (V_array_rise >= V_20)
    V_rise_20_80 = V_array_rise[rise_20_80_mask]
    t_rise_20_80 = np.arange(len(V_rise_20_80)) / sampling_rate * 1000  # ms

    # Rising phase check
    if len(t_rise_20_80) > 1 and np.ptp(V_rise_20_80) > 1e-3:
        rise_dvdt, intercept, r_value, p_value, std_err = linregress(t_rise_20_80, V_rise_20_80)
    else:
        # print(f"AP too fast, <2 points between 20-80% rising: rise_dvdt set to np.nan")
        rise_dvdt = np.nan

    # max dvdt between upshoot and peak 
    derivative_rise = np.gradient(V_array_rise)
    max_dvdt_index = np.argmax(derivative_rise) + upshoot_location
    max_dvdt = derivative_rise[max_dvdt_index - upshoot_location] * sampling_rate / 1000  # convert to mV/ms

    # #Debug Plot 
    # plt.figure()
    # plt.plot(t_rise_20_80, V_rise_20_80, 'ko', label='20–80% Rise Phase')
    # plt.plot(t_rise_20_80, rise_dvdt * t_rise_20_80 + intercept, 'r-', label=f'Fit: {rise_dvdt:.2f} mV/ms')
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Voltage (mV)")
    # plt.title("Linear Fit of 20–80% Rise Phase")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    #decay phase 
    AP_latency_in_samples = int(AP_latency * sampling_rate / 1000)  # Convert latency in ms to samples in V array
    V_array_post_peak = V_array[peak_location:min(peak_location + 7 * AP_latency_in_samples, len(V_array))]
    V_decay_derivative = np.diff(V_array_post_peak)
    binary_v_decay_derivative = np.heaviside(V_decay_derivative, 0)

    positive_derivative_indices = np.where(binary_v_decay_derivative == 1)[0]
    valid_decay_candidates = positive_derivative_indices[positive_derivative_indices >= 2*AP_latency_in_samples] # trim to after 2* latency after peak avoiding noise at peak
    if len(valid_decay_candidates)>0:
        decay_end_index = peak_location + valid_decay_candidates[0] 
    else:
        decay_end_index = min(peak_location + 3 * AP_latency_in_samples, (len(V_array)-1)) # HARDCODE 3 * latency crop if dvdt doent cross 0

    if V_array[decay_end_index] > V_array[peak_location] - 0.3 * ap_height:
        try:
            decay_threshold_crossings = np.where(V_array_post_peak <= voltage_threshold)[0] #using return to voltage threshold as end
            decay_end_index = peak_location + decay_threshold_crossings[0]
        except IndexError:
            decay_end_index = peak_location + AP_latency_in_samples 

    V_array_decay = V_array[peak_location:decay_end_index]
    V_80_decay = V_array[peak_location] - 0.2 * ap_height
    V_20_decay = V_array[peak_location] - 0.8 * ap_height
    within_20_80_mask_decay = (V_array_decay <= V_80_decay) & (V_array_decay >= V_20_decay)
    V_20_80_decay = V_array_decay[within_20_80_mask_decay]
    t_array_20_80_decay = np.arange(len(V_20_80_decay)) / sampling_rate * 1000  # ms

    # Decay phase check
    if len(t_array_20_80_decay) > 1 and np.ptp(V_20_80_decay) > 1e-3:
        decay_dvdt, intercept, r_value, p_value, std_err = linregress(t_array_20_80_decay, V_20_80_decay)
    else:
        # print(f"AP too fast, <2 points between 20-80% decay: decay_dvdt set to np.nan")
        decay_dvdt = np.nan



    # # #Debug Plot
    # plt.figure()
    # plt.plot(t_array_20_80_decay, V_20_80_decay, 'bo', label='20–80% Decay Phase')
    # plt.plot(t_array_20_80_decay,
    #          decay_dvdt * t_array_20_80_decay + V_20_80_decay[0],  # line anchored to first point
    #          'g-', label=f'Fit: {decay_dvdt:.2f} mV/ms')
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Voltage (mV)")
    # plt.title("Linear Fit of 20–80% Decay Phase")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return decay_dvdt, rise_dvdt, max_dvdt, max_dvdt_index


# def old_calculate_ap_slope_and_max_dvdt(V_array, upshoot_index, latency, sampling_rate): #24/4/25
#     """
#     Calculate both the linear fit slope and maximum dV/dt of the action potential (AP) 
#     and return the index of max dV/dt.

#     Parameters:
#     V_array (numpy.ndarray): The array containing voltage data for a single sweep.
#     upshoot_index (int): The index in V_array corresponding to the AP upshoot.
#     latency (float): The latency of the AP in milliseconds.
#     sampling_rate (float): The sampling rate at which the data was recorded (in Hz).

#     Returns:
#     tuple: 
#         - slope (float): The slope from the linear fit between 1/10 to 9/10 AP latency (in mV/ms).
#         - max_dvdt (float): The maximum rate of voltage change (dV/dt) in that window (in mV/ms).
#         - max_dvdt_index (int): The index in V_array where the max dV/dt occurs.
#     """
#     # Adjust the start and end index for calculating the slope
#     latency_samples = int(latency * sampling_rate / 1000)
#     start_slope_index = upshoot_index + int(latency_samples * 2/10)
#     end_slope_index = upshoot_index + int(latency_samples * 8/10)

#     # Adjust the window for calculating max_dvdt to start 1ms before the upshoot
#     pre_upshoot_samples = int(1 * sampling_rate / 1000)  # Convert 5ms to samples
#     start_dvdt_index = max(0, upshoot_index - pre_upshoot_samples)
#     end_dvdt_index = upshoot_index + latency_samples  # Extend to the full latency period
    
#     # Calculate derivative over the adjusted window
#     derivative_window = np.gradient(V_array[start_dvdt_index:end_dvdt_index])
#     max_dvdt_index = np.argmax(derivative_window) + start_dvdt_index
#     max_dvdt = derivative_window[max_dvdt_index - start_dvdt_index] * sampling_rate / 1000  # convert to mV/ms

    
#     # Time array for linear regression, in milliseconds 
#     time_array = np.arange(start_slope_index, end_slope_index) / sampling_rate * 1000

#     if len(time_array) < 3:
#         # Adjust time_array to include the max_dvdt_index and its nearest points
#         nearest_points = [max(0, max_dvdt_index - 1), max_dvdt_index, min(len(V_array) - 1, max_dvdt_index + 1)]
#         time_array = np.array(nearest_points) / sampling_rate * 1000
#         V_array_for_slope = V_array[nearest_points]
#         # Correct the time_array to start from the first point's time
#         time_array -= time_array[0]
#         # print("Insufficient points for full linear regression. Using points surrounding max dv/dt for slope calculation.")
#     else:
#         V_array_for_slope = V_array[start_slope_index:end_slope_index]

#     # Perform linear regression
#     slope, intercept, _, _, _ = linregress(time_array, V_array_for_slope)

#     return slope, max_dvdt, max_dvdt_index

def ap_characteristics_extractor_main(folder_file, V_array, sampling_rate=2e4):
    '''
    Extracts action potential (AP) features from multiple voltage sweeps.

    Parameters:
        folder_file : str - Identifier for the data file/folder.
        V_array : 2D array - Voltage traces (time x sweeps).
        sampling_rate : float - Sampling rate in Hz.

    Returns:
        peak_voltages_all : list of float — AP peak voltages (mV)
        peak_latencies_all : list of float — Latency of AP peaks (ms)
        v_thresholds_all : list of float — Voltage thresholds (mV)
        peak_rise_all : list of float — Rise speed (20–80% of height, mV/ms)
        peak_max_dvdt_all : list of float — Max dV/dt during upstroke (mV/ms)
        peak_locs_corr_all : list of int — Index of AP peaks (indices within sweep) 
        upshoot_locs_all : list of int — Index of AP thresholds
        peak_heights_all : list of float — AP height (mV)
        peak_fw_all : list of float — AP width (ms)
        peak_indices_all : list of int — Index within total list of peaks
        sweep_indices_all : list of int — Sweep index for each AP
        peak_decay_all : list of float — Decay speed (20–80% of height, mV/ms)
    '''

    #ensure 2D_array
    if V_array.ndim == 1:
        V_array = V_array[:, np.newaxis]  # shape (30000,) -> (30000, 1)

    # itterating over sweeps
    sweep_indices = [i for i in range(V_array.shape[1])]

    # Initialise lists 
    peak_locs_corr_all   = []  # Locations of peaks corrected
    upshoot_locs_all     = []  # Locations of upshoots
    peak_indices_all     = []  # Indices of peaks
    sweep_indices_all    = []  # Indices of sweeps
    peak_voltages_all    = []  # Peak voltages
    peak_heights_all     = []  # Peak heights
    peak_latencies_all   = []  # Latencies of peaks
    v_thresholds_all     = []  # Voltage thresholds
    peak_rise_all        = []  # Rise rates (dV/dt)
    peak_decay_all       = []  # Decay rates (dV/dt)
    peak_fw_all          = []  # Full width
    peak_max_dvdt_all    = []  # Maximum dV/dt
    
    for sweep_index in sweep_indices: 

        peak_voltages_, peak_locs_corr_, upshoot_locs_, v_thresholds_, peak_heights_ ,  peak_latencies_ , peak_rise_dvdt_ , peak_fw_,  peak_max_dvdt_, peak_decay_dvdt_  =  ap_characteristics_extractor_subroutine_derivative(folder_file, V_array, sweep_index, sampling_rate=sampling_rate, input_backwards_window=10) #HFD was 10 but its too much

        if peak_locs_corr_  == [] : # if any list is empty 
            # print(f"No APs in sweep number {sweep_index+1}, index {sweep_index}.")
            pass 
        else: 
            peak_voltages_all  += peak_voltages_
            peak_locs_corr_all += peak_locs_corr_
            upshoot_locs_all   += upshoot_locs_
            peak_latencies_all += peak_latencies_
            v_thresholds_all   += v_thresholds_
            peak_rise_all     += peak_rise_dvdt_
            peak_decay_all     += peak_decay_dvdt_
            peak_heights_all   += peak_heights_
            peak_fw_all        += peak_fw_
            peak_max_dvdt_all    += peak_max_dvdt_
            peak_indices_all   +=  list(np.arange(0, len(peak_locs_corr_all))) 
            sweep_indices_all +=   [sweep_index]*len(peak_locs_corr_)

    return peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_rise_all  , peak_max_dvdt_all,  peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all   , peak_indices_all , sweep_indices_all , peak_decay_all


########################      RA DETECTION FUNCTION(S)  ####################


# def build_AP_DF(folder_file, V_array, I_array, v_thresh):
#     '''
#     BASE FUNCTION
#     Builds a df for a single file where each row is an AP with columns for AP charecteristics.

#     Input: 
#         folder_file (str)  : name of unique file identifier
#         V_array (np.ndarray) : 2D voltage array for folder_file, if not supplied fetched
#         I_array (np.ndarray) : 2D voltage array for folder_file, if not supplied fetched
#         v_thresh

#     Output: 
#         AP_df (pd.DataFrame): 
#                 'folder_file':          string inentifier
#                 'peak_location':        peak location within sweep
#                 'upshoot_location':     upshoot location within sweep
#                 'voltage_threshold':    voltage at detected upshoot
#                 'slope':                slope of AP 
#                 'latency':              time (s) from upshoot to peak             
#                 'peak_voltage':         voltage at AP peak
#                 'height':               mV height from upshoot to peak
#                 'width':                peak full width at half maximum
#                 'sweep':                sweep index
#                 'I_injected':           pA of current (I) injected
#                 'AP_type'               defult to np.NaN otherwise set in this finction to RA

#     '''

#     V_array_adj, I_array_adj = normalise_array_length(V_array, I_array, columns_match=True)
    
#     # Extract AP characteristics
#     peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_rise_all  , peak_max_dvdt_all,  peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all   , peak_indices_all , sweep_indices_all , peak_decay_all = ap_characteristics_extractor_main(folder_file, V_array)

#     # peak_voltages_all, peak_latencies_all, v_thresholds_all, peak_slope_all, peak_dvdt_max_all, peak_locs_corr_all, upshoot_locs_all, peak_heights_all, peak_fw_all, peak_indices_all, sweep_indices_all = ap_characteristics_extractor_main(folder_file, V_array)
    
#     # Early return if no APs found
#     if np.all(np.isnan(peak_latencies_all)):
#         print (f"No APs detected in voltage trace {folder_file}.")
#         return pd.DataFrame(columns=['folder_file', 'peak_location', 'upshoot_location', 'voltage_threshold',
#            'slope', 'latency', 'peak_voltage', 'height', 'width', 'sweep',
#            'I_injected', 'AP_type'])

#     # create list of same length peak_locs_corr_all of the current injected at that peak location #GPT HERE IS MY QUESTION

#     # Create DataFrame of APs
#     AP_df = pd.DataFrame({
#         'folder_file': folder_file,
#         'peak_location': peak_locs_corr_all,
#         'upshoot_location': upshoot_locs_all,
#         'voltage_threshold': v_thresholds_all,
#         'rise_dvdt': peak_rise_all,
#         'max_dvdt': peak_max_dvdt_all,
#         'decay_dvdt': peak_decay_all,
#         'latency': peak_latencies_all,
#         'peak_voltage': peak_voltages_all,
#         'height': peak_heights_all,
#         'width': peak_fw_all,
#         'sweep': sweep_indices_all,
#         'I_injected': [I_array[loc, 0] for loc in peak_locs_corr_all], #sweep index is 0 as I_array is indentical
#         'AP_type': np.NaN  # default 
#     })

#     # Classify APs as 'RA_true' if threshold < -65 mV and AP_turn around > 20mV  

#     AP_df.loc[(v_thresh < -65) & (AP_df['peak_voltage'] > 20), 'AP_type'] = 'RA'
    
#     return AP_df




def pAD_detection(folder_file, V_array, sampling_rate=2e4): #old and unused?
    '''
    Main pAD detection algorithm.
    Input: 
        V_array: np.array : Voltage array of the trace to be analysed.
    Output: 
        peak_latencies_all, v_thresholds_all, peak_slope_all, peak_heights_all, pAD_df
    '''

    # Extract AP characteristics
    peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_rise_all  , peak_max_dvdt_all,  peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all   , peak_indices_all , sweep_indices_all , peak_decay_all = ap_characteristics_extractor_main(folder_file, V_array, sampling_rate=sampling_rate)

    #old peak_voltages_all, peak_latencies_all, v_thresholds_all, peak_slope_all, peak_dvdt_max_all, peak_locs_corr_all, upshoot_locs_all, peak_heights_all, peak_fw_all, peak_indices_all, sweep_indices_all = ap_characteristics_extractor_main(folder_file, V_array)
    
    # Early return if no APs found
    if np.all(np.isnan(peak_latencies_all)):
        print (f"No APs detected in voltage trace.")
        return peak_voltages_all, peak_latencies_all, peak_locs_corr_all, v_thresholds_all, peak_rise_all, peak_heights_all, np.nan

    # Create DataFrame of APs
    pAD_df = pd.DataFrame({
        'AP_loc': peak_locs_corr_all,
        'upshoot_loc': upshoot_locs_all,
        'AP_threshold': v_thresholds_all,
        'AP_slope': peak_rise_all,
        'AP_latency': peak_latencies_all,
        'AP_turn_around': peak_voltages_all,
        'AP_height': peak_heights_all,
        'AP_width': peak_fw_all,
        'AP_sweep_num': sweep_indices_all,
        'AP_type': 'somatic'  # default to 'somatic'
    })

    # Classify APs as 'pAD_true' if threshold < -65 mV and AP_turn around > 20mV                        #HARD CODE
    pAD_df.loc[(pAD_df['AP_threshold'] < -65) & (pAD_df['AP_turn_around'] > 20), 'AP_type'] = 'pAD_true'

    # Prepare data for clustering
    pAD_df_uncertain = pAD_df[pAD_df['AP_type'] != 'pAD_true']
    if len(pAD_df_uncertain) < 2:
        print (f"Fewer than 2 APs with voltage threshold > -65mV.")
        return peak_voltages_all, peak_latencies_all, peak_locs_corr_all, v_thresholds_all, peak_rise_all, peak_heights_all, pAD_df

    
    #OLD TO IDENTIFY pAD / RA APs that do not fir base criteria 
    # # Clustering with KMeans
    # X = pAD_df_uncertain[['AP_slope', 'AP_threshold', 'AP_latency']]
    # kmeans = KMeans(n_clusters=2, n_init=1).fit(X)
    # labels = kmeans.labels_

    # # Clustering with GMM
    # X = pAD_df_uncertain[['AP_slope', 'AP_threshold', 'AP_latency']]
    # gmm = GaussianMixture(n_components=2, n_init=1).fit(X)
    # labels = gmm.predict(X)

    # # Assign GMM labels as 'pAD_possible' or 'somatic'
    # pAD_df.loc[pAD_df['AP_type'] == 'somatic', 'AP_type'] = np.where(labels == 0, 'pAD_possible', 'somatic')

    return peak_voltages_all, peak_latencies_all, peak_locs_corr_all, v_thresholds_all, peak_rise_all, peak_heights_all, pAD_df


########## HANDELIN FIRING PROPERTY DATA (FP)  --  FI curves
def plot_APs_off_step(folder_file, V_array, I_array, peak_locs_corr_all, sweep_indices_all, sweep_to_plot):
    '''
    Plots action potentials (APs) that occur off the current step for a specific sweep.

    Input:
        V_array (numpy array): 2D array containing voltage recordings for different current steps.
        I_array (numpy array): 2D array containing current recordings for different current steps.
        peak_locs_corr_all (list): List of peak locations.
        sweep_indices_all (list): List of sweep indices corresponding to peak_locs_corr_all.
        sweep_to_plot (int): The sweep index for which to plot APs off the current step.

    Returns:
        None (plots the APs).
    '''
    V_array_adj, I_array_adj = normalise_array_length(V_array, I_array, columns_match=True)

    # Get the sweep-specific peak locations and corresponding sweep indices
    sweep_peak_locs = [peak_locs_corr_all[i] for i, sweep_index in enumerate(sweep_indices_all) if sweep_index == sweep_to_plot]
    current_sweep = I_array_adj[:, sweep_to_plot]
    voltage_sweep = V_array_adj[:, sweep_to_plot]

    # Identifying the current injection step
    current_injection_indices = np.where(current_sweep != 0)[0]
    if len(current_injection_indices) == 0:
        print(f"No current injection step detected in sweep {sweep_to_plot}.")
        return

    first_current_point = current_injection_indices[0]
    last_current_point = current_injection_indices[-1]

    # Check for spikes off the current step
    ap_off_step = [peak for peak in sweep_peak_locs if peak < first_current_point or peak > last_current_point]

    if ap_off_step:
        # Plot APs off the current step
        plt.figure(figsize=(10, 6))
        plt.plot(voltage_sweep, label='Voltage (mV)')
        plt.scatter(ap_off_step, voltage_sweep[ap_off_step], color='red', marker='o', label='APs off current step')
        plt.axvline(first_current_point, color='green', linestyle='--', label='Start of Current Injection')
        plt.axvline(last_current_point, color='blue', linestyle='--', label='End of Current Injection')
        plt.xlabel('Time')
        plt.ylabel('Membrane Potential (mV)')
        plt.title(f'APs Off Current Step {folder_file} - Sweep {sweep_to_plot + 1}')
        plt.legend()
        plt.show()
    else:
        print(f"No APs detected off the current step in sweep {sweep_to_plot}.")


def _step_indices_from_command_trace(trace, threshold=1.0):
    """
    Detect the active command window relative to the command baseline.
    """
    if trace is None:
        return None, np.nan, None

    trace = np.asarray(trace, dtype=float).flatten()
    if trace.size < 2 or np.all(np.isnan(trace)):
        return None, np.nan, None

    edge_n = min(max(5, int(0.05 * trace.size)), max(1, trace.size // 2))
    baseline = np.nanmedian(np.concatenate([trace[:edge_n], trace[-edge_n:]]))
    delta = trace - baseline
    max_delta = np.nanmax(np.abs(delta))
    if not np.isfinite(max_delta) or max_delta < threshold:
        return None, np.nan, None

    active_threshold = max(threshold, 0.1 * max_delta)
    active_indices = np.where(np.abs(delta) >= active_threshold)[0]
    if active_indices.size < 2:
        return None, np.nan, None

    segments = np.split(active_indices, np.where(np.diff(active_indices) > 1)[0] + 1)
    step_indices = max(segments, key=len)
    if step_indices.size < 2:
        return None, np.nan, None

    rest_mask = np.ones(trace.size, dtype=bool)
    rest_mask[step_indices] = False
    rest_indices = np.where(rest_mask)[0]
    step_value = np.nanmedian(trace[step_indices]) - baseline
    return step_indices, step_value, rest_indices


def _as_2d_array(array):
    """Return an array as time x sweeps, or None when missing."""
    if array is None:
        return None
    array = np.asarray(array, dtype=float)
    if array.size == 0:
        return None
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def has_protocol_steps(array, threshold=1.0):
    """True when any sweep has a command/protocol deflection from baseline."""
    array = _as_2d_array(array)
    if array is None:
        return False
    for sweep in range(array.shape[1]):
        step_indices, _, _ = _step_indices_from_command_trace(array[:, sweep], threshold=threshold)
        if step_indices is not None:
            return True
    return False


def _first_command_step_from_array(command_array):
    command_array = _as_2d_array(command_array)
    if command_array is None:
        return None, np.nan, None

    for sweep in range(command_array.shape[1]):
        step_indices, step_value, rest_indices = _step_indices_from_command_trace(
            command_array[:, sweep]
        )
        if step_indices is not None:
            return step_indices, step_value, rest_indices
    return None, np.nan, None


def protocol_array_to_match_V(V_array, protocol_array):
    """
    Align protocol/command samples and sweeps to a voltage array.

    A single protocol sweep is tiled across voltage sweeps. If both arrays have
    multiple unequal sweep counts, both are cropped to their shared sweep count.
    """
    V_array = _as_2d_array(V_array)
    protocol_array = _as_2d_array(protocol_array)
    if protocol_array is None:
        return V_array, None

    V_array_adj, protocol_array_adj = normalise_array_length(
        V_array,
        protocol_array,
        columns_match=False,
        verbose=False
    )
    V_array_adj = _as_2d_array(V_array_adj)
    protocol_array_adj = _as_2d_array(protocol_array_adj)

    if protocol_array_adj.shape[1] == 1 and V_array_adj.shape[1] > 1:
        protocol_array_adj = np.tile(protocol_array_adj, (1, V_array_adj.shape[1]))
    elif protocol_array_adj.shape[1] != V_array_adj.shape[1]:
        min_cols = min(V_array_adj.shape[1], protocol_array_adj.shape[1])
        V_array_adj = V_array_adj[:, :min_cols]
        protocol_array_adj = protocol_array_adj[:, :min_cols]

    return V_array_adj, protocol_array_adj


def command_array_to_match_V(V_array, command_array):
    """Return command array aligned to voltage array, preserving per-sweep commands."""
    _, command_array_adj = protocol_array_to_match_V(V_array, command_array)
    return command_array_adj


def _match_command_array_to_v(V_array, command_array):
    """Backward-compatible alias for command_array_to_match_V."""
    return command_array_to_match_V(V_array, command_array)


def extract_FI_x_y(
    folder_file,
    V_array,
    I_array,
    sampeling_rate,
    peak_locs_corr_all=None,
    sweep_indices_all=None,
    command_array=None,
    return_details=False
):
    '''
    Extracts data for Frequency-Current (FI) relationship from voltage recordings.

    Input:
        V_array (np.ndarray):  2D array containing voltage recordings for different current steps (sweeps).
        I_array (np.ndarray): measured current, used only as a fallback protocol source when command_array is absent.
        peak_locs_corr_all (list, optional): AP peak locations already extracted for folder_file.
        sweep_indices_all (list, optional): Sweep index for each precomputed AP peak.
        command_array (np.ndarray, optional): Clamp command used for step timing and size when present.

    Returns:
        step_current_values (list): List of injected current values in picoamperes (pA) for each sweep.
        ap_counts (list): List of action potential counts for each sweep.
        V_rest (float): Average resting membrane potential (in millivolts) calculated when no current is injected.
    '''
    V_array_adj, protocol_array, step_source = select_protocol_array(
        V_array,
        command_array=command_array,
        I_array=I_array,
        clean_I_fallback=False
    )
    if protocol_array is None:
        print(f"No step detected in {folder_file}, unable to calculate FI properties.")
        if return_details:
            return np.nan, np.nan, np.nan, np.nan, {
                "used_command_array_for_steps": False,
                "used_I_array_fallback": False,
                "step_source": step_source,
            }
        return np.nan, np.nan, np.nan, np.nan

    use_precomputed_aps = peak_locs_corr_all is not None and sweep_indices_all is not None
    peaks_by_sweep = {}
    if use_precomputed_aps:
        for peak_loc, sweep_index in zip(peak_locs_corr_all, sweep_indices_all):
            try:
                peaks_by_sweep.setdefault(int(sweep_index), []).append(int(peak_loc))
            except (TypeError, ValueError):
                continue

    I_steps = [] 
    AP_frequencies_Hz = []
    V_rest_values = []
    off_step_peak_locs = [] # off step

    template_step_indices, _, template_rest_indices = _first_command_step_from_array(protocol_array)

    for sweep in range(protocol_array.shape[1]):
        protocol_sweep = protocol_array[:, sweep]
        V_sweep = V_array_adj[:, sweep]

        if use_precomputed_aps:
            sweep_peak_locs = peaks_by_sweep.get(sweep, [])
        else:
            (
                peak_voltages_all,
                peak_latencies_all,
                v_thresholds_all,
                peak_rise_all,
                peak_max_dvdt_all,
                sweep_peak_locs,
                upshoot_locs_all,
                peak_heights_all,
                peak_fw_all,
                peak_indices_all,
                sweep_indices_all,
                peak_decay_all,
            ) = ap_characteristics_extractor_main(folder_file, V_sweep, sampling_rate=sampeling_rate)

        # index protocol step
        non_zero_indices, I_step, V_rest_indices = _step_indices_from_command_trace(protocol_sweep)
        if non_zero_indices is None:
            if template_step_indices is None:
                print(f"No step detected in {folder_file}, unable to calculate FI properties.")
                if return_details:
                    return np.nan, np.nan, np.nan, np.nan, {
                        "used_command_array_for_steps": step_source == "command_array",
                        "used_I_array_fallback": step_source == "I_array_fallback",
                        "step_source": step_source,
                    }
                return np.nan, np.nan, np.nan, np.nan
            non_zero_indices = template_step_indices
            V_rest_indices = template_rest_indices
            I_step = 0
        I_step = int(round(I_step / 10.0)) * 10 # round to nearest 10pA
        I_steps.append(int(I_step))

        # f_Hz
        ap_on_step = sum(
            1 for peak_loc in sweep_peak_locs
            if non_zero_indices[0] <= peak_loc <= non_zero_indices[-1]
        )
        setep_in_seconds = len(non_zero_indices)/sampeling_rate
        ap_frequency_Hz = ap_on_step/setep_in_seconds
        AP_frequencies_Hz.append(ap_frequency_Hz)

        # V_rest_step
        if len(V_rest_indices) > 0:
            APnan_sweep = spike_remover_nan(V_sweep[V_rest_indices], threshold_sd=2)
            V_rest_values.append(np.nanmean(APnan_sweep))


        # Check for spikes off the current step
        ap_off_step = [peak for peak in sweep_peak_locs if peak < (non_zero_indices[0]) or peak > (non_zero_indices[-1]+10)] # 10ms buffer added after step
        if ap_off_step:
            # print(f"APs detected off current step at {np.mean(V_sweep[V_rest_indices]):.2f}mV in sweep {sweep+1}. ") #TODO 
            # plot_APs_off_step(folder_file, V_array, I_array, peak_locs_corr_all, sweep_indices_all, sweep_to_plot=sweep)
            off_step_peak_locs.extend(ap_off_step)
            

    V_rest = np.nanmean(V_rest_values) if len(V_rest_values) > 0 else np.nan

    if return_details:
        return I_steps , AP_frequencies_Hz, V_rest , off_step_peak_locs, {
            "used_command_array_for_steps": step_source == "command_array",
            "used_I_array_fallback": step_source == "I_array_fallback",
            "step_source": step_source,
        }
    return I_steps , AP_frequencies_Hz, V_rest , off_step_peak_locs


def correct_I_offset_IF(I_array_adj, threshold_pA=1.0, step_threshold=5):
    """
    Corrects baseline offset (holding current) in I_array_adj for IF steps.
    """
    # find flattest sweep
    stds = [np.std(I_array_adj[:, i]) for i in range(I_array_adj.shape[1])]
    flattest_idx = np.argmin(stds)
    sweep = I_array_adj[:, flattest_idx]
    
    if np.ptp(sweep) < step_threshold:  # peak-to-peak < threshold
        offset = np.mean(sweep)
    else:
        # fallback using pre/post step points - detect step by largest change
        diffs = np.max(I_array_adj, axis=0) - np.min(I_array_adj, axis=0)
        step_idx = np.argmax(diffs)
        pre_step = I_array_adj[:5, step_idx]   
        post_step = I_array_adj[-5:, step_idx] 
        offset = np.mean(np.concatenate([pre_step, post_step]))

    if abs(offset) > threshold_pA:
        I_array_adj -= offset
    else:
        offset = 0

    return I_array_adj, offset



def denoise_steps(I_array_adj):
    """
    Replaces each sweep's segments (pre-step, on-step, post-step) with median-based integers.
    Assumes offset has already been corrected.
    """
    n_time, n_sweeps = I_array_adj.shape
    denoised = np.zeros_like(I_array_adj)

    for i in range(n_sweeps):
        trace = I_array_adj[:, i]
        if np.all(trace == 0): # clean all 0 trace
            denoised[:, i] = 0
            continue
        d = np.diff(trace)
        threshold = np.max(np.abs(d)) * 0.3
        step_idx = np.where(np.abs(d) > threshold)[0]

        # Default to full sweep if no step detected
        if len(step_idx) >= 2:
            start = step_idx[0] + 1
            end = step_idx[-1] + 1
        else:
            start, end = n_time, n_time

        # # Use rounded medians - too noisy I get 182 instead of 180 for instance
        # denoised[:start, i] = int(round(np.median(trace[:start])))
        # denoised[start:end, i] = int(round(np.median(trace[start:end])))
        # denoised[end:, i] = int(round(np.median(trace[end:])))
        
        # Use rounded medians to nearest 5
        denoised[:start, i] = int(5 * round(np.median(trace[:start]) / 5))
        denoised[start:end, i] = int(5 * round(np.median(trace[start:end]) / 5))
        denoised[end:, i] = int(5 * round(np.median(trace[end:]) / 5))

    return denoised


def select_protocol_array(V_array, command_array=None, I_array=None, clean_I_fallback=True):
    """
    Return the best available protocol trace aligned to V_array.

    command_array is preferred because it is the intended clamp command. If it
    is missing, measured I_array can be cleaned and used as a legacy fallback
    to infer protocol timing/size. Returns (V_array_adj, protocol_array, source).
    """
    V_array = _as_2d_array(V_array)
    V_command, command_array_adj = protocol_array_to_match_V(V_array, command_array)
    if has_protocol_steps(command_array_adj):
        return V_command, command_array_adj, "command_array"

    measured_protocol = _as_2d_array(I_array)
    if measured_protocol is not None:
        measured_protocol = measured_protocol.copy()
        if clean_I_fallback:
            measured_protocol, _ = correct_I_offset_IF(measured_protocol)
            measured_protocol = denoise_steps(measured_protocol)
        V_I, I_array_adj = protocol_array_to_match_V(V_array, measured_protocol)
        if has_protocol_steps(I_array_adj):
            return V_I, I_array_adj, "I_array_fallback"

    return V_array, None, "missing"


# def correct_current_offset_and_denoise(I_array_adj, folder_file, threshold_pA=1.0):
#     """
#     Denoises and baseline-corrects I_array_adj.
#     Replaces each sweep's segments (pre-step, on-step, post-step) with rounded mean.
#     """
#     n_time, n_sweeps = I_array_adj.shape
#     denoised = np.zeros_like(I_array_adj)
    

#     for i in range(n_sweeps):
#         trace = I_array_adj[:, i]
#         d = np.diff(trace)
#         threshold = np.max(np.abs(d)) * 0.3
#         step_idx = np.where(np.abs(d) > threshold)[0]

#         # Default to full sweep if no step detected
#         start, end = (step_idx[0]+1, step_idx[-1]+1) if len(step_idx) >= 2 else (n_time, n_time)

#         denoised[:start, i] = int(np.median(trace[:start]))
#         denoised[start:end, i] = int(np.median(trace[start:end]))
#         denoised[end:, i] = int(np.median(trace[end:]))

#     flattest_idx = np.argmin([np.std(denoised[:, i]) for i in range(n_sweeps)])
#     offset = np.mean(denoised[:, flattest_idx])

#     if abs(offset) > threshold_pA:
#         print(f"I_array for {folder_file} is offset by {offset:.2f} pA, correcting. Verify if holding I used.")
#         denoised -= offset
#     else:
#         offset = 0

#     return denoised, offset



def FI_slope_and_rheobase(folder_file, x, y, min_consecutive=3, return_details=False, verbose=True):
    """
    Calculate IF slope and rheobase threshold based on the first APs.
    
    Parameters:
        folder_file (str): Unique identifier.
        x (np.ndarray): Current injection (pA).
        y (np.ndarray): Firing rate (Hz).
    
    Returns:
        tuple: (FI_slope, rheobase_threshold, valid_FP) by default.
        If return_details=True, adds a fourth fit-details dictionary.
    """

    fit_details = {
        "status": None,
        "method": None,
        "fit_quality": np.nan,
        "last_I": np.nan,
        "first_I": np.nan,
        "fit_points": np.nan,
    }

    def finish(slope, rheo, valid, status, method=None, message=None, **updates):
        details = fit_details.copy()
        details.update({
            "status": status,
            "method": method,
        })
        details.update(updates)
        if message and verbose:
            print(message)
        if return_details:
            return slope, rheo, valid, details
        return slope, rheo, valid

    def valid_fit(slope, intercept, x_fit, y_fit, var_y, last_I, first_I, min_fit_quality=0.5, margin_pA=5):
        """
        Evaluates whether a linear fit is valid based on normalized residuals and rheobase bounds.

        Parameters:
            slope (float): Slope of the linear fit.
            intercept (float): Intercept of the linear fit.
            x_fit (np.ndarray): x values used for fit.
            y_fit (np.ndarray): y values used for fit.
            var_y (float): Variance of y_fit.
            last_I (float): Last current step without action potentials.
            first_I (float): First current step with action potentials.
            min_fit_quality (float, default=0.2):  Normalized residual.
            min_consecutive (float, default=2): number of consecutive sweeps required to start slope calculation.

        Returns:
            is_valid (bool): True if fit passes quality check and rheobase falls between
                the last step without APs and the first step with APs.
            rheo (float): Estimated rheobase (x-intercept of the fit).
            fit_quality (float): Normalized residuals of the fit.
        """
        residuals = np.sum((np.polyval([slope, intercept], x_fit) - y_fit) ** 2)
        fit_quality = residuals / var_y if var_y else np.inf
        rheo = -intercept / slope if slope != 0 else np.nan
        if not np.isfinite(rheo):
            return False, rheo, fit_quality

        return (fit_quality <= min_fit_quality and last_I <= rheo < first_I+margin_pA), rheo, fit_quality

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
        return finish(
            np.nan,
            np.nan,
            False,
            "invalid_input",
            message=f"Invalid FI input for {folder_file}."
        )

    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return finish(
            np.nan,
            np.nan,
            False,
            "non_finite",
            message=f"Non-finite values detected in {folder_file}. Skipping fit."
        )

    list_of_non_zero = np.flatnonzero(y > 0) #indexes with APs
    if len(list_of_non_zero) == 0:
        return finish(
            np.nan,
            np.nan,
            False,
            "no_APs",
            message=f'NO APs DETECTED: {folder_file} check FP data or AP health.'
        )

    # Find first sustained APs (ignore single AP outliers)
    first_idx = None
    seq_len = 0
    for i in range(len(list_of_non_zero) - min_consecutive + 1):
        if np.all(np.diff(list_of_non_zero[i:i + min_consecutive]) == 1):
            first_idx = list_of_non_zero[i]
            
            seq_end = first_idx # find how long this consecutive run actually lasts
            while (seq_end + 1 < len(y)) and (y[seq_end + 1] > 0):
                seq_end += 1
            seq_len = seq_end - first_idx + 1
            break        
    else:
        return finish(
            np.nan,
            np.nan,
            False,
            "no_consecutive_APs",
            message=f"No consecutive APs detected: {folder_file}"
        )

    if first_idx is None or first_idx == 0:
        return finish(
            np.nan,
            np.nan,
            False,
            "missing_last_no_AP_step",
            message=f'Cannot determine last I without APs for: {folder_file}'
        )

    last_I = x[first_idx - 1] # last current step with NO spikes
    first_I = x[first_idx] # first current step WITH spikes

    last_slope = np.nan
    last_fit_quality = np.nan
    last_points = np.nan

    for points in range(min(7, seq_len)  , min_consecutive - 1, -1):
        x_fit = x[first_idx-1:first_idx + points]
        y_fit = y[first_idx-1:first_idx + points]
        y_fit = np.round(y_fit, 3)

        var_y = np.var(y_fit)
        # if var_y == 0 or np.isnan(var_y):
        if var_y < (np.mean(y_fit) * 1e-6) ** 2 or np.isnan(var_y): 
            rheo = (last_I + first_I) / 2
            return finish(
                np.nan,
                rheo,
                True,
                "fallback_flat_firing",
                method="bracket_midpoint",
                message=f"IF_slope incalculable non-variable firing frequency for {folder_file}; using bracket midpoint rheobase.",
                last_I=last_I,
                first_I=first_I,
                fit_points=len(x_fit)
            )
        
        # OPTION 1
        slope, intercept, r_value, p_value, std_err = linregress(x_fit, y_fit)
        last_slope = slope
        last_points = len(x_fit)

        # OPTION 2
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore', np.RankWarning)
        #     slope, intercept = np.polyfit(x_fit, y_fit, 1)
        # slope, intercept = np.polyfit(x_fit, y_fit, 1) # RankWarning: Polyfit may be poorly conditioned

        #OPTION 3 
        # A = np.vstack([x_fit, np.ones_like(x_fit)]).T
        # slope, intercept = np.linalg.lstsq(A, y_fit, rcond=None)[0]

        # plot_FI_curve_and_fit(folder_file, x, y, slope, intercept) 
        is_good_fit, rheo, fit_quality = valid_fit(slope, intercept, x_fit, y_fit, var_y, last_I, first_I)
        last_fit_quality = fit_quality
        

        if is_good_fit:
            return finish(
                slope,
                rheo,
                True,
                "ran",
                method="linear_intercept",
                fit_quality=fit_quality,
                last_I=last_I,
                first_I=first_I,
                fit_points=len(x_fit)
            )

    # Fallback: threshold is bracketed even when the early FI curve is not linear enough.
    bracket_rheo = (last_I + first_I) / 2
    if np.isfinite(last_slope) and last_slope > 0:
        return finish(
            last_slope,
            bracket_rheo,
            True,
            "fallback_bracket",
            method="bracket_midpoint",
            message=f"Unable to calculate strict FI fit for {folder_file}; using bracket midpoint rheobase.",
            fit_quality=last_fit_quality,
            last_I=last_I,
            first_I=first_I,
            fit_points=last_points
        )

    return finish(
        np.nan,
        bracket_rheo,
        True,
        "fallback_bracket_no_slope",
        method="bracket_midpoint",
        message=f"Unable to calculate FI slope for {folder_file}; using bracket midpoint rheobase.",
        fit_quality=last_fit_quality,
        last_I=last_I,
        first_I=first_I,
        fit_points=last_points
    )



def plot_FI_curve_and_fit(folder_file, x, y, slope, intercept):
    """
    Plot the Frequency-Current (F-I) curve with a linear fit and mark the x-intercept.

    Parameters:
    - x: numpy array of current inputs.
    - y: numpy array of corresponding firing rates.
    - slope: slope of the linear fit.
    - intercept: y-intercept of the linear fit.
    """
    # Plot the original data points
    plt.scatter(x, y, label='Data points')
    
    # Generate x values for the fit line
    x_fit = np.linspace(min(x), max(x), 100)
    # Calculate the corresponding y values from the linear fit equation
    y_fit = slope * x_fit + intercept
    
    # Plot the linear fit line
    plt.plot(x_fit, y_fit, 'r-', label=f'Linear fit: y = {slope:.2f}x + {intercept:.2f}')
    
    # Mark the x-intercept (rheobase threshold) if it's within the plotted x range
    if intercept <= 0:
        rheobase = -intercept / slope
        if min(x) <= rheobase <= max(x):
            plt.axvline(rheobase, color='g', linestyle='--', label=f'Rheobase: {rheobase:.2f} pA')

    # Labeling the plot
    plt.xlabel('Current (pA)')
    plt.ylabel('Firing Rate (Hz)')
    plt.title(f'{folder_file} F-I Curve')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

def replace_nan_with_mean(array): #OLD 26_5_25
        '''
        Replaces nan values in an array with the mean of the column (for 2D arrays)
        or with the mean of the array (for 1D arrays).
        '''
        if array.ndim == 1:
            # For a 1D array, replace NaNs with the mean of the array
            array_mean = np.nanmean(array)
            array = np.where(np.isnan(array), array_mean, array)
        else:
            # For a 2D array, replace NaNs in each column with the mean of that column
            nan_indices = np.isnan(array)
            column_means = np.nanmean(array, axis=0)
            array[nan_indices] = np.take(column_means, np.where(nan_indices)[1])

        return array


def spike_remover(array): #old 26_5_25 THIS ASSUMES THAT THE MEAN OF THE entire sweep is a good thing to replace the data with :/
    '''
    removes points >2SD from the mean of any column in array
    input: V_array (1d / 2d)
    returns: V_array_cleaned (identical shape as input with spikes removed) 
    '''
    array_cleaned  = array.copy()
    #
    if array.ndim == 1:
        #  1D array 
        array_cleaned[array > np.mean(array) + 2 * np.std(array)] = np.nan

        # array_diff_abs = np.abs(np.diff(array)) # OLD CODE USING DIFF
        # threshold = np.mean(array_diff_abs) + 2 * np.std(array_diff_abs)
        # spikes = array_diff_abs > threshold
        # array_cleaned = array.copy()
        # array_cleaned[:-1][spikes] = np.nan #slicing here to align lengths
        # array_cleaned = replace_nan_with_mean(array_cleaned)

    else: # 2D array case
        
        if array.shape[0] <= 1:
            print('Voltage array has 1 or fewer points. MISSING DATA!')
            return array_cleaned

        # array_diff_abs = np.abs(np.diff(array, axis = 0)) #diff between consecutive points # OLD CODE USING DIFF
        # array_diff_abs = np.vstack([array_diff_abs, np.mean(array_diff_abs, axis  = 0 ).reshape(1,-1) ])
        # array_cleaned[array_diff_abs > np.mean(array_diff_abs , axis = 0 ) + 2*np.std(array_diff_abs , axis =  0 )] = np.nan

        array_cleaned[array > np.mean(array , axis = 0 ) + 2*np.std(array, axis =  0 )] = np.nan
        
    array_cleaned = replace_nan_with_mean(array_cleaned)
    return array_cleaned  








def APP_splitter(V_array_or_list, drug_in, drug_out):
    '''
    inputs: V_array_or_list  :  voltage np.array, shape: length x num_sweeps | list of mean values for each sweep
            drug_in  :  integer , sweep number when drug was applied (included in APP)
            drug_out :  integer , sweep number when drug was washed out (included in WASH)

    Returns: list_PRE, list_APP, list_WASH : each a list of input file values for that condition. 
    '''
    if drug_in > 0 and drug_out > 0: #check drug in and out exist 
        if isinstance(V_array_or_list, list):
            V_PRE  = V_array_or_list[:drug_in-1]
            V_APP  = V_array_or_list[drug_in-1:drug_out-1]
            V_WASH = V_array_or_list[drug_out-1:]

        elif isinstance(V_array_or_list, np.ndarray):
            V_PRE  = V_array_or_list[:, :drug_in-1]
            V_APP  = V_array_or_list[:, drug_in-1:drug_out-1]
            V_WASH = V_array_or_list[:, drug_out-1:]

        return V_PRE, V_APP, V_WASH
    else:
        return [], [], V_array_or_list  # all wash values if no drug in or out 


def mean_RMP_APP_calculator(V_array, drug_in, drug_out, I_array=None, command_array=None, folder_file=None, print_warnings=False):
    '''
    inputs: V_array (2D array of V_df),
            drug_in  :  integer , sweep number when drug was applied (included in APP)
            drug_out :  integer , sweep number when drug was washed out (included in WASH)
            command_array : protocol/command trace used to exclude command steps from RMP.
            I_array : measured current, used only as a fallback protocol source.

    return: input_R_PRE, input_R_APP, input_R_WASH
            lists of mean RMP for each sweep in PRE APP or WASH
    
    '''
    mean_RMP_sweep_list = sweep_mean_RMP_calculator(
        V_array,
        command_array=command_array,
        I_array=I_array,
        folder_file=folder_file,
        print_warnings=print_warnings,
    )

    mean_RMP_PRE, mean_RMP_APP, mean_RMP_WASH = APP_splitter(mean_RMP_sweep_list, drug_in, drug_out)
    return mean_RMP_PRE, mean_RMP_APP, mean_RMP_WASH

def sweep_mean_RMP_calculator(V_array, command_array=None, I_array=None, folder_file=None, print_warnings=False):
    '''
    inputs: V_array (2D array of voltage)
            command_array (2D array of clamp command, preferred)
            I_array (2D array of measured current, fallback only)

    return: list of mean RMP for each sweep 
    
    '''
    V_array_cleaned = spike_remover_nan(_as_2d_array(V_array))
    V_array_adj, protocol_array, source = select_protocol_array(
        V_array_cleaned,
        command_array=command_array,
        I_array=I_array,
        clean_I_fallback=True,
    )

    if protocol_array is None:
        if print_warnings and folder_file is not None:
            print(
                f"[WARNING] APP RMP whole-sweep fallback | folder_file: {folder_file} | "
                "no command_array or measured-I protocol steps detected"
            )
        return list(np.nanmean(V_array_cleaned, axis=0))

    mean_RMP_sweep_list = []
    for sweep in range(V_array_adj.shape[1]):
        V_sweep = V_array_adj[:, sweep]
        protocol_sweep = protocol_array[:, sweep]
        _, _, rest_indices = _step_indices_from_command_trace(protocol_sweep)
        if rest_indices is None or len(rest_indices) == 0:
            mean_RMP_sweep_list.append(np.nanmean(V_sweep))
        else:
            mean_RMP_sweep_list.append(np.nanmean(V_sweep[rest_indices]))
    return mean_RMP_sweep_list


def sweep_mean_inputR_calculator(V_array, command_array=None, I_array=None, folder_file=None, print_warnings=False):
    '''
    input:      V_array 2D array of voltage
                command_array 2D command trace, preferred for step timing/size
                I_array 2D measured current, fallback only

    returns :   list: mean input R for each sweep = current injected / change in V 
    '''
    original_V = _as_2d_array(V_array)
    V_array_adj, protocol_array, source = select_protocol_array(
        original_V,
        command_array=command_array,
        I_array=I_array,
        clean_I_fallback=True,
    )
    if protocol_array is None:
        if print_warnings and folder_file is not None:
            print(
                f"[WARNING] APP inputR skipped | folder_file: {folder_file} | "
                "no command_array or measured-I protocol steps detected"
            )
        return [np.nan] * original_V.shape[1]

    input_R_ohms_V_array = []
    
    for index, V_sweep in enumerate(V_array_adj.T):  # Transpose V_array to iterate over columns/sweeps
        
        if index < 1:  # skip first sweep (or first 2 if you change <1 to <2)
            input_R_ohms_V_array.append(np.nan)
            continue

        V_cleaned = spike_remover_nan(V_sweep)

        V_cleaned = V_cleaned.flatten()

        try:
            # fetch delta_V
            protocol_sweep = protocol_array[:, index]
            step_indices, step_value, rest_indices = _step_indices_from_command_trace(protocol_sweep)
            if step_indices is None or rest_indices is None:
                input_R_ohms_V_array.append(np.nan)
                continue

            steady_state, hyper, first_current_point, last_current_point = steady_state_value(V_sweep, protocol_sweep, step_value)
            if first_current_point is None or last_current_point is None:
                input_R_ohms_V_array.append(np.nan)
                continue

            rmp = np.nanmean(V_cleaned[rest_indices])
            delta_V_mV = abs(steady_state - rmp)

            # fetch I injected
            delta_I_pA = abs(step_value)
            if not np.isfinite(delta_I_pA) or delta_I_pA < 1:
                input_R_ohms_V_array.append(np.nan)
                continue
            delta_I_A = delta_I_pA * 1e-12  # convert pA to A
            delta_V_V = delta_V_mV * 1e-3  # convert mV to V

            sweep_input_R_ohms = delta_V_V / delta_I_A
            input_R_ohms_V_array.append(sweep_input_R_ohms)

        except Exception as e:
            print(f"Sweep {index} skipped due to error: {e}")
            input_R_ohms_V_array.append(np.nan)

    # Convert all to MΩ
    input_R_MOhms_array = [r / 1e6 if r is not np.nan else np.nan for r in input_R_ohms_V_array]

    return input_R_MOhms_array

def normalise_array_length(V_array, I_array, columns_match=False, verbose=True): #2/5/25 changed
    """
    Adjusts V_array and I_array to have matching row lengths (samples) and optionally columns (sweeps).
    Trims or stretches I_array if needed to match V_array.
    """
    v_len, v_cols = (V_array.shape[0], 1) if len(V_array.shape) == 1 else V_array.shape
    i_len, i_cols = (I_array.shape[0], 1) if len(I_array.shape) == 1 else I_array.shape

    if abs(v_len - i_len) == 1: #diff of 1
        min_len = min(v_len, i_len)
        V_array, I_array = V_array[:min_len], I_array[:min_len]

    elif i_len < v_len:
        if verbose:
            print(f"⚠️ Stretching I_array: V {v_len}, I {i_len}") 
        if I_array.ndim == 1: 
            I_array = I_array[:, np.newaxis]  # convert to 2D with one column
        x_old = np.linspace(0, 1, i_len)
        x_new = np.linspace(0, 1, v_len)
        I_stretched = np.zeros((v_len, i_cols))
        for c in range(i_cols):
            I_stretched[:, c] = np.interp(x_new, x_old, I_array[:, c])  # Interpolate for each column
        I_array = I_stretched

    elif v_len < i_len:
        if verbose:
            print(f"⚠️ Trimming I_array: V {v_len}, I {i_len}")
        I_array = I_array[:v_len]

    if columns_match and v_cols != i_cols:
        min_cols = min(v_cols, i_cols)
        V_array, I_array = V_array[:, :min_cols], I_array[:, :min_cols]

    # print(f"returning V_array shape {V_array.shape} and I_array shape {I_array.shape}")
    return V_array, I_array

# def normalise_array_length(V_array, I_array, columns_match=False):
#     '''
#     Adjusts the lengths of V_array and I_array to have the same number of rows and, optionally, the same number of columns.

#     Input:
#         V_array (numpy.ndarray): 2D array containing voltage recordings (sweeps).
#         I_array (numpy.ndarray): 2D array containing corresponding current recordings (sweeps).
#         columns_match (bool): If True, adjusts the arrays to have the same number of columns as well.

#     Returns:
#         V_adj (numpy.ndarray): Adjusted voltage array.
#         I_adj (numpy.ndarray): Adjusted current array.
#     '''
#     # #ensure V_sweep and I_sweep are the same length
#     # if len(V_array) != len(I_array):
#     #     # print(f"Length of V_sweep: {len(V_sweep)}, Length of I_sweep: {len(I_sweep)}") #V is usaly 400001 and I 400000
#     #     V_adj = V_array[:min(len(V_array), len(I_array))]
#     #     I_adj = I_array[:min(len(V_array), len(I_array))]
#     min_rows = min(V_array.shape[0], I_array.shape[0])
#     V_adj = V_array[:min_rows]
#     I_adj = I_array[:min_rows]

#     if columns_match:
#         min_cols = min(V_array.shape[1], I_array.shape[1])
#         V_adj = V_adj[:, :min_cols]
#         I_adj = I_adj[:, :min_cols]
    
#     return V_adj, I_adj

def getI_array_sweep(I_array):
    if I_array.shape[1] > 1: 
    # Check if all columns in I_array are identical
        if not np.all(I_array == I_array[:, [0]]):
            raise ValueError("Columns in I_array are not identical.")
        I_sweep = I_array[:, 0]  # Use the first column if they are identical
    else:
        I_sweep = I_array[:, 0] # only 1 column
    return I_sweep

def I_array_to_match_V (V_array, I_array):
    I_sweep = getI_array_sweep(I_array)
    #set rows the same 
    V_array_adj, I_sweep_adj = normalise_array_length(V_array, I_sweep)
    #duplicate I_sweep
    # I_sweep_adj = I_sweep_adj[:, np.newaxis] #old 5 may 2025
    I_sweep_adj = I_sweep_adj.reshape(-1, 1)
    I_array_adj = np.tile(I_sweep_adj, (1, V_array_adj.shape[1])) 
    return I_array_adj, V_array_adj
