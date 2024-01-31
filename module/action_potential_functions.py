
#change name to something more appropriate later 

import warnings
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter1d
import scipy.signal as sg
from scipy import stats
from collections import namedtuple
from module.utils import *
from scipy.optimize import curve_fit
import sys

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from sklearn.mixture import GaussianMixture

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler 

# from module.plotters import plot_ap_window

from scipy.signal import find_peaks
import numpy as np

########## TESTERS
def generate_V_pAD_df(folder_file): 
    '''
    Input : 
           folder_file : str 
    Ouput : 
           pAD_df  : pAD dataframe built from pAD_detection
           V_array : v array     
    '''
    path_V, path_I = make_path(folder_file)
    V_list, V_df = igor_exporter(path_V)
    V_array      = np.array(V_df) 
    peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df  =   pAD_detection(V_array)
    return pAD_df , V_array

from scipy.ndimage import gaussian_filter1d
def plot_ap_window(v_array, peak_location, upshoot_location, threshold_voltage, latency, average_slope, max_dvdt, max_dvdt_location, input_sampling_rate, sec_to_ms):
    """
    Plot the action potential (AP) window centered around the upshoot location, including the slope and the point of maximum derivative (dV/dt).
    
    Inputs:
        v_array (numpy.ndarray): The array containing voltage data for a single sweep.
        upshoot_location (int): The index in v_array corresponding to the AP upshoot.
        threshold_voltage (float): The voltage value at the upshoot.
        latency (float): Half of the latency period for the AP in milliseconds.
        average_slope (float): The average slope of the AP within the window.
        max_dvdt (float): The maximum derivative (dV/dt) within the window.
        max_dvdt_location (int): The index in v_array where max dV/dt occurs.
        input_sampling_rate (float): The sampling rate at which the data was recorded (in Hz).
        sec_to_ms (float): Conversion factor from seconds to milliseconds.
    """
    # Define the window around the upshoot location
    window_size_samples = int(latency * (input_sampling_rate / 1000))
    window_start = max(0, upshoot_location - window_size_samples)
    window_end = min(len(v_array), peak_location + window_size_samples)
    
    # Calculate 3/4 latency in samples
    three_quarter_latency_samples = int((latency * 3/4) * (input_sampling_rate / 1000))
    three_quarter_latency_time = upshoot_location / input_sampling_rate * sec_to_ms + (latency * 3/4)


    # Create time points for the entire sweep
    time_points = np.arange(len(v_array)) / input_sampling_rate * sec_to_ms
    window_time_points = time_points[window_start:window_end]

    # Extracting the window of interest for plotting
    v_window = v_array[window_start:window_end]
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting the raw voltage trace and upshoot
    ax.plot(window_time_points, v_window, label='Raw Voltage Trace', color='blue')
    ax.axvline(time_points[upshoot_location], color='red', linestyle='--', label='Upshoot', linewidth=2)

    # Marking the 3/4 latency point on the plot
    ax.axvline(three_quarter_latency_time, color='orange', linestyle='--', label='3/4 Latency', linewidth=2)

    # Annotating the 3/4 latency
    ax.annotate(f'3/4 Latency', xy=(three_quarter_latency_time, threshold_voltage), xytext=(10, 10),
                textcoords='offset points', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.5'))

    # Plotting the slope as a line
    slope_line_x = [time_points[upshoot_location], time_points[upshoot_location] + latency]
    slope_line_y = [threshold_voltage, threshold_voltage + average_slope * latency]
    ax.plot(slope_line_x, slope_line_y, label='Slope', color='green', linewidth=2)

    # Indicating the max dV/dt point
    ax.axvline(time_points[max_dvdt_location], color='purple', linestyle='--', label=f'Max dV/dt: {max_dvdt} mV/ms', linewidth=2)
    
    # Annotating max dV/dt value
    ax.annotate(f'{max_dvdt:.2f} mV/ms', xy=(time_points[max_dvdt_location], max_dvdt), xytext=(10, 3),
                textcoords='offset points', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.5'))

    # Setting up the plot
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title('AP Characteristics in the Focused Window')
    ax.legend()
    ax.grid(True)

    # Display the plot
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
    Calculates the steady state value of a voltage trace during a current injection step.

    Parameters:
    - V_sweep (array-like): Voltage trace for a single sweep.
    - I_sweep (array-like): Current trace corresponding to the voltage sweep.
    - step_current_val (float, optional): The value of the step current injection in pA. If None, it's derived from I_sweep.
    - avg_window (float, optional): Fraction of the step current duration used for averaging. Default is 0.5 (50%).

    Returns:
    - asym_current (float): The steady state value of the voltage trace.
    - hyper (bool): True if the step current is hyperpolarizing; False otherwise.
    - first_current_point (int): Index of the start of the step current injection.
    - last_current_point (int): Index of the end of the step current injection.

    The function calculates the steady state value ('asym_current') by averaging the voltage trace over a window at the end of the current injection step.
    """

    # Check for empty inputs
    if len(V_sweep) == 0 or len(I_sweep) == 0:
        return np.nan, False, None, None

    # Determine the step current value if not provided
    if step_current_val is None:
        non_zero_I = I_sweep[I_sweep != 0]
        if len(non_zero_I) > 0:
            step_current_val = np.unique(non_zero_I)[0]
        else:
            return np.nan, False, None, None

    # Determine if the current step is hyperpolarizing
    hyper = step_current_val < 0
    # Find the indices where the current equals its maximum (or minimum for hyper)
    current_points = np.where(I_sweep == (np.min(I_sweep) if hyper else np.max(I_sweep)))[0]
    
    # Check if there are no current points found
    if len(current_points) == 0:
        return np.nan, hyper, None, None

    # Calculate the first and last points of the current injection
    first_current_point = current_points[0]
    last_current_point = current_points[-1]

    # Calculate the duration for averaging, ensuring it does not exceed array bounds
    current_avg_duration = int(avg_window * (last_current_point - first_current_point))
    if last_current_point - current_avg_duration < 0:
        return np.nan, hyper, first_current_point, last_current_point

    # Calculate the steady state value by averaging over the determined window
    asym_current = np.mean(V_sweep[last_current_point - current_avg_duration:last_current_point])
    return asym_current, hyper, first_current_point, last_current_point
# #OLD 
# def steady_state_value(V_sweep, I_sweep, step_current_val=None,  avg_window = 0.5):
#     ''' 
#     """
#     Calculates the steady state value of a voltage trace during a current injection step.

#     Input:
#         V_sweep (array-like): A 1D time series representing a single voltage trace (single sweep).
#         I_sweep (array-like): The corresponding 1D time series for the current trace (single sweep).
#         step_current_val (float, optional): The value of the step current injection in pA. If not provided, it is derived from the unique non-zero value in I_sweep.
#         avg_window (float, optional): The fraction of the step current duration used for averaging to determine the steady state value. Default is 0.5.

#     Output:
#         asym_current (float): The steady state value of the voltage trace during the step current injection.
#         hyper (bool): Indicates whether the step current is hyperpolarizing (True) or not (False).
#         first_current_point (int): The index of the first timeframe of the step current injection.
#         last_current_point (int): The index of the last timeframe of the step current injection.

#     Note:
#         The function determines whether the step current is hyperpolarizing based on the sign of 'step_current_val'.
#         It calculates the steady state value ('asym_current') by averaging the voltage trace over a window at the end of the current injection step.
#   '''
#     if step_current_val == None:
#         if np.count_nonzero(I_sweep) > 0:
#             step_current_val = np.unique(I_sweep[I_sweep != 0])[0]
#         else:
#             print("Multiple I values in step, unable to calculate steady state.")
#             return

#     if step_current_val >= 0: #note that 0pA of I injected is not hyperpolarising 
#         hyper = False
#         first_current_point = np.where(I_sweep == np.max(I_sweep) )[0][0] 
#         last_current_point = np.where(I_sweep == np.max(I_sweep) )[0][-1]

#     elif step_current_val < 0 : 
#         hyper = True 
#         first_current_point = np.where(I_sweep == np.min(I_sweep) )[0][0] 
#         last_current_point = np.where(I_sweep == np.min(I_sweep) )[0][-1]

#     current_avg_duration = int(avg_window*(last_current_point - first_current_point))
#     asym_current  = np.mean(V_sweep[ last_current_point - current_avg_duration : last_current_point  ])

#     return asym_current , hyper  , first_current_point, last_current_point


def calculate_max_firing(voltage_array, input_sampling_rate=1e4): #HARD CODE sampeling rate
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


########## TAU and SAG
from scipy.optimize import curve_fit

def exponential_decay(x, tau, baseline):
    """Exponential decay function for curve fitting."""
    return baseline * (1 - np.exp(-x / tau))

def plot_tau(time, normalized_voltage, popt, fit_start, fit_end):
    """
    Plots the normalized voltage trace and the fitted exponential decay curve, including the fitting window.
    Parameters:
    - time (array-like): Time array corresponding to the voltage trace.
    - normalized_voltage (array-like): Normalized voltage trace.
    - popt (array-like): Optimized parameters from curve fitting.
    - fit_start (float): Start time of the fitting window in seconds.
    - fit_end (float): End time of the fitting window in seconds.
    """
    tau_ms, baseline = popt
    plt.figure(figsize=(12, 6))
    plt.plot(time * 1000, normalized_voltage, label='Data', color='b')  # time in milliseconds
    plt.plot(time * 1000, exponential_decay(time, *popt), 'r-', label=f'Fit, tau={tau_ms:.2f} ms', linewidth=2)
    
    # Add vertical lines for the fitting window
    plt.axvline(fit_start * 1000, color='g', linestyle='--', label='Fit start')
    plt.axvline(fit_end * 1000, color='y', linestyle='--', label='Fit end')
    
    # Optionally, add a shaded area for the fitting window
    plt.axvspan(fit_start * 1000, fit_end * 1000, color='grey', alpha=0.2, label='Fitting Window')

    plt.xlabel('Time (ms)')
    plt.ylabel('Normalized Voltage (mV)')
    plt.title(f'Tau fitting with tau = {tau_ms:.2f} ms')
    plt.legend()
    plt.show()

def tau_analyser(V_array, I_array, step_current_values, ap_counts, sampling_rate=1e4, avg_window=0.5):
    """
    Estimates the membrane time constant (tau) and additional parameters for a neuron from voltage traces during step current injections.
    """
    voltage_array, current_array = normalise_array_length(V_array, I_array, columns_match=True)

    # Ensure the number of sweeps in V_array and I_array matches the length of ap_counts
    num_sweeps = voltage_array.shape[1]
    if num_sweeps != len(ap_counts):
        print("Error: Number of sweeps in V_array/I_array does not match length of ap_counts.")
        return [np.nan, np.nan, np.nan, np.nan]

    #  index of the last sweep with zero action potentials within the first 50% of the ap_counts list
    # Find the threshold index for the first 50% of the ap_counts list
    threshold_index = len(ap_counts) // 2

    # Remove trailing zeros from the end of the ap_counts list
    while ap_counts and ap_counts[-1] == 0:
        ap_counts.pop()

    # Find the index of the last sweep with zero APs within the first 50% of the list
    last_zero_AP_sweep_index = None
    for i in range(threshold_index):
        if ap_counts[i] == 0:
            last_zero_AP_sweep_index = i

    # If no such index is found, or it's the very first sweep, we cannot calculate tau
    if last_zero_AP_sweep_index is None or last_zero_AP_sweep_index == 0:
        print('No suitable sweep found for tau calculation within the first 50% of the list.')
        return [np.nan, np.nan, np.nan, np.nan]
    
    # # Reverse the ap_counts list, remove trailing zeros, and reverse it back
    # reversed_ap_counts = ap_counts[::-1]
    # while reversed_ap_counts and reversed_ap_counts[0] == 0:
    #     reversed_ap_counts.pop(0)
    # ap_counts_without_trailing_zeros = reversed_ap_counts[::-1]

    # # Find the sweep with the last non-zero step current value before APs start appearing
    # first_AP_sweep_index = len(ap_counts_without_trailing_zeros) - next((i for i, ap_count in enumerate(reversed(ap_counts_without_trailing_zeros)) if ap_count == 0), len(ap_counts_without_trailing_zeros))
    
   

    if last_zero_AP_sweep_index >= voltage_array.shape[1] or last_zero_AP_sweep_index<=0:
        print(f'Unable to calculate tau.')
        print(f"first_AP_sweep_index ({last_zero_AP_sweep_index}) is 0 or larger than the number of columns in voltage_array ({voltage_array.shape[1]}).")
        print(f"ap_count: {ap_counts[last_zero_AP_sweep_index]}")
        return [np.nan, np.nan, np.nan, np.nan]
    else:
        # Extract the relevant voltage trace and corresponding current trace
        voltage_trace = voltage_array[:, last_zero_AP_sweep_index]
        current_trace = current_array[:, last_zero_AP_sweep_index]
        step_current = step_current_values[last_zero_AP_sweep_index]

    # Calculate RMP as the mean voltage before current injection
    zero_current_indices = current_trace == 0
    if not zero_current_indices.any():
        print(' No zero current segment found in the trace, unable to measure RMP.')
        RMP=np.nan
    RMP = np.mean(voltage_trace[zero_current_indices])

    # Find where the current changes
    changes = np.where(np.diff(current_trace) != 0)[0]

    # Assuming there is only one step current applied, the start and end indices
    # would be right before and after the changes
    if len(changes) == 2:
        start_index = changes[0] + 1  # The index after the first change
        end_index = changes[1]  # The index before the second change
    else:
        print('Unexpected number of steps in the current trace.')
        return [np.nan, np.nan, np.nan, RMP]

    # Use only the portion of the voltage trace where the current is applied
    voltage_during_step = voltage_trace[start_index:end_index+1]

    # Normalize the voltage by subtracting the initial value (align to 0)
    normalized_voltage = voltage_during_step - voltage_trace[start_index] 

    # Time array for fitting, in seconds
    time = np.arange(len(normalized_voltage)) / sampling_rate

    # Define a fitting window to capture the transient response
    fit_start = 0.001  # Start fitting 2 ms after the current step, for example
    fit_end = 0.080  # End fitting 50 ms into the step, before the voltage stabilizes

    # Create a boolean mask to select the fitting window
    fit_mask = (time >= fit_start) & (time <= fit_end)

    # Apply the mask to time and voltage arrays
    time_fit = time[fit_mask]
    normalized_voltage_fit = normalized_voltage[fit_mask]

    # Calculate the steady state voltage during the current step
    steady_state_voltage, hyper, first_current_point, last_current_point = steady_state_value(voltage_trace, current_trace, step_current, avg_window)
    
    # Set initial guess values for the curve fitting
    initial_tau_guess = 0.02  # Initial guess for tau
    initial_baseline_guess = RMP  # Or use the steady-state value from the end of the step

    # Fit the exponential decay model to the normalized voltage trace
    try:
        popt, _ = curve_fit(exponential_decay, time_fit, normalized_voltage_fit, p0=[initial_tau_guess, initial_baseline_guess])
        tau_ms = popt[0] * 1000  # Convert tau from seconds to milliseconds
        fitted_baseline = popt[1]  # This is the fitted baseline value
    except RuntimeError as e:
        # Curve fitting failed, return NaNs
        print("Curve fitting error:", e)
        return [np.nan, np.nan, np.nan, RMP]

    
    
    if not 2<= tau_ms <=180: #HARD CODE # 10 - 110 idealy but not optimised #TODO
        print(f"Tau calculated at {tau_ms} ms is outside the the physiolgoical range 10-110ms, plotting fit.")
        plot_tau(time, normalized_voltage, popt, fit_start=fit_start, fit_end=fit_end)


    return [tau_ms, steady_state_voltage, step_current, RMP]

# def tau_analyser(voltage_array, current_array, input_step_current_values, plotting_viz = False, verbose = False , analysis_mode  = 'max' ,input_sampling_rate = 1e4): 

#     '''
#     Functin to calculate tau associated with RC charging of cell under positive step current 

#     Input : voltage_array : 2d array containing sweeps of voltage recordings / traces for different step currents
#             current_array : 2d array containing sweeps of current recordings / traces for different step currents
#             input_step_current_values : list containing step current values (in pA) 
#             plotting_viz  :  boolean : whether or not we want to plot/viz 
#             analysis_mode : str , either 'max' or 'asym'. If 'max' uses max of voltage to find tau, otw used v_steady
#             input_sampling_rate : float : data acquisition rate as given from Igor. 
#             verbose : Boolean : if False, suppresses print outputs 

#     #TODO currenly returning only the 1st element as nan in the second and seems inferior quality 
#     Returns : [ [tau (ms), steady state with I , I step (pA) , steady stats no I (RMP)] , [...] ]
#     ''' 
#     # noisy error msg and chunk stuff / exclusion criterion, calculation based on abs max / steady state . 

#     sampling_rate = input_sampling_rate
#     sec_to_ms = 1e3  
#     # First we get the tau indices from the voltage trace: 
#     num_aps_all = num_ap_finder(voltage_array)

#     aps_found = np.array([1 if elem  > 0 else 0 for elem in num_aps_all])
#     aps_found_idx = np.where(aps_found > 0)[0] 
#     true_ap_start_idx = voltage_array.shape[-1]

#     for idx in aps_found_idx : 
#         if idx + 1 in aps_found_idx:
#             if idx + 2 in aps_found_idx:   # once the 1st instance of spiking is found if 2 subsequent ones are found thats good
#                             true_ap_start_idx = min(idx, true_ap_start_idx)  
    
#     visualisation = plotting_viz
#     tau_array = [] 

#     num_tau_analysed_counter = 0   
#     counter = 0                                                      #   as we might potentially skip some step currents as they are too noisy, need to keep track of which/ how many taus are actually analysed (we need two)
#     if verbose: 
#         print('Cell starts spiking at trace index start %s' % true_ap_start_idx)
#     # check noise level: 
#     while num_tau_analysed_counter <  2 : 

#         tau_idx  =  true_ap_start_idx - 1  -  counter 
#         if verbose: 
#             print('Analysing for sweep index %s' % tau_idx)
#         step_current_val  = input_step_current_values[tau_idx]
#         asym_current, hyper , current_inj_first_point, current_inj_last_point  = steady_state_value(voltage_array[:, tau_idx], current_array[:, tau_idx], step_current_val)
#         max_current    = np.max(voltage_array[current_inj_first_point:current_inj_last_point,tau_idx ]) 
        
#         thresh_current_asym = (1 - np.exp(-1))*( asym_current - voltage_array[current_inj_first_point - 1 , tau_idx])  + voltage_array[current_inj_first_point - 1 , tau_idx] 
#         thresh_current_max =  (1 - np.exp(-1))*( max_current - voltage_array[current_inj_first_point - 1 , tau_idx])  + voltage_array[current_inj_first_point - 1 , tau_idx] 
        
#         # check first how noisy the signal is : 
#         current_noise = np.std(voltage_array[ current_inj_first_point:current_inj_last_point,tau_idx] )
        
#         # Calculate v resting potential 
#         v_resting_membrane = np.mean(voltage_array[ 0 : current_inj_first_point, tau_idx]) 
        
#         if abs(current_noise + thresh_current_asym - asym_current) <= 0.5 :          # we use asymtotic current for noise characterisation rather than max current 
#             if verbose: 
#                 print('Too noisy, skipping current current step...')
#         else: 
#             num_tau_analysed_counter += 1 
#             if visualisation: 
#                 # Plot it out for sanity check 
#                 plt.figure(figsize = (12,12))
#                 time_series = sec_to_ms*np.arange(0, len(voltage_array[:, tau_idx]))/sampling_rate
#                 plt.plot(time_series, voltage_array[:, tau_idx])
#                 if analysis_mode == 'max': 
#                     plt.axhline( (1 - 0*np.exp(-1))*( max_current - voltage_array[current_inj_first_point - 1 , tau_idx])  + voltage_array[current_inj_first_point - 1 , tau_idx] , c = 'r', label = 'Max Current')
#                     plt.axhline( (1 - np.exp(-1))*( max_current - voltage_array[current_inj_first_point - 1 , tau_idx])  + voltage_array[current_inj_first_point - 1 , tau_idx] , c = 'b',  label = 'Threshold' )
#                 elif analysis_mode == 'asym':
#                     plt.axhline( (1 - 0*np.exp(-1))*( asym_current - voltage_array[current_inj_first_point - 1 , tau_idx])  + voltage_array[current_inj_first_point - 1 , tau_idx] , c = 'r', label = 'Asymtotic Current')
#                     plt.axhline( (1 - np.exp(-1))*( asym_current - voltage_array[current_inj_first_point - 1 , tau_idx])  + voltage_array[current_inj_first_point - 1 , tau_idx] , c = 'b',  label = 'Threshold' )
#                 else: 
#                     raise ValueError('Invalid Analysis Mode, can be either max or asym')
#                 plt.legend()
#                 plt.ylabel('Membrane Potential (mV)')
#                 plt.xlabel('Time (ms)')
#                 plt.show()
                
#             if hyper: 
#                 if verbose: 
#                     print('Positive current step not used! Hence breaking')
#                 break
#             else:  
#                 # find all time points where voltage is at least 1 - 1/e or ~ 63% of max / asym val depending on analysis_mode
#                 if analysis_mode == 'max': 
#                     time_frames =  np.where(voltage_array[:, tau_idx] > thresh_current_max )[0]
#                 elif analysis_mode == 'asym':
#                     time_frames =  np.where(voltage_array[:, tau_idx] > thresh_current_asym )[0]
#                 else : 
#                     raise ValueError('Invalid Analysis Mode, can be either max or asym')

#                 time_frames_ = time_frames[time_frames > current_inj_first_point]
#                 tau = sec_to_ms*(time_frames_[0] - current_inj_first_point) / sampling_rate

#                 # tau (ms), steady state with I , I step (pA) , steady stats no I
#                 tau_temp = [tau, asym_current, step_current_val, v_resting_membrane]

#                 tau_array.append(tau_temp)
        
#         counter += 1

#     if num_tau_analysed_counter == 0 : 
#         return [np.nan, np.nan, np.nan, np.nan]
#     return tau_array[0]


def plot_sag(voltage_trace, time_trace, RMP, steady_state_voltage, min_sag_voltage, sag_ratio):
    """
    Plot the sag with the minimum sag voltage and sag ratio annotated.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time_trace, voltage_trace, label='Voltage Trace', color='blue')
    plt.axhline(y=min_sag_voltage, color='red', linestyle='--', label='Min Sag Voltage')
    plt.axhline(y=steady_state_voltage, color='green', linestyle='--', label='Steady State Voltage')
    plt.axhline(y=RMP, color='orange', linestyle='--', label='RMP')
    plt.title(f"Sag Analysis (Sag Ratio: {sag_ratio:.2f})")
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (mV)')
    plt.legend()
    plt.show()

def sag_current_analyser(voltage_array, current_array, step_current_values, ap_counts, avg_window=0.5, visualise=False):
    """
    Function to calculate and plot the sag current from voltage and current traces under the first hyperpolarizing current step without action potentials.
    
    Parameters:
    - voltage_array (2D array): 2D array containing voltage recordings for different current steps.
    - current_array (2D array): 2D array containing current recordings for different current steps.
    - step_current_values (list): List of injected current values for each sweep.
    - ap_counts (list): List of action potential counts for each sweep.
    - avg_window (float): Fraction of the step current duration used for averaging.
    
    Returns:
    - List containing [sag ratio, steady state with I, I step (pA), RMP].
    """
    for sweep_index, ap_count in enumerate(ap_counts):
        if ap_count == 0 and step_current_values[sweep_index] < 0:  # Check for no APs and negative current
            V_sweep = voltage_array[:, sweep_index]
            I_sweep = current_array[:, sweep_index]
            step_current = step_current_values[sweep_index]

            # Calculate steady state value using the steady_state_value function
            asym_current, hyper, first_current_point, last_current_point = steady_state_value(V_sweep, I_sweep, step_current, avg_window)
            
            # Calculate RMP before current injection
            RMP = np.mean(V_sweep[:first_current_point])

            # Calculate sag ratio
            min_sag_voltage = min(V_sweep[first_current_point:last_current_point])
            sag_ratio = (asym_current - min_sag_voltage) / (RMP - min_sag_voltage)

            # Plot the sag for visualization
            if visualise==True:
                plot_sag(V_sweep[first_current_point:last_current_point], np.arange(first_current_point, last_current_point) / 1000, RMP, asym_current)

            return [sag_ratio, asym_current, step_current, RMP]
        
    print("No sweep found with negative current injecttion and without action potentials, unable to calculate sag.")
    return [np.nan, np.nan, np.nan, np.nan]
# #OLD
# def sag_current_analyser(voltage_array, current_array, input_step_current_values,input_current_avging_window = 0.5): 

#     '''
#     Function to calculate sag current from voltage and current traces under hyper polarising current steps

#     Input:
#     volatge array : 2d array containing sweeps of voltage recordings / traces for different step currents
#     current array : 2d array containing sweeps of current recordings / traces for different step currents
#     input_step_current_values : list containing different step currents 
#     input_current_avging_window : float / fraction representating what portion of step current duration is used for calculating 
    
#     Returns : [[ sag(0-1) , steady state with I , I step (pA) , steady stats no I (RMP) ] , [...] ] 
#     '''
#     # sag : 0, 1 : works with the hyperpolarising current injection steps and 0 and 1 indices correspond to
#     # the 1st two (least absolute value) step current injections. Later expand for positive current inj too? 
#     sag_indices = [0,1,2]
#     sag_current_all =  [] 
#     current_avging_window = input_current_avging_window 

#     for sag_idx in sag_indices: 

#         v_smooth, peak_locs , peak_info , num_peaks  = ap_finder(voltage_array[:,sag_idx])
#         if num_peaks == 0: 

#             asym_current, _ , _ , _  = steady_state_value(voltage_array[:, sag_idx], current_array[:, sag_idx], input_step_current_values[sag_idx])
#             i_min = min(current_array[:,sag_idx])
#             first_min_current_point = np.where(current_array[:,sag_idx] == i_min )[0][0] 
#             last_min_current_point = np.where(current_array[:,sag_idx] == i_min )[0][-1]

#             step_current_dur = last_min_current_point - first_min_current_point
#             step_current_avg_window = int(current_avging_window*step_current_dur)
            
#             # Calculate v resting potential 
#             v_resting_membrane = np.mean(voltage_array[ 0 : first_min_current_point , sag_idx]) 

#             asym_sag_current = np.mean(voltage_array[  last_min_current_point - step_current_avg_window: last_min_current_point, sag_idx])
#             min_sag_current_timepoint = np.where( voltage_array[:,sag_idx]  ==  min(voltage_array[:,sag_idx]  ) )[0][0] 
#             min_sag_current =  min(voltage_array[:,sag_idx]  )
#             max_sag_current = np.mean(voltage_array[0 : first_min_current_point  - 1   , sag_idx] )

#             sag_current = (asym_sag_current - min_sag_current)  / (  max_sag_current  - asym_sag_current )

#             sag_current_temp = [sag_current, asym_current, input_step_current_values[sag_idx], v_resting_membrane] 

#         elif num_peaks > 0: 
#             print('Spike found in Sag analysis, skipping')
#             sag_current_temp = [np.nan, np.nan, np.nan, np.nan] 
#         # Append Value to existing named tuple
#         sag_current_all.append(sag_current_temp) 

#     return sag_current_all


########## ACTION POTENTIAL

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



########## ACTION POTENTIAL RETROAXONAL / ANTIDROMIC

def ap_characteristics_extractor_subroutine_derivative(df_V_arr, sweep_index, main_plot = False, input_sampling_rate = 1e4 , input_smoothing_kernel = 1.5, input_plot_window = 500 , input_slide_window = 200, input_gradient_order  = 1, input_backwards_window = 50 , input_ap_fwhm_window = 100 , input_pre_ap_baseline_forwards_window = 50, input_significance_factor = 8 ):
    '''
    Extracts detailed characteristics of action potentials (APs) from voltage data within a specified sweep.

    Input:
        df_V_arr (2d array): A pandas DataFrame containing voltage data from electrophysiological recordings. cols == sweeps
        sweep_index (int): The index of the sweep in the DataFrame from which AP characteristics are to be extracted.

                main_plot (bool, optional): If set to True, generates a main plot. Defaults to False.
                input_sampling_rate (float, optional): The sampling rate of the data in Hz. Defaults to 10000 Hz.
                input_smoothing_kernel (float, optional): The size of the smoothing kernel to apply to the voltage data. Defaults to 1.5.
                input_plot_window (int, optional): The window size for plotting the data. Defaults to 500.
                input_slide_window (int, optional): The sliding window size for analyzing APs. Defaults to 200.
                input_gradient_order (int, optional): The order of the gradient used in derivative calculation. Defaults to 1.
                input_backwards_window (int, optional): The window size used for searching backward from a peak to find the AP upshoot. Defaults to 100.
                input_ap_fwhm_window (int, optional): The window size for calculating the full width at half maximum (FWHM) of APs. Defaults to 100.
                input_pre_ap_baseline_forwards_window (int, optional): The window size for determining the baseline before an AP upshoot. Defaults to 50.
                input_significance_factor (float, optional): A factor used to determine the significance of AP characteristics. Defaults to 8.

    Returns:
        peak_locs_corr (list): Corrected locations of AP peaks.
        upshoot_locs (list): Locations of AP thresholds, where the voltage trace begins to rise steeply.
        v_thresholds (list): Voltage values at each AP upshoot (mV).
        peak_heights (list): Differences in voltage between each AP's peak and its threshold (mV).
        peak_latencies (list): Time delays between the AP threshold and its peak (ms).
        peak_slope (list): Rate of change of the membrane potential at the AP upshoot location (mV/ms).
        peak_fw (list): Full width at half maximum (FWHM) of each AP, if <0 set as peak_latency (ms).
           
    '''
    # Other Input Hyperparameters 
    # need the first two sweeps from 1st spike and 2nd spike. if 1st ap has around 3 spikes (parameter) then only first is good enough otw go to the 2nd trace 
    
    ap_backwards_window = input_backwards_window
    pre_ap_baseline_forwards_window = input_pre_ap_baseline_forwards_window 
    slide_window     = input_slide_window      # slides 200 frames from the prev. AP  200 frames ~ 20 ms 
    sampling_rate    = input_sampling_rate
    sec_to_ms        = 1e3 
    ap_spike_min_height =  0 # if value if less than 0 mV - very unhealthy spike  
    ap_width_min      = 0.1 # in ms minimum width of AP to be considered biologically plasuible
    ap_width_max      = 2   # in ms maximum width of AP to be considered biologically plasuible

    # initialise output lists
    AP_locations_list            = []            #  location of peaks *exact^
    AP_upshoot_locations_list    = []            #  location of AP upshoot  
    AP_voltage_thresholds_list   = []            #  voltage at upshoot (mV)
    AP_heights_list              = []            #  peak heights in mV, voltage diff from peak to upshoot 
    AP_latencies_list            = []            #  latency or time delay between peak and threshold points
    AP_fwhm_list                 = []            #  width of peak at half maximum (FWHM)
    AP_slope_list                = []            #  mean slope from upshoot to 3/4 latency
    ap_max_dvdt_list             = []            # max dv/dt of AP from upshoot to 3/4 latency


    # df_V_arr = np.array(V_dataframe)
    sweep_idx = sweep_index

    # estimate AP locations for given sweep
    v_array = df_V_arr[:,sweep_idx]
    v_smooth, peak_locs , peak_info , num_peaks  = ap_finder(v_array) 
    v_deriv_transformed = np.heaviside( -np.diff(v_array)+ np.exp(1), 0 )
    
    # exit if no APs
    if len(peak_locs) == 0 :
        print("No peaks found in trace.")
        return  [] ,   [] ,  []  , [] ,  [] ,  [] , [] , []
    # print (f"Analising {len(peak_locs)} APs in trace.")

    for peak_idx in range(len(peak_locs)) : # find AP locations - correct for offsets 
        while peak_locs[peak_idx]  < ap_backwards_window: 
            ap_backwards_window = int(ap_backwards_window - 10)
        
        v_max  = np.max(v_array[peak_locs[peak_idx] - ap_backwards_window : peak_locs[peak_idx] + pre_ap_baseline_forwards_window]) 
        peak_locs_shift = ap_backwards_window - np.where(v_array[peak_locs[peak_idx] - ap_backwards_window: peak_locs[peak_idx] + pre_ap_baseline_forwards_window] == v_max)[0][0]
        AP_locations_list += [ peak_locs[peak_idx] - peak_locs_shift ]  
    
    ls = list(np.where(v_array[AP_locations_list] >= ap_spike_min_height)[0])
    AP_locations_list  = [AP_locations_list[ls_ ] for ls_ in ls ] # list of accurate AP peak locations

   
    # redefine ap_backwards_window to mean inter spike interval if <  ap_backwards_window
    if len(AP_locations_list) >= 2 : 
        ap_backwards_window = int(min(ap_backwards_window ,  np.mean(np.diff(AP_locations_list))))

    # looping APs in sweep
    for peak_idx in range(len(AP_locations_list)): 

        peak_location = AP_locations_list[peak_idx] #peak location for given AP
        
        # define window v_temp for AP and derivitive in that window v_derivative_temp
        if ap_backwards_window <  peak_location: 
            #print('debugging')
            #print(peak_location - ap_backwards_window,  peak_location )
            if peak_location - ap_backwards_window ==  peak_location and  peak_location >= 50 :
                ap_backwards_window = ap_backwards_window 
                v_temp = v_array[peak_location - ap_backwards_window: peak_location ]
                v_derivative_temp = v_deriv_transformed[peak_location - ap_backwards_window: peak_location ]
            else:
                v_temp = v_array[peak_location - ap_backwards_window: peak_location ]
                v_derivative_temp = v_deriv_transformed[peak_location - ap_backwards_window: peak_location ]
        else: 
            v_temp = v_array[0:peak_location]
            v_derivative_temp = v_deriv_transformed[0:peak_location]
        
          
        # find changes in dv/dt in AP window
        x_                = np.diff(v_derivative_temp)
        upshoot_loc_array = np.where(x_  <  0)[0]   # indices where the values in the x_ array are less than zero, indicating a negative change in derivitive
       
        #check for artifacts by max voltage
        AP_peak_voltage   = v_array[peak_location]         #np.where(v_temp == np.max(v_temp))[0][0] # ? idk why dj ever hasd this
        # print(f" peak voltage is {AP_peak_voltage} mV ")
        if AP_peak_voltage > 120 :
            print(f"Artifact detected at index {peak_location}, {AP_peak_voltage} > 120mV skipping.")
            continue

        if len(upshoot_loc_array) > 0 : 
            upshoot_loc_in_window  = np.where(x_  <  0)[0][0]
        elif len(upshoot_loc_array) == 0 : # return empty lists if no upshoot detected 
            return [] ,   [] ,  []  , [] ,  [] ,  [] , [] , []   
        
        # UPSHOOT
        upshoot_location  =   peak_location - ap_backwards_window + upshoot_loc_in_window
        AP_upshoot_locations_list += [   upshoot_location  ]
        # VOLTAGE THRESHOLD
        voltage_threshold = v_array[upshoot_location]
        AP_voltage_thresholds_list += [voltage_threshold]
        # AP HEIGHT
        AP_height = v_array[peak_location]  - voltage_threshold
        AP_heights_list    += [  AP_height  ]
        if AP_height > 150 or AP_height < 10:
            print(f"Artifact detected at index {peak_location}, AP height of {AP_height} mV, skipping.")
            continue
        # LATENCY
        AP_latency   = sec_to_ms * (peak_location - upshoot_location)  / sampling_rate
        AP_latencies_list  += [ AP_latency ]
        
        #SLOPE
        average_slope, max_dvdt, max_dvdt_location = calculate_ap_slope_and_max_dvdt(v_array, upshoot_location, AP_latency, sampling_rate)
        if average_slope <= 0 or max_dvdt <= 0:
            warnings.warn("Slope/derivative of AP is not positive.")
            AP_slope_list += [np.nan]
            ap_max_dvdt_list += [np.nan]
            plot_ap_window(v_array,peak_location, upshoot_location, voltage_threshold, AP_latency, average_slope, max_dvdt, max_dvdt_location, input_sampling_rate, sec_to_ms)
        else:
            AP_slope_list +=  [ average_slope ]
            ap_max_dvdt_list += [ max_dvdt ]

        #WIDTH
        ap_width_min = 0.1 # ms
        ap_width_max = 3 # ms
        fwhm_AP = calculate_fwhm(v_array, peak_location, upshoot_location, AP_height, sampling_rate, sec_to_ms, ap_width_min, ap_width_max)
        AP_fwhm_list       += [ fwhm_AP ]
        
    return AP_locations_list , AP_upshoot_locations_list, AP_voltage_thresholds_list , AP_heights_list , AP_latencies_list , AP_slope_list , AP_fwhm_list, ap_max_dvdt_list

########## AP EXTRACTOR MODULES
def get_window_bounds(peak_location, upshoot_location, array_length, isi_multiplier=3):
    """
    Define the window around the action potential using a conservative estimate based on the peak location.
    """
    # Calculate average ISI to estimate the duration of an individual AP
    average_isi = (peak_location - upshoot_location) * isi_multiplier
    window_start = max(0, upshoot_location - average_isi)
    window_end = min(array_length, peak_location + average_isi)
    return window_start, window_end

def calculate_fwhm(v_array, peak_location, upshoot_location, AP_height, sampling_rate, sec_to_ms, ap_width_min, ap_width_max):
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
    window_start, window_end = get_window_bounds(peak_location, upshoot_location, len(v_array))


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
        plot_fwhm(v_array, upshoot_location, peak_location, sampling_rate, sec_to_ms)
        return np.nan

    # Find the crossing after the peak (descending phase)
    fwhm_end_candidates = crossings[crossings > peak_location]
    if fwhm_end_candidates.size > 0:
        fwhm_end = fwhm_end_candidates[0]  # The first crossing after the peak
    else:
        warnings.warn("No crossing found after peak for FWHM calculation, setting to NaN.")
        plot_fwhm(v_array, upshoot_location, peak_location, sampling_rate, sec_to_ms)
        return np.nan
    
    # Calculate FWHM in ms
    fwhm_ms = (fwhm_end - fwhm_start) / sampling_rate * sec_to_ms

    #AP height to width ratio to exclude poorly compensated recordings
    height_to_width_ratio = AP_height/fwhm_ms

    # Check if FWHM is within plausible biological limits
    if fwhm_ms < ap_width_min or fwhm_ms > ap_width_max:
        print(f"Calculated FWHM is {fwhm_ms} outside of plausible limits ({ap_width_min}, {ap_width_max} ms), checking ratio.")
        if 40 > height_to_width_ratio >100:
            print(f"AP height to width ratio is {height_to_width_ratio}, outside relevant bounds 40-100, poor compensation, setting fwhm to nan.")
            # plot_fwhm(v_array, upshoot_location, peak_location, sampling_rate, sec_to_ms)
            return np.nan

    return fwhm_ms

def plot_fwhm(v_array, upshoot_location, peak_location, sampling_rate, sec_to_ms):
    """
    Plot the action potential and the half-maximum level to visualize the FWHM calculation.

    Parameters:
    v_array (numpy.ndarray): The array containing voltage data for a single sweep.
    upshoot_location (int): The index in v_array corresponding to the upshoot of the AP.
    peak_location (int): The index in v_array corresponding to the peak of the AP.
    sampling_rate (float): The sampling rate at which the data was recorded (in Hz).
    sec_to_ms (float): Conversion factor from seconds to milliseconds.
    """

    window_start, window_end = get_window_bounds(peak_location, upshoot_location, len(v_array))

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
    plt.title('FWHM Visualization')
    plt.legend()
    plt.show()

def calculate_ap_slope_and_max_dvdt(v_array, upshoot_index, latency, sampling_rate):
    """
    Calculate both the average slope and maximum dV/dt of the action potential (AP) 
    from the upshoot to three quarters of the latency period, and return the index of max dV/dt.

    Parameters:
    v_array (numpy.ndarray): The array containing voltage data for a single sweep.
    upshoot_index (int): The index in v_array corresponding to the AP upshoot.
    latency (float): The latency of the AP in milliseconds.
    sampling_rate (float): The sampling rate at which the data was recorded (in Hz).

    Returns:
    tuple: 
        - average_slope (float): The average slope from the upshoot to 3/4 latency (in mV/ms).
        - max_dvdt (float): The maximum rate of voltage change (dV/dt) in that window (in mV/ms).
        - max_dvdt_index (int): The index in v_array where the max dV/dt occurs.
    """
    # Define the end point as three quarters of the latency period
    three_quarter_latency_samples = int((latency * 3/4) * (sampling_rate / 1000))
    end_index = min(len(v_array), upshoot_index + three_quarter_latency_samples)

    # Calculate the derivative over the window
    derivative_window = np.gradient(v_array[upshoot_index:end_index])

    # Calculate average slope
    voltage_change = v_array[end_index-1] - v_array[upshoot_index]
    time_change_ms = (end_index - upshoot_index) / sampling_rate * 1000  # convert to ms
    average_slope = voltage_change / time_change_ms

    # Find maximum dV/dt and its index
    max_dvdt_index = np.argmax(derivative_window) + upshoot_index
    max_dvdt = derivative_window[max_dvdt_index - upshoot_index] * sampling_rate  # convert to mV/ms

    return average_slope, max_dvdt, max_dvdt_index


def ap_characteristics_extractor_main(V_array, critical_num_spikes = 1, all_sweeps = False, method = "derivative"): 
    '''
      Main function for extracting action potential (AP) characteristics across multiple sweeps of electrophysiological data. 
      It iteratively calls the 'ap_characteristics_extractor_subroutine_derivative' for each selected sweep.

    Input:
        V_array (2d array): A pandas DataFrame containing voltage data from electrophysiological recordings.
        critical_num_spikes (int, optional): The minimum number of spikes required in a sweep to consider it for analysis. Defaults to 1.
        all_sweeps (bool, optional): If True, analyzes all sweeps; otherwise, selects sweeps based on the number of spikes. Defaults to False.
        method (str, optional): The method used for extracting AP characteristics. Currently, only "derivative" is supported. Defaults to "derivative".

    Returns:
        peak_latencies_all (list):  AP latency across all analyzed sweeps (ms).
        v_thresholds_all (list):  AP Voltage thresholds across all analyzed sweeps (mV).
        peak_slope_all (list):  AP slope rates of change upshoot locations across all analyzed sweeps (mV/ms).
        peak_locs_corr_all (list):  Corrected locations of AP peaks across all analyzed sweeps.
        upshoot_locs_all (list):  Locations of AP thresholds across all analyzed sweeps.
        peak_heights_all (list): AP height (peak to threshold) across all analyzed sweeps (mV).
        peak_fw_all (list):  AP full widths at half maximum (FWHM) of each AP across all analyzed sweeps, if <0 set as peak_latency (ms).
        peak_indices_all (list): Indices of the peaks within the aggregated list.
        sweep_indices_all (list): Indices of the sweeps corresponding to each peak in the aggregated list.
        

    '''

    # V_array       = np.array(V_dataframe)
    num_aps_all   = np.array(num_ap_finder(V_array))
    sweep_indices = None 
    min_spikes    = critical_num_spikes 
    # Find which trace has first spiking and if that is more than min_spikes just take the first, else take first two sweeps with spiking
    
    if sum(num_aps_all) == 0 : 
        return np.nan , np.nan , np.nan , np.nan , np.nan , np.nan , np.nan , np.nan , np.nan 
    
    first_spike_idx  = np.where(num_aps_all > 0)[0][0]
    num_first_spikes = num_aps_all[first_spike_idx] 

    if all_sweeps : 
        sweep_indices =  list(np.arange(0, len(num_aps_all)))
        sweep_indices = np.array(sweep_indices)[num_aps_all>0] 
    else: 
        if num_first_spikes >= min_spikes : 
            sweep_indices  = [first_spike_idx]
        else:
            sweep_indices  = [first_spike_idx, first_spike_idx + 1] 
    
    # Make sure list is in range 
    sweep_indices =  np.array(sweep_indices)
    sweep_indices = sweep_indices[sweep_indices < len(num_aps_all)] 
    
    if len(sweep_indices) == 0 : 
        return np.nan , np.nan , np.nan , np.nan , np.nan , np.nan , np.nan , np.nan , np.nan 

    # Initialise lists 
    peak_latencies_all  = [] 
    v_thresholds_all    = [] 
    peak_slope_all      = []
    peak_locs_corr_all  = []
    upshoot_locs_all    = []
    peak_heights_all    = []
    peak_fw_all         = [] 
    sweep_indices_all   = [] 
    peak_indices_all    = []
    
    for sweep_num in sweep_indices: 
        
        if method == "derivative":
            
            peak_locs_corr_, upshoot_locs_, v_thresholds_, peak_heights_ ,  peak_latencies_ , peak_slope_ , peak_fw_,  ap_max_dvdt_list  =  ap_characteristics_extractor_subroutine_derivative(V_array, sweep_num)
        else:
            print("Invalid thresholding method")
            sys.exit(1)
            
        
        if v_thresholds_  == [] :
            pass 
        else: 
            peak_locs_corr_all += peak_locs_corr_
            upshoot_locs_all   += upshoot_locs_
            peak_latencies_all += peak_latencies_
            v_thresholds_all   += v_thresholds_
            peak_slope_all     += peak_slope_
            peak_heights_all   += peak_heights_
            peak_fw_all        += peak_fw_
            # AP_max_dvdt        += ap_max_dvdt_list
            peak_indices_all   +=  list(np.arange(0, len(peak_locs_corr_all)))
            
            sweep_indices_all +=   [sweep_num]*len(peak_locs_corr_)


    return peak_latencies_all  , v_thresholds_all  , peak_slope_all  , peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all   , peak_indices_all , sweep_indices_all 


########################      pAD DETECTION FUNCTION(S)  ####################
 

def pAD_detection(V_array):
    '''
    Main pAD detection algorithm.
    Input: 
        V_array: np.array : Voltage array of the trace to be analysed.
    Output: 
        peak_latencies_all, v_thresholds_all, peak_slope_all, peak_heights_all, pAD_df
    '''

    # Extract AP characteristics
    peak_latencies_all, v_thresholds_all, peak_slope_all, peak_locs_corr_all, upshoot_locs_all, peak_heights_all, peak_fw_all, peak_indices_all, sweep_indices_all = ap_characteristics_extractor_main(V_array, all_sweeps=True)
    
    # Early return if no APs found
    if np.all(np.isnan(peak_latencies_all)):
        print (f"No APs detected in voltage trace.")
        return peak_latencies_all, v_thresholds_all, peak_slope_all, peak_heights_all, np.nan

    # Create DataFrame
    pAD_df = pd.DataFrame({
        'AP_loc': peak_locs_corr_all,
        'upshoot_loc': upshoot_locs_all,
        'AP_threshold': v_thresholds_all,
        'AP_slope': peak_slope_all,
        'AP_latency': peak_latencies_all,
        'AP_height': peak_heights_all,
        'AP_width': peak_fw_all,
        'AP_sweep_num': sweep_indices_all,
        'AP_type': 'somatic'  # default to 'somatic'
    })

    # Classify APs as 'pAD_true' if threshold < -60 mV
    pAD_df.loc[pAD_df['AP_threshold'] < -60, 'AP_type'] = 'pAD_true'
    # Count pADs
    pAD_df['pAD_count'] = np.sum(pAD_df['AP_type'] == 'pAD_true')

    # Prepare data for clustering
    pAD_df_uncertain = pAD_df[pAD_df['AP_type'] == 'somatic']

    if len(pAD_df_uncertain) < 2:
        print (f"Fewer than 2 APs with voltage threshold > -60mV.")
        return peak_latencies_all, v_thresholds_all, peak_slope_all, peak_heights_all, pAD_df

    # # Clustering with KMeans
    # X = pAD_df_uncertain[['AP_slope', 'AP_threshold', 'AP_latency']]
    # kmeans = KMeans(n_clusters=2, n_init=1).fit(X)
    # labels = kmeans.labels_

    # Clustering with GMM
    X = pAD_df_uncertain[['AP_slope', 'AP_threshold', 'AP_latency']]
    gmm = GaussianMixture(n_components=2, n_init=1).fit(X)
    labels = gmm.predict(X)

    # Assign GMM labels as 'pAD_possible' or 'somatic'
    pAD_df.loc[pAD_df['AP_type'] == 'somatic', 'AP_type'] = np.where(labels == 0, 'pAD_possible', 'somatic')


    return peak_latencies_all, v_thresholds_all, peak_slope_all, peak_heights_all, pAD_df


########## HANDELIN FIRING PROPERTY DATA (FP)  --  FI curves
def plot_APs_off_step(V_array, I_array, peak_locs_corr_all, sweep_indices_all, sweep_to_plot):
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
        plt.title(f'APs Off Current Step - Sweep {sweep_to_plot + 1}')
        plt.legend()
        plt.show()
    else:
        print(f"No APs detected off the current step in sweep {sweep_to_plot}.")


def extract_FI_x_y(V_array, I_array, peak_locs_corr_all, sweep_indices_all):
    '''
    Extracts data for Frequency-Current (FI) relationship from voltage (V_array) and current (I_array) recordings.

    Input:
        V_array (numpy array): 2D array containing voltage recordings for different current steps.
        I_array (numpy array): 2D array containing current recordings for different current steps.
        peak_locs_corr_all (): peak locations  

    Returns:
        step_current_values (list): List of injected current values in picoamperes (pA) for each sweep.
        ap_counts (list): List of action potential counts for each sweep.
        V_rest (float): Average resting membrane potential (in millivolts) calculated when no current is injected.
    '''
    V_array_adj, I_array_adj = normalise_array_length(V_array, I_array, columns_match=True)

    ap_counts_per_sweep = [] #on step
    step_current_values = []
    V_rest_values = []

    for sweep in range(I_array_adj.shape[1]):
        current_sweep = I_array_adj[:, sweep]
        voltage_sweep = V_array_adj[:, sweep]

        # Get the sweep-specific peak locations and corresponding sweep indices
        sweep_peak_locs = [peak_locs_corr_all[i] for i, sweep_index in enumerate(sweep_indices_all) if sweep_index == sweep]

        # Identifying the current injection step
        non_zero_indices = np.where(current_sweep != 0)[0]
        if len(non_zero_indices) == 0:
            
            previous_current = I_array_adj[:, sweep-1]
            previous_non_zero_indices = np.where(previous_current != 0)[0]
            previous_current_step = previous_current[previous_non_zero_indices[0]]
            next_current = I_array_adj[:, sweep+1]
            next_current_step = next_current[previous_non_zero_indices[0]]
            if previous_current_step+ next_current_step==0:
                step_current_values.append(0)
                ap_counts_per_sweep.append(len(sweep_peak_locs))
                V_rest_values.append(np.mean(voltage_sweep))  
                continue
            else:
                print ('Error in current steps, missing step value that should not be 0.')
                return step_current_values, ap_counts_per_sweep,  V_rest 
                
        first_non_zero_index = non_zero_indices[0]
        last_non_zero_index = non_zero_indices[-1]
        
        # Count of APs within the current injection step of this sweep
        ap_on_step = len([peak_loc for peak_loc in sweep_peak_locs if first_non_zero_index <= peak_loc <= last_non_zero_index])
        ap_counts_per_sweep.append(ap_on_step)
         

        # Calculating the resting membrane potential
        V_rest_indices = np.where(current_sweep == 0)[0]
        if len(V_rest_indices) > 0:
            V_rest_values.append(np.mean(voltage_sweep[V_rest_indices]))

        # Calculating the step current value
        first_non_zero_index = non_zero_indices[0]
        first_non_zero_current = current_sweep[first_non_zero_index]
        step_current_values.append(first_non_zero_current)

        buffer = 50
        # Check for spikes off the current step
        ap_off_step = [peak for peak in sweep_peak_locs if peak < (first_non_zero_index) or peak > (last_non_zero_index+buffer)]
        if ap_off_step:
            # add plotter to check for pAD
            plot_APs_off_step(V_array, I_array, peak_locs_corr_all, sweep_indices_all, sweep_to_plot=sweep)
            print(f"APs detected off current step at {np.mean(voltage_sweep[V_rest_indices]):.2f}mV in sweep {sweep+1}. ")

    V_rest = np.nanmean(V_rest_values) if len(V_rest_values) > 0 else np.nan

    return step_current_values, ap_counts_per_sweep,  V_rest 
#OLD 
# def extract_FI_x_y (path_I, path_V):
#     '''
#     Extracts data for Frequency-Current (FI) relationship from voltage and current recordings.

#     Input:
#         path_I (str): Path to the current (I) data file.
#         path_V (str): Path to the voltage (V) data file.

#     Returns:
#         x (list): List of injected current values in picoamperes (pA) for each sweep.
#         y (list): List of action potential counts for each sweep.
#         v_rest (float): Average resting membrane potential (in millivolts) calculated when no current is injected.
#     '''
#     _, df_V = igor_exporter(path_V) # _ will be the continious wave which is no use here
#     _, df_I = igor_exporter(path_I)
    
#     # setting df_V and df_I to be the same dimentions
#     df_I_test = df_I.iloc[:, 0:df_V.shape[1]]
#     df_V_test = df_V.iloc[0:df_I.shape[0],:]
    
#     #pulling and averageing all V values  when I = 0
#     v_rest = np.nanmean(df_V_test[df_I_test == 0])

#     y = [] #APs per sweep
#     for i in range (len(df_V.columns)): #count APs in each column (sweep/ current step and append to list y)
    
#         y_ = df_V.iloc[:,i].tolist()
#         v_smooth, peak_locs , _ , num_peaks  = ap_finder(y_) #props cotains dict with height/width for each AP
#         y.append(num_peaks) 
    
#     x = [] #current injection per sweep (taking max)
#     for i in range (len(df_I.columns)):
#         x_ = df_I.iloc[:,i].tolist()
#         #I_modes = max(set(x_), key=x_.count)
#         I_modes = np.max(np.abs(x_))
#         x.append(I_modes)
    
#     #now df is trimmed to fit eachother should not be relevant 
#     if len(x) < len(y):
#         print("F file > I file        Data Error: Igor Sucks") #THIS SHOULDNT HAPPEN
        
#     if len(x) is not len(y):    #this may happen because when you plan a series of sweeps all are saved even if you esc before all are run
#         # print (len(x), len(y))
#         del x[len(y):]
#         # print("inequal_adjusting")
#         # print (len(x), len(y))

#     return x, y,  v_rest 



def extract_FI_slope_and_rheobased_threshold(x,y, slope_liniar = True):
    '''

    Calculates the slope of the frequency-current (FI) curve and the rheobase threshold.

    input:
        x (numpy.ndarray): 1D array representing the current input (usually in pA).
        y (numpy.ndarray): 1D array representing the number of action potentials per sweep.
        slope_linear (bool): If True, a linear fit is used to calculate the slope; if False, a sigmoid fit is used, and the slope is determined from the 'k' value of the sigmoid (default True).

    Returns:
        slope (float): The slope of the FI curve. It's either the linear slope or the 'k' value from the sigmoid fit.
        rheobase_threshold (float): The calculated rheobase threshold in pA, determined as the x-intercept of the linear fit of the FI curve.


    '''
    list_of_non_zero = [i for i, element in enumerate(y) if element!=0] #indicies of non-0 element in y

    #need to handle pAD here 
    x_threshold = np.array (x[list_of_non_zero[0]:list_of_non_zero[0]+3])# taking the first 3 non zero points to build linear fit corisponsing to the first 3 steps with APs
    y_threshold = np.array(y[list_of_non_zero[0]:list_of_non_zero[0]+3])
    
    # rehobased threshold with linear
    coef = np.polyfit(x_threshold,y_threshold,1) #coef = m, b #rheobase: x when y = 0 (nA) 
    rheobase_threshold = -coef[1]/coef[0] #-b/m
    #liniar_FI_slope
    if slope_liniar == True:
        FI_slope_linear = coef[0]  
    else: #sigmoid_FI_slope
        x_sig, y_sig = trim_after_AP_dropoff(x,y) #remove data after depolarisation block/AP dropoff
        x_sig, y_sig = sigmoid_fit_0_0_trim ( x_sig, y_sig, zeros_to_keep = 3) # remove excessive (0,0) point > 3 preceeding first AP
        x_fit, y_fit , popt = fit_sigmoid( x_sig, y_sig, maxfev = 1000000, visualise = False) #calculate sigmoid best fit #popt =  [L ,x0, k]  
        FI_slope_linear = popt[2]
        
    return FI_slope_linear, rheobase_threshold



                       
########## PLOTTTING -- can we move it to plotters 
    if main_big_plot: 
        plot_window = 500 

        for idx in range(len(peak_locs_corr)):    

            fig  = plt.figure(figsize = (20,20))
            plt.plot(v_array[peak_locs_corr[idx] - plot_window : peak_locs_corr[idx] + plot_window    ])
            plt.plot( upshoot_locs[idx] -  peak_locs_corr[idx] + plot_window , v_array[upshoot_locs[idx]  ]  , '*', label = 'Threshold')
            plt.plot(  plot_window , v_array[peak_locs_corr[idx]]  , '*', label = 'Peak')
            plt.legend()
            plt.show()

    return peak_locs_corr , upshoot_locs, v_thresholds , peak_heights , peak_latencies , peak_slope , peak_fw

def replace_nan_with_mean(array):
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


def spike_remover(array):
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
    inputs: V_array_or_list  :  voltage np.array, shape: length x num_sweeps
            drug_in  :  integer , sweep number when drug was applied (included in APP)
            drug_out :  integer , sweep number when drug was washed out (included in WASH)

    Returns: list_PRE, list_APP, list_WASH : each a list of input file values for that condition. 
    '''
    if isinstance(V_array_or_list, list):
        V_PRE  = V_array_or_list[:drug_in-1]
        V_APP  = V_array_or_list[drug_in-1:drug_out-1]
        V_WASH = V_array_or_list[drug_out-1:]

    elif isinstance(V_array_or_list, np.ndarray):
        V_PRE  = V_array_or_list[:, 0:drug_in-1]
        V_APP  = V_array_or_list[:, drug_in-1:drug_out-1]
        V_WASH = V_array_or_list[:, drug_out-1:]

    return V_PRE, V_APP, V_WASH 


def mean_RMP_APP_calculator(V_array, drug_in, drug_out, I_array=None):
    '''
    inputs: V_array (2D array of V_df),
            drug_in  :  integer , sweep number when drug was applied (included in APP)
            drug_out :  integer , sweep number when drug was washed out (included in WASH)

    return: input_R_PRE, input_R_APP, input_R_WASH
            lists of mean RMP for each sweep in PRE APP or WASH
    
    '''
    V_array_cleaned  = spike_remover(V_array) #NOT WORKING JJB210427/t8

    if I_array is None or (I_array == 0).all() :
        print(" No I injected or no I data, taking RMP as all.")
        # list of means of every column
        mean_RMP_sweep_list  = list(np.mean(V_array_cleaned  , axis = 0))
        
    else: # I step in I_array
        print('I step detected, averaging V when no I injected for each sweep.')
        I_array_adj, V_array_adj = I_array_to_match_V (V_array, I_array)
        zero_current_mask = (I_array_adj == 0) #boolian mask where I == 0
        V_masked = np.where(zero_current_mask, V_array_adj, np.nan) # V where I -- 0
        mean_RMP_sweep_list = np.nanmean(V_masked, axis=0).tolist()
        

    mean_RMP_PRE, mean_RMP_APP, mean_RMP_WASH = APP_splitter(mean_RMP_sweep_list, drug_in, drug_out)
    return mean_RMP_PRE, mean_RMP_APP, mean_RMP_WASH

def mean_inputR_APP_calculator(V_array, I_array, drug_in, drug_out):
    '''
    input:      V_array, I_array (2D array of df shape)
                drug_in  :  integer , sweep number when drug was applied (included in APP)
                drug_out :  integer , sweep number when drug was washed out (included in WASH)

    returns :   input_R_PRE, input_R_APP, input_R_WASH 
                input R for each sweep = current injected / change in V in a list/array 
    '''

    I_sweep = getI_array_sweep(I_array)

    input_R_ohms_V_array = []
    
    for index, V_sweep in enumerate(V_array.T):  # Transpose V_array to iterate over columns/sweeps
        # print(input_R_ohms_V_array)
        #skip first 3 sweeps as often holding I was being set or cell was not stabelised
        if index < 2:
            input_R_ohms_V_array.append(np.nan) #preserve sweeps for APP_splitter
            # print(f"Appended NaN for index {index}.")
            continue

        V_sweep, I_sweep =normalise_array_length(V_sweep, I_sweep)

        V_cleaned  = spike_remover(V_sweep) #remove spiking 

        #fetch delta_V
        steady_state , hyper  , first_current_point, last_current_point= steady_state_value(V_sweep, I_sweep)  #with I injection
        rmp = np.nanmean(V_cleaned[I_sweep == 0]) #without I injection
        delta_V_mV = abs(steady_state - rmp)
        #fetch I injected 
        delta_I_pA = abs(np.unique(I_sweep[I_sweep != 0])[0])
        # fectch sweep input R (ohm)
        delta_I_A = delta_I_pA * 1e-12  #  picoamperes to amperes
        delta_V_V = delta_V_mV * 1e-3  # millivolts to volts

        sweep_input_R_ohms = (delta_V_V / delta_I_A) 
        input_R_ohms_V_array.append(sweep_input_R_ohms)
        # print(f"Appended value for index {index}. ")

    input_R_PRE, input_R_APP, input_R_WASH =APP_splitter(input_R_ohms_V_array, drug_in, drug_out)

    return input_R_PRE, input_R_APP, input_R_WASH 


def normalise_array_length(V_array, I_array, columns_match=False):
    '''
    Adjusts the lengths of V_array and I_array to have the same number of rows and, optionally, the same number of columns.

    Input:
        V_array (numpy.ndarray): 2D array containing voltage recordings (sweeps).
        I_array (numpy.ndarray): 2D array containing corresponding current recordings (sweeps).
        columns_match (bool): If True, adjusts the arrays to have the same number of columns as well.

    Returns:
        V_adj (numpy.ndarray): Adjusted voltage array.
        I_adj (numpy.ndarray): Adjusted current array.
    '''
    # #ensure V_sweep and I_sweep are the same length
    # if len(V_array) != len(I_array):
    #     # print(f"Length of V_sweep: {len(V_sweep)}, Length of I_sweep: {len(I_sweep)}") #V is usaly 400001 and I 400000
    #     V_adj = V_array[:min(len(V_array), len(I_array))]
    #     I_adj = I_array[:min(len(V_array), len(I_array))]
    min_rows = min(V_array.shape[0], I_array.shape[0])
    V_adj = V_array[:min_rows]
    I_adj = I_array[:min_rows]

    if columns_match:
        min_cols = min(V_array.shape[1], I_array.shape[1])
        V_adj = V_adj[:, :min_cols]
        I_adj = I_adj[:, :min_cols]
    
    return V_adj, I_adj

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
    I_sweep_adj = I_sweep_adj[:, np.newaxis]
    I_array_adj = np.tile(I_sweep_adj, (1, V_array_adj.shape[1])) 
    return I_array_adj, V_array_adj