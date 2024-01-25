
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



from scipy.signal import find_peaks
import numpy as np

########## USER ?
#FIX ME POU IN RIGHT PLACE
 
def generate_V_pAD_df(folder_file): 
    '''
    Generates pAD_df, V_array  
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

def steady_state_value(V_sweep, I_sweep, step_current_val=None,  avg_window = 0.5):
    ''' 
    """
    Calculates the steady state value of a voltage trace during a current injection step.

    Input:
        V_sweep (array-like): A 1D time series representing a single voltage trace (single sweep).
        I_sweep (array-like): The corresponding 1D time series for the current trace (single sweep).
        step_current_val (float, optional): The value of the step current injection in pA. If not provided, it is derived from the unique non-zero value in I_sweep.
        avg_window (float, optional): The fraction of the step current duration used for averaging to determine the steady state value. Default is 0.5.

    Output:
        asym_current (float): The steady state value of the voltage trace during the step current injection.
        hyper (bool): Indicates whether the step current is hyperpolarizing (True) or not (False).
        first_current_point (int): The index of the first timeframe of the step current injection.
        last_current_point (int): The index of the last timeframe of the step current injection.

    Note:
        The function determines whether the step current is hyperpolarizing based on the sign of 'step_current_val'.
        It calculates the steady state value ('asym_current') by averaging the voltage trace over a window at the end of the current injection step.
  '''
    if step_current_val == None:
        if np.count_nonzero(I_sweep) > 0:
            step_current_val = np.unique(I_sweep[I_sweep != 0])[0]
        else:
            print("Multiple I values in step, unable to calculate steady state.")
            return

    if step_current_val >= 0: #note that 0pA of I injected is not hyperpolarising 
        hyper = False
        first_current_point = np.where(I_sweep == np.max(I_sweep) )[0][0] 
        last_current_point = np.where(I_sweep == np.max(I_sweep) )[0][-1]

    elif step_current_val < 0 : 
        hyper = True 
        first_current_point = np.where(I_sweep == np.min(I_sweep) )[0][0] 
        last_current_point = np.where(I_sweep == np.min(I_sweep) )[0][-1]

    current_avg_duration = int(avg_window*(last_current_point - first_current_point))
    asym_current  = np.mean(V_sweep[ last_current_point - current_avg_duration : last_current_point  ])

    return asym_current , hyper  , first_current_point, last_current_point

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

def tau_analyser(voltage_array, current_array, input_step_current_values, plotting_viz = False, verbose = False , analysis_mode  = 'max' ,input_sampling_rate = 1e4): 

    '''
    Functin to calculate tau associated with RC charging of cell under positive step current 

    Input : voltage_array : 2d array containing sweeps of voltage recordings / traces for different step currents
            current_array : 2d array containing sweeps of current recordings / traces for different step currents
            input_step_current_values : list containing step current values (in pA) 
            plotting_viz  :  boolean : whether or not we want to plot/viz 
            analysis_mode : str , either 'max' or 'asym'. If 'max' uses max of voltage to find tau, otw used v_steady
            input_sampling_rate : float : data acquisition rate as given from Igor. 
            verbose : Boolean : if False, suppresses print outputs 

    #TODO currenly returning only the 1st element as nan in the second and seems inferior quality 
    Returns : [ [tau (ms), steady state with I , I step (pA) , steady stats no I (RMP)] , [...] ]
    ''' 
    # noisy error msg and chunk stuff / exclusion criterion, calculation based on abs max / steady state . 

    sampling_rate = input_sampling_rate
    sec_to_ms = 1e3  
    # First we get the tau indices from the voltage trace: 
    num_aps_all = num_ap_finder(voltage_array)

    aps_found = np.array([1 if elem  > 0 else 0 for elem in num_aps_all])
    aps_found_idx = np.where(aps_found > 0)[0] 
    true_ap_start_idx = voltage_array.shape[-1]

    for idx in aps_found_idx : 
        if idx + 1 in aps_found_idx:
            if idx + 2 in aps_found_idx:   # once the 1st instance of spiking is found if 2 subsequent ones are found thats good
                            true_ap_start_idx = min(idx, true_ap_start_idx)  
    
    visualisation = plotting_viz
    tau_array = [] 

    num_tau_analysed_counter = 0   
    counter = 0                                                      #   as we might potentially skip some step currents as they are too noisy, need to keep track of which/ how many taus are actually analysed (we need two)
    if verbose: 
        print('Cell starts spiking at trace index start %s' % true_ap_start_idx)
    # check noise level: 
    while num_tau_analysed_counter <  2 : 

        tau_idx  =  true_ap_start_idx - 1  -  counter 
        if verbose: 
            print('Analysing for sweep index %s' % tau_idx)
        step_current_val  = input_step_current_values[tau_idx]
        asym_current, hyper , current_inj_first_point, current_inj_last_point  = steady_state_value(voltage_array[:, tau_idx], current_array[:, tau_idx], step_current_val)
        max_current    = np.max(voltage_array[current_inj_first_point:current_inj_last_point,tau_idx ]) 
        
        thresh_current_asym = (1 - np.exp(-1))*( asym_current - voltage_array[current_inj_first_point - 1 , tau_idx])  + voltage_array[current_inj_first_point - 1 , tau_idx] 
        thresh_current_max =  (1 - np.exp(-1))*( max_current - voltage_array[current_inj_first_point - 1 , tau_idx])  + voltage_array[current_inj_first_point - 1 , tau_idx] 
        
        # check first how noisy the signal is : 
        current_noise = np.std(voltage_array[ current_inj_first_point:current_inj_last_point,tau_idx] )
        
        # Calculate v resting potential 
        v_resting_membrane = np.mean(voltage_array[ 0 : current_inj_first_point, tau_idx]) 
        
        if abs(current_noise + thresh_current_asym - asym_current) <= 0.5 :          # we use asymtotic current for noise characterisation rather than max current 
            if verbose: 
                print('Too noisy, skipping current current step...')
        else: 
            num_tau_analysed_counter += 1 
            if visualisation: 
                # Plot it out for sanity check 
                plt.figure(figsize = (12,12))
                time_series = sec_to_ms*np.arange(0, len(voltage_array[:, tau_idx]))/sampling_rate
                plt.plot(time_series, voltage_array[:, tau_idx])
                if analysis_mode == 'max': 
                    plt.axhline( (1 - 0*np.exp(-1))*( max_current - voltage_array[current_inj_first_point - 1 , tau_idx])  + voltage_array[current_inj_first_point - 1 , tau_idx] , c = 'r', label = 'Max Current')
                    plt.axhline( (1 - np.exp(-1))*( max_current - voltage_array[current_inj_first_point - 1 , tau_idx])  + voltage_array[current_inj_first_point - 1 , tau_idx] , c = 'b',  label = 'Threshold' )
                elif analysis_mode == 'asym':
                    plt.axhline( (1 - 0*np.exp(-1))*( asym_current - voltage_array[current_inj_first_point - 1 , tau_idx])  + voltage_array[current_inj_first_point - 1 , tau_idx] , c = 'r', label = 'Asymtotic Current')
                    plt.axhline( (1 - np.exp(-1))*( asym_current - voltage_array[current_inj_first_point - 1 , tau_idx])  + voltage_array[current_inj_first_point - 1 , tau_idx] , c = 'b',  label = 'Threshold' )
                else: 
                    raise ValueError('Invalid Analysis Mode, can be either max or asym')
                plt.legend()
                plt.ylabel('Membrane Potential (mV)')
                plt.xlabel('Time (ms)')
                plt.show()
                
            if hyper: 
                if verbose: 
                    print('Positive current step not used! Hence breaking')
                break
            else:  
                # find all time points where voltage is at least 1 - 1/e or ~ 63% of max / asym val depending on analysis_mode
                if analysis_mode == 'max': 
                    time_frames =  np.where(voltage_array[:, tau_idx] > thresh_current_max )[0]
                elif analysis_mode == 'asym':
                    time_frames =  np.where(voltage_array[:, tau_idx] > thresh_current_asym )[0]
                else : 
                    raise ValueError('Invalid Analysis Mode, can be either max or asym')

                time_frames_ = time_frames[time_frames > current_inj_first_point]
                tau = sec_to_ms*(time_frames_[0] - current_inj_first_point) / sampling_rate

                # tau (ms), steady state with I , I step (pA) , steady stats no I
                tau_temp = [tau, asym_current, step_current_val, v_resting_membrane]

                tau_array.append(tau_temp)
        
        counter += 1

    if num_tau_analysed_counter == 0 : 
        return [np.nan, np.nan, np.nan, np.nan]
    return tau_array[0]


def sag_current_analyser(voltage_array, current_array, input_step_current_values,input_current_avging_window = 0.5): 

    '''
    Function to calculate sag current from voltage and current traces under hyper polarising current steps

    Input:
    volatge array : 2d array containing sweeps of voltage recordings / traces for different step currents
    current array : 2d array containing sweeps of current recordings / traces for different step currents
    input_step_current_values : list containing different step currents 
    input_current_avging_window : float / fraction representating what portion of step current duration is used for calculating 
    
    Returns : [[ sag(0-1) , steady state with I , I step (pA) , steady stats no I (RMP) ] , [...] ] 
    '''
    # sag : 0, 1 : works with the hyperpolarising current injection steps and 0 and 1 indices correspond to
    # the 1st two (least absolute value) step current injections. Later expand for positive current inj too? 
    sag_indices = [0,1]
    sag_current_all =  [] 
    # sag_tuple = namedtuple('Sag', ['val', 'steady_state', 'current_inj'])
    current_avging_window = input_current_avging_window 

    for sag_idx in sag_indices: 

        v_smooth, peak_locs , peak_info , num_peaks  = ap_finder(voltage_array[:,sag_idx])
        if num_peaks == 0: 

            # print('No Spike in sag trace, all good in da hood')

            asym_current, _ , _ , _  = steady_state_value(voltage_array[:, sag_idx], current_array[:, sag_idx], input_step_current_values[sag_idx])
            i_min = min(current_array[:,sag_idx])
            first_min_current_point = np.where(current_array[:,sag_idx] == i_min )[0][0] 
            last_min_current_point = np.where(current_array[:,sag_idx] == i_min )[0][-1]

            step_current_dur = last_min_current_point - first_min_current_point
            step_current_avg_window = int(current_avging_window*step_current_dur)
            
            # Calculate v resting potential 
            v_resting_membrane = np.mean(voltage_array[ 0 : first_min_current_point , sag_idx]) 

            asym_sag_current = np.mean(voltage_array[  last_min_current_point - step_current_avg_window: last_min_current_point, sag_idx])
            min_sag_current_timepoint = np.where( voltage_array[:,sag_idx]  ==  min(voltage_array[:,sag_idx]  ) )[0][0] 
            min_sag_current =  min(voltage_array[:,sag_idx]  )
            max_sag_current = np.mean(voltage_array[0 : first_min_current_point  - 1   , sag_idx] )

            sag_current = (asym_sag_current - min_sag_current)  / (  max_sag_current  - asym_sag_current )

            sag_current_temp = [sag_current, asym_current, input_step_current_values[sag_idx], v_resting_membrane] 

        elif num_peaks > 0: 
            print('Spike found in Sag analysis, skipping')
            sag_current_temp = [np.nan, np.nan, np.nan, np.nan] 
        # Append Value to existing named tuple
        sag_current_all.append(sag_current_temp) 

    return sag_current_all


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

def ap_characteristics_extractor_subroutine_derivative(V_dataframe, sweep_index, main_plot = False, input_sampling_rate = 1e4 , input_smoothing_kernel = 1.5, input_plot_window = 500 , input_slide_window = 200, input_gradient_order  = 1, input_backwards_window = 100 , input_ap_fwhm_window = 100 , input_pre_ap_baseline_forwards_window = 50, input_significance_factor = 8 ):
    '''
    Subroutine to extract AP characteristics from a given voltage trace.

    input: 
    V_dataframe : Voltage dataframe obtained from running igor extractor 
    sweep_index : integer representing the sweep number of the voltage file. 
    
    outputs: 
    peak_locs_corr : location of exact AP peaks 
    upshoot_locs   : location of thresholds for AP i.e. where the voltage trace takes the upshoot from   
    v_thresholds   : voltage at the upshoot poiny
    peak_heights   : voltage diff or height from threshold point to peak 
    peak_latencies : time delay or latency for membrane potential to reach peak from threshold
    peak_slope     : Rate of change/derivative of membrane potential AT the upshoot location
    peak_fwhm      : AP width at half maximum, # FIX ME currently has error that values are too large and are then set to latency 
   
    '''
    # Other Input Hyperparameters 
    # need the first two sweeps from 1st spike and 2nd spike. if 1st ap has around 3 spikes (parameter) then only first is good enough otw go to the 2nd trace 
    
    ap_backwards_window = input_backwards_window
    pre_ap_baseline_forwards_window = input_pre_ap_baseline_forwards_window 
    ap_fwhm_window  = input_ap_fwhm_window
    gradient_order = input_gradient_order
    significance_factor = input_significance_factor 
    plot_window      = input_smoothing_kernel 
    slide_window     = input_slide_window      # slides 200 frames from the prev. AP  200 frames ~ 20 ms 
    smoothing_kernel = input_smoothing_kernel
    main_big_plot    = main_plot
    inner_loop_plot  = True
    sampling_rate    = input_sampling_rate
    sec_to_ms        = 1e3 
    ap_spike_min_height =  0 # if value if less than 0 mV - very unhealthy spike  
    ap_width_min      = 0.1 # in ms minimum width of AP to be considered biologically plasuible
    ap_width_max      = 2   # in ms maximum width of AP to be considered biologically plasuible

    ## Outputs on Peak Statistics 

    peak_locs_corr  = []            #  location of peaks *exact^
    upshoot_locs    = []            #  location of voltage thresholds 
    v_thresholds    = []            #  value of voltage thresholds in mV
    peak_heights    = []            #  peak heights in mV, measured as difference of voltage values at peak and threshold 
    peak_latencies  = []            #  latency or time delay between peak and threshold points
    peak_fw         = []            #  width of peak at HALF MAXIMUM (FWHM abbreviated here to fw for brevity)
    peak_slope      = []            #  peak height / peak latency

    # Make np array of dataframe

    df_V_arr = np.array(V_dataframe)
    sweep_idx = sweep_index

    # Get rough estimate of peak locations 

    v_array = df_V_arr[:,sweep_idx]
    v_smooth, peak_locs , peak_info , num_peaks  = ap_finder(v_array) 
    v_deriv_transformed = np.heaviside( -np.diff(v_array)+ np.exp(1), 0 )
    
    if len(peak_locs) == 0 :
        print("no peaks found in trace..")
        return  [] ,   [] ,  []  , [] ,  [] ,  [] , [] 

    for peak_idx in range(len(peak_locs)) : 


        # Correct for peak offsets 
        while peak_locs[peak_idx]  < ap_backwards_window: 
            ap_backwards_window = int(ap_backwards_window - 10)
        
        v_max  = np.max(v_array[peak_locs[peak_idx] - ap_backwards_window : peak_locs[peak_idx] + pre_ap_baseline_forwards_window]) 
        peak_locs_shift = ap_backwards_window - np.where(v_array[peak_locs[peak_idx] - ap_backwards_window: peak_locs[peak_idx] + pre_ap_baseline_forwards_window] == v_max)[0][0]

        peak_locs_corr += [ peak_locs[peak_idx] - peak_locs_shift ]  

    
    ls = list(np.where(v_array[peak_locs_corr] >= ap_spike_min_height)[0])

    peak_locs_corr  = [peak_locs_corr[ls_ ] for ls_ in ls ] 

    if len(peak_locs_corr) >= 2 : 
        ap_backwards_window = int(min(ap_backwards_window ,  np.mean(np.diff(peak_locs_corr))))
    
    

    for peak_idx in range(len(peak_locs_corr)): # looping for peaks within a specified sweep 

        if peak_idx == 0 : 
            start_loc = slide_window
            
        else:
            start_loc = peak_locs_corr[peak_idx - 1 ]  + slide_window
        end_loc = peak_locs_corr[peak_idx]

        # Get AP slice  
        
        if ap_backwards_window <  end_loc: 
            #print('debugging')
            #print(end_loc - ap_backwards_window,  end_loc )
            if end_loc - ap_backwards_window ==  end_loc and  end_loc >= 50 :
                ap_backwards_window = 50 ### HARD CODED TO MAKE IT WORK FOR NOW!!
                v_temp = v_array[end_loc - ap_backwards_window: end_loc ]
                v_derivative_temp = v_deriv_transformed[end_loc - ap_backwards_window: end_loc ]
        
            else:
                # Normally 
                v_temp = v_array[end_loc - ap_backwards_window: end_loc ]
                v_derivative_temp = v_deriv_transformed[end_loc - ap_backwards_window: end_loc ]
        else: 
            v_temp = v_array[0:end_loc]
            v_derivative_temp = v_deriv_transformed[0:end_loc]
        
          

        x_                = np.diff(v_derivative_temp)
        
        upshoot_loc_array = np.where(x_  <  0)[0]   # define array where the dip / upshoot happens 
        
         
        v_peak   = np.where(v_temp == np.max(v_temp))[0][0]

        
        if len(upshoot_loc_array) > 0 : 
            upshoot_loc  = np.where(x_  <  0)[0][0]
        elif len(upshoot_loc_array) == 0 :
            return [] ,   [] ,  []  , [] ,  [] ,  [] , []    

        
        
        # Calculate AP metrics here 
        upshoot_loc  =   peak_locs_corr[peak_idx] - ap_backwards_window + upshoot_loc
        upshoot_locs += [   upshoot_loc  ]
        v_thresholds += [v_array[upshoot_locs[peak_idx]]]


        peak_heights    += [  v_array[peak_locs_corr[peak_idx]]  - v_array[upshoot_locs[peak_idx]]   ]
        peak_latency_   = sec_to_ms * (peak_locs_corr[peak_idx] - upshoot_locs[peak_idx])  / sampling_rate
        peak_latencies  += [ peak_latency_ ]
        
        #NEW CORRECTION

        smoothed_v_array = gaussian_filter1d(v_array, 20)
        smoothed_gradient = np.gradient(smoothed_v_array)

        # Check if derivative is positive
        if smoothed_gradient[upshoot_loc] <= 0:
            warnings.warn("Derivative at upshoot_loc is not positive")
            peak_slope += [np.nan] # FIXME : negative peak slopes should not occur, if they do, need more smoothing
        else:
            # Calculate derivative at upshoot_loc
            derivative_at_upshoot = smoothed_gradient[upshoot_loc]

            dt = sec_to_ms / sampling_rate                      # in ms should be 0.1 
            peak_slope      += [ derivative_at_upshoot / dt ]   # in mV  / ms or V / s 
        
        
        # Calculation of AP widths : use a linear fit for now  - really an exponential fit be the best, and since we use a linear fit, not a problem to calculate full width rather than half width  

        upshoot_x = peak_locs_corr[peak_idx] - ap_backwards_window + upshoot_loc 
        upshoot_y = v_array[upshoot_locs[peak_idx]]

        peak_x    = peak_locs_corr[peak_idx]
        peak_y    = v_array[peak_locs_corr[peak_idx]]
        
        half_y =  upshoot_y + (peak_y - upshoot_y) / 2
        
        
        first_x_array    =  np.where( v_array >= half_y  )[0] 
        return_x_array   = np.where( v_array <= half_y  )[0] 
        
        
        return_x_ = return_x_array[return_x_array > peak_x]
        first_x_  = first_x_array[first_x_array   < peak_x]
        
        # check these have values actually!!
        if len(first_x_ ) == 0 or len(return_x_) == 0: 
            print('AP width calculation not accurate!!')
            first_x  = upshoot_x + (peak_x - upshoot_x) / 2 
            return_x = peak_x + (peak_x - upshoot_x) / 2 
            
            
        else: 
            
            return_x = return_x_[0]
            first_x  = first_x_[0]     

        peak_fw_ =  sec_to_ms *  (return_x  - first_x)    / sampling_rate 
        
        if peak_fw_    < ap_width_min : 
            print('AP width negative, setting FWHM approx. as latency!!') #FIXME
            peak_fw_   = peak_latency_
        elif peak_fw_  > ap_width_max : # a bit of a hack to then have a simpler calculation for the widths FIXME
            # FIXME : happens way too often so hid the print out, needs to be fixed
            #print('AP width too large, setting FWHM approx. as latency!!')
            peak_fw_   = peak_latency_        
        peak_fw       += [ peak_fw_ ]

    return peak_locs_corr , upshoot_locs, v_thresholds , peak_heights , peak_latencies , peak_slope , peak_fw




def ap_characteristics_extractor_main(V_dataframe, critical_num_spikes = 4, all_sweeps = False, method = "derivative"): 
    '''
    Main function that calls the ap_characteristics_extractor_subroutine. Separated into main for future flexibility/adaptability. 

    Input  : V_dataframe         : pandas dataframe : voltage dataframe containing all sweeps of given Igor run 
           : critical_num_spikes : int : critical num of spikes that we want beyond which intrinsic properties are probably way off
           : all_sweeps          : boolean : whether or not we analyse all current injection sweeps

    Output : output of lists as generated by subroutine iterated over chosen sweeps AND analysed sweeps
    '''

    v_array       = np.array(V_dataframe)
    num_aps_all   = np.array(num_ap_finder(v_array))
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
            peak_locs_corr_, upshoot_locs_, v_thresholds_, peak_heights_ ,  peak_latencies_ , peak_slope_ , peak_fw_  =  ap_characteristics_extractor_subroutine_derivative(V_dataframe, sweep_num)
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
            peak_indices_all   +=  list(np.arange(0, len(peak_locs_corr_all)))
            
            sweep_indices_all +=   [sweep_num]*len(peak_locs_corr_)

    return peak_latencies_all  , v_thresholds_all  , peak_slope_all  , peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all   , peak_indices_all , sweep_indices_all 


########################      pAD DETECTION FUNCTION(S)  ####################
 
    
def pAD_detection(V_dataframe):
    '''
    Main pAD detection algorithm that first applies a pAD filter to  those APs with threshold < -65 mV and then uses a classifier to classify the remaining APs as pAD or not.
    Clustering algorithm is a KNN classifier with at most 2 clusters, but can be later changed to better account for outlier or noise variability. 
    Input : 
            V_dataframe   : np.array : Voltage dataframe of the trace to be analysed
            
    Output : 
            peak_latencies_all: list : List of peak latencies of all APs in the trace
            v_thresholds_all  : list: List of voltage thresholds of all APs in the trace
            peak_slope_all    : list: List of peak slopes of all APs in the trace 
            pAD_df            :  pandas dataframe : Dataframe containing all AP characteristics and pAD classification

    '''
      
    ### STEP 1 :  Get the action potential (AP) values and create pAD dataframe with columns and values 
        
    peak_latencies_all_ , v_thresholds_all_  , peak_slope_all_  , peak_locs_corr_, upshoot_locs_all_ , peak_heights_all_  , peak_fw_all_ ,  peak_indices_all_, sweep_indices_all_  =  ap_characteristics_extractor_main(V_dataframe, all_sweeps =True)
    
    # Create Dataframe and append values
    pAD_df = pd.DataFrame(columns=['pAD', 'AP_loc', 'AP_sweep_num', 'AP_slope', 'AP_threshold',  'AP_width' , 'AP_latency'])
    
    # V_threshold nans

    non_nan_locs =  np.where(np.isnan(v_thresholds_all_) == False)[0] 
    
    if len (non_nan_locs) == 0 : 
        pAD_df["AP_loc"] = peak_locs_corr_ 
        pAD_df["AP_threshold"] =  v_thresholds_all_
        pAD_df["AP_slope"] = peak_slope_all_
        pAD_df["AP_latency"] = peak_latencies_all_
        pAD_df["AP_width"] = peak_fw_all_
        pAD_df['AP_sweep_num'] = sweep_indices_all_
        pAD_df['upshoot_loc']  = upshoot_locs_all_ 
        pAD_df["pAD"] = ""
        pAD_df["pAD_count"] = np.nan 
        
        return peak_latencies_all_ , v_thresholds_all_  , peak_slope_all_  ,  pAD_df
        

    peak_latencies_all = np.array(peak_latencies_all_)[non_nan_locs]
    v_thresholds_all   = np.array(v_thresholds_all_ )[non_nan_locs]
    peak_slope_all     = np.array(peak_slope_all_  )[non_nan_locs]
    peak_locs_corr     = np.array(peak_locs_corr_ )[non_nan_locs]
    upshoot_locs_all   = np.array(upshoot_locs_all_)[non_nan_locs]
    peak_fw_all        = np.array(peak_fw_all_)[non_nan_locs]
    peak_indices_all    = np.array(peak_indices_all_)[non_nan_locs]
    sweep_indices_all  = np.array(sweep_indices_all_)[non_nan_locs]

    # peak slope nans 

    non_nan_locs =  np.where(np.isnan(peak_slope_all) == False)[0] 

    # remove those nan values from the dataframe

    peak_latencies_all = np.array(peak_latencies_all_)[non_nan_locs]
    v_thresholds_all   = np.array(v_thresholds_all_ )[non_nan_locs]
    peak_slope_all     = np.array(peak_slope_all_  )[non_nan_locs]
    peak_locs_corr     = np.array(peak_locs_corr_ )[non_nan_locs]
    upshoot_locs_all   = np.array(upshoot_locs_all_)[non_nan_locs]
    peak_fw_all        = np.array(peak_fw_all_)[non_nan_locs]
    peak_indices_all    = np.array(peak_indices_all_)[non_nan_locs]
    sweep_indices_all  = np.array(sweep_indices_all_)[non_nan_locs]
    
    assert len(peak_latencies_all) == len(peak_slope_all) == len(v_thresholds_all) == len(peak_locs_corr) == len(upshoot_locs_all) ==  len(peak_fw_all) == len(peak_indices_all)== len(sweep_indices_all) 
    
    pAD_df["AP_loc"] = peak_locs_corr 
    pAD_df["upshoot_loc"] =  upshoot_locs_all   
    pAD_df["AP_threshold"] =  v_thresholds_all 
    pAD_df["AP_slope"] = peak_slope_all 
    pAD_df["AP_latency"] = peak_latencies_all
    pAD_df["AP_width"] = peak_fw_all
    
    pAD_df['AP_sweep_num'] = sweep_indices_all
    pAD_class_labels = np.array(["pAD" , "Somatic"] )
    
    if len (peak_locs_corr) == 0 :
        print("No APs found in trace")
        pAD_df["pAD"] = ""
        pAD_df["pAD_count"] = np.nan 
        
        return   peak_latencies_all , v_thresholds_all  , peak_slope_all  , pAD_df
    elif len (peak_locs_corr) == 1 : 
        if v_thresholds_all[0]  < -65.0 :
            # pAD 
            pAD_df["pAD"] = "pAD"
            pAD_df["pAD_count"] = 1 
        else: 
            pAD_df["pAD"]       = "Somatic"
            pAD_df["pAD_count"] = 0 
        return peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  pAD_df
    else : 
        pass
    

    # STEP 2 : Make all labels for upshoot potential < -65 mV   = pAD  
    
    # Label those rows of pAD_df as pAD
    pAD_df.loc[pAD_df["AP_threshold"] < -65, "pAD"] = "True pAD"

    # df of all APs that are not certain pADs so we need a classifier 
    
    pAD_df_uncertain_via_threshold = pAD_df[pAD_df[ "AP_threshold"]  > -65.0 ]
    
    # We can use a clustering on the certain/true pADs to get a better estimate on accuracy of the classifier
    pAD_df_pAD_via_threshold = pAD_df[pAD_df[ "AP_threshold"]  < -65.0 ]    # TRUE pADs
    
    pAD_df_pAD_via_threshold["pAD"] = "True pAD"
    
    # Data for fitting procedure : Only those that are uncertain go into the classifier 
    
    X = pAD_df_uncertain_via_threshold[["AP_slope", "AP_threshold",  "AP_latency"]]
    
    # Check that there are enough APs to do a clustering

    if len(X) < 2 :
        print("Not enough APs to do a clustering")
        pAD_df["pAD_count"] = pAD_df["pAD"].apply(lambda x: np.sum(x == "True pAD"))
        
        return peak_latencies_all , v_thresholds_all  , peak_slope_all  , pAD_df
    # Check if 2 clusters are better than 1 
    
    kmeans_1 = KMeans(n_clusters=1, n_init = 1).fit(X)
    kmeans_2 = KMeans(n_clusters=2, n_init = 1 ).fit(X)
    
    wcss_1 = kmeans_1.inertia_
    wcss_2 = kmeans_2.inertia_
    
    
    if wcss_2 < wcss_1 :
        # 2 CLUSTERS  are a better fit than 1  
        v_thresholds_all_uncertain = np.array(pAD_df_uncertain_via_threshold["AP_threshold"])
        if np.nanmean( v_thresholds_all_uncertain [kmeans_2.labels_ == 0] )  < np.nanmean( v_thresholds_all_uncertain  [kmeans_2.labels_ == 1] ):
            
            pAD_df.loc[pAD_df["AP_threshold"] > -65, "pAD"] = pAD_class_labels[np.array(kmeans_2.labels_)]
            pAD_df["pAD_count"] = np.sum(kmeans_2.labels_)
        else:
            pAD_df.loc[pAD_df["AP_threshold"] > -65, "pAD"]  = pAD_class_labels[np.mod( np.array( kmeans_2.labels_ + 1 )     , 2  )]
            pAD_df["pAD_count"] = np.sum(np.mod( np.array( kmeans_2.labels_ + 1 )     , 2  ))
              
        pAD_df["num_ap_clusters"] = 2 
        
    else :
         # JUST 1 CLUSTER
         if np.nanmean(v_thresholds_all )  <= -65.0 : 
             print("check file all APs seems pAD")
             pAD_df["pAD"] = "pAD"
             pAD_df["pAD_count"] = len(v_thresholds_all)
         else : 
          pAD_df["pAD"] = "Somatic"
         pAD_df["num_ap_clusters"] = 1
         pAD_df["pAD_count"] = 0 
    
    return peak_latencies_all , v_thresholds_all  , peak_slope_all  ,   pAD_df  



########## HANDELIN FIRING PROPERTY DATA (FP)  --  FI curves


def extract_FI_x_y (path_I, path_V):
    '''
    Extracts data for Frequency-Current (FI) relationship from voltage and current recordings.

    Input:
        path_I (str): Path to the current (I) data file.
        path_V (str): Path to the voltage (V) data file.

    Returns:
        x (list): List of injected current values in picoamperes (pA) for each sweep.
        y (list): List of action potential counts for each sweep.
        v_rest (float): Average resting membrane potential (in millivolts) calculated when no current is injected.
    '''
    _, df_V = igor_exporter(path_V) # _ will be the continious wave which is no use here
    _, df_I = igor_exporter(path_I)
    
    # setting df_V and df_I to be the same dimentions
    df_I_test = df_I.iloc[:, 0:df_V.shape[1]]
    df_V_test = df_V.iloc[0:df_I.shape[0],:]
    
    #pulling and averageing all V values  when I = 0
    v_rest = np.nanmean(df_V_test[df_I_test == 0])

    y = [] #APs per sweep
    for i in range (len(df_V.columns)): #count APs in each column (sweep/ current step and append to list y)
    
        y_ = df_V.iloc[:,i].tolist()
        v_smooth, peak_locs , _ , num_peaks  = ap_finder(y_) #props cotains dict with height/width for each AP
        y.append(num_peaks) 
    
    x = [] #current injection per sweep (taking max)
    for i in range (len(df_I.columns)):
        x_ = df_I.iloc[:,i].tolist()
        #I_modes = max(set(x_), key=x_.count)
        I_modes = np.max(np.abs(x_))
        x.append(I_modes)
    
    #now df is trimmed to fit eachother should not be relevant 
    if len(x) < len(y):
        print("F file > I file        Data Error Igor Sucks") #THIS SHOULDNT HAPPEN
        
    if len(x) is not len(y):    #this may happen because when you plan a series of sweeps all are saved even if you esc before all are run
        # print (len(x), len(y))
        del x[len(y):]
        # print("inequal_adjusting")
        # print (len(x), len(y))

    return x, y,  v_rest 



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


def normalise_array_length(V_array, I_array):
    '''
    takes two arrays (1d / 2d) and sets the length (rows) to be same as shortest
    '''
    #ensure V_sweep and I_sweep are the same length
    if len(V_array) != len(I_array):
        # print(f"Length of V_sweep: {len(V_sweep)}, Length of I_sweep: {len(I_sweep)}") #V is usaly 400001 and I 400000
        V_adj = V_array[:min(len(V_array), len(I_array))]
        I_adj = I_array[:min(len(V_array), len(I_array))]
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