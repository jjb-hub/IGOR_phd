#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:11:46 2023

@author: jasminebutler

This code is to handel data collected in current clamp from Igor stored as .ibw formats:
    cleaned version of IGOR_analysis.py
    
1 PLOTING:
    FP data is plotted for a single cell PRE or after drug aplication  ###### pAD AP charecterisation!? ######
    AP data is plotted with a bra indicating aplication period
    FP_AP data is plotted on a single  plot with colour gradients showing drug aplication
    
2 ANALYSIS OF INTRINSIC PROPERTIES:
    Extension of feature_df into FP_df which will extract the following values from the FP data (each replication number) and extend the feature_df
        max_firing : x       single number of max firing reached on FI curve in Hz
        rheobased_threshold: x      pA value where liniar fit of the first three non zero values in FI curve crosses 0
        voltage_threshold: [x,x,x]       length of APs in the first I steps after threshold (unless is 1 AP then take second I step also) mV
        AP_height: [x,x,x]       length of APs in the first I steps after threshold (unless is 1 AP then take second I step also) mV
        AP_width: [x,x,x]       length of APs in the first I steps after threshold (unless is 1 AP then take second I step also) ms
        AP_slope: [x,x,x]       length of APs in the first I steps after threshold (unless is 1 AP then take second I step also) 
        FI_slope: x             slope liniar fit of the first three non zero values in FI curve (could also do k value of sigmoid fit) 
        tau_rc: [(t, V, I) , (t, V, I)]          trouple of two values from two just subthreshold I steps: time constant (ms), steady state V, I injected in pA
        sag: [(s, V, I) , (s, V, I)]             trouple of two values from two lowest I injection steps: time constant (ms), steady state V, I injected in pA
        
        IV_curve_slope: 
        pAD: T/F
        
        
"""
#%% IMPORTS

import igor.packed
#to open the .pxp
import igor.binarywave
#data science package to open .ibw
import pandas as pd
#ploting package 
from matplotlib import pyplot as plt
import glob
import numpy as np
import scipy.signal as sg
#detect APs / peaks 
import os
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path    
import timeit
import time
from scipy.ndimage import gaussian_filter1d

from scipy.signal import butter, lfilter
from scipy.signal import freqz
import scipy.io.wavfile
import scipy.signal

from scipy.optimize import curve_fit
import sys
from sklearn.linear_model import LinearRegression
import scipy.signal as sg

from scipy import stats
from collections import namedtuple
import seaborn as sns
from collections import OrderedDict

#%% BASIC FUNCTIONS:


def igor_exporter(path):
    ''' 
     Parameters
    ----------
    path: path of data, .ibw files
    
    Returns
    -------
    'point_list' : a continious points  (combining sweeps into continious wave)  
    'igor_df' : a df with each column corisponding to one sweep  

     '''
    igor_file = igor.binarywave.load(path)
    wave = igor_file["wave"]["wData"]
    igor_df = pd.DataFrame(wave)
    
    point_list = list()
    counter = len(igor_df.columns)
    
    for i in range(len(igor_df.columns)):
        temp_list = igor_df.iloc[:,i].tolist()
        point_list.extend(temp_list)
        counter = counter - 1
    
    return (point_list, igor_df)


def make_path(folder_file):
    '''
    Parameters
    ----------
    folder_file : TYPE
        DESCRIPTION.

    Returns
    -------
    path_V : string - path for V data 
    path_I : string - path for I data 
    '''
    extension_V = "Soma.ibw"
    extension_I = "Soma_outwave.ibw" 
    path_V = base_path + folder_file + extension_V
    path_I = base_path + folder_file + extension_I
    return path_V, path_I

def ap_finder(voltage_trace, smoothing_kernel = 10):
    '''
    most basic AP detector for a 1D array of voltage 
    Parameters
    ----------
    voltage_trace : 1D array of Voltage
    smoothing_kernel : int, optional, default is 10.

    Returns
    -------
    v_smooth : array of smoothed points
    peak_locs : array of peak locations 
    peak_info : dict scipy auto generatde dictionary not used in this code
    num_peaks : int the number of peaks in trace 
    '''
    v_smooth = gaussian_filter1d(voltage_trace , smoothing_kernel)
    peak_locs , peak_info = sg.find_peaks(v_smooth, height = 10 + np.average(v_smooth), distance = 2, 
                                    prominence = [20,150], width = 1, rel_height= 1)
    
    num_peaks = len(peak_locs)
    
    return  v_smooth, peak_locs , peak_info , num_peaks 

def extract_FI_x_y (path_I, path_V):
    '''

    Parameters
    ----------
    path_I : string - path to I data 
    path_V : string - path to V data 

    Returns
    -------
    x : list - I injectionin pA
    y : list - AP number  ### currently per sweep but needs to be p in Hz (APs/sec) FIX ME 
    v_rest : int - V when ni I injected

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
        I_modes = max(set(x_), key=x_.count)
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

def sigmoid_fit_0_0_trim (x,y, zeros_to_keep = 3):
    '''
    trims 0's in FI x,y data to be consistent for sigmoid fit'
    Parameters
    ----------
    x : list - I values for FI curve 
    y : list - firing frequency in Hz
    zeros_to_keep : int - default is 3 - number of 0's to keep for 0APs in a negative I injection steps'

    Returns
    -------
    x_cut : list - trimed data 
    y_cut : list - trimed data 

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
    ''' sigmoid function: 
    # https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python '''
    y = L / (1 + np.exp(-k*(x-x0))) # +b #b is the y intercept and has been removed as b = 0 for FI curves
    return (y)

def fit_sigmoid(xdata, ydata, maxfev = 5000, visualise = False): 
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

#%% DJ intergrarting ... 

def num_ap_finder(voltage_array): #not so sure why we nee dthis fun maybe DJ explain
    '''
    takes in as input full voltage array, i.e. single ibw file containing 

    Parameters
    ----------
    voltage_array : 1D array of Voltage

    Returns
    -------
    num_aps_all : int number of all present APs

    '''
    num_aps_all = []
    for idx in range(voltage_array.shape[-1]): 
        _, _, _,  num_peaks  = ap_finder(voltage_array[:, idx])
        num_aps_all += [num_peaks]
    return num_aps_all


def extract_FI_slope_and_rheobased_threshold(x,y, slope_liniar = True):
    '''
    Takes the x,y data of the FI curve and calculates rheobase (I in pA at x intercept) and the slope of the FI curve either a liniar fit or  the k of a sigmoid

    Parameters
    ----------
    x : 1D array (usualy I in pA)
    y : 1D array (usualy number of APs per sweep )
    slope_liniar : True / False if false thank than a sigmod fit will be done and the k value taken

    Returns
    -------
    slope : int
    rheobase_threshold : int (in pA)

    '''
    list_of_non_zero = [i for i, element in enumerate(y) if element!=0]
    x_threshold = np.array (x[list_of_non_zero[0]:list_of_non_zero[0]+3])# taking the first 3 non zero points to build linear fit
    y_threshold = np.array(y[list_of_non_zero[0]:list_of_non_zero[0]+3])
    
    #liniar_FI_slope
    coef = np.polyfit(x_threshold,y_threshold,1) #coef = m, b #rheobase: x when y = 0 (nA) 
    rheobase_threshold = -coef[1]/coef[0] #-b/m
    
    if slope_liniar == True:
        FI_slope_linear = coef[0]
        
    else:
    #sigmoid_FI_slope
        x_sig, y_sig = trim_after_AP_dropoff(x,y) #remove data after depolarisation block/AP dropoff
        x_sig, y_sig = sigmoid_fit_0_0_trim ( x_sig, y_sig, zeros_to_keep = 3) # remove excessive (0,0) point > 3 preceeding first AP
        x_fit, y_fit , popt = fit_sigmoid( x_sig, y_sig, maxfev = 1000000, visualise = False) #calculate sigmoid best fit #popt =  [L ,x0, k]  
        FI_slope_linear = popt[2]
        
    return FI_slope_linear, rheobase_threshold



#newer DJ 'subroutine functions'
def ap_characteristics_extractor_subroutine_linear(V_dataframe, sweep_index, main_plot = False, input_sampling_rate = 1e4 , input_smoothing_kernel = 1.5, input_plot_window = 500 , input_slide_window = 200, input_gradient_order  = 1, input_backwards_window = 100 , input_ap_fwhm_window = 100 , input_pre_ap_baseline_forwards_window = 50, input_significance_factor = 8 ):
    '''
    input: 
    V_dataframe : Voltage dataframe obtained from running igor extractor 
    sweep_index : integer representing the sweep number of the voltage file. 
    
    outputs: 
    peak_locs_corr : location of exact AP peaks 
    upshoot_locs   : location of thresholds for AP i.e. where the voltage trace takes the upshoot from   
    v_thresholds   : voltage at the upshoot poiny
    peak_heights   : voltage diff or height from threshold point to peak 
    peak_latencies : time delay or latency for membrane potential to reach peak from threshold
    peak_slope     : rate of change of membrane potential 
    peak_fwhm      : DUMMIES for now, change later: 
   
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
    inner_loop_plot  = False
    sampling_rate    = input_sampling_rate
    sec_to_ms        = 1e3 

    ## Outputs on Peak Statistics 

    peak_locs_corr  = []            #  location of peaks *exact^
    upshoot_locs    = []            #  location of voltage thresholds 
    v_thresholds    = []            #  value of voltage thresholds in mV
    peak_heights    = []            #  peak heights in mV, measured as difference of voltage values at peak and threshold 
    peak_latencies  = []            #  latency or time delay between peak and threshold points
    peak_fw         = []            #  width of peak at half maximum
    peak_slope      = []            #  peak height / peak latency

    # Make np array of dataframe

    df_V_arr = np.array(V_dataframe)
    sweep_idx = sweep_index

    # Get rough estimate of peak locations 

    v_array = df_V_arr[:,sweep_idx]
    v_smooth, peak_locs , peak_info , num_peaks  = ap_finder(v_array) 

    for peak_idx in range(len(peak_locs)) : 


        # Correct for peak offsets 

        v_max  = np.max(v_array[peak_locs[peak_idx] - ap_backwards_window : peak_locs[peak_idx] + pre_ap_baseline_forwards_window]) 
        peak_locs_shift = ap_backwards_window - np.where(v_array[peak_locs[peak_idx] - ap_backwards_window: peak_locs[peak_idx] + pre_ap_baseline_forwards_window] == v_max)[0][0]

        peak_locs_corr += [ peak_locs[peak_idx] - peak_locs_shift ]   


        if peak_idx == 0 : 
            start_loc = slide_window
            
        else:
            start_loc = peak_locs_corr[peak_idx - 1 ]  + slide_window
        end_loc = peak_locs_corr[peak_idx]

        # Get AP slice

        v_temp = v_array[end_loc - ap_backwards_window: end_loc ]

        # Define x, y for linear regression fit

        y_all = gaussian_filter1d(v_temp   , smoothing_kernel)        # entire AP slice 
        y     = gaussian_filter1d(v_temp[0:pre_ap_baseline_forwards_window]   , smoothing_kernel)        # pre AP baseline

        x = np.arange(0, len(y))
        x_all = np.arange(0, len(y_all))
        res = stats.linregress(x, y)                # lin reg fit
        # residual error fit 
        res_error = abs(y_all - res.intercept + res.slope*x_all) 
        res_err_mean , res_err_std = np.mean(res_error[0:pre_ap_baseline_forwards_window]) , np.std(res_error[0:pre_ap_baseline_forwards_window])
        
        upshoot_loc_array = np.where( res_error > res_err_mean + significance_factor*res_err_std ) [0]# define an array first in case its empty
        
        # print(upshoot_loc_array)
        if len(upshoot_loc_array) ==  0:
            
            return [np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan] #returning nan as was unable to find upshoot from linear method
        
        else:
            upshoot_loc = upshoot_loc_array[0] #location of first AP upshoot
            
    
        if inner_loop_plot: 
            plt.plot(v_temp)
            plt.plot(upshoot_loc, v_temp[upshoot_loc], '*')
            plt.show()

        
        # Calculate AP metrics here 
        upshoot_locs += [    peak_locs_corr[peak_idx] - ap_backwards_window + upshoot_loc ]
        v_thresholds += [v_array[upshoot_locs[peak_idx]]]


        peak_heights    += [  v_array[peak_locs_corr[peak_idx]]  - v_array[upshoot_locs[peak_idx]]   ]
        peak_latencies  += [ sec_to_ms * (peak_locs_corr[peak_idx] - upshoot_locs[peak_idx])  / sampling_rate ]
        peak_slope      += [ peak_heights[peak_idx]  / peak_latencies[peak_idx] ] 
        
        # Calculation of AP widths : use a linear fit for now  - really an exponential fit be the best, and since we use a linear fit, not a problem to calculate full width rather than half width  

        upshoot_x = peak_locs_corr[peak_idx] - ap_backwards_window + upshoot_loc 
        upshoot_y = v_array[upshoot_locs[peak_idx]]

        peak_x    = peak_locs_corr[peak_idx]
        peak_y    = v_array[peak_locs_corr[peak_idx]]

        return_x_array   = np.where( v_array <= upshoot_y  )[0] 
        return_x_ = return_x_array[return_x_array > peak_x]
        return_x = return_x_[0]
        return_y  =  v_array[return_x]


        
        peak_fw       += [ sec_to_ms * (return_x  - upshoot_x)  / sampling_rate ]

                       
    # Plotting Stuff
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

def ap_characteristics_extractor_main(V_dataframe, critical_num_spikes = 4): 
    '''
    Main function that calls the ap_characteristics_extractor_subroutine_linear. Separated into main for future flexibility/adaptability. 

    Input  : V_dataframe 

    Output : output of lists as generated by subroutine iterated over chosen sweeps
    '''

    v_array       = np.array(V_dataframe)
    num_aps_all   = np.array(num_ap_finder(v_array))
    sweep_indices = None 
    min_spikes    = critical_num_spikes 
    # Find which trace has first spiking and if that is more than min_spikes just take the first, else take first two sweeps with spiking
    first_spike_idx  = np.where(num_aps_all > 0)[0][0]
    num_first_spikes = num_aps_all[first_spike_idx] 

    if num_first_spikes >= min_spikes : 
        sweep_indices  = [first_spike_idx]
    else:
        sweep_indices  = [first_spike_idx, first_spike_idx + 1]
    sweep_indices = np.array(sweep_indices)
    sweep_indices = sweep_indices[sweep_indices < v_array.shape[-1]]
    # Initialise lists 
    peak_latencies_all  = [] 
    v_thresholds_all    = [] 
    peak_slope_all      = []
    peak_locs_corr_all  = []
    upshoot_locs_all    = []
    peak_heights_all    = []
    peak_fw_all         = [] 
    for sweep_num in sweep_indices: 
        # print('\n')
        # print('\n')
        # print('Analysing for sweep %s' % sweep_num) 

        peak_locs_corr_, upshoot_locs_, v_thresholds_, peak_heights_ ,  peak_latencies_ , peak_slope_ , peak_fw_  =  ap_characteristics_extractor_subroutine_linear(V_dataframe, sweep_num)
    
        peak_locs_corr_all += peak_locs_corr_
        upshoot_locs_all   += upshoot_locs_
        peak_latencies_all += peak_latencies_
        v_thresholds_all   += v_thresholds_
        peak_slope_all     += peak_slope_
        peak_heights_all   += peak_heights_
        peak_fw_all        += peak_fw_

    return peak_latencies_all  , v_thresholds_all  , peak_slope_all  , peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all

def steady_state_value(voltage_trace, current_trace, step_current_val,  avg_window = 0.5):

    ''' 
    Takes in the following 
    voltage_trace:  a single voltage trace so 1d time series 
    current_trace: its corresponding current trace (again 1d timeseries) 
    step_current_val: step current value so a scalar float 
    avg_window: what fraction of step current duration do we average? 

    Returns: 
    asym_current: the steady state value from any step current injection
    hyper : boolen value     indicating whether step current is hyperpolarising  or not 
    first_current_point = first timeframe (so integer) of step current injection 
    '''

    if step_current_val >= 0: #note that 0pA of I injected is not hyperpolarising 
        hyper = False
        first_current_point = np.where(current_trace == np.max(current_trace) )[0][0] 
        last_current_point = np.where(current_trace == np.max(current_trace) )[0][-1]

    elif step_current_val < 0 : 
        hyper = True 
        first_current_point = np.where(current_trace == np.min(current_trace) )[0][0] 
        last_current_point = np.where(current_trace == np.min(current_trace) )[0][-1]


    current_avg_duration = int(avg_window*(last_current_point - first_current_point))

    asym_current  = np.mean(voltage_trace[ last_current_point - current_avg_duration : last_current_point  ])
    

    return asym_current , hyper  , first_current_point, last_current_point



def calculate_max_firing(voltage_array, input_sampling_rate=1e4): 
    '''
    Function to calculate max firing of given voltage traces in Hz
    Input : 
            voltage_array : 2d array containing sweeps of voltage recordings / traces for different step currents

    Output: 
            max_firing : float in Hz 
    '''
    num_aps_all  = np.array(num_ap_finder(voltage_array))                               # get num aps from each sweep
    index_max = np.where(num_aps_all == max(num_aps_all)) [0][0]    # get trace number with max aps 
    sampling_rate = input_sampling_rate
    _ , peak_locs , _ , _   =  ap_finder(voltage_array[:, index_max ] , smoothing_kernel = 10)
    
    return np.nanmean(sampling_rate / np.diff(peak_locs)) 



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


    Returns : tuple in the format : ( tau (in ms) , asymptotic steady-state voltage , step current)
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
    # Define Named Tuple 
    tau_tuple  = namedtuple('Tau', ['val', 'steady_state', 'current_inj'])
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

                tau_temp = tau_tuple(val = tau , steady_state = asym_current, current_inj=step_current_val)

                tau_array.append(tau_temp)
        
        counter += 1

    if num_tau_analysed_counter == 0 : 
        return tau_tuple(val = np.nan , steady_state = np.nan, current_inj=np.nan)

    return tau_array


def sag_current_analyser(voltage_array, current_array, input_step_current_values,input_current_avging_window = 0.5): 

    '''
    Function to calculate sag current from voltage and current traces under hyper polarising current steps
    Takes in: 
    volatge array : 2d array containing sweeps of voltage recordings / traces for different step currents
    current array : 2d array containing sweeps of current recordings / traces for different step currents
    input_step_current_values : list containing different step currents 
    input_current_avging_window : float / fraction representating what portion of step current duration is used for calculating 
    
    Returns : tuple 
    sag_current_all : List containing tuples in the form (Sag, steady state voltage , step current)
    '''
    # sag : 0, 1 : works with the hyperpolarising current injection steps and 0 and 1 indices correspond to
    # the 1st two (least absolute value) step current injections. Later expand for positive current inj too? 
    sag_indices = [0,1]
    sag_current_all =  [] 
    sag_tuple = namedtuple('Sag', ['val', 'steady_state', 'current_inj'])
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
        

            asym_sag_current = np.mean(voltage_array[  last_min_current_point - step_current_avg_window: last_min_current_point, sag_idx])
            min_sag_current_timepoint = np.where( voltage_array[:,sag_idx]  ==  min(voltage_array[:,sag_idx]  ) )[0][0] 
            min_sag_current =  min(voltage_array[:,sag_idx]  )
            max_sag_current = np.mean(voltage_array[0 : first_min_current_point  - 1   , sag_idx] )

            sag_current = (asym_sag_current - min_sag_current)  / (  max_sag_current  - asym_sag_current )

            

            sag_current_temp = sag_tuple(val=sag_current, steady_state=asym_current , current_inj=input_step_current_values[sag_idx])

        elif num_peaks > 0: 
            print('Spike found in Sag analysis, skipping')
            sag_current_temp = sag_tuple(val= np.nan, steady_state=np.nan, current_inj = np.nan)
        
        # Append Value to existing named tuple
        sag_current_all.append(sag_current_temp) 

    return sag_current_all

#%% TESTING FUNCTIONS

# path = make_path ('    ')
# _, dfVI = igor_exporter()

def plot_single_df_I_or_V_by_col (df_to_plot, y_label = 'V (mV) or I injected (pA)'): 
    '''
    Opens a single figure and plots each column on top of eachother (nice for I FP data)
    Parameters
    ----------
    df_to_plot : df where sweeps are columns
    y_label : string -   'V (mV)' or 'I injected (pA)'

    Returns
    -------
    plot

    '''
    fig, ax = plt.subplots(1,1, figsize = (10, 8))
    # for i in df_to_plot.columns:
    for i in [ 14]:
        print('I is = ', i)
        x_ = df_to_plot.iloc[:,i].tolist() #create list of values for each col 
        #quick plot all I steps free of APs
        y_to_fit = np.arange(len(x_)) * 0.0001 #sampeling at 10KHz will give time in seconds
        ax.plot(y_to_fit, x_) 
        ax.set_xlabel( "Time (s)", fontsize = 15)
        ax.set_ylabel( y_label, fontsize = 15)
        
    return

def single_drug_aplication_visualisation(feature_df,  color_dict, cell = 'DRD230104b', drug = 'TCB2'):
    '''
    Generates cell + '_' + drug + '_sweeps.pdf' with each sweep from IGOR displayed on a single page, colour change at drug aplication (voltage data )

    Parameters
    ----------
    feature_df : df including all factors needed to distinguish data 
    color_dict : dict with a colour for each drug to be plotted

    Returns
    -------
    None.

    '''
    
    start = timeit.default_timer()
    with PdfPages(cell + '_' + drug + '_sweeps.pdf') as pdf:
        
        aplication_df = feature_df[feature_df.data_type == 'AP'] #create sub dataframe of aplications
        
        for row_ind, row in aplication_df.iterrows():  #row is a series that can be called row['colname']
            if row['cell_ID'] == cell and row['drug'] == drug:
                path_V, path_I = make_path(row['folder_file'])
                print (path_V)
                _ , y_df = igor_exporter(path_V)
                
                drug_in = row['drug_in']
                drug_out = row['drug_out']
                
                for col in  y_df.columns:
                    fig, axs = plt.subplots(1,1, figsize = (10, 5))
                    y = y_df[col]
                    x = np.arange(len(y)) * 0.0001 #sampeling at 10KHz frequency if igor?
                    sweep = col+1
                    
                    if drug_in <= sweep <= drug_out:
                        axs.plot(x,y, c = color_dict[row['drug']], lw=1)
                    else:
                        axs.plot(x,y, c = color_dict['PRE'], lw=1)
    
                    axs.set_xlabel( "Time (s)", fontsize = 15)
                    axs.set_ylabel( "Membrane Potential (mV)", fontsize = 15)
                    axs.set_title(row['cell_ID'] + ' '+ row['drug'] +' '+ " Aplication" + " (" + str(row['aplication_order']) + ")", fontsize = 25)
                    pdf.savefig(fig)
                    plt.close("all") 

    stop = timeit.default_timer()
    print('Time: ', stop - start)      
    return


#%% POSIBLY REDUNDANT 

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]


def reject_outliers(data, m = 2.):
    '''reject outliers outside 2 SD of the median '''
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

#%% PLOTTING FUNCS: AP, FI, FI_AP


def drug_aplication_visualisation(feature_df,  color_dict):
    '''
    Plots continuious points (sweeps combined, voltage data)
    Generates 'drug_aplications_all_cells.pdf' with a single AP recording plot per page, drug aplication by bath shown in grey bar

    Parameters
    ----------
    feature_df : df including all factors needed to distinguish data 
    color_dict : dict with a colour for each drug to be plotted

    Returns
    -------
    None.

    '''
  
    start = timeit.default_timer()

    with PdfPages('drug_aplications_all_cells.pdf') as pdf:
        
        aplication_df = feature_df[feature_df.data_type == 'AP'] #create sub dataframe of aplications
        
        for row_ind, row in aplication_df.iterrows():  #row is a series that can be called row['colname']
        
            path_V, path_I = make_path(row['folder_file'])
            y, df_y = igor_exporter(path_V) # df_y each sweep is a column
            fig, axs = plt.subplots(1,1, figsize = (10, 5))
            
            x_scaler_drug_bar = len(df_y[0]) * 0.0001 # multiplying this  by drug_in/out will give you the point at the end of the sweep in seconds
            x = np.arange(len(y)) * 0.0001 #sampeling at 10KHz will give time in seconds
            
            axs.plot(x,y, c = color_dict[row['drug']], lw=1)
            # plt.ylim(-65, -35)
            
            axs.axvspan((int((row['drug_in'])* x_scaler_drug_bar) - x_scaler_drug_bar), (int(row['drug_out'])* x_scaler_drug_bar), facecolor = "grey", alpha = 0.2) #drug bar shows start of drug_in sweep to end of drug_out sweep 
            axs.set_xlabel( "Time (s)", fontsize = 15)
            axs.set_ylabel( "Membrane Potential (mV)", fontsize = 15)
            axs.set_title(row['cell_ID'] + ' '+ row['drug'] +' '+ " Aplication" + " (" + str(row['aplication_order']) + ")", fontsize = 25)
            pdf.savefig(fig)
            plt.close("all")
    stop = timeit.default_timer()
    print('Time: ', stop - start)  
    return

def plot_all_FI_curves(feature_df,  color_dict):
    '''
    Generates 'FI_curves_all_cells.pdf' with all FI curves for a single cell polotted on  one page , coloured to indicate drug

    Parameters
    ----------
    feature_df : df including all factors needed to distinguish data 
    color_dict : dict with a colour for each drug to be plotted

    Returns
    -------
    None.

    '''

    start = timeit.default_timer()
    
    with PdfPages('FI_curves_all_cells.pdf') as pdf:
        
        FI_df = feature_df[feature_df.data_type == 'FP'] #create sub dataframe of firing properties
        
        for cell_ID_name, cell_df in FI_df.groupby('cell_ID'): # looping for each cell 
            fig, ax = plt.subplots(1,1, figsize = (10,5)) #generate empty figure for each cell to plot all FI curves on
            
            for row_ind, row in cell_df.iterrows(): 
                
              path_V, path_I = make_path(row['folder_file'])  #get paths
              x, y, v_rest = extract_FI_x_y (path_I, path_V)
              
              #add V_rest to  label plot 
              ax.plot(x, y, lw = 1, label = row['drug'] + ' ' + str(row['aplication_order']) + " " + str(np.round(v_rest)) + " mV RMP", c = color_dict[row['drug']])

              
              # handles, labels = plt.gca().get_legend_handles_labels() #remove duplicate labels in legend #FIX ME 
              # by_label = dict(zip(labels, handles))
              # ax.legend(by_label.values(), by_label.keys()) https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend

            ax.set_xlabel( "Current (nA)", fontsize = 15)
            ax.set_ylabel( "AP frequency", fontsize = 15)
            ax.set_title( cell_ID_name + " FI curves", fontsize = 20)
            ax.legend(fontsize = 10)
            pdf.savefig(fig)
            plt.close("all") 
                
        stop = timeit.default_timer()
        print('Time: ', stop - start)    
    return 


def plot_FI_AP_curves(feature_df):
    '''
    Generates 'aplication_FI_curves_all_cells.pdf' with a plot of all FI_AP curves for a single cell on one plot per page, colour gradient after startof drug AP

    Parameters
    ----------
    feature_df : df including all factors needed to distinguish data 

    Returns
    -------
    None.

    '''

    start = timeit.default_timer()
    
    color1 =  '#0000EE' #"#8A5AC2" #'#AAFF32' #blue
    color2 = '#FF4040'  #"#3575D5" #'#C20078' #red
    with PdfPages('aplication_FI_curves_all_cells.pdf') as pdf:
        
        FI_AP_df = feature_df[feature_df.data_type == 'FP_AP'] #create sub dataframe of firing properties
        
        for cell_ID_name, cell_df in FI_AP_df.groupby('cell_ID'): # looping for each cell 
            fig, ax = plt.subplots(1,1, figsize = (10,5)) #generate empty figure for each cell to plot all FI curves on
            color = get_color_gradient(color1, color2, len(cell_df['replication_no']))
            
            for row_ind, row in cell_df.iterrows(): 
                
              path_V, path_I = make_path(row['folder_file'])  #get paths
              x, y, v_rest = extract_FI_x_y (path_I, path_V)
              
              if row['drug'] == 'PRE':
                  plt_color = color[0]
              else:
                  plt_color = color[row['replication_no']-1]
                  
              ax.plot(x, y, lw = 1, label = str(row['replication_no']) + " " + row['drug'] + str(np.round(v_rest)) + ' mV RMP', color = plt_color)

            ax.set_xlabel( "Current (nA)", fontsize = 15)
            ax.set_ylabel( "AP frequency", fontsize = 15)
            ax.set_title( cell_ID_name + " FI curves", fontsize = 20)
            ax.legend(fontsize = 10)
            pdf.savefig(fig)
            plt.close("all") 
                
        stop = timeit.default_timer()

        print('Time: ', stop - start) 
        
    return 



#%% LOAD FEATURE_DF + START INPUTS

color_dict = {"PRE":"black", "CONTROL": 'grey', "TCB2":'green', "DMT":"teal", "PSIL":"orange", "LSD":"purple", "MDL":'blue'}

# color_dict = OrderedDict([("PRE","black"), 
#                           ("CONTROL", 'grey'), 
#                           ("TCB2",'green'), 
#                           ("DMT","teal"), 
#                           ("PSIL","orange"), 
#                           ("LSD","purple"),  
#                           ("MDL",'blue')]
# )

base_path = "/Users/jasminebutler/Desktop/PatchData/"

feature_df = pd.read_excel('/Users/jasminebutler/Desktop/feature_df_py.xlsx') #create cell_directory_df



#%% RUN PLOTS

drug_aplication_visualisation(feature_df, color_dict) # generates PDF of drug aplications

plot_all_FI_curves(feature_df,  color_dict)  # generates PDF with all FI curves for single cell labed with drug and aplication order #### MAKE HZ NOT APs per sweep also isnt it in pA not nA??

plot_FI_AP_curves(feature_df) #generated PDF with FI-AP for each cell


#%%   EXPAND FEATURE_DF
feature_df_ex = feature_df.copy()



def apply_group_by_funcs(df, groupby_cols, handleFn):
    res_dfs_li = []
    for group_info, group_df in df.groupby(groupby_cols):
        
        # print("group_info:", group_info, "Group len:", len(group_df))
        res_df = handleFn(group_info, group_df)
        res_dfs_li.append(res_df)
        
    new_df = pd.concat(res_dfs_li)
    return new_df


def _handleFile(row): #takes a row of the df (a single file) and extractes values based on the data type  FP or AP then appends values to df
    # print(row.index)
    # print(row.folder_file)# "- num entries:", row)
    row = row.copy()
    if row.data_type in ["FP", "FP_AP"]:
        print(row.folder_file)
        # row["new_data_type"] = ["Boo", 0, 1 ,2] #creating a new column or filling a pre existing column with values or list of values
        
        #data files to be fed into extraction funcs
        path_V, path_I = make_path(row.folder_file)
        V_list, V_df = igor_exporter(path_V)
        I_list, I_df = igor_exporter(path_I)
        V_array , I_array = np.array(V_df) , np.array(I_df)
        
        
        row["max_firing"] = calculate_max_firing(V_array)
        
        x, y, v_rest = extract_FI_x_y (path_I, path_V) # x = current injected, y = number of APs, v_rest = the steady state voltage
        FI_slope, rheobase_threshold = extract_FI_slope_and_rheobased_threshold(x,y, slope_liniar = True) ## FIX ME some values are negative! this should not be so better to use slope or I step? Needs a pAD detector 
        row["rheobased_threshold"] = rheobase_threshold
        row["FI_slope"] = FI_slope
        
        

        peak_latencies_all  , v_thresholds_all  , peak_slope_all  , peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all = ap_characteristics_extractor_main(V_df)
        row["voltage_threshold"] = v_thresholds_all 
        row["AP_height"] = peak_heights_all
        row["AP_width"] = peak_fw_all
        row["AP_slope"] = peak_slope_all 

        #not yet working    #UnboundLocalError: local variable 'last_current_point' referenced before assignment
        #extract_FI_x_y has been used differently by DJ check this is correct  # step_current_values  == x
        tau_all         =  tau_analyser(V_array, I_array, x, plotting_viz= False, analysis_mode = 'max')
        sag_current_all =  sag_current_analyser(V_array, I_array, x)

        row["tau_rc"] = tau_all
        row["sag"] = sag_current_all
        
        
    elif row.data_type == "AP":
        # row["new_data_type"] = ["Boo", 0, 1 ,2]
        pass
    else:
        raise NotADirectoryError(f"Didn't handle: {row.data_type}")
    return row

def _handleCellID(cell_id_drug, cell_df):
    cell_id, drug = cell_id_drug
    # fp_idx = cell_df.data_type == "FP" #indexes of FP data in cell_df
    # cell_df.loc[fp_idx, "new_col"] = calcValue()

    return cell_df


def loopCombinations(df):
    og_columns = df.columns.copy() #origional columns #for ordering columns
    df['mouseline'] = df.cell_ID.str[:3]
    # print("Columns: ", df.columns[-40:])
    # df = pd.read_excel(r'E:\OneDrive - Floating Reality\analysis\feature_df_py.xlsx')
    df = df.apply(_handleFile, axis=1) #Apply a function along an axis (rows = 1) of the DataFrame
    # display(df)
    combinations = [
                    (["cell_ID", "drug"], _handleCellID), #finding all combination in df and applying a function to them #here could average and plot or add to new df for stats
                    #(["cell_type", "drug"], _poltbygroup) #same as df.apply as folder_file is unique for each row
    ]
    # for col_names, handlingFn in combinations:
    #     df = apply_group_by_funcs(df, col_names, handlingFn) #handel function is the generic function to be applied to different column group by
    
    #ORDERING DF internal: like this new columns added will appear at the end of the df in the order they were created in _handelfile()
    all_cur_columns = df.columns.copy()
    new_colums_set = set(all_cur_columns ) - set(og_columns) # Subtract mathematical sets of all cols - old columns to get a set of the new columns
    new_colums_li = list(new_colums_set)
    reorder_cols_li =  list(og_columns) + new_colums_li # ['mouseline'] + ordering
    df_reordered = df[reorder_cols_li]    
    
    return df_reordered
#RUN
feature_df_expanded_raw = loopCombinations(feature_df_ex)

# feature_df_ex_tau.to_csv('middle_TCB_data.xls')

#%% stats/plotting 

def apply_group_by_funcs(df, groupby_cols, handleFn): #creating a list of new values and adding them to the existign df
    res_dfs_li = [] #list of dfs
    for group_info, group_df in df.groupby(groupby_cols):
        
        res_df = handleFn(group_info, group_df, color_dict)
        res_dfs_li.append(res_df)
        
    new_df = pd.concat(res_dfs_li)
    return new_df



def _getstats_FP(mouseline_drug_datatype, df, color_dict):
    
    mouse_line, drug, data_type = mouseline_drug_datatype
    
    if data_type == 'AP': #if data type is not firing properties (FP then return df)
            return df
        
    df = df.copy()
    #for entire mouseline_drug combination
    df['mean_max_firing'] = df['max_firing'].mean() #dealing with pd.series a single df column
    df['SD_max_firing'] = df['max_firing'].std()
    
    df['mean_FI_slope'] = df['FI_slope'].mean() #dealing with pd.series a single df column
    df['SD_FI_slope'] = df['FI_slope'].std()
    
    df['mean_rheobased_threshold'] =df['rheobased_threshold'].mean()
    
    #AP Charecteristics i.e. mean/SD e.c.t. for a single file (replications not combined consider STATISTICAL IMPLICATIONS FORPLOTTING)
    df['mean_voltage_threshold_file'] = df.voltage_threshold.apply(np.mean) #apply 'loops' through the slied df and applies the function np.mean
    df['SD_voltage_threshold_file'] = df.voltage_threshold.apply(np.std)
    
    df['mean_AP_height_file'] = df.AP_height.apply(np.mean)
    
    df['mean_AP_slope_file'] = df.AP_slope.apply(np.mean)
    
    df['mean_AP_width'] = df.AP_width.apply(np.mean)
    
    return df

def _plotwithstats_FP(mouseline_datatype, df, color_dict):
    
    
    mouse_line, data_type = mouseline_datatype
    order = list(color_dict.keys()) #ploting in order of dict keys
    
    if data_type == 'AP': #if data type is not firing properties (FP then return df)
        return df
    if data_type == 'FP_AP': #if data type is not firing properties (FP then return df)
        return df
    
    # generating plot by single file values e.g. max_firing or mean_voltage_threshold_file
    
    factors_to_plot = ['max_firing', 'rheobased_threshold', 'FI_slope', 'mean_voltage_threshold_file', 'mean_AP_height_file', 'mean_AP_slope_file', 'mean_AP_width']
    names_to_plot = ['_max_firing_Hz', 'rheobased_threshold_pA' , '_FI_slope_linear', '_voltage_threshold_mV', '_AP_height_mV', '_AP_slope', '_AP_width_ms']
    
    for _y, name in zip(factors_to_plot, names_to_plot):
    
        sns_barplot_swarmplot (df, order, mouse_line, _x='drug', _y=_y, name = name)
    

    plt.close("all") #close open figures
    
    return df 

def sns_barplot_swarmplot (df, order, mouse_line, _x='drug', _y='max_firing', name = '_max_firing_Hz'):
    '''
    Generates figure and plot and saves to patch_daddy_output/
    '''
    fig, axs = plt.subplots(1,1, figsize = (20, 10))
    sns.barplot(data = df, x=_x, y=_y,  order=order, palette=color_dict, capsize=.1, 
                             alpha=0.8, errcolor=".2", edgecolor=".2" )
    sns.swarmplot(data = df,x=_x, y=_y, order=order, palette=color_dict, linewidth=1, linestyle='-')
    axs.set_xlabel( "Drug applied", fontsize = 20)
    axs.set_ylabel( name, fontsize = 20)
    axs.set_title( mouse_line + name + '  (CI 95%)', fontsize = 30)
    
    fig.savefig('patch_daddy_output/' + mouse_line + name + '.pdf')
    
    return

def loopCombinations_stats(df):
    #create a copy of file_folder column to use at end of looping to restore  origional row order !!! NEEDS TO BE DONE
    
    #keepingin mind that the order is vital  as the df is passed through againeach one
    combinations = [
                    (["mouseline", "drug", "data_type"], _getstats_FP), #stats_df to be fed to next function mouseline 
                    (["mouseline",  "data_type"], _plotwithstats_FP) #finding all combination in df and applying a function to them #here could average and plot or add to new df for stats
                    # (["cell_ID", "drug", "data_type"], _plotstats_FP)
    ]
    
    # pdf = PdfPages('patch_daddy_output.pdf') # open pdf for plotsto be saved in
    # pdf.savefig(fig)
    # pdf.close()

    for col_names, handlingFn in combinations:

    
        df = apply_group_by_funcs(df, col_names, handlingFn) #note that as each function is run the updated df is fed to the next function
    
    
    
    return df

#RUN
feature_df_expanded_stats = loopCombinations_stats(feature_df_expanded_raw)

#REMI: the df that i gave you is the current output here you can use it to rerun from this cell (currently i would like you to help me to understand how ot moidularise this cell as I am beginning to herad code shit) .... 
# but actualy values you will not be able to work on until the OG PatchData file is online in inputs to code

#%% EXPANDIN DF ALTERNATE SOLOUTION //// insecured to delete stats  plot structure
def apply_group_by_funcs(df, groupby_cols, handleFn): #creating a list of new values and adding them to the existign df
    res_dfs_li = [] #list of dfs
    for group_info, group_df in df.groupby(groupby_cols):
        
        # print("group_info:", group_info, "Group len:", len(group_df))
        res_df = handleFn(group_info, group_df)
        res_dfs_li.append(res_df)
        
    new_df = pd.concat(res_dfs_li)
    return new_df



def _getstats(mouse_line_drug, df):
    mouse_line, drug = mouse_line_drug
    # fp_idx = cell_df.data_type == "FP" #indexes of FP data in cell_df
    # cell_df.loc[fp_idx, "new_col"] = calcValue()
    df = df.copy()
    df['mean_max_firing'] = df['max_firing'].mean() #dealing with pd.series a ingld df column
    df['mean_voltage_thresh'] = df.voltage_threshold.apply(np.mean) #apply 'loops' through the slied df and applies the function np.mean
    return df


    # list_of_all_cell_IDs = list(df['cell_ID'])
    # unique_mouse_line =  [list_of_all_cell_IDs[i][0:3] for i in range(len(list_of_all_cell_IDs))]
    # unique_mouse_line = np.unique(unique_mouse_line)
    
    # unique_drug_list = [] #do same as above 
    
    # unique_condition_list = [unique_mouse_line, unique_drug_list]
    
    # #  dj pseudo
    # for mouseline in unique_mouse_line:
    #     df_ = df[df['cell_ID'].str[:3] == mouseline]
        
        
    #     #[x for l in list for x in l]
        
    #     # once df is sliced we perwrform some plots and stats
    #     # look at voltage threshold
    #     list_list_voltage_threshold =[*df_['voltage_threshold']]
    #     list_voltage_threshold = []
    #     for idx in range(len(list_list_voltage_threshold )):
    #         list_voltage_threshold  += list_list_voltage_threshold[idx]
        
    #         plt.plot(list_voltage_threshold, label = mouseline)
    #     plt.legend()
    #     plt.show()
            
        
    #     return None 
    
def loopCombinations_stats(df):
    
    combinations = [
                    (["mouseline", "drug"], _getstats), #stats_df to be fed to next function 
                    (["mouseline", "drug"], _plotwithstats)#finding all combination in df and applying a function to them #here could average and plot or add to new df for stats
                    #(["cell_type", "drug"], _poltbygroup) #same as df.apply as folder_file is unique for each row
    ]
    
    # df_stat = df.copy()
    for col_names, handlingFn in combinations:
        
#here would be adding to the df and 
        # #creating new df roto be fed to next func
        # df_stat = apply_group_by_funcs(df_stat, col_names, handlingFn) #handel function is the generic function to be applied to different column group by
        
        #expand the df expanded
        df = apply_group_by_funcs(df, col_names, handlingFn) #handel function is the generic function to be applied to different column group by

  
    
    return df

#RUN
feature_df_ex_tau = loopCombinations(feature_df_ex)

# feature_df_ex_tau.to_csv('middle_TCB_data.xls')


# #MAXs soloution
# #add emptyy columns to append extracted values to FP data 
# feature_df = feature_df.assign(max_firing=np.NaN, 
#                                rheobased_threshold = np.NaN, 
#                                voltag_threshold = np.NaN, 
#                                AP_height = np.NaN, 
#                                AP_width = np.NaN, 
#                                AP_slope = np.NaN, 
#                                FI_slope = np.NaN, 
#                                tau_rc = np.NaN, 
#                                sag = np.NaN)



# # combinations = feature_df.folder_file.unique()

# combinations = product(feature_df.data_type.unique(),
#                 feature_df.drug.unique())


# for data_type, drug in combinations:
    
#     #take slice of data 
#     sliced_df= feature_df.loc[(feature_df.data_type== 'FP') &
#                               (feature_df.drug == 'PRE')]
    
#     # sliced_df= feature_df.loc[(feature_df.folder_file== file ) ] # single row
    
#     print(sliced_df.data_type)

#     if sliced_df.data_type[0] == 'FP' :
#         print('hi')
    
    
#     or 'FP_AP':
        
#         path = 
#         func
#         folder_file = ' '  
#         out = []
#         ...
        
#     elif data_type == 'AP':
#         continue


#     # call on a specific set of rows and set to whatever 
#     feature_df = feature_df.loc[(feature_df.data_type==a) &
#                    (feature_df.drug == b), 'max_firing'] = max_firing

              
#%%TEST PATHS / FUNCS

#FP tester paths 
path_I =  '/Users/jasminebutler/Desktop/PatchData/JJB221230/t15Soma_outwave.ibw'
path_V =  '/Users/jasminebutler/Desktop/PatchData/JJB221230/t15Soma.ibw'
#AP tester paths
path_I =  '/Users/jasminebutler/Desktop/PatchData/JJB230222/t17Soma_outwave.ibw'
path_V = '/Users/jasminebutler/Desktop/PatchData/JJB230222/t17Soma.ibw'

#to plot I steps or V responce to steps all in 1 
_, dfV = igor_exporter(path_V)
_, dfI = igor_exporter(path_I)

plot_single_df_I_or_V_by_col (dfV.iloc[:, 0:4], y_label = 'V (mV)') ### fix me : can only section df from 0:x otherwise IndexError: single positional indexer is out-of-bounds
plot_single_df_I_or_V_by_col (dfI, y_label = 'I injected (pA)')

