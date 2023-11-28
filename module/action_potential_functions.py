# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:30:00 2023

@author: Debapratim Jana, Jasmine Butler
"""


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



def ap_characteristics_extractor_subroutine_derivative(V_dataframe, sweep_index, main_plot = False, input_sampling_rate = 1e4 , input_smoothing_kernel = 1.5, input_plot_window = 500 , input_slide_window = 200, input_gradient_order  = 1, input_backwards_window = 100 , input_ap_fwhm_window = 100 , input_pre_ap_baseline_forwards_window = 50, input_significance_factor = 8 ):
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
    inner_loop_plot  = True
    sampling_rate    = input_sampling_rate
    sec_to_ms        = 1e3 
    ap_spike_min_height =  0 # if value if less than 0 mV - very unhealthy spike  

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
        
        #print(v_temp) 
        v_peak   = np.where(v_temp == np.max(v_temp))[0][0]

        
        if len(upshoot_loc_array) > 0 : 
            upshoot_loc  = np.where(x_  <  0)[0][0]
        elif len(upshoot_loc_array) == 0 :
            return [] ,   [] ,  []  , [] ,  [] ,  [] , []    

        
        
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
            
        peak_fw       += [ sec_to_ms * (return_x  - first_x)  / sampling_rate ]

            
        
        
                       
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
        
        if method == "linear": 
            peak_locs_corr_, upshoot_locs_, v_thresholds_, peak_heights_ ,  peak_latencies_ , peak_slope_ , peak_fw_  =  ap_characteristics_extractor_subroutine_linear(V_dataframe, sweep_num)
        elif method == "derivative":
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



def calculate_max_firing(voltage_array, input_sampling_rate=1e4): #HARD CODE sampeling rate
    '''
    Function to calculate max firing of given voltage traces in Hz
    Input : 
            voltage_array : 2d array containing sweeps of voltage recordings / traces for different step currents
    Output: 
            max_firing : float in Hz 
    '''
    num_aps_all  = np.array(num_ap_finder(voltage_array))           # get num aps from each sweep
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
    # tau_tuple  = namedtuple('Tau', ['val', 'steady_state', 'current_inj'])

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

                # tau_temp = tau_tuple(val = tau , steady_state = asym_current, current_inj=step_current_val)
                tau_temp = [tau, asym_current, step_current_val, v_resting_membrane]

                tau_array.append(tau_temp)
        
        counter += 1

    if num_tau_analysed_counter == 0 : 
        # return tau_tuple(val = np.nan , steady_state = np.nan, current_inj=np.nan)
        return [np.nan, np.nan, np.nan, np.nan]
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

            

            # sag_current_temp = sag_tuple(val=sag_current, steady_state=asym_current , current_inj=input_step_current_values[sag_idx])
            sag_current_temp = [sag_current, asym_current, input_step_current_values[sag_idx], v_resting_membrane] 

        elif num_peaks > 0: 
            print('Spike found in Sag analysis, skipping')
            # sag_current_temp = sag_tuple(val= np.nan, steady_state=np.nan, current_inj = np.nan)
            sag_current_temp = [np.nan, np.nan, np.nan, np.nan] 
        
        # Append Value to existing named tuple
        sag_current_all.append(sag_current_temp) 

    return sag_current_all



#DJ: this is here so we do not lose your plotting settings for pca, kmeans and his even tho they wont run for me - each needs to be its own simple func like buildMeanAPFig()
#and they should be in plotters
def pAD_detection_old(V_dataframe, pca_plotting = False , kmeans_plotting = False , histogram_plotting=False):

    '''
    Input : 
            V_dataframe   : np.array : Voltage dataframe as  np array
            
    Output : 
            ap_slopes     : list of lists containing action_potential slopes (num 0f spikes x sweeps analysed)
            ap_thresholds : list of lists containing ap_thresholds or turning points of the spike (num 0f spikes x sweeps analysed)
            ap_width      :  
            ap_height     :
    '''

    peak_latencies_all_ , v_thresholds_all_  , peak_slope_all_  , peak_locs_corr_, upshoot_locs_all_ , peak_heights_all_  , peak_fw_all_ ,  peak_indices_all_, sweep_indices_all_  =  ap_characteristics_extractor_main(V_dataframe, all_sweeps =True)
    
    # Create Dataframe and append values
    
    pAD_df = pd.DataFrame(columns=['pAD', 'AP_loc', 'AP_sweep_num', 'AP_slope', 'AP_threshold', 'AP_height' , 'AP_latency'])
    
    
    non_nan_locs =  np.where(np.isnan(v_thresholds_all_) == False)[0] 
    
    if len (non_nan_locs) == 0 : 
        pAD_df["AP_loc"] = peak_locs_corr_ 
        pAD_df["AP_threshold"] =  v_thresholds_all_
        pAD_df["AP_slope"] = peak_slope_all_
        pAD_df["AP_latency"] = peak_latencies_all_
        pAD_df["AP_height"] =  peak_heights_all_ 
        pAD_df['AP_sweep_num'] = sweep_indices_all_
        pAD_df["pAD"] = ""
        pAD_df["pAD_count"] = np.nan 
        
        return peak_latencies_all_ , v_thresholds_all_  , peak_slope_all_  ,  peak_heights_all_ , pAD_df, None
        
        
    
    
    peak_latencies_all = np.array(peak_latencies_all_)[non_nan_locs]
    v_thresholds_all   = np.array(v_thresholds_all_ )[non_nan_locs]
    peak_slope_all     = np.array(peak_slope_all_  )[non_nan_locs]
    peak_locs_corr     = np.array(peak_locs_corr_ )[non_nan_locs]
    upshoot_locs_all   = np.array(upshoot_locs_all_)[non_nan_locs]
    peak_heights_all   = np.array(peak_heights_all_)[non_nan_locs]
    peak_fw_all        = np.array(peak_fw_all_)[non_nan_locs]
    peak_indices_all    = np.array(peak_indices_all_)[non_nan_locs]
    sweep_indices_all  = np.array(sweep_indices_all_)[non_nan_locs]
    
    assert len(peak_latencies_all) == len(peak_slope_all) == len(v_thresholds_all) == len(peak_locs_corr) == len(upshoot_locs_all) == len(peak_heights_all) == len(peak_fw_all) == len(peak_indices_all)== len(sweep_indices_all) 
    
    
        
        
    
    
    pAD_df["AP_loc"] = peak_locs_corr 
    pAD_df["AP_threshold"] =  v_thresholds_all 
    pAD_df["AP_slope"] = peak_slope_all 
    pAD_df["AP_latency"] = peak_latencies_all
    pAD_df["AP_height"] =  peak_heights_all 
    pAD_df['AP_sweep_num'] = sweep_indices_all
    pAD_class_labels = np.array(["pAD" , "Somatic"] )
    
    
    
    if len (peak_locs_corr) == 0 :
        print("No APs found in trace")
        pAD_df["pAD"] = ""
        pAD_df["pAD_count"] = np.nan 
        
        return   peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df, None 
    elif len (peak_locs_corr) == 1 : 
        if v_thresholds_all[0]  < -65.0 :
            # pAD 
            pAD_df["pAD"] = "pAD"
            pAD_df["pAD_count"] = 1 
        else: 
            pAD_df["pAD"]       = "Somatic"
            pAD_df["pAD_count"] = 0 
        return peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df, None
    else : 
        pass

def beta_pAD_detection(V_dataframe):
    '''
    Input : 
            V_dataframe   : np.array : Voltage dataframe as  np array
            
    Output : 
            ap_slopes     : list of lists containing action_potential slopes (num 0f spikes x sweeps analysed)
            ap_thresholds : list of lists containing ap_thresholds or turning points of the spike (num 0f spikes x sweeps analysed)
            ap_width      :  
            ap_height     :
    '''
    
    #  ESTABLISHED CONVENTION:  !!! pAD label == 1   !!!    
    
    
    ### STEP 1 :  Get the action potential (AP) values and create pAD dataframe with columns and values 
        
    peak_latencies_all_ , v_thresholds_all_  , peak_slope_all_  , peak_locs_corr_, upshoot_locs_all_ , peak_heights_all_  , peak_fw_all_ ,  peak_indices_all_, sweep_indices_all_  =  ap_characteristics_extractor_main(V_dataframe, all_sweeps =True)
    
    # Create Dataframe and append values
    pAD_df = pd.DataFrame(columns=['pAD', 'AP_loc', 'AP_sweep_num', 'AP_slope', 'AP_threshold', 'AP_height' , 'AP_latency'])
    
    non_nan_locs =  np.where(np.isnan(v_thresholds_all_) == False)[0] 
    
    if len (non_nan_locs) == 0 : 
        pAD_df["AP_loc"] = peak_locs_corr_ 
        pAD_df["AP_threshold"] =  v_thresholds_all_
        pAD_df["AP_slope"] = peak_slope_all_
        pAD_df["AP_latency"] = peak_latencies_all_
        pAD_df["AP_height"] =  peak_heights_all_ 
        pAD_df['AP_sweep_num'] = sweep_indices_all_
        pAD_df['upshoot_loc']  = upshoot_locs_all_ 
        pAD_df["pAD"] = ""
        pAD_df["pAD_count"] = np.nan 
        
        return peak_latencies_all_ , v_thresholds_all_  , peak_slope_all_  ,  peak_heights_all_ , pAD_df
        

    peak_latencies_all = np.array(peak_latencies_all_)[non_nan_locs]
    v_thresholds_all   = np.array(v_thresholds_all_ )[non_nan_locs]
    peak_slope_all     = np.array(peak_slope_all_  )[non_nan_locs]
    peak_locs_corr     = np.array(peak_locs_corr_ )[non_nan_locs]
    upshoot_locs_all   = np.array(upshoot_locs_all_)[non_nan_locs]
    peak_heights_all   = np.array(peak_heights_all_)[non_nan_locs]
    peak_fw_all        = np.array(peak_fw_all_)[non_nan_locs]
    peak_indices_all    = np.array(peak_indices_all_)[non_nan_locs]
    sweep_indices_all  = np.array(sweep_indices_all_)[non_nan_locs]
    
    assert len(peak_latencies_all) == len(peak_slope_all) == len(v_thresholds_all) == len(peak_locs_corr) == len(upshoot_locs_all) == len(peak_heights_all) == len(peak_fw_all) == len(peak_indices_all)== len(sweep_indices_all) 
    
    pAD_df["AP_loc"] = peak_locs_corr 
    pAD_df["upshoot_loc"] =  upshoot_locs_all   
    pAD_df["AP_threshold"] =  v_thresholds_all 
    pAD_df["AP_slope"] = peak_slope_all 
    pAD_df["AP_latency"] = peak_latencies_all
    pAD_df["AP_height"] =  peak_heights_all 
    pAD_df['AP_sweep_num'] = sweep_indices_all
    pAD_class_labels = np.array(["pAD" , "Somatic"] )
    
    if len (peak_locs_corr) == 0 :
        print("No APs found in trace")
        pAD_df["pAD"] = ""
        pAD_df["pAD_count"] = np.nan 
        
        return   peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df
    elif len (peak_locs_corr) == 1 : 
        if v_thresholds_all[0]  < -65.0 :
            # pAD 
            pAD_df["pAD"] = "pAD"
            pAD_df["pAD_count"] = 1 
        else: 
            pAD_df["pAD"]       = "Somatic"
            pAD_df["pAD_count"] = 0 
        return peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df
    else : 
        pass
    
    
    
    # STEP 2 : Make all labels for upshoot potential < -65 mV   = pAD  
    
    pAD_df_pAD_via_threshold = pAD_df[pAD_df[ "AP_threshold"]  < -65.0 ]
    pAD_df_uncertain_via_threshold = pAD_df[pAD_df[ "AP_threshold"]  > -65.0 ]
    
    pAD_df_mod = pd.concat([ pAD_df_pAD_via_threshold , pAD_df_uncertain_via_threshold] , axis = 0) 
    
    
    
    #pAD_df["AP_loc"][np.where(pAD_df["AP_threshold"] < - 65)[0]]
    
    
    
    # Data for fitting procedure
    
    X = pAD_df_mod[["AP_slope", "AP_threshold", "AP_height", "AP_latency"]]
    
    # Check if 2 clusters are better than 1 
    
    kmeans_1 = KMeans(n_clusters=1, n_init = 1).fit(X)
    kmeans_2 = KMeans(n_clusters=2, n_init = 1 ).fit(X)
    
    wcss_1 = kmeans_1.inertia_
    wcss_2 = kmeans_2.inertia_
    
    
    true_labels_combined = None 
    
    #print(type(kmeans_2.labels_))
    #print(wcss_1, wcss_2)

    if wcss_2 < wcss_1 :
        true_labels_combined = kmeans_2.labels_
        # 2 CLUSTERS  are a better fit than 1  
        # Append label column to type 
        #print("Mean Voltage Threshold of Cluster 1")
        #print(np.nanmean( v_thresholds_all  [kmeans_2.labels_ == 0] ))
        #print("Mean Voltage Threshold of Cluster 2")
        #print(np.nanmean( v_thresholds_all  [kmeans_2.labels_ == 1] ))
        #print(np.nanmean( peak_slope_all  [kmeans_2.labels_ == 0] ) , np.nanmean( peak_slope_all  [kmeans_2.labels_ == 1] ))
        
        if np.nanmean( v_thresholds_all  [kmeans_2.labels_ == 0] )  < np.nanmean( v_thresholds_all  [kmeans_2.labels_ == 1] ):
            pAD_df["pAD"] = pAD_class_labels[np.array(kmeans_2.labels_)]
            pAD_df["pAD_count"] = np.sum(kmeans_2.labels_)
        else:
            pAD_df["pAD"] = pAD_class_labels[np.mod( np.array( kmeans_2.labels_ + 1 )     , 2  )]
            pAD_df["pAD_count"] = np.sum(np.mod( np.array( kmeans_2.labels_ + 1 )     , 2  ))
              
        pAD_df["num_ap_clusters"] = 2 
        
    else :
         true_labels_combined = kmeans_1.labels_
         # JUST 1 CLUSTER
         if np.nanmean(v_thresholds_all )  <= -65.0 : 
             print("check file all APs seems pAD")
             pAD_df["pAD"] = "pAD"
             pAD_df["pAD_count"] = len(v_thresholds_all)
         else : 
          pAD_df["pAD"] = "Somatic"
         pAD_df["num_ap_clusters"] = 1
         pAD_df["pAD_count"] = 0 
         
    
    
    # Do the same but for the uncertain bit 
         
    X = pAD_df_uncertain_via_threshold [["AP_slope", "AP_threshold", "AP_height", "AP_latency"]]
    
    # Check if 2 clusters are better than 1 
    
    kmeans_1 = KMeans(n_clusters=1, n_init = 1).fit(X)
    kmeans_2 = KMeans(n_clusters=2, n_init = 1 ).fit(X)
    
    wcss_1 = kmeans_1.inertia_
    wcss_2 = kmeans_2.inertia_
    
    
    true_labels_split = None 
    
    #print(type(kmeans_2.labels_))
    #print(wcss_1, wcss_2)

    if wcss_2 < wcss_1 :
        true_labels_split  = kmeans_2.labels_
        
    else :
         true_labels_split = kmeans_1.labels_
        
    # Print % overlap 
    flip = False 
    
    if pAD_df_pAD_via_threshold.shape[0] > 0 : 
        if true_labels_combined[0] == 1 :
            pass
        else: 
            flip = True 
    
    true_labels_ = np.hstack((np.ones(pAD_df_pAD_via_threshold.shape[0])  ,  true_labels_split  ) )
    
    if flip :
        true_labels_combined = 1 - true_labels_combined
    
    
    
    print(sum( true_labels_  == true_labels_combined  )  / len(  true_labels_combined   )   , pAD_df_pAD_via_threshold.shape[0] /  pAD_df.shape[0]  )
    
    
    return true_labels_, true_labels_split, true_labels_combined 

    


def pAD_detection(V_dataframe):
    '''
    Input : 
            V_dataframe   : np.array : Voltage dataframe as  np array
            
    Output : 
            ap_slopes     : list of lists containing action_potential slopes (num 0f spikes x sweeps analysed)
            ap_thresholds : list of lists containing ap_thresholds or turning points of the spike (num 0f spikes x sweeps analysed)
            ap_width      :  
            ap_height     :
    '''
    
    ### STEP 1 :  Get the action potential (AP) values and create pAD dataframe with columns and values 
        
    peak_latencies_all_ , v_thresholds_all_  , peak_slope_all_  , peak_locs_corr_, upshoot_locs_all_ , peak_heights_all_  , peak_fw_all_ ,  peak_indices_all_, sweep_indices_all_  =  ap_characteristics_extractor_main(V_dataframe, all_sweeps =True)
    
    # Create Dataframe and append values
    pAD_df = pd.DataFrame(columns=['pAD', 'AP_loc', 'AP_sweep_num', 'AP_slope', 'AP_threshold', 'AP_height' , 'AP_latency'])
    
    non_nan_locs =  np.where(np.isnan(v_thresholds_all_) == False)[0] 
    
    if len (non_nan_locs) == 0 : 
        pAD_df["AP_loc"] = peak_locs_corr_ 
        pAD_df["AP_threshold"] =  v_thresholds_all_
        pAD_df["AP_slope"] = peak_slope_all_
        pAD_df["AP_latency"] = peak_latencies_all_
        pAD_df["AP_height"] =  peak_heights_all_ 
        pAD_df['AP_sweep_num'] = sweep_indices_all_
        pAD_df['upshoot_loc']  = upshoot_locs_all_ 
        pAD_df["pAD"] = ""
        pAD_df["pAD_count"] = np.nan 
        
        return peak_latencies_all_ , v_thresholds_all_  , peak_slope_all_  ,  peak_heights_all_ , pAD_df
        

    peak_latencies_all = np.array(peak_latencies_all_)[non_nan_locs]
    v_thresholds_all   = np.array(v_thresholds_all_ )[non_nan_locs]
    peak_slope_all     = np.array(peak_slope_all_  )[non_nan_locs]
    peak_locs_corr     = np.array(peak_locs_corr_ )[non_nan_locs]
    upshoot_locs_all   = np.array(upshoot_locs_all_)[non_nan_locs]
    peak_heights_all   = np.array(peak_heights_all_)[non_nan_locs]
    peak_fw_all        = np.array(peak_fw_all_)[non_nan_locs]
    peak_indices_all    = np.array(peak_indices_all_)[non_nan_locs]
    sweep_indices_all  = np.array(sweep_indices_all_)[non_nan_locs]
    
    assert len(peak_latencies_all) == len(peak_slope_all) == len(v_thresholds_all) == len(peak_locs_corr) == len(upshoot_locs_all) == len(peak_heights_all) == len(peak_fw_all) == len(peak_indices_all)== len(sweep_indices_all) 
    
    pAD_df["AP_loc"] = peak_locs_corr 
    pAD_df["upshoot_loc"] =  upshoot_locs_all   
    pAD_df["AP_threshold"] =  v_thresholds_all 
    pAD_df["AP_slope"] = peak_slope_all 
    pAD_df["AP_latency"] = peak_latencies_all
    pAD_df["AP_height"] =  peak_heights_all 
    pAD_df['AP_sweep_num'] = sweep_indices_all
    pAD_class_labels = np.array(["pAD" , "Somatic"] )
    
    if len (peak_locs_corr) == 0 :
        print("No APs found in trace")
        pAD_df["pAD"] = ""
        pAD_df["pAD_count"] = np.nan 
        
        return   peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df
    elif len (peak_locs_corr) == 1 : 
        if v_thresholds_all[0]  < -65.0 :
            # pAD 
            pAD_df["pAD"] = "pAD"
            pAD_df["pAD_count"] = 1 
        else: 
            pAD_df["pAD"]       = "Somatic"
            pAD_df["pAD_count"] = 0 
        return peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df
    else : 
        pass
    
    
    
    # Data for fitting procedure
    
    X = pAD_df[["AP_slope", "AP_threshold", "AP_height", "AP_latency"]]
    
    # Check if 2 clusters are better than 1 
    
    kmeans_1 = KMeans(n_clusters=1, n_init = 1).fit(X)
    kmeans_2 = KMeans(n_clusters=2, n_init = 1 ).fit(X)
    
    wcss_1 = kmeans_1.inertia_
    wcss_2 = kmeans_2.inertia_
    
    #print(type(kmeans_2.labels_))
    #print(wcss_1, wcss_2)

    if wcss_2 < wcss_1 :
        # 2 CLUSTERS  are a better fit than 1  
        # Append label column to type 
        #print("Mean Voltage Threshold of Cluster 1")
        #print(np.nanmean( v_thresholds_all  [kmeans_2.labels_ == 0] ))
        #print("Mean Voltage Threshold of Cluster 2")
        #print(np.nanmean( v_thresholds_all  [kmeans_2.labels_ == 1] ))
        #print(np.nanmean( peak_slope_all  [kmeans_2.labels_ == 0] ) , np.nanmean( peak_slope_all  [kmeans_2.labels_ == 1] ))
        
        if np.nanmean( v_thresholds_all  [kmeans_2.labels_ == 0] )  < np.nanmean( v_thresholds_all  [kmeans_2.labels_ == 1] ):
            pAD_df["pAD"] = pAD_class_labels[np.array(kmeans_2.labels_)]
            pAD_df["pAD_count"] = np.sum(kmeans_2.labels_)
        else:
            pAD_df["pAD"] = pAD_class_labels[np.mod( np.array( kmeans_2.labels_ + 1 )     , 2  )]
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
    
    return peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df  


def array_peak_cleaner(input_array, prominence = 10 , threshold = -55 ,  peak_pre_window = 5 ):
    
   
    array_cleaned = input_array.copy()
    
    array_diff_abs = np.diff(input_array) 
    array_diff_abs = np.hstack([array_diff_abs , np.mean(array_diff_abs)   ])
    
   
    
    array_cleaned[array_diff_abs  > np.mean(array_diff_abs) + 2*np.std(array_diff_abs) ] = np.nan
    array_cleaned = array_cleaned[~np.isnan(array_cleaned) ] 
     
    
    return array_cleaned, None 


        





