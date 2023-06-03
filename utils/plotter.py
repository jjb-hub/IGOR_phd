# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:29:46 2023

@author: GUEST1
"""
#myshit 
from utils.igor_utils import igor_exporter, make_path
from utils.helper_functions import extract_FI_x_y
#shitshit
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import timeit

#simp funcs
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


#%% PLOTTING FUNCS: AP, FI, FI_AP, pAD

# https://www.google.com/search?q=how+to+create+a+small+plot+inside+plot+in+python+ax.plot&rlz=1C5CHFA_enAU794AU794&sxsrf=APwXEdcAmrqZK5nDrVeiza4rtKgMqeIQKQ%3A1681904232199&ei=aNI_ZMvLC_KTxc8P7Z-a-A8&ved=0ahUKEwjLn7nC7bX-AhXySfEDHe2PBv8Q4dUDCA8&uact=5&oq=how+to+create+a+small+plot+inside+plot+in+python+ax.plot&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzoKCAAQRxDWBBCwAzoECCMQJzoHCCMQsAIQJzoFCAAQogRKBAhBGABQ_gRYuWpgwmtoBHABeACAAacBiAGiHZIBBDEuMjiYAQCgAQHIAQjAAQE&sclient=gws-wiz-serp#bsht=CgRmYnNtEgQIBDAB&kpvalbx=_oNI_ZJSCFPbOxc8Pwf-MkA8_30
# add the I injected as a subplot to show the AP method #FIX ME 
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
            
            array_V, df_V = igor_exporter(path_V) # df_y each sweep is a column
            I_color = 'cornflowerblue'
            try:
                array_I, df_I = igor_exporter(path_I) #df_I has only 1 column and is the same as array_I
            except FileNotFoundError: #if no I file exists 
                print(f"no I file found for {row['cell_ID']}, I setting used was: {row['I_set']}")
                array_I = np.zeros(len(df_V)-1)
                I_color='grey'
                
            x_scaler_drug_bar = len(df_V[0]) * 0.0001 # multiplying this  by drug_in/out will give you the point at the end of the sweep in seconds
            x_V = np.arange(len(array_V)) * 0.0001 #sampeling at 10KHz will give time in seconds
            x_I = np.arange(len(array_I))*0.0001
            

            plt.figure(figsize = (12,9))
            # ax1 = plt.subplot2grid((20, 20), (0, 0), rowspan = 15, colspan =20) #(nrows, ncols)
            # ax2 = plt.subplot2grid((20, 20), (17, 0), rowspan = 5, colspan=20)
            
            ax1 = plt.subplot2grid((11, 8), (0, 0), rowspan = 8, colspan =11) #(nrows, ncols)
            ax2 = plt.subplot2grid((11, 8), (8, 0), rowspan = 2, colspan=11)
            
            ax1.plot(x_V,array_V, c = color_dict[row['drug']], lw=1) #voltage trace plot
            ax2.plot(x_I, array_I, label = row['I_set'], color=I_color )#label=
            ax2.legend()
            
            # ax2.axis('off')
            ax1.spines['top'].set_visible(False) # 'top', 'right', 'bottom', 'left'
            ax1.spines['right'].set_visible(False)
            
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            # ax2.spines['left'].set_visible(False)
            # ax2.spines['bottom'].set_visible(False)
            
            
            ax1.axvspan((int((row['drug_in'])* x_scaler_drug_bar) - x_scaler_drug_bar), (int(row['drug_out'])* x_scaler_drug_bar), facecolor = "grey", alpha = 0.2) #drug bar shows start of drug_in sweep to end of drug_out sweep 
            ax1.set_xlabel( "Time (s)", fontsize = 12) #, fontsize = 15
            ax1.set_ylabel( "Membrane Potential (mV)", fontsize = 12) #, fontsize = 15
            ax2.set_xlabel( "Time (s)", fontsize = 10) #, fontsize = 15
            ax2.set_ylabel( "Current (pA)", fontsize = 10) #, fontsize = 15
            ax1.set_title(row['cell_ID'] + ' '+ row['drug'] +' '+ " Application" + " (" + str(row['application_order']) + ")", fontsize = 16) # , fontsize = 25
            plt.tight_layout()
            pdf.savefig()

            plt.close("all")
    stop = timeit.default_timer()
    print('Time: ', stop - start)  
    return

def drug_aplication_visualisation_old(feature_df,  color_dict):
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
            array_V, df_V = igor_exporter(path_V) # df_y each sweep is a column
            array_I, df_I = igor_exporter(path_I) 
            
            fig, axs = plt.subplots(1,1, figsize = (10, 5))
            
            ##
            # figure = plt.Figure()
            # ax1 = plt.subplot2grid((10, 10), (0, 0), rowspan = 7, colspan =10) #(nrows, ncols)
            # ax2 = plt.subplot2grid((10, 10), (5, 0), rowspan = 2, colspan=10)
            
            # ax1.plot(x,array_V, c = color_dict[row['drug']], lw=1) #voltage trace plot
            
            # fig, axs = plt.subplots(1,2, figsize = (10, 5)) to make subplot ?
            #https://matplotlib.org/stable/tutorials/intermediate/tight_layout_guide.html#sphx-glr-tutorials-intermediate-tight-layout-guide-py
  
            x_scaler_drug_bar = len(df_V[0]) * 0.0001 # multiplying this  by drug_in/out will give you the point at the end of the sweep in seconds
            x = np.arange(len(array_V)) * 0.0001 #sampeling at 10KHz will give time in seconds
            
            axs.plot(x,array_V, c = color_dict[row['drug']], lw=1)
            # plt.ylim(-65, -35)
            
            axs.axvspan((int((row['drug_in'])* x_scaler_drug_bar) - x_scaler_drug_bar), (int(row['drug_out'])* x_scaler_drug_bar), facecolor = "grey", alpha = 0.2) #drug bar shows start of drug_in sweep to end of drug_out sweep 
            axs.set_xlabel( "Time (s)", fontsize = 15)
            axs.set_ylabel( "Membrane Potential (mV)", fontsize = 15)
            axs.set_title(row['cell_ID'] + ' '+ row['drug'] +' '+ " Application" + " (" + str(row['application_order']) + ")", fontsize = 25)
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

            ax.set_xlabel( "Current (pA)", fontsize = 15)
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

            ax.set_xlabel( "Current (pA)", fontsize = 15)
            ax.set_ylabel( "AP frequency", fontsize = 15)
            ax.set_title( cell_ID_name + " FI curves", fontsize = 20)
            ax.legend(fontsize = 10)
            pdf.savefig(fig)
            plt.close("all") 
                
        stop = timeit.default_timer()

        print('Time: ', stop - start) 
        
    return 

