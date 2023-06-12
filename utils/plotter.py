# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:29:46 2023

@author: GUEST1
"""
#myshit 

from utils.mettabuild_functions import expandFeatureDF, loopCombinations_stats
from utils.base_utils import *
from utils.mettabuild_functions import extract_FI_x_y
from ephys.ap_functions import pAD_detection
#shitshit
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D 
import numpy as np

import timeit

#external functions #CHECK USEAGE
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


# My Functions

def getorBuildApplicationFig(filename, cell_ID_or_cell_df, from_scratch=None):
    color_dict = getColors(filename)

    if not isinstance(cell_ID_or_cell_df, pd.DataFrame):
        expanded_df = getorBuildExpandedDF(filename, 'feature_df_expanded', expandFeatureDF, from_scratch=False)
        cell_df = getCellDF(expanded_df, cell_ID_or_cell_df, data_type = 'AP')
    else:
        cell_df = cell_ID_or_cell_df
    cell_ID = cell_df['cell_ID'].iloc[0]

    from_scratch = from_scratch if from_scratch is not None else input("Rebuild Fig even if previous version exists? (y/n)") == 'y'
    if from_scratch or not isCached(filename, cell_ID):
        print(f'BUILDING "{cell_ID} Application Figure"')    #Build useing callback otherwise and cache result
        #inputs to builder if not cached:
        folder_file = cell_df['folder_file'].values[0]
        I_set = cell_df['I_set'].values[0]
        drug = cell_df['drug'].values[0]
        drug_in = cell_df['drug_in'].values[0]
        drug_out = cell_df['drug_out'].values[0]
        application_order = cell_df['application_order'].values[0]
        pAD_locs = cell_df['APP_pAD_AP_locs'].values[0]  #FIX ME perhaps this should also be in try so can run without pAD! or add pAD == True in vairables
        
        fig = buildApplicationFig(color_dict, cell_ID=cell_ID, folder_file=folder_file, I_set=I_set, drug=drug, drug_in=drug_in, drug_out=drug_out, application_order=application_order, pAD_locs=None)
        saveAplicationFig(fig, cell_ID)
    else : fig = getCache(filename, cell_ID)
    fig.show()
    
    
    

def buildApplicationFig(color_dict, cell_ID=None, folder_file=None, I_set=None, drug=None, drug_in=None, drug_out=None, application_order=None, pAD_locs=None):
    #load raw data
    path_V, path_I = make_path(folder_file)
    array_V, df_V = igor_exporter(path_V) # df_y each sweep is a column
    try:
        array_I, df_I = igor_exporter(path_I) #df_I has only 1 column and is the same as array_I
    except FileNotFoundError: #if no I file exists 
        print(f"no I file found for {cell_ID}, I setting used was: {I_set}")
        array_I = np.zeros(len(df_V)-1)
    #scale data
    x_scaler_drug_bar = len(df_V[0]) * 0.0001 # multiplying this  by drug_in/out will give you the point at the end of the sweep in seconds
    x_V = np.arange(len(array_V)) * 0.0001 #sampeling at 10KHz will give time in seconds
    x_I = np.arange(len(array_I))*0.00005 #20kHz igor 
    #plot 
    fig = plt.figure(figsize = (12,9))
    ax1 = plt.subplot2grid((11, 8), (0, 0), rowspan = 8, colspan =11) #(nrows, ncols)
    ax2 = plt.subplot2grid((11, 8), (8, 0), rowspan = 2, colspan=11)
    ax1.plot(x_V,array_V, c = color_dict[drug], lw=1) #voltage trace plot # "d", markevery=pAD_locs
    
    if pAD_locs is not None: #FIX ME
        print('plotting pAD')
    
    ax2.plot(x_I, array_I, label = I_set, color=color_dict['I_display'] )#label=
    ax2.legend()
    # ax2.axis('off')
    ax1.spines['top'].set_visible(False) # 'top', 'right', 'bottom', 'left'
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)
    ax1.axvspan((int((drug_in)* x_scaler_drug_bar) - x_scaler_drug_bar), (int(drug_out)* x_scaler_drug_bar), facecolor = "grey", alpha = 0.2) #drug bar shows start of drug_in sweep to end of drug_out sweep 
    ax1.set_xlabel( "Time (s)", fontsize = 12) #, fontsize = 15
    ax1.set_ylabel( "Membrane Potential (mV)", fontsize = 12) #, fontsize = 15
    ax2.set_xlabel( "Time (s)", fontsize = 10) #, fontsize = 15
    ax2.set_ylabel( "Current (pA)", fontsize = 10) #, fontsize = 15
    ax1.set_title(cell_ID + ' '+ drug +' '+ " Application" + " (" + str(application_order) + ")", fontsize = 16) # , fontsize = 25
    plt.tight_layout()
    return fig

def loopBuildAplicationFigs(filename):
    df = getorBuildExpandedDF(filename, 'feature_df_expanded', expandFeatureDF, from_scratch=False)
    color_dict = getColors(filename)
    application_df = df[df.data_type == 'AP'] 
    for row_ind, row in application_df.iterrows():  #row is a series that can be called row['colname']
        #inputs to builder if not cached:
        cell_ID = row['cell_ID']
        folder_file = row['folder_file']
        I_set = row['I_set']
        drug = row['drug']
        drug_in = row['drug_in']
        drug_out = row['drug_out']
        application_order = row['application_order']
        pAD_locs = row['APP_pAD_AP_locs']

        buildApplicationFig(color_dict, cell_ID=cell_ID, folder_file=folder_file, I_set=I_set, drug=drug, drug_in=drug_in, drug_out=drug_out, application_order=application_order, pAD_locs=None)
        plt.close()
    return

#PLOTTERS FOR pAD

def getorbuildMeanAPFig(filename, cell_ID_or_cell_df, from_scratch=None):
        if not isinstance(cell_ID_or_cell_df, pd.DataFrame):
            expanded_df = getorBuildExpandedDF(filename, 'feature_df_expanded', expandFeatureDF, from_scratch=False)
            cell_df = getCellDF(expanded_df, cell_ID_or_cell_df, data_type = 'AP')
        else:
            cell_df = cell_ID_or_cell_df
        cell_ID = cell_df['cell_ID'].iloc[0]

        from_scratch = from_scratch if from_scratch is not None else input("Rebuild Fig even if previous version exists? (y/n)") == 'y'
        if from_scratch or not isCached(filename, cell_ID):
            print(f'BUILDING "{cell_ID} Mean APs Figure"') 
            folder_file = cell_df['folder_file'].values[0]
            path_V, path_I = make_path(folder_file)
            listV, dfV = igor_exporter(path_V)
            V_array = np.array(dfV)
            peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df  = pAD_detection(V_array)
            if peak_heights_all <=1:
                return print(f'No APs in trace for {cell_ID}')
            fig = buildMeanAPFig(cell_ID, pAD_df, V_array, input_plot_forwards_window  = 50, input_plot_backwards_window= 100)
            saveMeanAPFig(fig, cell_ID)
        else : fig = getCache(filename, cell_ID)
        fig.show()

def buildMeanAPFig(cell_id, pAD_dataframe, V_array, input_plot_forwards_window  = 50, input_plot_backwards_window= 100):

    # Rename vars: 
    pAD_df = pAD_dataframe
    V      = V_array  
    plot_forwards_window        =  input_plot_forwards_window  
    plot_backwards_window       =  input_plot_backwards_window
    plot_window = plot_forwards_window + plot_backwards_window
    sampling_rate               = 1e4 
    sec_to_ms                   = 1e3
    time_                       = sec_to_ms* np.arange(0,150) / sampling_rate  
    # pAD subdataframe and indices
    pAD_sub_df = pAD_df[pAD_df.pAD =="pAD"] 
    pAD_ap_indices = pAD_sub_df[["AP_loc", "AP_sweep_num"]].values

    # Somatic subdataframe and indices
    Somatic_sub_df = pAD_df[pAD_df.pAD =="Somatic"] 
    Somatic_ap_indices = Somatic_sub_df[["AP_loc", "AP_sweep_num"]].values

    pAD_spike_array = np.zeros([len(pAD_ap_indices), plot_window  ])
    Somatic_spike_array = np.zeros([len(Somatic_ap_indices), plot_window ])

    # Plotter for pAD and Somatic Spikes 
    fig, ax = plt.subplots()
    lines  = []  # initialise empty line list 

    for idx in range(len(pAD_ap_indices)): 
        pAD_spike_array[idx,:] = V[ pAD_ap_indices[:,0][idx] - plot_backwards_window :  pAD_ap_indices[:,0][idx] +  plot_forwards_window  ,  pAD_ap_indices[:,1][idx]    ]
        line, = ax.plot(time_, pAD_spike_array[idx,:] , color = 'salmon', alpha=0.05)
        lines.append(line)
        #plt.plot(pAD_spike_array[idx,:], color ='grey', label = 'pAD')

    line, = ax.plot(time_, np.mean(pAD_spike_array , axis = 0)  , color = 'red')
    lines.append(line)

    for idx_ in range(len(Somatic_ap_indices)): 
        Somatic_spike_array[idx_,:] = V[ Somatic_ap_indices[:,0][idx_] - plot_backwards_window :  Somatic_ap_indices[:,0][idx_] + plot_forwards_window   ,  Somatic_ap_indices[:,1][idx_]    ]
        line, = ax.plot(time_,Somatic_spike_array[idx_,:] , color = 'cornflowerblue', alpha=0.05)
        lines.append(line)

    line, = ax.plot(time_, np.mean(Somatic_spike_array , axis = 0)  , c = 'blue')
    lines.append(line)

    # Create the custom legend with the correct colors
    legend_elements = [Line2D([0], [0], color='salmon', lw=2, label='pAD Ensemble', alpha=0.2),
                       Line2D([0], [0], color='red', lw=2, label= 'pAD Mean'),
                       Line2D([0], [0], color='cornflowerblue', lw=2, label='Somatic Ensemble', alpha=0.2),
                       Line2D([0], [0], color='blue', lw=2, label='Somatic Mean')]

    # Set the legend with the custom elements
    ax.legend(handles=legend_elements)

        
    #plt.plot(np.mean(Somatic_spike_array, axis = 0 ) , c = 'blue', label = 'Somatic Mean')
    plt.title(cell_id)
    plt.ylabel('Membrane Potential (mV)')
    plt.xlabel('Time (ms)')
    plt.tight_layout
    plt.show()    
    return fig 
    
    
    

# OLD SHIT BELLOW WILL DELETE ONCE metta looper works
#%% PLOTTING FUNCS: AP, FI, FI_AP, pAD

# https://www.google.com/search?q=how+to+create+a+small+plot+inside+plot+in+python+ax.plot&rlz=1C5CHFA_enAU794AU794&sxsrf=APwXEdcAmrqZK5nDrVeiza4rtKgMqeIQKQ%3A1681904232199&ei=aNI_ZMvLC_KTxc8P7Z-a-A8&ved=0ahUKEwjLn7nC7bX-AhXySfEDHe2PBv8Q4dUDCA8&uact=5&oq=how+to+create+a+small+plot+inside+plot+in+python+ax.plot&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzoKCAAQRxDWBBCwAzoECCMQJzoHCCMQsAIQJzoFCAAQogRKBAhBGABQ_gRYuWpgwmtoBHABeACAAacBiAGiHZIBBDEuMjiYAQCgAQHIAQjAAQE&sclient=gws-wiz-serp#bsht=CgRmYnNtEgQIBDAB&kpvalbx=_oNI_ZJSCFPbOxc8Pwf-MkA8_30
# add the I injected as a subplot to show the AP method #FIX ME 
def drug_aplication_visualisation(feature_df,  OUTPUT_DIR, color_dict):
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

    # with PdfPages(f'{OUTPUT_DIR}/drug_aplications_all_cells.pdf') as pdf:
        
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
        plt.savefig(f"{OUTPUT_DIR}/AP/{row['cell_ID']}.svg") #not run remi comented stuff bellow also didnt run : application memory
            # pdf.savefig() #rewite as svg

            # plt.close("all")
    stop = timeit.default_timer()
    print('Time: ', stop - start)  
    return



def plot_FI_AP_curves(feature_df, OUTPUT_DIR):
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
    with PdfPages(f'{OUTPUT_DIR}aplication_FI_curves_all_cells.pdf') as pdf:
        
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

