# module
from module.metabuild_functions import expandFeatureDF, loopCombinations_stats
from module.utils import *
from module.getters import getorbuildExpandedDF, getCellDF
from module.metabuild_functions import extract_FI_x_y
from module.action_potential_functions import pAD_detection
#external
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D 
import numpy as np

import timeit


########## BASE PLOTTERS ##########

#  plots any waveform based off fold_file
def quick_plot_file(filename, folder_file):
    df=getRawDF(filename)
    path_V, path_I = make_path(folder_file)
    listV, dfV = igor_exporter(path_V)
    V_array = np.array(dfV)
    try:
        listI, dfI = igor_exporter(path_I)
        I_array = np.array(dfI)
    except:
        print('no I file found', path_I)

    #incase you want to output the data_type for instance
    row = df[df['folder_file']==folder_file]
    print (row)

    quick_line_plot(listV, f'Voltage trace for {folder_file}')
    quick_line_plot(listI, f'Current (I) trace for {folder_file}')

def quick_line_plot(plotlist, plottitle):
    plt.plot( range(len(plotlist)), plotlist, marker='o', linestyle='-')
    plt.title(plottitle)
    plt.show()



    




## LOOPERS ## 
# #right now need expanded df to make ap figuures not ideal

def loopbuildAplicationFigs(filename):
    df = getorbuildExpandedDF(filename, 'feature_df_expanded', expandFeatureDF, from_scratch=False)
    color_dict = getColors(filename)
    application_df = df[df.data_type == 'AP'] 
    for row_ind, row in application_df.iterrows():  #row is a series that can be called row['colname']
        #inputs to builder if not cached:
        cell_id = row['cell_id']
        folder_file = row['folder_file']
        I_set = row['I_set']
        drug = row['drug']
        drug_in = row['drug_in']
        drug_out = row['drug_out']
        application_order = row['application_order']
        pAD_locs = row['APP_pAD_AP_locs']

        buildApplicationFig(color_dict, cell_id=cell_id, folder_file=folder_file, I_set=I_set, drug=drug, drug_in=drug_in, drug_out=drug_out, application_order=application_order, pAD_locs=None)
        plt.close()
    return

## GETTERS ##

def getorbuildApplicationFig(filename, cell_id_or_cell_df, from_scratch=None):
    color_dict = getColors(filename) 

    if not isinstance(cell_id_or_cell_df, pd.DataFrame):
        expanded_df = getorbuildExpandedDF(filename, 'feature_df_expanded', expandFeatureDF, from_scratch=False)
        cell_df = getCellDF(expanded_df, cell_id_or_cell_df, data_type = 'AP')
    else:
        cell_df = cell_id_or_cell_df
    cell_id = cell_df['cell_id'].iloc[0]

    from_scratch = from_scratch if from_scratch is not None else input("Rebuild Fig even if previous version exists? (y/n)") == 'y'
    if from_scratch or not isCached(filename, cell_id):
        for index, row in cell_df.iterrows(): #loops rows from multiple applications
            #inputs to builder if not cached:
            folder_file = row['folder_file']
            I_set = row['I_set']
            drug = row['drug']
            drug_in = row['drug_in']
            drug_out = row['drug_out']
            application_order = row['application_order']
            print(f'BUILDING "{cell_id} Application {application_order} Figure"') #Build useing callback otherwise and cache result
            fig = buildApplicationFig(color_dict, cell_id=cell_id, folder_file=folder_file, I_set=I_set, drug=drug, drug_in=drug_in, drug_out=drug_out, application_order=application_order, pAD_locs=True)
            saveAplicationFig(fig, f'{cell_id}_application_{application_order}')
            plt.close(fig)
            
    else : 
        fig = getCache(filename, cell_id)
    # fig.show()
    

def getorbuildAP_MeanFig(filename, cell_id_or_cell_df, from_scratch=None):
        if not isinstance(cell_id_or_cell_df, pd.DataFrame):
            expanded_df = getorbuildExpandedDF(filename, 'feature_df_expanded', expandFeatureDF, from_scratch=False)
            cell_df = getCellDF(expanded_df, cell_id_or_cell_df, data_type = 'AP')
        else:
            cell_df = cell_id_or_cell_df
        cell_id = cell_df['cell_id'].iloc[0]

        from_scratch = from_scratch if from_scratch is not None else input("Rebuild Fig even if previous version exists? (y/n)") == 'y'
        if from_scratch or not isCached(filename, cell_id):
            print(f'BUILDING "{cell_id} Mean APs Figure"') 
            folder_file = cell_df['folder_file'].values[0]
            path_V, path_I = make_path(folder_file)
            listV, dfV = igor_exporter(path_V)
            V_array = np.array(dfV)
            peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df  = pAD_detection(V_array)
            if len(peak_heights_all) <=1:
                return print(f'No APs in trace for {cell_id}')
            fig = buildAP_MeanFig(cell_id, pAD_df, V_array, input_plot_forwards_window  = 50, input_plot_backwards_window= 100)
            saveAP_MeanFig(fig, cell_id)
        else : fig = getCache(filename, cell_id)
        fig.show()
        
def getorbuildAP_PhasePlotFig(filename, cell_id_or_cell_df, from_scratch=None):
        if not isinstance(cell_id_or_cell_df, pd.DataFrame):
            expanded_df = getorbuildExpandedDF(filename, 'feature_df_expanded', expandFeatureDF, from_scratch=False)
            cell_df = getCellDF(expanded_df, cell_id_or_cell_df, data_type = 'AP')
        else:
            cell_df = cell_id_or_cell_df
        cell_id = cell_df['cell_id'].iloc[0]

        from_scratch = from_scratch if from_scratch is not None else input("Rebuild Fig even if previous version exists? (y/n)") == 'y'
        if from_scratch or not isCached(filename, cell_id):
            print(f'BUILDING "{cell_id} Phase Plot Figure"') 
            folder_file = cell_df['folder_file'].values[0]
            path_V, path_I = make_path(folder_file)
            listV, dfV = igor_exporter(path_V)
            V_array = np.array(dfV)
            peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df  = pAD_detection(V_array)
            if len(peak_heights_all) <=1:
                return print(f'No APs in trace for {cell_id}')
            fig =buildAP_PhasePlotFig(cell_id, pAD_df, V_array)
            saveAP_PhasePlotFig(fig, cell_id)
        else : fig = getCache(filename, cell_id)
        fig.show()


def getorbuildAP_PCAFig(filename, cell_id_or_cell_df, from_scratch=None):
        if not isinstance(cell_id_or_cell_df, pd.DataFrame):
            expanded_df = getorbuildExpandedDF(filename, 'feature_df_expanded', expandFeatureDF, from_scratch=False)
            cell_df = getCellDF(expanded_df, cell_id_or_cell_df, data_type = 'AP')
        else:
            cell_df = cell_id_or_cell_df
        cell_id = cell_df['cell_id'].iloc[0]

        from_scratch = from_scratch if from_scratch is not None else input("Rebuild Fig even if previous version exists? (y/n)") == 'y'
        if from_scratch or not isCached(filename, cell_id):
            print(f'BUILDING "{cell_id} PCA Figure"') 
            folder_file = cell_df['folder_file'].values[0]
            path_V, path_I = make_path(folder_file)
            listV, dfV = igor_exporter(path_V)
            V_array = np.array(dfV)
            peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df  = pAD_detection(V_array)
            if len(peak_heights_all) <=1:
                return print(f'No APs in trace for {cell_id}')
            fig =buildAP_PCAFig(cell_id, pAD_df, V_array)
            saveAP_PCAFig(fig, cell_id)
        else : fig = getCache(filename, cell_id)
        fig.show()

def getorbuildAP_HistogramFig(filename, cell_id_or_cell_df, from_scratch=None):
        if not isinstance(cell_id_or_cell_df, pd.DataFrame):
            expanded_df = getorbuildExpandedDF(filename, 'feature_df_expanded', expandFeatureDF, from_scratch=False)
            cell_df = getCellDF(expanded_df, cell_id_or_cell_df, data_type = 'AP')
        else:
            cell_df = cell_id_or_cell_df
        cell_id = cell_df['cell_id'].iloc[0]

        from_scratch = from_scratch if from_scratch is not None else input("Rebuild Fig even if previous version exists? (y/n)") == 'y'
        if from_scratch or not isCached(filename, cell_id):
            print(f'BUILDING "{cell_id} AP Histogram Figure"') 
            folder_file = cell_df['folder_file'].values[0]
            path_V, path_I = make_path(folder_file)
            listV, dfV = igor_exporter(path_V)
            V_array = np.array(dfV)
            peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df  = pAD_detection(V_array)
            if len(peak_heights_all) <=1:
                return print(f'No APs in trace for {cell_id}')
            fig =buildAP_HistogramFig(cell_id, pAD_df, V_array)
            saveAP_HistogramFig(fig, cell_id)
        else : fig = getCache(filename, cell_id)
        fig.show()

## BUILDERS ##
def buildApplicationFig(color_dict, cell_id=None, folder_file=None, I_set=None, drug=None, drug_in=None, drug_out=None, application_order=None, pAD_locs=None):
    #load raw data 
    color_dict = {"pAD":"orange","Somatic":"blue","WASH":"lightsteelblue", "PRE":"black", "CONTROL": 'grey', "TCB2":'green', "DMT":"teal", "PSIL":"orange", "LSD":"purple", "MDL":'blue', 'I_display':'cornflowerblue'} 
    if drug is None:
        plot_color = 'k'
    else:
        plot_color = color_dict[drug]
    path_V, path_I = make_path(folder_file)
    array_V, df_V = igor_exporter(path_V) # df_y each sweep is a column
    try:
        array_I, df_I = igor_exporter(path_I) #df_I has only 1 column and is the same as array_I
    except FileNotFoundError: #if no I file exists 
        print(f"no I file found for {cell_id}, I setting used was: {I_set}")
        array_I = np.zeros(len(df_V)-1)
    #scale data
    x_scaler_drug_bar = len(df_V[0]) * 0.0001 # multiplying this  by drug_in/out will give you the point at the end of the sweep in seconds
    x_V = np.arange(len(array_V)) * 0.0001 #sampeling at 10KHz will give time in seconds
    x_I = np.arange(len(array_I))*0.00005 #20kHz igor 
    #plot 
    fig = plt.figure(figsize = (12,9))
    ax1 = plt.subplot2grid((11, 8), (0, 0), rowspan = 8, colspan =11) #(nrows, ncols)
    ax2 = plt.subplot2grid((11, 8), (8, 0), rowspan = 2, colspan=11)
    ax1.plot(x_V, array_V, c = plot_color, lw=1, alpha=0.8) #voltage trace plot # "d", markevery=pAD_locs
    pAD_plot_pre_window = 50
    pAD_plot_post_window = 50
    
    if pAD_locs is None: 
        # Get pAD_locs
        peak_latencies_all , v_thresholds_all  , peak_slope_all  ,  peak_heights_all , pAD_df  = pAD_detection(df_V) 
        
        # pAD subdataframe and indices
        pAD_sub_df = pAD_df[pAD_df.pAD =="pAD"] 
        pAD_ap_indices = pAD_sub_df[["upshoot_loc", "AP_sweep_num", "AP_loc"]].values

        # Somatic subdataframe and indices
        Somatic_sub_df = pAD_df[pAD_df.pAD =="Somatic"] 
        Somatic_ap_indices = Somatic_sub_df[["AP_loc", "AP_sweep_num", "AP_loc"]].values
    
        for pAD_spike_idx in range(len(pAD_ap_indices)):
            pAD_upshoot_loc , sweep_num , pAD_AP_loc =  pAD_ap_indices[pAD_spike_idx][0], pAD_ap_indices[pAD_spike_idx][1], pAD_ap_indices[pAD_spike_idx][2]
            v_temp = np.array(array_V[sweep_num*df_V.shape[0] +  pAD_upshoot_loc - pAD_plot_pre_window : sweep_num*df_V.shape[0] +  pAD_AP_loc + pAD_plot_post_window  ] )
            time_temp = np.linspace((sweep_num*df_V.shape[0] +  pAD_upshoot_loc  - pAD_plot_pre_window )*0.0001 , (sweep_num*df_V.shape[0] +  pAD_AP_loc + pAD_plot_post_window  )*0.0001 , len(v_temp) )  
            ax1.plot(time_temp, v_temp, c  = 'red', lw = 2, alpha=0.5 )
            
    
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
    #ax1.set_title(cell_id + ' '+ drug +' '+ " Application" + " (" + str(application_order) + ")", fontsize = 16) # , fontsize = 25
    plt.tight_layout()
    plt.show()
    return fig

def buildAP_MeanFig(cell_id, pAD_dataframe, V_array, input_plot_forwards_window  = 50, input_plot_backwards_window= 100):

    # Rename vars: 
    pAD_df = pAD_dataframe
    V      = V_array  
    plot_forwards_window        =  input_plot_forwards_window  
    plot_backwards_window       =  input_plot_backwards_window
    plot_window = plot_forwards_window + plot_backwards_window
    sampling_rate               = 1e4 
    sec_to_ms                   = 1e3
    
    # pAD subdataframe and indices
    pAD_sub_df = pAD_df[pAD_df.pAD =="pAD"] 
    pAD_ap_indices = pAD_sub_df[["upshoot_loc" , "AP_loc", "AP_sweep_num"]].values

    # Somatic subdataframe and indices
    Somatic_sub_df = pAD_df[pAD_df.pAD =="Somatic"] 
    Somatic_ap_indices = Somatic_sub_df[["upshoot_loc" , "AP_loc", "AP_sweep_num"]].values

    pAD_spike_array = np.zeros([len(pAD_ap_indices), plot_window  ])
    Somatic_spike_array = np.zeros([len(Somatic_ap_indices), plot_window ])
    
    # Plotter for pAD and Somatic Spikes 
    fig, ax = plt.subplots()
    lines  = []  # initialise empty line list 

    for idx in range(len(pAD_ap_indices)): 
        if plot_backwards_window >=   pAD_ap_indices[:,0][idx]:
            plot_backwards_window_ = 0
            plot_forwards_window_  = plot_forwards_window + plot_backwards_window  
        else: 
            plot_backwards_window_ = plot_backwards_window
            plot_forwards_window_  = plot_forwards_window
        
        
        pAD_spike_array[idx,:] = V[ pAD_ap_indices[:,0][idx] - plot_backwards_window_ :  pAD_ap_indices[:,0][idx] +  plot_forwards_window_  ,  pAD_ap_indices[:,-1][idx]    ]
        time_                       = sec_to_ms* np.arange(0, len(pAD_spike_array[idx,:])) / sampling_rate  
        line, = ax.plot(time_, pAD_spike_array[idx,:] , color = 'salmon', alpha=0.05)
        lines.append(line)
        #plt.plot(pAD_spike_array[idx,:], color ='grey', label = 'pAD')
    
    if pAD_spike_array.shape[0] > 0 :
        line, = ax.plot(time_, np.mean(pAD_spike_array , axis = 0)  , color = 'red')
        lines.append(line)
    else : # no spikes to plot
        pass

    
    for idx_ in range(len(Somatic_ap_indices)): 
        
        if plot_backwards_window >=   Somatic_ap_indices[:,0][idx_]:
            plot_backwards_window_ = 0
            plot_forwards_window_  = plot_forwards_window + plot_backwards_window  
        else: 
            plot_backwards_window_ = plot_backwards_window
            plot_forwards_window_  = plot_forwards_window
            
        Somatic_spike_array[idx_,:] = V[ Somatic_ap_indices[:,0][idx_] - plot_backwards_window_ :  Somatic_ap_indices[:,0][idx_] + plot_forwards_window_   ,  Somatic_ap_indices[:,-1][idx_]    ]
        time_                       = sec_to_ms* np.arange(0, len(Somatic_spike_array[idx_,:])) / sampling_rate  
        line, = ax.plot(time_,Somatic_spike_array[idx_,:] , color = 'cornflowerblue', alpha=0.05)
        lines.append(line)
    if pAD_spike_array.shape[0] > 0 :
        line, = ax.plot(time_, np.mean(Somatic_spike_array , axis = 0)  , c = 'blue')
        lines.append(line)
    else : # no spikes to plot
        pass

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
    plt.tight_layout()
    plt.show()    
    return fig 
    

def buildAP_PhasePlotFig(cell_id, pAD_dataframe, V_array) :
    '''
    Input pAD_dataframe corresponding to cell_id and V_array
    '''
    # Rename vars: 
    pAD_df = pAD_dataframe
    V      = V_array  
    plot_forwards_window = 50 
    voltage_max = 60.0 
    voltage_min = -120.0
    
    # pAD subdataframe and indices
    pAD_sub_df = pAD_df[pAD_df.pAD =="pAD"] 
    pAD_upshoot_indices = pAD_sub_df[["upshoot_loc", "AP_sweep_num"]].values

    # Somatic subdataframe and indices
    Somatic_sub_df = pAD_df[pAD_df.pAD =="Somatic"] 
    Somatic_upshoot_indices = Somatic_sub_df[["upshoot_loc", "AP_sweep_num"]].values
    
    # # Plotter for pAD and Somatic Spikes but separated into DRD, CTG, TLX celltypes
    
    fig, ax = plt.subplots()
    lines  = []  # initialise empty line list 
    
    for idx in range(len(pAD_upshoot_indices)):
        
        
        
        v_temp = V[ pAD_upshoot_indices[:,0][idx] :  pAD_upshoot_indices[:,0][idx] +  plot_forwards_window  ,  pAD_upshoot_indices[:,1][idx]    ]
        dv_temp = np.diff(v_temp) 
        
        if max(v_temp) > voltage_max or min(v_temp) < voltage_min:             # don't plot artifacts
            pass
        else:
            line, = ax.plot(v_temp[:-1], dv_temp , color = 'salmon', alpha=0.05)        
            lines.append(line)
        
    for idx_ in range(len(Somatic_upshoot_indices)): 
        
        
        v_temp = V[ Somatic_upshoot_indices[:,0][idx_]  :  Somatic_upshoot_indices[:,0][idx_] + plot_forwards_window   ,  Somatic_upshoot_indices[:,1][idx_]    ]
        dv_temp = np.diff(v_temp) 
        
        if max(v_temp) > voltage_max or min(v_temp) < voltage_min:             # don't plot artifacts
            pass
        else:
            line, = ax.plot(v_temp[:-1], dv_temp , color = 'cornflowerblue' , alpha=0.05)
            lines.append(line)
            
        
        # Create the custom legend with the correct colors
        legend_elements = [Line2D([0], [0], color='salmon', lw=2, label='pAD Ensemble'),
                           Line2D([0], [0], color='cornflowerblue', lw=2, label='Somatic Ensemble')]

        # Set the legend with the custom elements
        ax.legend(handles=legend_elements)
    
    plt.title(cell_id)
    plt.ylabel(' dV (mV)')
    plt.xlabel(' Membrane Potential (mV)')
    plt.tight_layout
    plt.show()    
    return fig 

def buildAP_PCAFig(cell_id, pAD_dataframe, V_array):

    '''
    PCA plotter build on top of pAD labelling
    '''
    
    # Rename vars: 
    pAD_df = pAD_dataframe
    V      = V_array  
    
    # pAD subdataframe and indices
    pAD_sub_df = pAD_df[pAD_df.pAD =="pAD"] 
    pAD_upshoot_indices = pAD_sub_df[["upshoot_loc", "AP_sweep_num"]].values

    # Somatic subdataframe and indices
    Somatic_sub_df = pAD_df[pAD_df.pAD =="Somatic"] 
    Somatic_upshoot_indices = Somatic_sub_df[["upshoot_loc", "AP_sweep_num"]].values
    
    X = pAD_df[["AP_slope", "AP_threshold", "AP_height", "AP_latency"]]
    
    
    y = pAD_df['pAD'] 
    
        
    # Standardize the features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    # Perform PCA with 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    
    # Plot the PCA results with different colors for each AP_type label
    
    fig, ax = plt.subplots()
    
    
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c= list(y.map({"Somatic": 'cornflowerblue' , "pAD": 'salmon'})))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(cell_id)
    
    # Create the custom legend with the correct colors
    legend_elements = [Line2D([0], [0], color='salmon', lw=2, label='pAD Ensemble'),
                       Line2D([0], [0], color='cornflowerblue', lw=2, label='Somatic Ensemble')]

    # Set the legend with the custom elements
    ax.legend(handles=legend_elements)
    plt.show()
    
    return fig 

def buildAP_HistogramFig(cell_id, pAD_dataframe, V_array):
    
    # Rename vars: 
    pAD_df = pAD_dataframe
    V      = V_array  
    
    # Define colors 
    colors = ['salmon', 'cornflowerblue' ]
    plot_labels = ['pAD' , 'Somatic' ]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Add each subplot to the figure
    
    for idx in [0,1] : 
        axs[0, 0].hist(pAD_df[pAD_df["pAD"] == plot_labels[idx] ]["AP_threshold"], bins=20, color= colors[idx], label =plot_labels[idx] )
        axs[0, 1].hist(pAD_df[pAD_df["pAD"] == plot_labels[idx] ]["AP_slope"], bins=20, color= colors[idx], label =plot_labels[idx])
        axs[1, 0].hist(pAD_df[pAD_df["pAD"] == plot_labels[idx] ]["AP_height"], bins=20, color= colors[idx], label =plot_labels[idx])
        axs[1, 1].hist(pAD_df[pAD_df["pAD"] == plot_labels[idx] ]["AP_latency"], bins=20, color= colors[idx], label =plot_labels[idx])

    # Add x and y labels to each subplot
    for ax in axs.flat:
        ax.set(ylabel='Counts')
    axs[0,0].set_xlabel('Membrane Potential (mV)')
    axs[0,1].set_xlabel('Volts/sec')
    axs[1,0].set_xlabel('Potential Difference (mV)')
    axs[1,1].set_xlabel('Latency (ms)')

    # Add a legend to each subplot
    for ax in axs.flat:
        ax.legend()

    # Add a title to each subplot
    axs[0, 0].set_title('Voltage Thresholds')
    axs[0, 1].set_title('AP Slopes')
    axs[1, 0].set_title('AP Heights')
    axs[1, 1].set_title('Peak Latency')
    
    
    fig.tight_layout()   
    plt.suptitle(cell_id)
    plt.show()
    
    return fig
    
    
