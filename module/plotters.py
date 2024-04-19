# module
from module.stats import loop_stats, add_statistical_annotation_hues
from module.utils import *
from module.getters import getRawDf, getExpandedDf, getExpandedSubsetDf, getCellDf, getFPStats, getAPPStats
from module.action_potential_functions import pAD_detection
from module.constants import color_dict, unit_dict, n_minimum
#external
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D 
import numpy as np
from IPython.display import display
import seaborn as sns



########## BASE PLOTTERS ##########
def quick_plot_file(filename, folder_file, stacked=False):
    '''
    Plots any waveform based off folder_file.
    Stacked will plot each column on top of each other, defaults to False.
    '''
    df = getRawDf(filename)
    path_V, path_I = make_path(folder_file)
    
    # Plot Voltage Trace
    _, dfV = igor_exporter(path_V)
    V_array = np.array(dfV)
    display(df[df['folder_file'] == folder_file])  # Show file info
    quick_line_plot(V_array, f'Voltage trace for {folder_file}', stacked=stacked)
    
    # Attempt to Plot Current Trace
    try:
        _, dfI = igor_exporter(path_I)
        I_array = np.array(dfI)
        quick_line_plot(I_array, f'Current (I) trace for {folder_file}', stacked=stacked)
    except FileNotFoundError:
        print('No I file found', path_I)

def quick_line_plot(plot_array, plottitle, stacked=False):
    '''
    Plots line plot for given array without adding a legend for stacked plots.
    
    Parameters:
        plot_array (numpy.ndarray): 2D array to plot, where each column is a sweep.
        plottitle (str): Title for the plot.
        stacked (bool): If True, plots each sweep stacked. If False, concatenates sweeps.
    '''
    plt.figure()
    num_sweeps = plot_array.shape[1]
    
    if stacked:
        for i in range(num_sweeps):
            plt.plot(plot_array[:, i])  # Plot each sweep
    else:
        # Concatenate sweeps for continuous plotting
        continuous_plot = plot_array.ravel(order='F')  # Flatten array in column-major order
        plt.plot(continuous_plot)  # Plot continuous
    
    plt.title(plottitle)
    plt.xlabel('Time in ms')
    plt.ylabel('Current in pA')
    plt.show()
    

def draw_paired_lines(ax, data, x_col, y_col, hue_col, identifier_col, order, hue_order):
    # Extract the positions of the points in the swarmplot for each hue level
    points_positions = [ax.collections[i].get_offsets() for i in range(len(hue_order))]

    # Calculate the mean x position for each hue, assuming the dodge is symmetrical
    dodge_widths = [(np.mean(positions[:, 0]) - ax.get_xticks()[0]) for positions in points_positions]
    
    # Iterate over the order of x_col and draw lines for each cell_id
    for i, x_level in enumerate(order):
        # Get the data for this x_level
        subset = data[data[x_col] == x_level]

        # Get the x position for this x_level from the Axes object
        x_base_position = ax.get_xticks()[i]
        
        # Draw lines for each cell_id
        for ident in subset[identifier_col].unique():
            ident_subset = subset[subset[identifier_col] == ident]
            # Make sure we have all hues for this identifier
            if ident_subset[hue_col].nunique() == len(hue_order):
                y_values = ident_subset.set_index(hue_col)[y_col].reindex(hue_order).values
                x_values = [x_base_position + dodge_width for dodge_width in dodge_widths]
                ax.plot(x_values, y_values, color='black', lw=0.5)


def build_FP_figs(filename, compensation_variance=None):
    '''
    Fetches FP_stats df for filename - plots with paired student t PRE vs POST for each cell_type and FP measure (i.e. max_firing)
    input: filename
    output: 
    '''
    df=getFPStats(filename)
    insufficient_data_tracking=getOrBuildDataTracking(filename)

    for (cell_type, measure), subset_raw in df.groupby(['cell_type', 'measure']):
        print (f"plotting {measure} for {cell_type}")
        subset = subset_raw.copy()
        #track groups with n < n_minimum 
        # insufficient_data_tracking = {}
        
        #impliment n_minimum
        treatment_counts = subset.groupby('treatment')['cell_id'].nunique() #n per treatment 
        insufficient_treatments = treatment_counts[treatment_counts < n_minimum].index.tolist()

        if insufficient_treatments:
            # Update the tracking dictionary
            if cell_type not in insufficient_data_tracking:
                insufficient_data_tracking[cell_type] = {}

            for treatment in insufficient_treatments:
                count = int(treatment_counts[treatment])
                if treatment not in insufficient_data_tracking[cell_type]:
                    insufficient_data_tracking[cell_type][treatment] = {}
                
                insufficient_data_tracking[cell_type][treatment]['FP_n'] = count


            print(f"Insufficient data for {cell_type} - Treatments: {', '.join(insufficient_treatments)} (n < {n_minimum})")
            subset = subset[~subset['treatment'].isin(insufficient_treatments)] #remove data 
            
            if len(subset['treatment'].unique()) < 2: #  remaining treatments for plotting
                print(f"Not enough remaining treatments for {cell_type} - {measure}")
                continue  

        saveDataTracking(filename, insufficient_data_tracking)
        order = [t for t in color_dict.keys() if t in subset['treatment'].unique()]

        filtered_df = pd.DataFrame()
        if compensation_variance:
            # check R_series pre and post and include only diff smaller than compensation_variance
            for (cell_id, measure), df_group in subset.groupby(['cell_id', 'measure']):
                pre_mean = df_group[df_group['pre_post'] == 'PRE']['R_series'].explode().mean()
                post_mean = df_group[df_group['pre_post'] == 'POST']['R_series'].explode().mean()
                percent_diff = ((post_mean - pre_mean) / pre_mean) if pre_mean else np.nan

                if pd.isna(pre_mean) or pd.isna(post_mean) or abs(percent_diff) > compensation_variance:
                    # print(f"{cell_id} does not satisfy R_series conditions, compensation_variance : {percent_diff}. Removed.")
                    continue  
                if abs(percent_diff) < compensation_variance:
                    filtered_df = pd.concat([filtered_df, df_group])            

            if filtered_df.empty:
                print(f"No data meets this requirement, plotting all.")
                subset_to_plot = subset.copy()
                legend = 'NO FILTER on data based on access.'
            elif not filtered_df.empty:
                subset_to_plot = filtered_df
                removed_cells = set(subset['cell_id'].unique()) - set(subset_to_plot['cell_id'].unique())
                legend = f'Filtered for access change < {compensation_variance}, {len(removed_cells)} out of {len(subset["cell_id"].unique())} cells removed.'

        else:
            subset_to_plot = subset.copy()
            legend = 'NO FILTER on data based on access.'
        
    
        Fig, ax = plt.subplots(figsize=(20, 10))
        sns.barplot(
            x='treatment',
            y='mean_value',
            hue='pre_post',
            hue_order=['PRE', 'POST'],
            data=subset_to_plot,
            errorbar=('ci', 68),
            palette='Set2',
            edgecolor="k",
            order=order,
            ax=ax,
            )
        sns.swarmplot(
            x='treatment',
            y='mean_value',
            hue='pre_post',
            hue_order=['PRE', 'POST'],
            data=subset_to_plot,
            palette='Set2',
            order=order,
            edgecolor="k",
            linewidth=1,
            linestyle="-",
            dodge=True,
            ax=ax, 
            legend=False,
        )

        # Add significance annotations for paired t-test
        add_statistical_annotation_hues(ax, subset, 'treatment', 'mean_value', 'pre_post', order, ['PRE', 'POST'],  test='paired_ttest', p_threshold=0.05)
        
        # Add line between hues
        draw_paired_lines(ax, subset_to_plot, 'treatment', 'mean_value', 'pre_post', 'cell_id', order, ['PRE', 'POST'])

        # Customize plot labels and titles using ax.set_ methods
        ax.set_xlabel('Treatment', fontsize=14)  # Adjust font size as needed
        ax.set_ylabel(unit_dict[measure], fontsize=14)  # Adjust font size as needed
        ax.set_title(f'{cell_type} - {measure}', fontsize=16)  # Adjust font size as needed
        ax.text(0.01, 0.01, legend, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', 
                bbox=dict(facecolor='white', alpha=0.9))

        # Optionally, adjust the font size of the tick labels
        ax.tick_params(axis='x', labelsize=12)  # Adjust font size for x-axis tick labels
        ax.tick_params(axis='y', labelsize=12)  # Adjust font size for y-axis tick labels
        saveFP_HistogramFig(Fig, f'{cell_type}_{measure}')
      

def build_APP_histogram_figures(filename):
    df=getAPPStats(filename)
    insufficient_data_tracking=getOrBuildDataTracking(filename) #track groups with n < n_minimum 

    for (cell_type,  measure), group in df.groupby(['cell_type', 'measure']):
        subset = group.copy() #avoid SettingWithCopyWarning
        print (f"plotting {measure} for {cell_type}")


        #impliment n_minimum
        treatment_counts = subset.groupby('treatment')['cell_id'].nunique() #n per treatment 
        insufficient_treatments = treatment_counts[treatment_counts < n_minimum].index.tolist()

        if insufficient_treatments:
            # Update the tracking dictionary where n<n_minimum
            if cell_type not in insufficient_data_tracking:
                insufficient_data_tracking[cell_type] = {}

            for treatment in insufficient_treatments:
                count = int(treatment_counts[treatment])
                if treatment not in insufficient_data_tracking[cell_type]:
                    insufficient_data_tracking[cell_type][treatment] = {}
                
                insufficient_data_tracking[cell_type][treatment]['APP_n'] = count

            print(f"Insufficient data for {cell_type} - Treatments: {', '.join(insufficient_treatments)} (n < {n_minimum})")
            subset = subset[~subset['treatment'].isin(insufficient_treatments)] #remove data 
            
            if len(subset['treatment'].unique()) < 2: #  remaining treatments for plotting
                print(f"Not enough remaining treatments for {cell_type} - {measure}")
                continue  

        saveDataTracking(filename, insufficient_data_tracking)

        subset['value'] = pd.to_numeric(subset['value'], errors='coerce') #for add_statistical_annotation_hues 
        n_by_treatment = subset.groupby('treatment')['cell_id'].nunique()  # Calculate 'n' for each treatment
        order = [t for t in color_dict.keys() if t in subset['treatment'].unique()]
        #protocol tracking
        unique_protocols = subset['protocol'].unique()
        markers = ['o', 's', 'D', '^', 'v', '<', '>']
        legend_handles_labels = {}

        


        Fig, ax = plt.subplots(figsize=(20, 10))
        sns.barplot(
                x='treatment',
                y='value',
                hue='pre_app_wash', 
                hue_order=['PRE', 'APP', 'WASH'], 
                data=subset,
                errorbar=('ci', 68),
                palette='Set2',
                edgecolor="k",
                order=order,
                ax=ax,
                )
        for i, protocol in enumerate(unique_protocols):
            marker = markers[i % len(markers)]  # Cycle through markers if more protocols than markers
            protocol_subset = subset[subset['protocol'] == protocol]
            
            sns.stripplot(
                x='treatment',
                y='value',
                hue='pre_app_wash',
                hue_order=['PRE', 'APP', 'WASH'],
                data=protocol_subset,
                palette='Set2',
                order=order,
                edgecolor='k',
                linewidth=1,
                linestyle="-",
                dodge=True,
                ax=ax,
                legend=False,
                size=4,
                marker=marker,  # Assign marker based on protocol
            )

            # Add protocol to legend dictionary
            legend_handles_labels[protocol] = plt.Line2D(
                [0], [0], marker=marker, label=protocol, color='black'
            )

        # Add significance annotations for paired t-test PRE vs APP/WASH
        add_statistical_annotation_hues(ax, subset, 'treatment', 'value', 'pre_app_wash', order,  ['PRE', 'APP', 'WASH'],  test='paired_ttest', p_threshold=0.05)
        # Add lines between same cell paired data 
        draw_paired_lines(ax, subset, 'treatment', 'value', 'pre_app_wash', 'cell_id', order, ['PRE', 'APP', 'WASH'])

        # Annotate each bar with 'n' value
        for i, treatment in enumerate(order):
            n_value = n_by_treatment.get(treatment, 0)
            # Use annotate to position text below the x-axis
            ax.annotate(
                f"n={n_value}", 
                xy=(i, 0), 
                xytext=(0, -20),  # Offset text position (x, y)
                textcoords="offset points",  # Interpret xytext as offset in points
                ha='center', va='top', fontsize=10, color='black',
                xycoords='data'  # Use data coordinates for the xy position
            )

        # Customize plot labels and titles using ax.set_ methods
        ax.set_xlabel('')
        ax.set_ylabel(unit_dict[measure], fontsize=14)  # Adjust font size as needed
        ax.set_title(f'{cell_type} {measure} ', fontsize=16)  # Adjust font size as needed

        current_handles, current_labels = ax.get_legend_handles_labels()

        # Create a combined legend with both hue labels and custom protocol markers
        combined_handles = current_handles + list(legend_handles_labels.values())
        combined_labels = current_labels + list(legend_handles_labels.keys())

        # Set the combined custom legend to the axes
        ax.legend(handles=combined_handles, labels=combined_labels, loc='best', title='Legend')

        # ax.legend(handles=legend_handles_labels.values(), labels=legend_handles_labels.keys()) #legend for protocols

        

        # Optionally, adjust the font size of the tick labels
        ax.tick_params(axis='x', labelsize=12)  # Adjust font size for x-axis tick labels
        ax.tick_params(axis='y', labelsize=12)  # Adjust font size for y-axis tick labels
        saveAPP_HistogramFig(Fig, f'{cell_type}_{measure}')


## LOOPER ## 

#for cell_type and drug

# #right now need expanded df to make ap figuures not ideal

def loopbuildAplicationFigs(filename):
    df = getExpandedDf(filename)
    color_dict = getColors(filename)
    application_df = df[df.data_type == 'APP'] 
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

        buildApplicationFig(cell_id=cell_id, folder_file=folder_file, I_set=I_set, drug=drug, drug_in=drug_in, drug_out=drug_out, application_order=application_order, pAD_locs=None)
        plt.close()
    return

## GETTERS ##

def getorbuildApplicationFig(filename, cell_id_or_cell_df, from_scratch=None):
    if not isinstance(cell_id_or_cell_df, pd.DataFrame):
        expanded_df = getExpandedDf(filename)
        cell_df = getCellDf(expanded_df, cell_id_or_cell_df, data_type = 'APP')
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
            fig = buildApplicationFig(cell_id=cell_id, folder_file=folder_file, I_set=I_set, drug=drug, drug_in=drug_in, drug_out=drug_out, application_order=application_order, pAD_locs=True)
            saveAplicationFig(fig, f'{cell_id}_application_{application_order}')
            plt.close(fig)
            
    else : 
        fig = getCache(filename, cell_id)
    # fig.show()
    
### AP CHARECTRISTICS PLOTTERS

def RA_AP_chatecteristics_plot(filename, cell_id_or_cell_df, data_type = None, from_scratch=None):
        '''
        For a single cell plots all RA/pAD defined APs against all somatic from APP file and first 3 form each FP file. 
        '''
        
        # get df of all files for single cell
        if not isinstance(cell_id_or_cell_df, pd.DataFrame):
            expanded_df = getExpandedDf(filename)
            cell_df = getCellDf(expanded_df, cell_id_or_cell_df, data_type = data_type)
        else:
            cell_df = cell_id_or_cell_df

        # filters FP files where APs occure off I step and threshold <-65mV
        cell_id = cell_df['cell_id'].iloc[0]

        from_scratch = from_scratch if from_scratch is not None else input("Rebuild Fig even if previous version exists? (y/n)") == 'y'
        if from_scratch or not isCached(filename, cell_id):

            print(f'BUILDING "{cell_id} Mean APs Figure"') 

            # for folder_file in 
            folder_file = cell_df['folder_file'].values[0]
            path_V, path_I = make_path(folder_file)
            listV, dfV = igor_exporter(path_V)
            V_array = np.array(dfV)
            peak_voltages_all, peak_latencies_all, peak_locs_corr_all, v_thresholds_all, peak_slope_all, peak_heights_all, pAD_df  = pAD_detection(folder_file,V_array)
            if len(peak_heights_all) <=1:
                return print(f'No APs in trace for {cell_id}')
            fig = buildAP_MeanFig(cell_id, pAD_df, V_array, input_plot_forwards_window  = 50, input_plot_backwards_window= 100)
            saveAP_MeanFig(fig, cell_id)
        else : fig = getCache(filename, cell_id)
        fig.show()


#old 11/4/24
def getorbuildAP_MeanFig(filename, cell_id_or_cell_df, from_scratch=None):
        
        if not isinstance(cell_id_or_cell_df, pd.DataFrame):
            expanded_df = getExpandedDf(filename)
            cell_df = getCellDf(expanded_df, cell_id_or_cell_df, data_type = 'APP')
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
            peak_voltages_all, peak_latencies_all, peak_locs_corr_all, v_thresholds_all, peak_slope_all, peak_heights_all, pAD_df  = pAD_detection(folder_file,V_array)
            if len(peak_heights_all) <=1:
                return print(f'No APs in trace for {cell_id}')
            fig = buildAP_MeanFig(cell_id, pAD_df, V_array, input_plot_forwards_window  = 50, input_plot_backwards_window= 100)
            saveAP_MeanFig(fig, cell_id)
        else : fig = getCache(filename, cell_id)
        fig.show()
        
def getorbuildAP_PhasePlotFig(filename, cell_id_or_cell_df, from_scratch=None):
        if not isinstance(cell_id_or_cell_df, pd.DataFrame):
            expanded_df = getExpandedDf(filename)
            cell_df = getCellDf(expanded_df, cell_id_or_cell_df, data_type = 'APP')
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
            peak_voltages_all, peak_latencies_all, peak_locs_corr_all, v_thresholds_all, peak_slope_all, peak_heights_all, pAD_df  = pAD_detection(folder_file,V_array)
            if len(peak_heights_all) <=1:
                return print(f'No APs in trace for {cell_id}')
            fig =buildAP_PhasePlotFig(cell_id, pAD_df, V_array)
            saveAP_PhasePlotFig(fig, cell_id)
        else : fig = getCache(filename, cell_id)
        fig.show()


def getorbuildAP_PCAFig(filename, cell_id_or_cell_df, from_scratch=None):
        if not isinstance(cell_id_or_cell_df, pd.DataFrame):
            expanded_df = getExpandedDf(filename)
            cell_df = getCellDf(expanded_df, cell_id_or_cell_df, data_type = 'APP')
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
            peak_voltages_all, peak_latencies_all, peak_locs_corr_all, v_thresholds_all, peak_slope_all, peak_heights_all, pAD_df  = pAD_detection(folder_file,V_array)
            if len(peak_heights_all) <=1:
                return print(f'No APs in trace for {cell_id}')
            fig =buildAP_PCAFig(cell_id, pAD_df, V_array)
            saveAP_PCAFig(fig, cell_id)
        else : fig = getCache(filename, cell_id)
        fig.show()

def getorbuildAP_HistogramFig(filename, cell_id_or_cell_df, from_scratch=None):
        if not isinstance(cell_id_or_cell_df, pd.DataFrame):
            expanded_df = getExpandedDf(filename)
            cell_df = getCellDf(expanded_df, cell_id_or_cell_df, data_type = 'APP')
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
            peak_voltages_all, peak_latencies_all, peak_locs_corr_all, v_thresholds_all, peak_slope_all, peak_heights_all, pAD_df  = pAD_detection(folder_file,V_array)
            if len(peak_heights_all) <=1:
                return print(f'No APs in trace for {cell_id}')
            fig =buildAP_HistogramFig(cell_id, pAD_df, V_array)
            saveAP_HistogramFig(fig, cell_id)
        else : fig = getCache(filename, cell_id)
        fig.show()

## BUILDERS ##
def buildApplicationFig( cell_id=None, folder_file=None, I_set=None, drug=None, drug_in=None, drug_out=None, application_order=None, pAD_locs=None):
    #color for faw voltage trace
    if drug is None:
        plot_color = 'k'
    else:
        plot_color = color_dict[drug]
    # colors for APs 
    colors = {'pAD_true': 'salmon', 'pAD_possible': 'orange', 'somatic': 'cornflowerblue'}


    #fectch data from folder_file
    path_V, path_I = make_path(folder_file)
    array_V, V_df = igor_exporter(path_V) # df_y each sweep is a column
    V_array      = np.array(V_df) 
    try:
        array_I, df_I = igor_exporter(path_I) #df_I has only 1 column and is the same as array_I
    except FileNotFoundError: #if no I file exists 
        print(f"no I file found for {cell_id}, I setting used was: {I_set}")
        array_I = np.zeros(len(V_df)-1)

    #scale data
    x_scaler_drug_bar = len(V_df[0]) * 0.0001 # multiplying this  by drug_in/out will give you the point at the end of the sweep in seconds
    x_V = np.arange(len(array_V)) * 0.0001 #sampeling at 10KHz will give time in seconds
    x_I = np.arange(len(array_I))*0.00005 #20kHz igor 

    #plot 
    fig = plt.figure(figsize = (12,9))
    ax1 = plt.subplot2grid((11, 8), (0, 0), rowspan = 8, colspan =11) #(nrows, ncols)
    ax2 = plt.subplot2grid((11, 8), (8, 0), rowspan = 2, colspan=11)
    ax1.plot(x_V, array_V, c = plot_color, lw=1, alpha=0.8) #voltage trace plot # "d", markevery=pAD_locs

    
    pAD_plot_pre_window = 50
    pAD_plot_post_window = 50
    legend_handles = []
    
    if pAD_locs is True: 
        # Get pAD_locs
        peak_voltages_all, peak_latencies_all, peak_locs_corr_all, v_thresholds_all, peak_slope_all, peak_heights_all, pAD_df  = pAD_detection(folder_file,V_array) 

        if np.all(np.isnan(peak_latencies_all)) == False: #if APs detected

            # Plot pADs and Somatics and create legend handles
            for ap_type in pAD_df['AP_type'].unique():
                ap_sub_df = pAD_df[pAD_df.AP_type == ap_type]
                ap_indices = ap_sub_df[['upshoot_loc', 'AP_sweep_num', 'AP_loc']].values

                # Plot each AP
                for ap_idx in range(len(ap_indices)):
                    ap_upshoot_loc, sweep_num, ap_loc = ap_indices[ap_idx]
                    v_temp = np.array(array_V[sweep_num * len(V_df) + ap_upshoot_loc - pAD_plot_pre_window:sweep_num * len(V_df) + ap_loc + pAD_plot_post_window])
                    time_temp = np.linspace((sweep_num * len(V_df) + ap_upshoot_loc - pAD_plot_pre_window) * 0.0001, (sweep_num * len(V_df) + ap_loc + pAD_plot_post_window) * 0.0001, len(v_temp))
                    label = ' '.join([word.title() if word != 'pAD' else 'pAD' for word in ap_type.split('_')])
                    ax1.plot(time_temp, v_temp, c=colors[ap_type], lw=2, alpha=0.5, label=label if ap_idx == 0 else "")

            # Add the legend to the plot
            ax1.legend(loc='upper right')

    
    ax2.plot(x_I, array_I, label = I_set, color=color_dict['I_display'] )
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
    ax1.set_title(cell_id + ' '+ drug +' '+ " Application" + " (" + str(application_order) + ")", fontsize = 16) # , fontsize = 25
    plt.tight_layout()

    return fig

########## AP MATRICS PLOTTERS
def buildAP_MeanFig(cell_id, pAD_dataframe, V_array, input_plot_forwards_window=50, input_plot_backwards_window=100):
    # Initialize necessary variables
    plot_window = input_plot_forwards_window + input_plot_backwards_window
    sampling_rate = 1e4 
    sec_to_ms = 1e3
    AP_types = ['pAD_true', 'pAD_possible', 'somatic']
    colors = {'pAD_true': 'salmon', 'pAD_possible': 'orange', 'somatic': 'cornflowerblue'}
    
    # Prepare the spike array for each AP type
    spike_arrays = {AP_type: [] for AP_type in AP_types}

    # Iterate over each AP type to populate spike_arrays
    for AP_type in AP_types:
        ap_indices = pAD_dataframe[pAD_dataframe['AP_type'] == AP_type][["upshoot_loc", "AP_sweep_num"]].values
        for idx in range(len(ap_indices)):
            lower_bound = max(0, ap_indices[idx, 0] - input_plot_backwards_window)
            upper_bound = ap_indices[idx, 0] + input_plot_forwards_window
            spike_arrays[AP_type].append(V_array[lower_bound:upper_bound, ap_indices[idx, 1]])

    # Generate the plot
    fig, ax = plt.subplots()
    for AP_type in AP_types:
        if spike_arrays[AP_type]:
            mean_spike = np.mean(np.array(spike_arrays[AP_type]), axis=0)
            time_ = sec_to_ms * np.arange(0, len(mean_spike)) / sampling_rate
            ax.plot(time_, mean_spike, color=colors[AP_type], label=f'{AP_type} Mean')

    # Set plot attributes and show
    ax.legend()
    plt.title(cell_id)
    plt.ylabel('Membrane Potential (mV)')
    plt.xlabel('Time (ms)')
    plt.tight_layout()
    plt.show()    
    return fig

# #OLD DJ
# def buildAP_MeanFig(cell_id, pAD_dataframe, V_array, input_plot_forwards_window  = 50, input_plot_backwards_window= 100):
#     # Rename vars: 
#     pAD_df = pAD_dataframe
#     V      = V_array  
#     plot_forwards_window        =  input_plot_forwards_window  
#     plot_backwards_window       =  input_plot_backwards_window
#     plot_window = plot_forwards_window + plot_backwards_window
#     sampling_rate               = 1e4 
#     sec_to_ms                   = 1e3
    
#     # pAD subdataframe and indices
#     pAD_sub_df = pAD_df[pAD_df.pAD =="pAD"] 
#     pAD_ap_indices = pAD_sub_df[["upshoot_loc" , "AP_loc", "AP_sweep_num"]].values

#     # Somatic subdataframe and indices
#     Somatic_sub_df = pAD_df[pAD_df.pAD =="Somatic"] 
#     Somatic_ap_indices = Somatic_sub_df[["upshoot_loc" , "AP_loc", "AP_sweep_num"]].values

#     pAD_spike_array = np.zeros([len(pAD_ap_indices), plot_window  ])
#     Somatic_spike_array = np.zeros([len(Somatic_ap_indices), plot_window ])
    
#     # Plotter for pAD and Somatic Spikes 
#     fig, ax = plt.subplots()
#     lines  = []  # initialise empty line list 

#     for idx in range(len(pAD_ap_indices)): 
#         if plot_backwards_window >=   pAD_ap_indices[:,0][idx]:
#             plot_backwards_window_ = 0
#             plot_forwards_window_  = plot_forwards_window + plot_backwards_window  
#         else: 
#             plot_backwards_window_ = plot_backwards_window
#             plot_forwards_window_  = plot_forwards_window
        
#         pAD_spike_array[idx,:] = V[ pAD_ap_indices[:,0][idx] - plot_backwards_window_ :  pAD_ap_indices[:,0][idx] +  plot_forwards_window_  ,  pAD_ap_indices[:,-1][idx]    ]
#         time_                       = sec_to_ms* np.arange(0, len(pAD_spike_array[idx,:])) / sampling_rate  
#         line, = ax.plot(time_, pAD_spike_array[idx,:] , color = 'salmon', alpha=0.05)
#         lines.append(line)
#         #plt.plot(pAD_spike_array[idx,:], color ='grey', label = 'pAD')
    
#     if pAD_spike_array.shape[0] > 0 :
#         line, = ax.plot(time_, np.mean(pAD_spike_array , axis = 0)  , color = 'red')
#         lines.append(line)
#     else : # no spikes to plot
#         pass

#     for idx_ in range(len(Somatic_ap_indices)): 
        
#         if plot_backwards_window >=   Somatic_ap_indices[:,0][idx_]:
#             plot_backwards_window_ = 0
#             plot_forwards_window_  = plot_forwards_window + plot_backwards_window  
#         else: 
#             plot_backwards_window_ = plot_backwards_window
#             plot_forwards_window_  = plot_forwards_window
            
#         Somatic_spike_array[idx_,:] = V[ Somatic_ap_indices[:,0][idx_] - plot_backwards_window_ :  Somatic_ap_indices[:,0][idx_] + plot_forwards_window_   ,  Somatic_ap_indices[:,-1][idx_]    ]
#         time_                       = sec_to_ms* np.arange(0, len(Somatic_spike_array[idx_,:])) / sampling_rate  
#         line, = ax.plot(time_,Somatic_spike_array[idx_,:] , color = 'cornflowerblue', alpha=0.05)
#         lines.append(line)
#     if pAD_spike_array.shape[0] > 0 :
#         line, = ax.plot(time_, np.mean(Somatic_spike_array , axis = 0)  , c = 'blue')
#         lines.append(line)
#     else : # no spikes to plot
#         pass

#     # Create the custom legend with the correct colors
#     legend_elements = [Line2D([0], [0], color='salmon', lw=2, label='pAD Ensemble', alpha=0.2),
#                        Line2D([0], [0], color='red', lw=2, label= 'pAD Mean'),
#                        Line2D([0], [0], color='cornflowerblue', lw=2, label='Somatic Ensemble', alpha=0.2),
#                        Line2D([0], [0], color='blue', lw=2, label='Somatic Mean')]
    # # Set the legend with the custom elements
    # ax.legend(handles=legend_elements)
        
    # #plt.plot(np.mean(Somatic_spike_array, axis = 0 ) , c = 'blue', label = 'Somatic Mean')
    # plt.title(cell_id)
    # plt.ylabel('Membrane Potential (mV)')
    # plt.xlabel('Time (ms)')
    # plt.tight_layout()
    # plt.show()    
    # return fig 

def buildAP_PhasePlotFig(cell_id, pAD_dataframe, V_array):
    # Initialize necessary variables
    plot_forwards_window = 50
    voltage_max = 60.0
    voltage_min = -120.0
    AP_types = ['pAD_true', 'pAD_possible', 'somatic']
    colors = {'pAD_true': 'salmon', 'pAD_possible': 'orange', 'somatic': 'cornflowerblue'}

    # Generate the plot
    fig, ax = plt.subplots()
    for AP_type in AP_types:
        ap_indices = pAD_dataframe[pAD_dataframe['AP_type'] == AP_type][["upshoot_loc", "AP_sweep_num"]].values
        for idx in range(len(ap_indices)):
            v_temp = V_array[ap_indices[idx, 0]: ap_indices[idx, 0] + plot_forwards_window, ap_indices[idx, 1]]
            dv_temp = np.diff(v_temp)
            if max(v_temp) <= voltage_max and min(v_temp) >= voltage_min:
                ax.plot(v_temp[:-1], dv_temp, color=colors[AP_type], alpha=0.05, label=f'{AP_type} Ensemble')

    # Set plot attributes and show
    plt.title(cell_id)
    plt.ylabel('dV (mV)')
    plt.xlabel('Membrane Potential (mV)')
    plt.legend(handles=[Line2D([0], [0], color=color, lw=2, label=f'{AP_type} Ensemble') for AP_type, color in colors.items()])
    plt.tight_layout()
    plt.show()
    return fig

# DJ OLD
def buildAP_PhasePlotFig(cell_id, pAD_dataframe, V_array) :
    '''
    Input pAD_dataframe corresponding to cell_id and V_array
    '''
    # Rename vars: 
    pAD_df = pAD_dataframe
    # V      = V_array  
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
        
        
        
        v_temp = V_array[ pAD_upshoot_indices[:,0][idx] :  pAD_upshoot_indices[:,0][idx] +  plot_forwards_window  ,  pAD_upshoot_indices[:,1][idx]    ]
        dv_temp = np.diff(v_temp) 
        
        if max(v_temp) > voltage_max or min(v_temp) < voltage_min:             # don't plot artifacts
            pass
        else:
            line, = ax.plot(v_temp[:-1], dv_temp , color = 'salmon', alpha=0.05)        
            lines.append(line)
        
    for idx_ in range(len(Somatic_upshoot_indices)): 
        
        
        v_temp = V_array[ Somatic_upshoot_indices[:,0][idx_]  :  Somatic_upshoot_indices[:,0][idx_] + plot_forwards_window   ,  Somatic_upshoot_indices[:,1][idx_]    ]
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
    
    
