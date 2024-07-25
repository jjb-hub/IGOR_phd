# module
from module.stats import buildAggregateDfs, add_statistical_annotation_hues
from module.utils import *
from module.getters import getRawDf, getExpandedDf, getExpandedSubsetDf, getSingleCellDf, getFPAggStats, getAPPAggStats, getfileinfo, get_VandI_arrays_lists
from module.action_potential_functions import pAD_detection, build_AP_DF
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
from itertools import cycle



########## BASE PLOTTERS ##########
def quick_plot_file(filename, folder_file, stacked=False):
    '''
    Plots any waveform based off folder_file.
    Stacked will plot each column on top of each other, defaults to False.
    '''
    df = getRawDf(filename)
    path_V, path_I = make_path(folder_file)
    
    # Plot Voltage Trace
    _, V_array = igor_exporter(path_V)
    # V_array = np.array(dfV)
    display(df[df['folder_file'] == folder_file])  # Show file info
    quick_line_plot(V_array, f'Voltage trace for {folder_file}', stacked=stacked)
    
    # Attempt to Plot Current Trace
    try:
        _, I_array = igor_exporter(path_I)
        # I_array = np.array(dfI)
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
    


def draw_paired_lines(ax, data, x_col, y_col, hue_col, identifier_col, subtype_col, order, hue_order):
    '''
    Draws lines between paired data on histogram.
    '''
    line_styles = cycle(['--', '-.', ':', (0, (3, 10, 1, 10))])  # More styles can be defined
    default_nan_style = '-'  # Simple line style for NaN subtypes
    subtype_styles = {np.nan: default_nan_style}  # Dictionary to store styles

    # Extract unique non-NaN subtypes
    non_nan_subtypes = data[subtype_col].dropna().unique()
    
    # Assign unique line styles to non-NaN subtypes
    for subtype in non_nan_subtypes:
        if subtype not in subtype_styles:
            subtype_styles[subtype] = next(line_styles)
    
    # Extract positions of the points in the swarmplot for each hue
    points_positions = [ax.collections[i].get_offsets() for i in range(len(hue_order))]
    dodge_widths = [np.mean(positions[:, 0]) - ax.get_xticks()[0] for positions in points_positions if positions.size > 0]

    # Iterate over each level of x_col to draw lines
    for i, x_level in enumerate(order):
        subset = data[data[x_col] == x_level]
        x_base_position = ax.get_xticks()[i]

        for ident in subset[identifier_col].unique():
            ident_subset = subset[subset[identifier_col] == ident]
            if ident_subset[hue_col].nunique() == len(hue_order):
                subtype = ident_subset[subtype_col].iloc[0] if not pd.isna(ident_subset[subtype_col].iloc[0]) else np.nan
                line_style = subtype_styles.get(subtype, default_nan_style)

                y_values = ident_subset.set_index(hue_col)[y_col].reindex(hue_order).values
                x_values = [x_base_position + dodge for dodge in dodge_widths]

                # Check if x_values are valid (not masked or NaN)
                if not np.any(np.isnan(x_values)) and not np.ma.is_masked(x_values):
                    ax.plot(x_values, y_values, linestyle=line_style, color='black', lw=0.5)

    # Return the styles for subtypes that need to be added to the legend
    return {k: v for k, v in subtype_styles.items() if k in non_nan_subtypes and k != np.nan}



def build_FP_figs(filename, compensation_variance=None):
    '''
    Fetches FP_agg_stats df for filename - plots with paired student t PRE vs POST for each cell_type and FP measure (i.e. max_firing)
    input: filename
    output: 
    '''
    df=getFPAggStats(filename)
    insufficient_data_tracking=getOrBuildDataTracking(filename)

    for (cell_type, measure), subset_raw in df.groupby(['cell_type', 'measure']):
        print (f"plotting {measure} for {cell_type}")
        subset = subset_raw.copy()
        
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

        # filter for access start subset df --> subset_to_plot
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
        
        
        n_by_treatment = subset_to_plot.groupby('treatment')['cell_id'].nunique()  # Calculate 'n' for each treatment

        #protocol tracking
        unique_protocols = subset_to_plot['protocol'].unique()
        markers = ['o', 's', 'D', '^', 'v', '<', '>']
        legend_handles_labels = {}

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
        
        sns.swarmplot( #TODO hacky fix to show all detaisl
            x='treatment',
            y='mean_value',
            hue='pre_post',
            hue_order=['PRE', 'POST'],
            data=subset_to_plot,
            palette='Set2',
            order=order,
            edgecolor="k",
            linewidth=0.1,
            linestyle="-",
            dodge=True,
            ax=ax, 
            legend=False,
            marker = ".", 
            size = 0.01,
            )

        for i, protocol in enumerate(unique_protocols):
            marker = markers[i % len(markers)]  # Cycle through markers if more protocols than markers
            protocol_subset = subset_to_plot[subset_to_plot['protocol'] == protocol]
            sns.stripplot(
                x='treatment',
                y='mean_value',
                hue='pre_post',
                hue_order=['PRE', 'POST'],
                data=protocol_subset,
                palette='Set2',
                order=order,
                edgecolor="k",
                linewidth=1,
                linestyle="-",
                dodge=True,
                ax=ax, 
                legend=False,
                marker=marker,
            )
            
            # Add protocol to legend dictionary
            legend_handles_labels[protocol] = plt.Line2D(
                [0], [0], marker=marker, label=protocol, color='black'
            )

        # Add significance annotations for paired t-test
        add_statistical_annotation_hues(ax, subset_to_plot, 'treatment', 'mean_value', 'pre_post', order, ['PRE', 'POST'],  test='paired_ttest', p_threshold=0.05)
        
        # Add line between hues  draw_paired_lines(ax, data, x_col, y_col, hue_col, identifier_col, subtype_col, order, hue_order)
        subtype_to_style = draw_paired_lines(ax, subset_to_plot, 'treatment', 'mean_value', 'pre_post', 'cell_id', 'cell_subtype', order, ['PRE', 'POST'])

        current_handles, current_labels = ax.get_legend_handles_labels()
        # Create custom legend handles for the subtypes based on the style dictionary
        subtype_legend_handles = [plt.Line2D([0], [0], color='black', lw=1, linestyle=style, label=subtype) for subtype, style in subtype_to_style.items()]
        
        # # Create a combined legend with both hue labels and custom protocol markers
    
        combined_handles = current_handles + list(legend_handles_labels.values()) + subtype_legend_handles
        combined_labels = current_labels + list(legend_handles_labels.keys())+ [handle.get_label() for handle in subtype_legend_handles]

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
        ax.set_ylabel(unit_dict[measure], fontsize=14)  # Adjust font size as needed
        ax.set_title(f'{cell_type} - {measure}', fontsize=16)  # Adjust font size as needed
        ax.text(0.01, 0.01, legend, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', 
                bbox=dict(facecolor='white', alpha=0.9))

        ax.legend(handles=combined_handles, labels=combined_labels, loc='best', title='Legend')
        # Optionally, adjust the font size of the tick labels
        ax.tick_params(axis='y', labelsize=12)  # Adjust font size for y-axis tick labels
        plt.tight_layout()
        saveFP_HistogramFig(Fig, f'{cell_type}_{measure}')
      

def build_APP_histogram_figures(filename):
    df=getAPPAggStats(filename)
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
        subtype_to_style = draw_paired_lines(ax, subset, 'treatment', 'value', 'pre_app_wash', 'cell_id', 'cell_subtype', order, ['PRE', 'APP', 'WASH'])


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

        # Create custom legend handles for the subtypes based on the style dictionary
        subtype_legend_handles = [plt.Line2D([0], [0], color='black', lw=1, linestyle=style, label=subtype) for subtype, style in subtype_to_style.items()]
        
        # Create a combined legend with both hue labels and custom protocol markers
        combined_handles = current_handles + list(legend_handles_labels.values()) + subtype_legend_handles
        combined_labels = current_labels + list(legend_handles_labels.keys())+ [handle.get_label() for handle in subtype_legend_handles]

        # Set the combined custom legend to the axes
        
        ax.legend(handles=combined_handles, labels=combined_labels, loc='best', title='Legend')
        # Optionally, adjust the font size of the tick labels
        ax.tick_params(axis='x', labelsize=12)  # Adjust font size for x-axis tick labels
        ax.tick_params(axis='y', labelsize=12)  # Adjust font size for y-axis tick labels
        saveAPP_HistogramFig(Fig, f'{cell_type}_{measure}')



## GETTERS ##

def getorbuildApplicationFig(filename, cell_id_or_cell_df, from_scratch=None):
    '''
    Input:
        filename (string)
        cell_id_or_cell_df (pd.DataFrame / string)          subset of expanded df or cell ID
        from_scratch (boolian)                              True / False  to rebuild figure

    Output: 
        Generates and caches / retrives cached figure for APP data.
        '''
    
    #fetch relevant rows in expanded_df
    if not isinstance(cell_id_or_cell_df, pd.DataFrame):
        expanded_df = getExpandedDf(filename)
        cell_df = getSingleCellDf(expanded_df, cell_id_or_cell_df, data_type = 'APP')
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
            cell_df = getSingleCellDf(expanded_df, cell_id_or_cell_df, data_type = data_type)
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
            listV, V_array = igor_exporter(path_V)
            # V_array = np.array(dfV)
            peak_voltages_all, peak_latencies_all, peak_locs_corr_all, v_thresholds_all, peak_slope_all, peak_heights_all, pAD_df  = pAD_detection(folder_file,V_array)
            if len(peak_heights_all) <=1:
                return print(f'No APs in trace for {cell_id}')
            fig = buildAP_MeanFig(cell_id, pAD_df, V_array, input_plot_forwards_window  = 50, input_plot_backwards_window= 100)
            saveAP_MeanFig(fig, cell_id)
        else : fig = getCache(filename, cell_id)
        fig.show()


# nEW 18/5/24
def getorbuildAP_MeanFig(filename, folder_file, from_scratch=None):
        '''
        Gets from cache / builds MeanAPFig for a single file.
        '''
        cashable_folder_file = valdidateIdentifier(folder_file)
        from_scratch = from_scratch if from_scratch is not None else input("Rebuild Fig even if previous version exists? (y/n)") == 'y'
        if from_scratch or not isCached(filename, cashable_folder_file):
            print(f'BUILDING "{folder_file} Mean APs Figure"') 

            V_array, I_array, V_list, I_list = get_VandI_arrays_lists(filename, folder_file)

            AP_df = build_AP_DF(folder_file, V_array, I_array) 
            if AP_df.empty:
                return print(f'No APs in trace for {folder_file}')
            
            fig = buildAP_MeanFig(filename, folder_file, AP_df, V_array, forwards_window=50, backwards_window=50)
            saveAP_MeanFig(fig, cashable_folder_file) 
        else : fig = getCache(filename, cashable_folder_file)
        fig.show()


#old 11/4/24
def getorbuildAP_PhasePlotFig(filename, cell_id_or_cell_df, from_scratch=None):
        if not isinstance(cell_id_or_cell_df, pd.DataFrame):
            expanded_df = getExpandedDf(filename)
            cell_df = getSingleCellDf(expanded_df, cell_id_or_cell_df, data_type = 'APP')
        else:
            cell_df = cell_id_or_cell_df
        cell_id = cell_df['cell_id'].iloc[0]

        from_scratch = from_scratch if from_scratch is not None else input("Rebuild Fig even if previous version exists? (y/n)") == 'y'
        if from_scratch or not isCached(filename, cell_id):
            print(f'BUILDING "{cell_id} Phase Plot Figure"') 
            folder_file = cell_df['folder_file'].values[0]
            path_V, path_I = make_path(folder_file)
            listV, V_array = igor_exporter(path_V)
            # V_array = np.array(dfV)
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
            cell_df = getSingleCellDf(expanded_df, cell_id_or_cell_df, data_type = 'APP')
        else:
            cell_df = cell_id_or_cell_df
        cell_id = cell_df['cell_id'].iloc[0]

        from_scratch = from_scratch if from_scratch is not None else input("Rebuild Fig even if previous version exists? (y/n)") == 'y'
        if from_scratch or not isCached(filename, cell_id):
            print(f'BUILDING "{cell_id} PCA Figure"') 
            folder_file = cell_df['folder_file'].values[0]
            path_V, path_I = make_path(folder_file)
            listV, V_array = igor_exporter(path_V)
            # V_array = np.array(dfV)
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
            cell_df = getSingleCellDf(expanded_df, cell_id_or_cell_df, data_type = 'APP')
        else:
            cell_df = cell_id_or_cell_df
        cell_id = cell_df['cell_id'].iloc[0]

        from_scratch = from_scratch if from_scratch is not None else input("Rebuild Fig even if previous version exists? (y/n)") == 'y'
        if from_scratch or not isCached(filename, cell_id):
            print(f'BUILDING "{cell_id} AP Histogram Figure"') 
            folder_file = cell_df['folder_file'].values[0]
            path_V, path_I = make_path(folder_file)
            listV, V_array = igor_exporter(path_V)
            # V_array = np.array(dfV)
            peak_voltages_all, peak_latencies_all, peak_locs_corr_all, v_thresholds_all, peak_slope_all, peak_heights_all, pAD_df  = pAD_detection(folder_file,V_array)
            if len(peak_heights_all) <=1:
                return print(f'No APs in trace for {cell_id}')
            fig =buildAP_HistogramFig(cell_id, pAD_df, V_array)
            saveAP_HistogramFig(fig, cell_id)
        else : fig = getCache(filename, cell_id)
        fig.show()

## BUILDERS ##
def buildApplicationFig( cell_id=None, folder_file=None, I_set=None, drug=None, drug_in=None, drug_out=None, application_order=None, pAD_locs=False):

    #fectch data from folder_file
    path_V, path_I = make_path(folder_file)
    V_list, V_array = igor_exporter(path_V) # V_array (2D) each sweep is a column

    try:
        I_list, I_array = igor_exporter(path_I) #df_I has only 1 column and is the same as array_I
    except FileNotFoundError: #if no I file exists 
        print(f"no I file found for {cell_id}, I setting used was: {I_set}")
        I_array = np.zeros((len(V_array), 1))

    # sampeling at 20KHz -->  time (s)
    seconds_per_sweep = len(V_array[:,0]) * 0.00005 # multiplying this  by drug_in/out will give you the point at the end of the sweep in seconds
    x_V = np.arange(len(V_list)) * 0.00005 
    x_I = np.arange(len(I_array)) * 0.00005 

    #build figure 
    fig = plt.figure(figsize = (12,9))
    ax1 = plt.subplot2grid((11, 8), (0, 0), rowspan = 8, colspan =11) #(nrows, ncols)
    ax2 = plt.subplot2grid((11, 8), (8, 0), rowspan = 2, colspan=11)

    #plot voltage / time
    ax1.plot(x_V, V_list, c = 'k' if drug is None else color_dict.get(drug, 'k'), lw=1, alpha=0.8) 

    AP_df = build_AP_DF(folder_file, V_array, I_array)
    #label RA APs
    if pAD_locs and not AP_df.empty: 
        AP_df = AP_df.dropna(subset=['AP_type'])
        if not AP_df.empty:
            color_cycle = cycle(['salmon', 'cornflowerblue', 'seagreen'])
            color_map = {}
            
            for ap_type in AP_df['AP_type'].unique(): # loop AP_type 
                if ap_type not in color_map:
                    color_map[ap_type] = next(color_cycle) # assign color 

                ap_sub_df = AP_df[AP_df.AP_type == ap_type]
                for upshoot_location, sweep, peak_location in ap_sub_df[['upshoot_location', 'sweep', 'peak_location']].values: # loop APs for AP_type
                    v_temp = np.array(V_array[ : , sweep] [upshoot_location : peak_location ]) #voltage trace of AP in sweep
                    time_temp = np.linspace(0, len(v_temp) * 0.00005, len(v_temp))  # time values in sweep
                    time_temp += seconds_per_sweep * sweep + upshoot_location * 0.00005 # time values in x_V
                    ax1.plot(time_temp, v_temp, color=color_map[ap_type], lw=2, alpha=0.5, label=ap_type)

            # LEGENDS
            handles, labels = ax1.get_legend_handles_labels()
            unique_labels = list(color_map.keys())  # Unique AP_type labels based on color_map
            ax1.legend(handles, unique_labels, loc='upper right')
    ax2.plot(x_I, I_array, label = I_set, color=color_dict['I_display'] )
    ax2.legend()

    # SPINES
    ax1.spines['top'].set_visible(False) # 'top', 'right', 'bottom', 'left'
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # DRUG APPLICATION BAR
    ax1.axvspan((int((drug_in)* seconds_per_sweep) - seconds_per_sweep), (int(drug_out)* seconds_per_sweep), facecolor = "grey", alpha = 0.2) #drug bar shows start of drug_in sweep to end of drug_out sweep 
    
    #LABELS / TITLES
    ax1.set_xlabel( "Time (s)", fontsize = 12) #, fontsize = 15
    ax1.set_ylabel( "Membrane Potential (mV)", fontsize = 12) #, fontsize = 15
    ax2.set_xlabel( "Time (s)", fontsize = 10) #, fontsize = 15
    ax2.set_ylabel( "Current (pA)", fontsize = 10) #, fontsize = 15
    ax1.set_title(cell_id + ' '+ drug +' '+ " Application" + " (" + str(application_order) + ")", fontsize = 16) # , fontsize = 25
    plt.tight_layout()
    return fig




########## AP MATRICS PLOTTERS

#NEW
def buildAP_MeanFig(filename, folder_file, AP_df, V_array, forwards_window=50, backwards_window=50):
    '''
    Input
        cell_id             (string)    :   used for labeling and caching figure #TODO should add data_type
        AP_df               (pd.DF)     :   each row a information on a single AP in the corrisponding V_array
        V_array             (2D array)  :   voltage trace (mV)
        forwards_window     (int)       :   window after AP peak to plot 
        backwards_window    (int)       :   window before AP peak to plot
    Returns
        fig                             :   figure of each trace plotted and mean trace aligned by peak
    '''
    # Initialize necessary variables
    plot_window = forwards_window + backwards_window
    sampling_rate = 2e4 # HARD CODE
    sec_to_ms = 1e3
    color_cycle = cycle(['cornflowerblue', 'salmon', 'seagreen'])
    color_map = {}

    # Label np.NaN APs in AP_df
    AP_df['AP_type'].fillna('undefined', inplace=True)

    # Prepare spike arrays and color mapping
    spike_arrays = {AP_type: [] for AP_type in AP_df['AP_type'].unique()}
    for ap_type in AP_df['AP_type'].unique():
        if ap_type not in color_map:
            color_map[ap_type] = next(color_cycle)

            # Populate spike arrays
            ap_indices = AP_df[AP_df['AP_type'] == ap_type][["upshoot_location", "sweep"]].values
            for idx in range(len(ap_indices)):
                upshoot_location = ap_indices[idx, 0]
                lower_bound = max(0, upshoot_location - backwards_window)
                upper_bound = upshoot_location + forwards_window
                spike_arrays[ap_type].append(V_array[lower_bound:upper_bound, ap_indices[idx, 1]])

    # Plot mean spikes and individual traces
    fig, ax = plt.subplots()
    for ap_type, spikes in spike_arrays.items():
        if spikes:
            # Plot individual traces with low opacity
            for trace in spikes:
                time_ms = sec_to_ms * np.arange(0, len(trace)) / sampling_rate
                ax.plot(time_ms, trace, color=color_map[ap_type], alpha=0.09, linewidth=1)

            # Plot mean spike
            mean_spike = np.mean(np.array(spikes), axis=0)
            ax.plot(time_ms, mean_spike, color=color_map[ap_type], label=f'{ap_type} mean voltage (n={len(spikes)})', linewidth=1.2, alpha=1)

    # Set plot attributes and show
    ax.legend()
    plt.title(f"{getfileinfo(filename, folder_file, 'cell_id')} {getfileinfo(filename, folder_file, 'data_type')}({folder_file})")
    plt.ylabel('Membrane Potential (mV)')
    plt.xlabel('Time (ms)')
    plt.tight_layout()
    plt.show()    
    return fig


#OLD
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
    
    
