

from module.constants import INPUT_DIR
from module.utils import *
from module.action_potential_functions import (
    calculate_max_firing, 
    extract_FI_x_y, 
    extract_FI_slope_and_rheobased_threshold, 
    ap_characteristics_extractor_main,
    tau_analyser,
    pAD_detection,
    sag_current_analyser,
    mean_inputR_APP_calculator,
    mean_RMP_APP_calculator, 
    )

import pandas as pd
import numpy as np 
import traceback


########## GETTERS ##########
def getfileinfo(filename, folder_file, colname):
    df = getExpandedDf(filename)
    if colname not in df.columns: # check valid colname
        raise ValueError(f"Column '{colname}' does not exist in the DataFrame.")
    filtered_df = df[df['folder_file'] == folder_file][colname]
    if len(filtered_df) == 1:
        return filtered_df.iloc[0]
    else:
        raise ValueError(f"Expected exactly one row for folder_file '{folder_file}', found {len(filtered_df)} rows.")
    
def get_VandI_arrays_lists(filename, folder_file):
    path_V, path_I = make_path(folder_file)
    I_set = getfileinfo(filename, folder_file, 'I_set')
    V_list, V_array = igor_exporter(path_V) # V_array (2D) each sweep is a column
    try:
        I_list, I_array = igor_exporter(path_I) 
    except FileNotFoundError: #if no I file exists 
        print(f"no I file found for {folder_file}, I setting used was: {I_set}")
        if I_set == 'none':
            I_array = np.zeros((len(V_array), 1))
            I_list = np.zeros(len(V_list))
        else:
            I_array = np.full((len(V_array), 1), 3.333) # for instance unknown holding_I != 0pA of I
            I_list = np.full(len(V_array), 3.333)
    return V_array, I_array, V_list, I_list



def getRawDf(filename):
    return getOrBuildDf(filename, "feature_df", buildRawDf)

def getExpandedDf(filename):
    return getOrBuildDf(filename, "expanded_df", buildExpandedDF)

def getCellDf (filename):
    return getOrBuildDf(filename, 'cell_df', buildCellDf)

def getExpandedSubsetDf(filename, cell_type, from_scratch=None):
    return getorbuildSubselectExpandedDF(filename, f"{cell_type}_expanded_df", buildExpandedDF, cell_type, from_scratch)

def getFPAggStats(filename):
    return getOrBuildDf(filename, "FP_agg_stats", buildFPAggStatsDf)

def getAPPAggStats(filename):
    return getOrBuildDf(filename, "APP_agg_stats", buildAPPStatsDf)

def getFPAPPAggStats(filename):
    return getOrBuildDf(filename, "FP_APP_agg_stats", buildFPAPPStatsDf)


def getSingleCellDf(filename_or_df, cell_id, data_type = None):
    df, filename=getExpandedDfIfFilename(filename_or_df)
    cell_df = df[df['cell_id']==cell_id]
    if data_type is not None:
        cell_df = cell_df[cell_df['data_type']== data_type]
    return cell_df





def getOrBuildDf(filename, df_identifier, builder_cb):
    filename_no_extension = filename.split(".")[0]
    # Check cache to avoid recalcuating from scratch if alreasy done
    if isCached(filename_no_extension, df_identifier):
        return getCache(filename_no_extension, df_identifier)
    # Build useing callback otherwise and cache result
    print(f'BUILDING "{df_identifier}"')
    df = builder_cb(filename)
    cache(filename_no_extension, df_identifier, df)
    return df

    
#expands featureDF into expandedDF if specififed for only a single cell_type 
def getorbuildSubselectExpandedDF(filename, identifier, builder_cb, cell_type, from_scratch):
    filename_no_extension = filename.split(".")[0]
    from_scratch = from_scratch if from_scratch is not None else input("Recalculate DF even if previous version exists? (y/n)") == 'y'
    if from_scratch or not isCached(filename, identifier):
        print(f'BUILDING "{identifier}"')    

        df_raw = getRawDf(filename)
        df_cell_type = df_raw[df_raw['cell_type']== cell_type].copy() #JJB NEW
        
        df = builder_cb(df_cell_type) # buildExpandedDF will be the builder_cb
        cache(filename_no_extension, identifier, df)
    else : df = getCache(filename, identifier)
    return df

def getExpandedDfIfFilename(filename_or_df):
    '''
    input: filename or df to be expanded (df useful in the case of a cell df or some subset of feature df)
    output: df
    '''
    if not isinstance(filename_or_df,  pd.DataFrame):
        filename=filename_or_df
        df = getExpandedDf(filename_or_df)
    else:
        df = filename_or_df
        filename= "feature_df_py.xlsx" #HARD CODED
        print ('using suplied expanded df')
    return df, filename



########## SETTERS ##########

# def updateFPAggStats(filename, rows):
#     FP_agg_stats_df = getFPAggStats(filename)
#     # Create a unique identifier in the existing DataFrame
#     FP_agg_stats_df['unique_id'] = FP_agg_stats_df.apply(lambda row: '_'.join([str(row[col]) for col in ["cell_type", "cell_subtype", "cell_id", "measure", "treatment",'protocol', "pre_post"]]), axis=1)
#     for row in rows:
#         unique_id = '_'.join([str(row[col]) for col in ["cell_type", "cell_subtype",  "cell_id", "measure", "treatment", 'protocol', "pre_post"]])
#         data_row = pd.DataFrame([row])
#         data_row['unique_id'] = unique_id

#         # Check if this unique_id already exists
#         if unique_id in FP_agg_stats_df['unique_id'].values:
#             # Update the existing row
#             match_index = FP_agg_stats_df[FP_agg_stats_df['unique_id'] == unique_id].index[0]
#             FP_agg_stats_df.at[match_index, 'mean_value'] = data_row.at[0, 'mean_value']
#             FP_agg_stats_df.at[match_index, 'file_values'] = data_row.at[0, 'file_values']
#         else:
#             # Append the new row
#             FP_agg_stats_df = pd.concat([FP_agg_stats_df, data_row], ignore_index=True)
#     # Drop the unique identifier column
#     FP_agg_stats_df.drop('unique_id', axis=1, inplace=True)
#     cache(filename, "FP_agg_stats", FP_agg_stats_df)

# def updateAPPAggStats(filename, rows):
#     APP_agg_stats_df = getAPPAggStats(filename)
    
#     # Create a unique identifier 
#     APP_agg_stats_df['unique_id'] = APP_agg_stats_df.apply(lambda row: '_'.join([str(row[col]) for col in ["folder_file", "cell_type", "cell_id", "measure",  "treatment",'protocol', "pre_app_wash"]]), axis=1)
    
#     for row in rows:
#         unique_id = '_'.join([str(row[col]) for col in ["folder_file", "cell_type", "cell_id", "measure",  "treatment", 'protocol', "pre_app_wash"]])
#         data_row = pd.DataFrame([row])
#         data_row['unique_id'] = unique_id

#         if unique_id in APP_agg_stats_df['unique_id'].values: #update row
#             match_index = APP_agg_stats_df[APP_agg_stats_df['unique_id'] == unique_id].index[0]
#             APP_agg_stats_df.at[match_index, 'value'] = data_row.at[0, 'value']
#             APP_agg_stats_df.at[match_index, 'sem'] = data_row.at[0, 'sem']
#         else: #add row
#             APP_agg_stats_df = pd.concat([APP_agg_stats_df, data_row], ignore_index=True)

#     APP_agg_stats_df.drop('unique_id', axis=1, inplace=True)# Drop unique identifier 
#     cache(filename, "APP_agg_stats", APP_agg_stats_df)

############ SETTERS IMPROVED #############

def getAggStats(filename, stat_type):
    if stat_type == 'FP':
        return getFPAggStats(filename)
    elif stat_type == 'APP':
        return getAPPAggStats(filename)
    elif stat_type == 'FP_APP':
        return getFPAPPAggStats(filename) # DEVELOP WHEN NEEDED
    else:
        raise ValueError(f"Unknown stat type: {stat_type}")
    
def updateAggStats(filename, rows, stat_type, unique_id_columns, value_columns):
    agg_stats_df = getAggStats(filename, stat_type)
    
    # Create a unique identifier in the existing DataFrame
    agg_stats_df['unique_id'] = agg_stats_df.apply(lambda row: '_'.join([str(row[col]) for col in unique_id_columns]), axis=1)
    
    for row in rows:
        unique_id = '_'.join([str(row[col]) for col in unique_id_columns])
        data_row = pd.DataFrame([row])
        data_row['unique_id'] = unique_id

        # Check if this unique_id already exists
        if unique_id in agg_stats_df['unique_id'].values:
            # Update the existing row
            match_index = agg_stats_df[agg_stats_df['unique_id'] == unique_id].index[0]
            for value_col in value_columns:
                agg_stats_df.at[match_index, value_col] = data_row.at[0, value_col]
        else:
            # Append the new row
            agg_stats_df = pd.concat([agg_stats_df, data_row], ignore_index=True)

    # Drop the unique identifier column
    agg_stats_df.drop('unique_id', axis=1, inplace=True)
    cache(filename, f"{stat_type}_agg_stats", agg_stats_df)

def updateFPAggStats(filename, rows):
    unique_id_columns = ["cell_type", "cell_subtype", "cell_id", "measure", "treatment", 'protocol', "pre_post"]
    value_columns = ['mean_value', 'file_values']
    updateAggStats(filename, rows, 'FP', unique_id_columns, value_columns)

def updateAPPAggStats(filename, rows):
    unique_id_columns = ["folder_file", "cell_type", "cell_id", "measure", "treatment", 'protocol', "pre_app_wash"]
    value_columns = ['value', 'sem']
    updateAggStats(filename, rows, 'APP', unique_id_columns, value_columns)
    
############ BUILDERS ##########

def buildCellDf(filename):
    ''' 
    builds cell_df with each row a cell_id and valid data is marked True in the columns 'FP' 'APP' or 'FP_APP'
    '''
    df = getRawDf(filename)
    df['treatment'] = df.apply(lambda row: row['drug'] if row['application_order'] == 1 else np.nan, axis=1) #make treatment column

    def check_unique(series, cell_id):
        unique_values = series.dropna().unique()
        if len(unique_values) == 1:
            return unique_values[0]
        else:
            raise ValueError(f"Non-unique values found for cell_id: {cell_id} with values: {unique_values}")

    def apply_check_unique(group):
        cell_id = group.name  
        return group.agg({
            'treatment': lambda series: check_unique(series, cell_id),
            'cell_type': lambda series: check_unique(series, cell_id), 
            'cell_subtype':lambda series: check_unique(series, cell_id)
        })

    # Group by 'cell_id' and apply the function
    cell_df = df.groupby('cell_id').apply(apply_check_unique).reset_index()

    cell_df['FP'] = np.nan
    # cell_df['APP'] = np.nan
    # cell_df['FP_APP'] = np.nan
    cell_df['access_change'] = np.nan


    return cell_df


#row handeler for expandDF
def buildExpandedDF(filename_or_df): 
    '''
    input: filename or df (to allow subsetting)
    output: expanded df with aditional columns 'mouseline' # pandas.errors.SettingWithCopyWarning:
    '''
    if not isinstance(filename_or_df, pd.DataFrame):
        print('fetchig raw df')
        df = getRawDf(filename_or_df)
        cell_df = getCellDf(filename_or_df)
    else:
        df=filename_or_df
        print ('expanding on provided df') #TODO remove find way to add data without rerun


    
    # df=propagate_I_set(df)
    og_columns = df.columns.copy() #origional column order

    df = df.apply(lambda row: _handleFile(row), axis=1) # np.find_common_type is deprecated used in .apply check versions

    #ORDERING DF internal: like this new columns added will appear at the end of the df in the order they were created in _handelfile()
    all_cur_columns = df.columns.copy()
    new_colums_set = set(all_cur_columns ) - set(og_columns) # Subtract mathematical sets of all cols - old columns to get a set of the new columns
    new_colums_li = list(new_colums_set)
    reorder_cols_li =  list(og_columns) + new_colums_li # ['mouseline'] + ordering
    df_reordered = df[reorder_cols_li]    

    return df_reordered

def _handleFile(row):
    '''
    input: single row from feature_df corrisponding to a single file
    output: updated row with aditonal columns
    '''
    row = row.copy() #conserve order
    error_msg = None
    error_traceback = None

    def log_error(msg, tb):
        nonlocal error_msg 
        error_msg=msg
        error_traceback = tb

    #open file
    path_V, path_I = make_path(row.folder_file) 
    V_list, V_array = igor_exporter(path_V)
    # V_array      = np.array(V_df) 
    try: # handel missing I traces (Soma_outwave)
        I_list, I_array = igor_exporter(path_I)
        # I_array      = np.array(I_df)
    except FileNotFoundError:
        I_list, I_df = None, None
        I_array      = None
        print('is this still called? I file not found, path:', path_I)

    try:
        # FP data handeling
        if row.data_type in ["FP", "FP_APP"]: 
            
            print("FP type file")
            print(row.folder_file)
 
            row["max_firing"] = calculate_max_firing(V_array) 

            # get AP data 
            peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_slope_all  ,AP_max_dvdt_all, peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all , sweep_indices , sweep_indices_all  = ap_characteristics_extractor_main(row.folder_file, V_array, all_sweeps=True)
        
            #pAD check for file
            if any(threshold <= -65 and peak_voltage > 20 for peak_voltage, threshold in zip(peak_voltages_all, v_thresholds_all)):
                row['pAD'] = True
                row['pAD_locs'] = [peak_locs_corr_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if threshold <= -65 and peak_voltage > 20]
            
            # x, y for FI curve i.e. step_current_values, ap_counts (on I step) #TODO use off step peak locs? or remove 
            step_current_values, ap_counts, V_rest, off_step_peak_locs, ap_frequencies_Hz = extract_FI_x_y (row.folder_file, V_array, I_array, peak_locs_corr_all, sweep_indices_all) 
            FI_slope, rheobase_threshold = extract_FI_slope_and_rheobased_threshold(row.folder_file, step_current_values, ap_counts) #TODO handel pAD APs better slope fit
            row["rheobased_threshold"] = rheobase_threshold
            row["FI_slope"] = FI_slope


            #only consider APs on a I step using off_step_peak_locs to filter before selecting for 10

            # take first 10 APs 
            row['AP_peak_voltages'] = peak_voltages_all[:10]
            row["voltage_threshold"] = v_thresholds_all[:10]
            row["AP_height"] = peak_heights_all [:10]
            row["AP_width"] = peak_fw_all[:10]
            row["AP_slope"] = peak_slope_all [:10]
            row["AP_latency"] = peak_latencies_all[:10]
            row["AP_dvdt_max"] = AP_max_dvdt_all[:10]

            # [  tau (ms)  ,  steady_state_I  ,  I_injected  ,  RMP  ]
            row["tau_rc"] = tau_analyser(row.folder_file, V_array, I_array, step_current_values, ap_counts)
            # [  sag (%)  ,  steady_state_I  ,  I_injected  ,  RMP  ]
            row["sag"]    = sag_current_analyser(row.folder_file, V_array, I_array, step_current_values, ap_counts)


        # APP handeling
        elif row.data_type == "APP":

            print("AP type file")
            print(row.folder_file)        
            
            #input_R 
            if I_array is not None: #if I exists
                if (I_array[:, 0] != 0).any(): #all columns are the same so check column 1 for non 0 value
                    print(f"I injected, calculating inputR ...")
                    input_R_PRE, input_R_APP, input_R_WASH = mean_inputR_APP_calculator(V_array, I_array, row.drug_in, row.drug_out)
                    row['inputR_PRE'] = input_R_PRE
                    row['inputR_APP'] = input_R_APP
                    row['inputR_WASH'] = input_R_WASH
                    pass_I_array = I_array
                else:
                    print("No I injected, cannot calculate input_R.")
                    row['inputR_PRE'] = []
                    row['inputR_APP'] = []
                    row['inputR_WASH'] = []
                    pass_I_array = None
            else:
                print("I_df does not exist, cannot calculate input_R.")
                row['inputR_PRE'] = []
                row['inputR_APP'] = []
                row['inputR_WASH'] = []
                pass_I_array = None
                
            mean_RMP_PRE, mean_RMP_APP, mean_RMP_WASH = mean_RMP_APP_calculator(V_array, row.drug_in, row.drug_out, I_array=pass_I_array) #with current (I) data 
            row['RMP_PRE'] = mean_RMP_PRE
            row['RMP_APP'] = mean_RMP_APP
            row['RMP_WASH'] = mean_RMP_WASH


            # get AP data 
            peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_slope_all  ,AP_max_dvdt_all, peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all , sweep_indices , sweep_indices_all  = ap_characteristics_extractor_main(row.folder_file, V_array, all_sweeps=True)
            
            #pAD check -- definition voltage threshold of -65mV and peak voltage of > 20mV
            pAD_condition = lambda peak_voltage, threshold: threshold <= -65 and peak_voltage > 20
            if any(pAD_condition(peak_voltage, threshold) for peak_voltage, threshold in zip(peak_voltages_all, v_thresholds_all)):
                row['pAD'] = True
                row['pAD_locs'] = [peak_locs_corr_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if threshold <= -65 and peak_voltage > 20]
                # filter the peak_locs_corr_all  with  pAD condition and sweep index
                row['PRE_pAD_locs'] = [peak_loc for peak_loc, sweep_index, peak_voltage, threshold in 
                                        zip(peak_locs_corr_all, sweep_indices, peak_voltages_all, v_thresholds_all) 
                                        if sweep_index < row['drug_in'] and pAD_condition(peak_voltage, threshold)]
                row['APP_pAD_locs'] = [peak_loc for peak_loc, sweep_index, peak_voltage, threshold in 
                                        zip(peak_locs_corr_all, sweep_indices, peak_voltages_all, v_thresholds_all) 
                                        if row['drug_in'] <= sweep_index <= row['drug_out'] and pAD_condition(peak_voltage, threshold)]
                row['WASH_pAD_locs'] = [peak_loc for peak_loc, sweep_index, peak_voltage, threshold in 
                                        zip(peak_locs_corr_all, sweep_indices, peak_voltages_all, v_thresholds_all) 
                                        if sweep_index > row['drug_out'] and pAD_condition(peak_voltage, threshold)]
            else:
                # row['pAD'] = False
                row['pAD_locs'] = []

            # AP handeling 
            row['AP_locs'] = peak_locs_corr_all
            if len(peak_locs_corr_all) > 0:
                row['PRE_AP_locs'] = [peak_loc for peak_loc, sweep_index in zip(peak_locs_corr_all, sweep_indices) if sweep_index < row['drug_in']]
                row['APP_AP_locs'] = [peak_loc for peak_loc, sweep_index in zip(peak_locs_corr_all, sweep_indices) if row['drug_out'] >= sweep_index >= row['drug_in']]
                row['WASH_AP_locs'] = [peak_loc for peak_loc, sweep_index in zip(peak_locs_corr_all, sweep_indices) if sweep_index > row['drug_out']]
            
            # if valid parameters add True to APP for that cell_if in cell_df
            pass
        
        elif row.data_type == "pAD_hunter":
            # get AP data 
            peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_slope_all  ,AP_max_dvdt_all, peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all , sweep_indices , sweep_indices_all  = ap_characteristics_extractor_main(row.folder_file, V_array, all_sweeps=True)

            #pAD check for file
            if any(threshold <= -65 and peak_voltage > 20 for peak_voltage, threshold in zip(peak_voltages_all, v_thresholds_all)):
                row['pAD'] = True
                row['pAD_locs'] = [peak_locs_corr_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if threshold <= -65 and peak_voltage > 20]
           

        else:
            raise NotADirectoryError(f"Didn't handle data type: {row.data_type}") # data_type can be AP, FP_AP or FP 

    # docuument errors
    except Exception as e:
        error_type = type(e).__name__
        error_tb = traceback.format_exc()
        lines = error_tb.split('\n')
        relevant_tb = []

        for idx, line in enumerate(lines):
            if f'{error_type}: {str(e)}' in line: #take the line before the error message
                relevant_tb = lines[idx - 1].strip()
                break

        log_error(f'{error_type}: {str(e)}', relevant_tb)
        error_traceback = relevant_tb  # Capture traceback within the except block #check redundancey after DJ #28 merge with gpt addition in log_error() def
    if error_msg:
        row['error'] = error_msg
        row['traceback'] = error_traceback
        print(f'{row.cell_id} error message logged: {error_msg}')
        print(f'{row.cell_id} traceback: {error_traceback}')
    else:
        row['error'] = 'ran'
        row['traceback'] = None

    return row


def buildRawDf(filename):
    file_name, file_type = filename.split(".")
    if file_type.lower() != "xlsx":  # Update the file type check for Excel files
        raise Exception(f'METHOD TO DEAL WITH FILE TYPE "{file_type}" ABSENT')
    if not os.path.isfile(f"{INPUT_DIR}/{filename}"):
        raise Exception(f'FILE {filename} IS ABSENT IN "input/" DIRECTORY')

    df = pd.read_excel (f'{INPUT_DIR}/{filename}', converters={'drug_in':int, 'drug_out':int})
    df['cell_subtype'].fillna(np.nan, inplace=True) #for consistency in lack of subtype specification
    return (df)


def buildFPAggStatsDf(filename):
    return pd.DataFrame(
        columns=[
            "cell_type",
            "cell_subtype",
            "cell_id",
            "measure",
            "treatment",
            'protocol', 
            "pre_post",
            "mean_value",
            "file_values",
            "folder_files",
            "R_series",
        ]
    )


def buildAPPStatsDf(filename):
    return pd.DataFrame(
        columns=[
            "folder_file",
            "cell_type",
            "cell_subtype",
            "cell_id",
            "measure",
            "treatment",
            "protocol",             #I_set
            "pre_app_wash",
            "value",
            "sem", # makes sense only for measure == inpur_R , RMP 
        ]
    )

def buildFPAPPStatsDf(filename): #DEVELOP WHEN NEEDED TODO
    return pd.DataFrame(
        columns=[
            "cell_type",
            "cell_subtype",
            "cell_id",
            "measure",
            "treatment",
            'protocol', 
            "pre_drug_post",
            "mean_value",
            "file_values",
            "folder_files",
            "R_series",
        ]
    )
