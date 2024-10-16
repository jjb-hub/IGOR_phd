from module.constants import INPUT_DIR, CACHE_DIR, OUTPUT_DIR
import os, shutil, itertools, json, time, functools, pickle
import pandas as pd
import igor2 as igor
import os
import re
import numpy as np
import matplotlib.pyplot as plt



######### IGOR ##########

def load_file(folder_file):
    path_V, path_I = make_path(folder_file)
    V_list, V_array = igor_exporter(path_V)
    I_list, I_array = None, None
    try:
        I_list, I_array = igor_exporter(path_I)
    except FileNotFoundError:
        I_array = None
        # print(f'I file not found, path: {path_I}')
    return V_array , I_array, V_list
        

def make_path(folder_file): 
    '''
    Parameters       
    ----------
    folder_file : 'folder_file'
    Returns
    -------
    path_V : string - path for V data 
    path_I : string - path for I data 
    '''
    data_path = f'{INPUT_DIR}/PatchData/'
    extension_V = "Soma.ibw" #HARD CODE  
    extension_I = "Soma_outwave.ibw" 
    path_V = data_path + folder_file + extension_V
    try:
        path_I = data_path + folder_file + extension_I
    except:
        path_I = np.nan 
        
    return path_V, path_I


def igor_exporter(path):
    ''' 
     Parameters
    ----------
    path: path to .ibw file
    Returns
    -------
    'point_list' (list): a continious points  (combining sweeps)  
    'V_array_2d' (array): a 2d array with each column corisponding to one sweep  
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
    
    V_array_2d = np.array(igor_df)
    
    return (point_list, V_array_2d)

def get_absolute_column_value (df, column_name):
    '''
    Checks for absoloute value in column and returns it.
    '''
    if df[column_name].isna().all():
        return np.nan
    else:
        unique_values = df[column_name].unique()
        if len(unique_values) == 1:
            return unique_values[0]
        else:
            print(f"Error: Multiple unique values found in column '{column_name}'.")
            return None

######### CACHE SYSTEM and SAVING ##########
#figures
IDENTIFIERS={}

def valdidateIdentifier(identifier):
    invalid_chars_pattern = r'[\/:*?"<>|%\&#$@!=+,;\'`~]'
    sanitized_identifier = re.sub(invalid_chars_pattern, "_", identifier)
    if re.search(invalid_chars_pattern, identifier):
        print("Invalid characters in identifier, replacing with '_' ")
    return sanitized_identifier



def saveDataTracking(filename, insufficient_data_tracking):
    subcache_dir = f"{CACHE_DIR}/{filename.split('.')[0]}"
    json_path = f"{subcache_dir}/insufficient_data_tracking.json"
    print(f"Saving data tracking to: {json_path}") 
    checkFileSystem(subcache_dir)
    saveJSON(f"{subcache_dir}/insufficient_data_tracking.json", insufficient_data_tracking)
    print(f"DATA TRACKING {insufficient_data_tracking} SAVED TO {subcache_dir} SUBCACHE")

def getOrBuildDataTracking(filename):
    subcache_dir = f"{CACHE_DIR}/{filename.split('.')[0]}"
    json_path = f"{subcache_dir}/insufficient_data_tracking.json"
    print(f"Attempting to load data tracking from: {json_path}") 
    # Check cache 
    if isCached(filename, 'insufficient_data_tracking', extension='json'):
        return getJSON(f"{subcache_dir}/insufficient_data_tracking.json")
    # Build 
    else:
        print (f"INITIATING DATA TRACKING")
        insufficient_data_tracking = {}
        return insufficient_data_tracking

#This function saves dictionnaries, JSON is a dictionnary text format that you use to not have to reintroduce dictionnaries as variables 
def saveJSON(path, dict_to_save):
    with open(path, 'w', encoding ='utf8') as json_file:
        json.dump(dict_to_save, json_file)

#This function gets JSON files and makes them into python dictionnaries
def getJSON(path):
    with open(path) as outfile:
        loaded_json = json.load(outfile)
    return loaded_json


def checkFileSystem(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)

#This checks that the filesystem has all the requisite folders (input, cache, etc..) and creates them if not
def initiateFileSystem():
    checkFileSystem(INPUT_DIR)
    checkFileSystem(CACHE_DIR)
    checkFileSystem(OUTPUT_DIR)

#This function deletes all cached files, it is used when you want to start from square one because all intermediary results will be cached
def resetCache():
    shutil.rmtree(CACHE_DIR)
    os.mkdir(CACHE_DIR)
    print('CACHE CLEARED')

#This function cahces (aka saves in a easily readable format) all dataframes used itdnetifier is the name of the .pkl and to_cache is the object
def cache(filename, identifier, to_cache):
    filename = filename.split(".")[0]
    cache_subdir = f'{CACHE_DIR}/{filename}'
    checkFileSystem(cache_subdir)
    with open(f'{cache_subdir}/{identifier}.pkl','wb') as file:
        pickle.dump(to_cache, file)
    print(f'CREATED {cache_subdir}/{identifier}.pkl CACHE')

def cache_excel(filename, identifier, to_cache):
    filename = filename.split(".")[0]
    cache_subdir = f'{CACHE_DIR}/{filename}'
    checkFileSystem(cache_subdir)
    excel_path = f'{cache_subdir}/{identifier}.xlsx'
    to_cache.to_excel(excel_path, index=False, engine='openpyxl')
    print(f'CREATED {excel_path} EXCEL CACHE')

#This function gets the dataframes that are cached
def getCache(filename, identifier):
    filename = filename.split(".")[0]
    print(f'GETTING "{identifier}" FROM "{filename}" CACHE')
    with open(f'{CACHE_DIR}/{filename}/{identifier}.pkl','rb') as file:
        return pickle.load(file)
    
#This checks if a particulat dataframe/dataset is cached, return boolean
def isCached(filename, identifier, extension='pkl'):
    filename = filename.split(".")[0]
    print(f"{CACHE_DIR}/{filename}/{identifier}.{extension}")
    return os.path.isfile(f"{CACHE_DIR}/{filename}/{identifier}.{extension}")



######### SAVE ##########

def saveFigure(fig, identifier, fig_type):
    output_subdir = f"{OUTPUT_DIR}/{fig_type}"
    checkFileSystem(output_subdir)
    fig.savefig(f"{output_subdir}/{identifier}.svg")
    print(f'SAVED {output_subdir}/{identifier}.svg') 
    fig.savefig(f"{output_subdir}/{identifier}.png")
    print(f'SAVED {output_subdir}/{identifier}.png')

# def saveHistogramFig(fig, identifier): #example structure of save function
#     saveFigure(fig, identifier, 'Histogram')
def saveAplicationFig(fig, identifier):
    saveFigure(fig, identifier, 'DrugApplication')

def saveAP_MeanFig(fig, identifier):
    saveFigure(fig, identifier, 'Mean_APs')

def saveAP_PhasePlotFig(fig, identifier):
    saveFigure(fig, identifier, 'PhasePlot_APs')

def saveFP_HistogramFig(fig, identifier):
    saveFigure(fig, identifier, 'FP_Histograms')

def saveAPP_HistogramFig(fig, identifier):
    saveFigure(fig, identifier, 'APP_Histograms')

def saveAP_PCAFig(fig, identifier):
    saveFigure(fig, identifier, 'PCA_APs')

def saveAP_HistogramFig(fig, identifier):
    saveFigure(fig, identifier, 'Histogram_APs')

########## SUBSELECTOR ##########
def maskDf(df, mask_conditions):
    complex_filter = True
    for column, value in mask_conditions.items():
        if isinstance(value, list):
            atomic_filter = df[column].isin(value)
        else:
            atomic_filter = df[column] == value
        complex_filter &= atomic_filter
    return complex_filter

def subselectDf(df, subselection):
    df = df[
        maskDf(df, subselection)
    ]
    return df