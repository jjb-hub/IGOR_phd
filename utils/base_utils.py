
import os, shutil, itertools, json, time, functools, pickle
import pandas as pd
import igor.packed
import igor.binarywave
import os

######## CONSTANTS ######################
# Constants are like variables that should not be rewritten, they are declared in all caps by convention
ROOT = os.getcwd() #This gives terminal location (terminal working dir)
INPUT_DIR = f'{ROOT}/input'
OUTPUT_DIR = f'{ROOT}/output'
CACHE_DIR = f'{ROOT}/cache'


#IGOR 

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
    # ROOT = os.getcwd() #This gives terminal location (terminal working dir)  #HARD CODE ISH
    # INPUT_DIR = f'{ROOT}/input'
    data_path = f'{INPUT_DIR}/PatchData/'

    extension_V = "Soma.ibw" #HARD CODE SPECIFIC 
    extension_I = "Soma_outwave.ibw" 
    path_V = data_path + folder_file + extension_V
    path_I = data_path + folder_file + extension_I
    return path_V, path_I


def igor_exporter(path):
    ''' 
     Parameters
    ----------
    path: path to .ibw file
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

if __name__ == "__main__": #inbuilt test that will not be excuted unless run inside this file
    print('igor tester being run...')
    point_list, igor_df = igor_exporter('/Users/jasminebutler/Desktop/IGOR_phd/input/PatchData/JJB221230/t15Soma.ibw')



#CACHE SYSTEM
#Check filesystem is set up for write operations
def saveColors(filename, color_dict):
    subcache_dir = f"{CACHE_DIR}/{filename.split('.')[0]}"
    checkFileSystem(subcache_dir)
    saveJSON(f"{subcache_dir}/color_dict.json", color_dict)
    print(f"COLORS {color_dict} SAVED TO {subcache_dir} SUBCACHE")

def getColors(filename):
    return getJSON(f"{CACHE_DIR}/{filename.split('.')[0]}/color_dict.json")

#This function gets JSON files and makes them into python dictionnaries
def getJSON(path):
    with open(path) as outfile:
        loaded_json = json.load(outfile)
    return loaded_json

#This function saves dictionnaries, JSON is a dictionnary text format that you use to not have to reintroduce dictionnaries as variables 
def saveJSON(path, dict_to_save):
    with open(path, 'w', encoding ='utf8') as json_file:
        json.dump(dict_to_save, json_file)

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

#This function gets the dataframes that are cached
def getCache(filename, identifier):
    filename = filename.split(".")[0]
    print(f'GETTING "{identifier}" FROM "{filename}" CACHE')
    with open(f'{CACHE_DIR}/{filename}/{identifier}.pkl','rb') as file:
        return pickle.load(file)
    
#This checks if a particulat dataframe/dataset is cached, return boolean
def isCached(filename, identifier):
    filename = filename.split(".")[0]
    return os.path.isfile(f'{CACHE_DIR}/{filename}/{identifier}.pkl')


#need to also add from_scratch = None 
#need to make fo rloop combinations
def getorBuildExpandedDF(filename, identifier, builder_cb, from_scratch=None):
    filename_no_extension = filename.split(".")[0]
    # if isCached(filename_no_extension, identifier): #Check cache to avoid recalcuating from scratch if alreasy done
    #     return getCache(filename_no_extension, identifier)
    from_scratch = from_scratch if from_scratch is not None else input("Recalculate DF even if previous version exists? (y/n)") == 'y'
    if from_scratch or not isCached(filename, identifier):
        print(f'BUILDING "{identifier}"')    #Build useing callback otherwise and cache result
        df = builder_cb(filename)
        cache(filename_no_extension, identifier, df)
    else : df = getCache(filename, identifier)
    return df

def saveFigure(fig, identifier, fig_type):
    output_subdir = f"{OUTPUT_DIR}/{fig_type}"
    checkFileSystem(output_subdir)
    fig.savefig(f"{output_subdir}/{identifier}.svg")
    print(f'SAVED {output_subdir}/{identifier}.svg') 
    fig.savefig(f"{output_subdir}/{identifier}.png")
    print(f'SAVED {output_subdir}/{identifier}.png') 

def saveAplicationFig(fig, identifier):
    saveFigure(fig, identifier, 'DrugApplication')

def saveHistogramFig(fig, identifier):
    saveFigure(fig, identifier, 'Histogram')


######## INIT ##########
#Start by checking filesystem has all the folders necessary for read/write operations (cache) or create them otherwise
initiateFileSystem()