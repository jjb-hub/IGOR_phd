import pandas as pd
import igor.packed
import igor.binarywave
import os

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
    igor_exporter('/Users/jasminebutler/Desktop/IGOR_phd/input/PatchData/JJB221230/t15Soma.ibw')


def make_path(folder_file): #base_path, 
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
    ROOT = os.getcwd() #This gives terminal location (terminal working dir)  #HARD CODE ISH
    INPUT_DIR = f'{ROOT}/input'
    data_path = f'{INPUT_DIR}/PatchData/'

    extension_V = "Soma.ibw"
    extension_I = "Soma_outwave.ibw" 
    path_V = data_path + folder_file + extension_V
    path_I = data_path + folder_file + extension_I
    return path_V, path_I


## Cache system:
