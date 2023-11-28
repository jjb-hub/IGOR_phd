########## GETTERS ##########

from module.constants import INPUT_DIR
from module.utils import *
from module.metadata import build_cell_type_dict
import pandas as pd






#takes filename or df,  the cell id and optional data type returns subset of df
def getCellDF(filename_or_df, cell_id, data_type = None):
    if not isinstance(filename_or_df, pd.DataFrame):
        df=getRawDF(filename_or_df)
    else:
        df=filename_or_df

    cell_df = df[df['cell_id']==cell_id]
    if data_type is not None:
        cell_df = cell_df[cell_df['data_type']== data_type]
    return cell_df

#take filename and runs full df expansion using 
def getorbuildExpandedDF(filename, identifier, builder_cb, from_scratch=None):
    filename_no_extension = filename.split(".")[0]
    from_scratch = from_scratch if from_scratch is not None else input("Recalculate DF even if previous version exists? (y/n)") == 'y'
    if from_scratch or not isCached(filename, identifier):
        print(f'BUILDING "{identifier}"')    #build useing callback otherwise and cache result
        df = builder_cb(filename)
        cache(filename_no_extension, identifier, df)
    else : df = getCache(filename, identifier)
    return df

def getorbuild_cell_type_dict(filename, from_scratch=None): #may be redundant dont know if will use or just to check input file
    filename_no_extension = filename.split(".")[0]
    identifier = 'cell_type_dict'
    from_scratch = from_scratch if from_scratch is not None else input("Recalculate DF even if previous version exists? (y/n)") == 'y'
    if from_scratch or not isCached(filename, identifier):
        print(f'BUILDING "{identifier}"') 
        cell_type_dict = build_cell_type_dict(filename)
        subcache_dir = f"{CACHE_DIR}/{filename.split('.')[0]}"
        saveJSON(f"{subcache_dir}/cell_type_dict.json", cell_type_dict)
        print (f"CELL TYPE DICT {cell_type_dict} SAVED TO {subcache_dir}")
    else:
        getJSON(f"{subcache_dir}/cell_type_dict.json")
    return cell_type_dict