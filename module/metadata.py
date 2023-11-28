####### FUNCTION THAT DEAL WITH SAVING INFORMATIONS ABOUT EXPERIMENTS ###########
import os
from module.constants import CACHE_DIR
from module.utils import checkFileSystem, getJSON, saveJSON, getRawDF



#creates a dictionay of cell ids for each cell type and subtype 
#TODO does not yet include different data type but should
def build_cell_type_dict(filename):
    df = getRawDF(filename)
    # check and report folder_file's missing a cell_type
    folder_files_missing_cell_type = df[df['cell_type'].isnull()]['folder_file'].tolist() if not df[df['cell_type'].isnull()].empty else []
    if folder_files_missing_cell_type:
        print(f'files missing cell type are {folder_files_missing_cell_type}')

    #create dictionary of cell id's by cell type / subtype
    cell_type_dict = {}
    cell_id_tracker = {} #dictionary to track cell_id and their locations ( type / subtype )

    for _, row in df.iterrows():
        cell_type = row['cell_type']
        cell_subtype = row['cell_subtype']
        cell_id = row['cell_id']
        
        if cell_id in cell_id_tracker: # Check if the cell_id is already tracked for a different combination type/subtype
            
            conflicting_combination = False
            for existing_type, existing_subtype in cell_id_tracker[cell_id]:
                if existing_type != cell_type or existing_subtype != cell_subtype:
                    conflicting_combination = True
                    break
            
            if conflicting_combination:
                raise ValueError(f"Duplicate cell_id '{cell_id}' with different combinations: "
                                f"{existing_type}, {existing_subtype} and {cell_type}, {cell_subtype}")
            
            if cell_subtype not in cell_type_dict[cell_type]:
                cell_type_dict[cell_type][cell_subtype] = [cell_id]
            elif cell_id not in cell_type_dict[cell_type][cell_subtype]:
                cell_type_dict[cell_type][cell_subtype].append(cell_id)
            
            cell_id_tracker[cell_id].append((cell_type, cell_subtype))
        else:
            cell_id_tracker[cell_id] = [(cell_type, cell_subtype)]
            cell_type_dict.setdefault(cell_type, {}).setdefault(cell_subtype, []).append(cell_id)

    return cell_type_dict


