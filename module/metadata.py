####### FUNCTION THAT DEAL WITH SAVING INFORMATIONS ABOUT EXPERIMENTS ###########

# create cell dict with cell_type > cell_subtype > cell_id #TODO
# prints list of : cell_types, cell_subtype, drug and data_type
# cell_dict = checkFeatureDF(filename, from_scratch=True) ### WAS IN RUN IF NEED TO REBUILD REBUILD will begood for checking n and logging outliers 



# import os
# from module.constants import CACHE_DIR
# from module.getters import getRawDf
# from module.utils import checkFileSystem, getJSON, saveJSON, isCached


# #takes featureDF and outputs a nested dict checking for unique celltype for aingle cell id and prints basic readout 
# def checkFeatureDF(filename, from_scratch=None): #may be redundant dont know if will use or just to check input file
#     filename_no_extension = filename.split(".")[0]
#     df = getRawDf(filename)
#     #prints
#     print(f"cell types : {df['cell_type'].unique()}")
#     print(f" drugs applied : {df['drug'].unique()}")
#     print(f"cell subtypes : {df['cell_subtype'].unique()}")
#     print(f"data types : {df['data_type'].unique()}")

#     identifier = 'cell_type_dict'
#     from_scratch = from_scratch if from_scratch is not None else input("Recalculate DF even if previous version exists? (y/n)") == 'y'
#     if from_scratch or not isCached(filename, identifier):
#         print(f'BUILDING "{identifier}"') 
#         cell_type_dict = build_cell_type_dict(df)
#         subcache_dir = f"{CACHE_DIR}/{filename.split('.')[0]}"
#         saveJSON(f"{subcache_dir}/cell_type_dict.json", cell_type_dict)
#         print (f"CELL TYPE DICT {cell_type_dict} SAVED TO {subcache_dir}")
#     else:
#         getJSON(f"{subcache_dir}/cell_type_dict.json")
#     return cell_type_dict



# #creates a dictionay of cell ids for each cell type and subtype 
# #TODO does not yet include different data type but should
# def build_cell_type_dict(df):
#     # check and report folder_file's missing a cell_type
#     folder_files_missing_cell_type = df[df['cell_type'].isnull()]['folder_file'].tolist() if not df[df['cell_type'].isnull()].empty else []
#     if folder_files_missing_cell_type:
#         print(f'files missing cell type are {folder_files_missing_cell_type}')

#     #create dictionary of cell id's by cell type / subtype
#     cell_type_dict = {}
#     cell_id_tracker = {} #dictionary to track cell_id and their locations ( type / subtype )

#     for _, row in df.iterrows():
#         cell_type = row['cell_type']
#         cell_subtype = row['cell_subtype']
#         cell_id = row['cell_id']
        
#         if cell_id in cell_id_tracker: # Check if the cell_id is already tracked for a different combination type/subtype
            
#             conflicting_combination = False
#             for existing_type, existing_subtype in cell_id_tracker[cell_id]:
#                 if existing_type != cell_type or existing_subtype != cell_subtype:
#                     conflicting_combination = True
#                     break
            
#             if conflicting_combination:
#                 raise ValueError(f"Duplicate cell_id '{cell_id}' with different combinations: "
#                                 f"{existing_type}, {existing_subtype} and {cell_type}, {cell_subtype}")
            
#             if cell_subtype not in cell_type_dict[cell_type]:
#                 cell_type_dict[cell_type][cell_subtype] = [cell_id]
#             elif cell_id not in cell_type_dict[cell_type][cell_subtype]:
#                 cell_type_dict[cell_type][cell_subtype].append(cell_id)
            
#             cell_id_tracker[cell_id].append((cell_type, cell_subtype))
#         else:
#             cell_id_tracker[cell_id] = [(cell_type, cell_subtype)]
#             cell_type_dict.setdefault(cell_type, {}).setdefault(cell_subtype, []).append(cell_id)

    
#     return cell_type_dict


