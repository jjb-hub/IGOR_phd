# README

All user functions can use run in the notebook RUN.ipynb

you are welcome, enjoy

jk

INPUT:

project_name>input>features.xlsx
contains nessairy columns: folder_file cell_id treatment data_type
optional columns added for data_type specific analysis or subsetting i.e. drug_in drug_out / cell_type

project_name>input>PatchData>folder_file .avf / .ibw
folder_files mapped to treatment, cell_id and data_type in features.xlsx

OUTPUT:

For each data_type a df is made to extract relevant parameters for folder_file this is handeled by the EphysData parent calss.

Cell_df aggregates some info for cell ? which ?
