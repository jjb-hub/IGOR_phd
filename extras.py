# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:43:24 2023 

extra code from patch_daddy (old version)

@author: Debapratim Jana, Jasmine Butler
"""

def buildAggregateDfs(df):
    
    combinations = [
                    (["mouseline", "drug"], _getstats), #stats_df to be fed to next function 
                    (["mouseline", "drug"], _plotwithstats)#finding all combination in df and applying a function to them #here could average and plot or add to new df for stats
                    #(["cell_type", "drug"], _poltbygroup) #same as df.apply as folder_file is unique for each row
    ]
    
    # df_stat = df.copy()
    for col_names, handlingFn in combinations:
        
#here would be adding to the df and 
        # #creating new df roto be fed to next func
        # df_stat = apply_group_by_funcs(df_stat, col_names, handlingFn) #handel function is the generic function to be applied to different column group by
        
        #expand the df expanded
        df = apply_group_by_funcs(df, col_names, handlingFn) #handel function is the generic function to be applied to different column group by
 
    
    return df

#%% EXPANDIN DF ALTERNATE SOLOUTION //// insecured to delete stats  plot structure
def apply_group_by_funcs(df, groupby_cols, handleFn): #creating a list of new values and adding them to the existign df
    res_dfs_li = [] #list of dfs
    for group_info, group_df in df.groupby(groupby_cols):
        
        # print("group_info:", group_info, "Group len:", len(group_df))
        res_df = handleFn(group_info, group_df)
        res_dfs_li.append(res_df)
        
    new_df = pd.concat(res_dfs_li)
    return new_df

def _getstats(mouse_line_drug, df):
    mouse_line, drug = mouse_line_drug
    # fp_idx = cell_df.data_type == "FP" #indexes of FP data in cell_df
    # cell_df.loc[fp_idx, "new_col"] = calcValue()
    df = df.copy()
    df['mean_max_firing'] = df['max_firing'].mean() #dealing with pd.series a ingld df column
    df['mean_voltage_thresh'] = df.voltage_threshold.apply(np.mean) #apply 'loops' through the slied df and applies the function np.mean
    return df    
    


