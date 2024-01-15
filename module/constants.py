######### HERE WE DECLARE THE CONSTANTS USED BY OTHER FILES ############
# Constant are meant to be constants, the should not changed, that's what variables or user are for

import os

### Constants that reflect the filesystem structure, used by util functions
ROOT = os.getcwd()  # This gives terminal location (terminal working dir)
INPUT_DIR = f"{ROOT}/input"
OUTPUT_DIR = f"{ROOT}/output"
CACHE_DIR = f"{INPUT_DIR}/cache"

color_dict = {"pAD":"orange",
              "Somatic":"blue",
              "WASH":"lightsteelblue", 
              "PRE":"black", 
              "CONTROL": 'grey', 
              "TCB2":'green', 
              "DMT":"teal", 
              "PSIL":"orange", 
              "LSD":"purple", 
              "MDL":'blue', 
              'I_display':'cornflowerblue'} 

#old comment bellow moved here from mettabuiold funcs now stats 
#FIX ME build these in with simple get or guild functions outside the meta loop also interrate with meta loop
n_minimum = 3 
p_value_threshold=0.05