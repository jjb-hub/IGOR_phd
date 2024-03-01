

import os

### Constants that reflect the filesystem structure, used by util functions
ROOT = os.getcwd()  # This gives terminal location (terminal working dir)
INPUT_DIR = f"{ROOT}/input"
OUTPUT_DIR = f"{ROOT}/output"
CACHE_DIR = f"{INPUT_DIR}/cache"

color_dict = { #drugs
              "CONTROL": 'grey', 
              "TCB2":'green', 
              "DMT":"teal", 
              "PSIL":"orange", 
              "LSD":"purple", 
              "MDL":'blue', 
                #AP figures
              "pAD":"orange",
              "Somatic":"blue",
                #APP figure
              "PRE":"black",
              "WASH":"lightsteelblue", 
              'I_display':'cornflowerblue'} 

unit_dict = {'max_firing': 'Firing_(Hz)', 
            'voltage_threshold':'Voltage_Threshold_(mV)', 
            'AP_height': ' AP_Height_(mV)', 
            'AP_slope': 'AP_slope_(V_s^-1)',
            'AP_width': 'AP_width_(s) ',
            'AP_latency': 'AP_latency_(ms)',
            'tau_rc':'ms',
            'sag':'%',
            'AP_count':'AP_count',
            'AP_count_Somatic_AP':'AP_count_Somatic',
            'AP_count_pAD_True':'AP_count_pAD_True',
            'AP_count_pAD_Possible':'AP_count_pAD_Posible',
            'inputR': 'input R (Ohm)',
            'RMP':'membrane potentiam (mV)',
            'AP_dvdt_max': 'dV/dt (V/s)'
            }

n_minimum = 3 
p_value_threshold=0.05