

import os

### Constants that reflect the filesystem structure, used by util functions
ROOT = os.getcwd()  # This gives terminal location (terminal working dir)
INPUT_DIR = f"{ROOT}/input"
OUTPUT_DIR = f"{ROOT}/output"
CACHE_DIR = f"{INPUT_DIR}/cache"

color_dict = { #drugs
              "CONTROL": 'grey', 
              "CONTROL_CNQX_APP5": 'grey',
              "TCB2":'green', 
              "TCB2_CNQX_APP5": 'green',
              "DMT":"teal", 
              "PSIL":"orange", 
              "PSIL_CNQX_APP5":'PSIL_CNQX_APP5',
              "LSD":"purple", 
              "LSD_CNQX_APP5": 'purple',
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
            'rheobased_threshold': 'Rheobase_Threshold_(pA)',
            'FI_slope': 'Firing_Frequency_/_Current_(pA)',
            'AP_height': ' AP_Height_(mV)', 
            'AP_slope': 'AP_slope_(V_s^-1)',
            'AP_width': 'AP_width_(s) ',
            'AP_latency': 'AP_latency_(ms)',
            'tau_rc':'ms',
            'sag':'%',
            'AP_count_AP_locs':'AP_count', #ODD TODO 
            'AP_count_pAD_locs':'AP_count',
            'inputR': 'input R (Ohm)',
            'RMP':'membrane potentiam (mV)',
            'AP_dvdt_max': 'dV/dt (V/s)'
            }

n_minimum = 3 
p_value_threshold=0.05