

import os

### Constants that reflect the filesystem structure, used by util functions
ROOT = os.getcwd()  # This gives terminal location (terminal working dir)
INPUT_DIR = f"{ROOT}/input"
OUTPUT_DIR = f"{ROOT}/output"
CACHE_DIR = f"{INPUT_DIR}/cache"


#all treatments should be included here 
# order used in histogram
# #TODO IMPROVE
color_dict = { 
              # sex
              "M":'lightblue',
              "F":'pink',
              #stress models
              "CTR": 'lightgrey',
              "ELS": 'orange',
              "SD": 'lightslategrey',
              "HCD": 'dodgerblue',
              #drugs
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
              "Vehicle": "#D3D2D3",
              "5MeO": "#FAAF40" ,
                #regions
              "aIC":'red',
              "pIC":'blue',
                #AP figures
              "RA":"orange",
              "Somatic":"blue",
                #APP figure
              "PRE":"azure",
              "APP": "teal",
              "WASH":"cadetblue", 
              'I_display':'cornflowerblue'} 

# unit_dict = {'max_firing': 'Firing_(Hz)', 
#             'voltage_threshold':'Voltage_Threshold_(mV)', 
#             'rheobased_threshold': 'Rheobase_Threshold_(pA)',
#             'FI_slope': 'Firing_Frequency_/_Current_(pA)',
#             'AP_height': ' AP_Height_(mV)', 
#             'AP_decay_dvdt': 'AP_decay_(V/s)',
#             'AP_rise_dvdt': 'AP_rise_(V/s)',
#             'AP_width': 'AP_width_(s) ',
#             'AP_latency': 'AP_latency_(ms)',
#             'tau_rc':'ms',
#             '%_sag':'%',
#             'AP_count':'AP_count', #ODD TODO 
#             'RA_count':'AP_count',
#             'inputR': 'input R (MOhm)',
#             'RMP':'membrane potential (mV)',
#             'AP_dvdt_max': 'dV/dt (V/s)',

#             #added but needs to be stadardised with FP and APP in project_type application
#             'v_thresh_mV':'Voltage_Threshold_(mV)', 
#             'AP_height_mV':' AP_Height_(mV)', 
#             'AP_rise_mV_ms': 'AP_rise_(V/s)', 
#             'AP_decay_mV_ms': 'AP_decay_(V/s)',
#             'rheobase_pA': 'Rheobase_Threshold_(pA)'
#             }

unit_dict = {
    "M":"Male",
    "F":"Female",

    "Rs_MOhm": "Series Resistance (MΩ)", 
    "Rm_MOhm": "Membrane Resistance (MΩ)",
    "tau_ms": "Time Constant (ms)",
    "Cm_pF": "Membrane Capacitance (pF)",

    "%_sag": "Sag (%)",

    "ramp_rheobase_pA": "Rheobase Threshold (pA)",
    "ramp_voltage_threshold_mV": "Voltage Threshold (mV)",

    "IF_rheobase_pA": "Rheobase Threshold (pA)",
    "IF_voltage_threshold_mV": "Voltage Threshold (mV)",
    "IF_slope": "Firing Frequency / Current (pA)",
    "AP_frequencies_Hz": "Firing Frequency (Hz)",
    "max_firing_Hz": "Max Firing (Hz)",
    "off_step_peak_locs": "Off Step Peak Locations",
    "holding_I": "Holding Current (pA)",

    "AP_peaks_mV": "AP Peaks (mV)",
    "AP_height_mV": "AP Height (mV)",
    "AP_width_ms": "AP Width (ms)",
    "AP_rise_mV_ms": "AP Rise (mV/ms)",
    "AP_decay_mV_ms": "AP Decay (mV/ms)",
    "AP_latency_ms": "AP Latency (ms)",
    "AP_max_rise_mV_ms": "AP Max Rise (mV/ms)",
    
    "I_steps_pA": "Injected Current (pA)", # in IC
    "V_steps_mV": "Voltage Steps (mV)", # in VC
    "I_steady_pA": "Steady-State Current (pA)",
    "RMP_mV": "Resting Membrane Potential (mV)",
    "V_step_steady_mV": "Steady-State Voltage on step (mV)",

    'sEPSP_frequency_Hz': 'sEPSP frequency (Hz)', 
    'sEPSP_rise_times_ms': 'sEPSP rise time (ms)', 
    'sEPSP_amplitudes_mV': 'sEPSP amplitude (mV)',

    "AP_count": "number of APs",
    "RA_count": "number of retroaxonal APs",
    "SAP_count": "number of somatic APs",
    "inputR_MOhm": "Input resistance (MΩ)",

    "PPR": "Paired Pulse Ratio",

}


n_minimum = 3 
p_value_threshold=0.05