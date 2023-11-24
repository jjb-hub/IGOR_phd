# README

Patch Daddy TO DO LIST:

JJB

    outlier checks pre stats (AP width terrible) - violin plotting fuck histograms
    tau and sage normalised by V dif for analysis
    finish meta loop for APP visuilisation
    handel if drug in/out is not specificed in APP file ... e.g control / put other data_type pAD to analise pAD from -100mV
    change AP where it means Application to APP also in excel file !

    Fringe cases: pAD: TLX221229a, TLX230416b, TLX230511b, TLX230530b
    Clear cases: TLX221230a, TLX230411a, TLX230416c, TLX230524b, TLX230524c, TLX230622a
    BAD DETECTION-No pAD: TLX230511a,

DJ

0.  add PC to AP_PCA, debug AP_Mean (remove prints), organise histogram count/width + x-axis for AP heights/latency wrong, add mean to phase plotting like AP_Mean, ADD to all handel if AP_count = 0 #notworking No APs in trace for TLX221230b, TLX230622b and TLX230416a both have 1 AP (i would like to see it plotted alone as Somatic/unknown) ... odd case: TLX210603a histogram has count2 + no pAD (bad detection), TLX230518b/c/d has large artifacts that need to be handeled,

1.  APP files compare V trace in PRE / APP / WASH with APs removed and if >2SD the report:
    CLASIFY CELL RESPONCE (depol - hyperpol - no change)
    TIMESCALE OF RESPONCE ()

2.  pAD_detector : add if V_threshold<=-60mV == pAD , no pAD in ANY SIM CELLS: use as failed cases for pAD detection (example loss of access=SIM230225b) good detection of pAD =

3.  add compare PRE and DRUG tau and sag [value, V_steady_state, I_injected, resting_membrant_potential] normalised for a similar defelction in V or comparable V value i.e. -100

4.  AP width correction : " AP width calculation not accurate!! "

5.  build IV curves for pre and post drug aplication compare liniar slope

###

1. First check all dependencies are installed and paths correct see: requirements.txt
2. Run ap_functions.py
3. Run helper_functions.py
4. Then run ephys_tests.py or patch_main.py. This according ensures that
   we do not have circular imports.
