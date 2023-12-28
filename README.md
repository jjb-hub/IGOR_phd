# README

All user functions can use run in the notebook RUN.ipynb

getorbuild_cell_type_dict()
check feature df :
make nested dict with cell_id < cell_subtype < cell_type check no cell_id in two places
print cell_types, cell_subtypes drugs and data_types

getorbuildExpandedDF()
expands input df and caches reporting 'error' and 'traceback' columns for debugging with the error andthe line that caused it

loopStats()
#incomplete

- generate addStats_df for each cell_type / maybe in same df
- outlier checks pre stats

loopFigure()
#incomplete
-take figure type and getorbuildFig() then can loop build for all
-figure types include: APPfig,

JJB

    finish meta loop for APP visuilisation
    handel if drug in/out is not specificed in APP file ... e.g control / put other data_type pAD to analise pAD from -100mV
    change AP where it means Application to APP also in excel file !

DJ
tau and sage normalised by V dif for analysis

0.  add PC to AP_PCA, debug AP_Mean (remove prints), organise histogram count/width + x-axis for AP heights/latency wrong, add mean to phase plotting like AP_Mean, ADD to all handel if AP_count = 0 #notworking No APs in trace for TLX221230b, TLX230622b and TLX230416a both have 1 AP (i would like to see it plotted alone as Somatic/unknown) ... odd case: TLX210603a histogram has count2 + no pAD (bad detection), TLX230518b/c/d has large artifacts that need to be handeled,

1.  APP files compare V trace in PRE / APP / WASH with APs removed and if >2SD the report:
    CLASIFY CELL RESPONCE (depol - hyperpol - no change)
    TIMESCALE OF RESPONCE ()

2.  pAD_detector : add if V_threshold<=-60mV == pAD , no pAD in ANY SIM CELLS: use as failed cases for pAD detection (example loss of access=SIM230225b) good detection of pAD =

3.  add compare PRE and DRUG tau and sag [value, V_steady_state, I_injected, resting_membrant_potential] normalised for a similar defelction in V or comparable V value i.e. -100

4.  AP width correction : " AP width calculation not accurate!! "

5.  build IV curves for pre and post drug aplication compare liniar slope
