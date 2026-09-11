from module.figure_common import *

@dataclass
class DataSelection (Cachable): 
    ''' 
    Dataselection for a single data_type based on the folder_files used in the cell_df.

    Attributes:
        - project (str): defining the project and feature mapping ie RAW_df in Ephys
        - data_type (str): The data type 
        - filters (dict): Generic factor filters, e.g. {"pre_patch": "EPM", "region": ["aIC"]}.
        - threshold_access_change (float): The threshold for access change filtering (optional, default 30).
        '''

    project: str  = field(kw_only=True)
    project_obj: Project = field(init=False, repr=False)

    data_type: str = field(kw_only=True, default=None)
    filters: dict = field(kw_only=True, default_factory=dict)
    threshold_access_change: float = field(kw_only = True, default=30) # Rs % change threshold


    def __post_init__(self):

        super().__init__(cache_dir=f"{ROOT}/{self.project}/cache")#HACKY SHIT 
        self.location = f"{ROOT}/{self.project}"
        self.input_dir = self._checkFileSystem("input")
        self.output_dir = self._checkFileSystem("output")
        self.figure_output_dir = self._checkFileSystem("figures")

        self.project_obj = Project(self.project) #gives self.project_type to include time or not 
        self.load_extractor(self.data_type)
        self.cell_df = Ephys(self.project).df         

        self.validate_inputs() #except dv
        self.valid_files, self.valid_cell_ids = self.get_valid_folder_files()
        self.agg_df = self.build_agg_df() #validates dv
        if  self.project_obj.project_type == "application":
            self.treatment_count_df = self.generate_treatment_count_df() # not generic enpough yet #TODO
        # elif self.project_obj.project_type == ""

    def load_extractor(self, data_type: str):
        """Dynamically load the extractor class from Ephys_Project by name (data_type)."""
        module = importlib.import_module("module.Ephys_Project")  
        try:
            cls = getattr(module, data_type)  
        except AttributeError:
            raise ValueError(f"No extractor class found for data_type '{data_type}' in Ephys_Project")
        
        df = cls(self.project).df
        setattr(self, f"{data_type}_df", df)  
        return df


    def validate_inputs(self):
        """
        Validate data type and requested filters.
        """
        data_types=['APP_IC', 'st_VC', 'ramp_IC', 'IV_VC', 'spont_IC', 'IF_IC', 'PPR_VC' ]
        if self.data_type not in data_types: #complete list of data types
            raise ValueError(f"Invalid data_type: {self.data_type}. Must be one of {data_types}.")
        
        #handel subgroup prefix
        folder_file_cols = [c for c in self.cell_df.columns if c.endswith(f"{self.data_type}_folder_files")]
        valid_df = self.cell_df[self.cell_df[folder_file_cols].notna().any(axis=1)]

        # else:
        #     valid_df = self.cell_df

        def validate_attribute(column_name, attribute):
            if column_name not in valid_df.columns:
                raise ValueError(
                    f"Cannot filter on '{column_name}' in project {self.project}. "
                    f"Use filters={{...}} for project-specific factors. "
                    f"Available filters are: {self.available_selection_filters(valid_df)}"
                )
            valid_values = valid_df[column_name].dropna().unique()
            requested_values = self.filter_values(attribute)
            invalid_values = [val for val in requested_values if val not in valid_values]
            if invalid_values:
                raise ValueError(
                    f"Invalid value(s) for filter '{column_name}': {invalid_values}. "
                    f"Available values are: {list(valid_values)}."
                )

        for column_name, attribute in self.selection_filters().items():
            validate_attribute(column_name, attribute)
   
    def get_valid_folder_files(self):
        """
        Filters the cell_df based on the input parameters including threshold_access_change if not None.
        Returns a list of valid folder_files and cell_ids.
        """
        # valid_column = f'{self.data_type}_folder_files'
        # if valid_column not in self.cell_df.columns:
        #     raise ValueError(f"{valid_column} column does not exist in cell_df.")
        
        # find all relevant folder_files columns for this data_type
        folder_file_cols = [c for c in self.cell_df.columns if c.endswith(f"{self.data_type}_folder_files")]
        if not folder_file_cols:
            raise ValueError(f"No columns found for {self.data_type}_folder_files in cell_df.")

        
        filtered_cell_df = self.cell_df.copy()

        def collect_folder_files(cell_df):
            cell_ids = cell_df['cell_id'].tolist()
            files = [
                f
                for row in cell_df[folder_file_cols].dropna().values.tolist()
                for cell in row
                for f in (cell if isinstance(cell, list) else [cell])
            ]
            return files, cell_ids

        def collect_cell_ids_for_files(cell_df, folder_files):
            folder_file_set = set(folder_files)

            def row_has_valid_file(row):
                for value in row[folder_file_cols].dropna():
                    files = value if isinstance(value, list) else [value]
                    if any(file in folder_file_set for file in files):
                        return True
                return False

            return cell_df[cell_df.apply(row_has_valid_file, axis=1)]['cell_id'].tolist()

        #apply filters if set
        for column_name, attribute in self.selection_filters().items():
            requested_values = self.filter_values(attribute)
            filtered_cell_df = filtered_cell_df[filtered_cell_df[column_name].isin(requested_values)]
       
        valid_files, valid_cell_ids = collect_folder_files(filtered_cell_df)

        if (
            self.threshold_access_change is not None
            and self.project_obj.project_type == "intrinsic_properties"
        ):
            valid_files, valid_cell_ids = self.access_filtered_folder_files(valid_files)

        elif (
            self.threshold_access_change is not None
            and self.project_obj.project_type == "application"
        ):
            file_valid_files, file_valid_cell_ids, used_file_access = self.access_filtered_folder_files(
                valid_files,
                return_used=True,
                skip_if_no_values=True
            )
            if used_file_access:
                valid_files, valid_cell_ids = file_valid_files, file_valid_cell_ids
                if not valid_cell_ids:
                    valid_cell_ids = collect_cell_ids_for_files(filtered_cell_df, valid_files)
                dropped_cell_ids = sorted(set(filtered_cell_df['cell_id'].dropna()) - set(valid_cell_ids))
                if dropped_cell_ids:
                    print(
                        f"Access filter dropped {len(dropped_cell_ids)} cells with no valid "
                        f"{self.data_type} folder_files: {self.short_list(dropped_cell_ids)}"
                    )
            elif "Rs_pct_change" in filtered_cell_df.columns:
                rs_pct = pd.to_numeric(filtered_cell_df["Rs_pct_change"], errors="coerce")
                if rs_pct.notna().any():
                    before_cell_ids = set(filtered_cell_df['cell_id'].dropna())
                    filtered_cell_df = filtered_cell_df[rs_pct.abs() <= self.threshold_access_change]
                    valid_files, valid_cell_ids = collect_folder_files(filtered_cell_df)
                    dropped_cell_ids = sorted(before_cell_ids - set(valid_cell_ids))
                    if dropped_cell_ids:
                        print(
                            f"Access filter dropped {len(dropped_cell_ids)} cells "
                            f"(threshold_access_change={self.threshold_access_change}%): "
                            f"{self.short_list(dropped_cell_ids)}"
                        )
                else:
                    print("Access filter skipped: no Rs_pct_change values found for this application selection.")
            else:
                print("Access filter skipped: no Rs_pct_change values found for this application selection.")

        if not valid_files:
            print("No valid files found for data selection.")
            return [],[]
        # valid_files = [item for sublist in valid_files for item in sublist] if isinstance(valid_files[0], list) else valid_files

        return valid_files, valid_cell_ids

    def short_list(self, values, max_items=12):
        values = list(values)
        if len(values) <= max_items:
            return values
        return values[:max_items] + [f"... +{len(values) - max_items} more"]

    def access_filtered_folder_files(self, folder_files, return_used=False, skip_if_no_values=False):
        def finish(files, cell_ids, used_access):
            if return_used:
                return files, cell_ids, used_access
            return files, cell_ids

        access_df = self.project_obj.load_access_df()
        if access_df.empty:
            print("Access filter found no access_by_file data.")
            if skip_if_no_values:
                return finish(folder_files, [], False)
            return finish([], [], False)

        candidate_files = pd.DataFrame({"folder_file": list(dict.fromkeys(folder_files))})
        candidate_access = candidate_files.merge(access_df, on="folder_file", how="left")
        candidate_access["Rs_pct_change"] = pd.to_numeric(
            candidate_access["Rs_pct_change"],
            errors="coerce"
        )
        if not candidate_access["Rs_pct_change"].notna().any():
            if skip_if_no_values:
                return finish(folder_files, [], False)
            print("Access filter found no numeric Rs_pct_change values.")
            return finish([], [], False)

        valid_mask = candidate_access["Rs_pct_change"].abs() <= self.threshold_access_change
        valid_access = candidate_access[valid_mask].copy()
        dropped_access = candidate_access[~valid_mask].copy()

        if not dropped_access.empty:
            status_counts = (
                dropped_access["Rs_access_status"]
                .fillna("missing_access_metadata")
                .value_counts()
                .to_dict()
            )
            print(
                f"Access filter dropped {len(dropped_access)} folder_files "
                f"(threshold_access_change={self.threshold_access_change}%): {status_counts}"
            )

        valid_files = valid_access["folder_file"].dropna().tolist()
        valid_cell_ids = valid_access["cell_id"].dropna().unique().tolist()
        return finish(valid_files, valid_cell_ids, True)

    def selection_filters(self):
        return {
            column_name: value
            for column_name, value in (self.filters or {}).items()
            if value is not None
        }

    def filter_values(self, value):
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def format_selection_value(self, value, for_filename=False):
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
            values = [str(item) for item in list(value)]
            return "-".join(values) if for_filename else ", ".join(values)
        return str(value)

    def selection_label_parts(self, keys=None, for_filename=False):
        active_filters = self.selection_filters()
        if keys is not None:
            active_filters = {
                key: active_filters[key]
                for key in keys
                if key in active_filters
            }

        labels = []
        for key, value in active_filters.items():
            value_label = self.format_selection_value(value, for_filename=for_filename)
            labels.append(
                f"{key}_{value_label}" if for_filename else f"{key} = {value_label}"
            )
        return labels

    def filter_label(self, key, default=None):
        value = self.selection_filters().get(key, default)
        if value is None:
            return default
        return self.format_selection_value(value, for_filename=False)

    def available_selection_filters(self, df=None):
        columns = []
        for source in [df, getattr(self, "cell_df", None)]:
            if source is not None:
                columns.extend(source.columns.tolist())

        if hasattr(self, "project_obj"):
            columns.extend(getattr(self.project_obj, "subject_independant_vairables", []))
            columns.extend(getattr(self.project_obj, "cell_independant_vairables", []))

        return list(dict.fromkeys(columns))

    def reduce_series(s):
            vals = s.dropna()
            if len(vals) == 0:
                return np.nan
            # flatten nested lists
            if any(isinstance(v, list) for v in vals):
                out = []
                for v in vals:
                    if isinstance(v, list):
                        out.extend(v)
                    else:
                        out.append(v)
                return out
            uniq = vals.unique()
            # collapse if identical
            if len(uniq) == 1:
                return uniq[0]
            # otherwise preserve structure
            return vals.tolist()

    def application_time_label(self, row):
        return self.project_obj.application_time_label(row)

    def sweep_index(self, row, column_name, default):
        value = row.get(column_name, default)
        if pd.isna(value):
            return default
        return int(value)

    def build_agg_df(self):
        """
        Filters self.{data_type}_df for foler_files in cell_df["f{data_type}_folder_files"] and restructures it to a long format for plotting.
        
        Returns:
          agg_df aggregate df for data_type for stats and plotting (one row per cell_id and time).

        """
        data_type_df = getattr(self, f"{self.data_type}_df")

        
        independant_vairables = self.project_obj.data_independant_columns()

        # Determine which dependent variables exist for this data_type
        valid_dvs_for_data_type = [col for col in data_type_df.columns if col not in independant_vairables]

        # Filter only valid folder_files
        filtered_df = data_type_df[data_type_df['folder_file'].isin(self.valid_files)].copy()

        # DATA TYPE SPECIFIC AGGREGATION #
        if self.data_type == "PPR_VC": # combine all sweeps per cell_id + ISI + treatment
            rows = []
            for keys, sub in filtered_df.groupby(["cell_id", "treatment", "ISI_ms"]):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                row = dict(zip(["cell_id", "treatment", "ISI_ms"], keys))
                for col in sub.columns:
                    if col in row or col in {"error", "traceback"}:
                        continue
                    row[col] = DataSelection.reduce_series(sub[col])
                row["folder_files"] = sub["folder_file"].tolist()
                if "PPR" in row and isinstance(row["PPR"], list): # seperate raw values and average first 8
                    row["PPR_raw"] = row["PPR"]
                    # row["PPR"] = row["PPR"][:8] if len(row["PPR"]) >= 8 else np.nan # HARD CODE
                    row["PPR"] = np.nanmean(row["PPR"][:8]) if len(row["PPR"]) >= 6 else np.nan
                rows.append(row)
            filtered_df = pd.DataFrame(rows)

        # ACCOUNTING FOR TWO PROJECT TYPES #
        if  self.project_obj.project_type == "intrinsic_properties":
            # Intrinsic properties: no time, just keep cell_id, folder_file, and dependent variables
            cols_to_keep = (
                ['cell_id', 'folder_file'] +
                self.data_type_selector_columns() +
                valid_dvs_for_data_type
            )
            cols_to_keep = [
                col for col in dict.fromkeys(cols_to_keep)
                if col in filtered_df.columns
            ]
            filtered_df = filtered_df[cols_to_keep]

            agg_df = self.add_cell_mapping(filtered_df)
            return agg_df

        elif self.project_obj.project_type == "application": # all data_type files other than APP_IC will be PRE and POST

            # should become a generic for loop for each data_type PRE and POST 
            if self.data_type == 'IF_IC': 
                filtered_df = self.IF_IC_df[self.IF_IC_df['folder_file'].isin(self.valid_files)].copy()
                filtered_df['time'] = filtered_df.apply(self.application_time_label, axis=1)


                #aggregate appropriate cols ie not "I_steps_pA" , "AP_frequencies_Hz" / 'V_step_steady_mV', 'I_steady_pA'
                for col in ['AP_max_rise_mV_ms', 'AP_height_mV', 'AP_latency_ms', 'AP_peaks_mV',
                            'AP_decay_mV_ms', 'AP_rise_mV_ms', 'AP_width_ms', '%_sag', 'IF_voltage_threshold_mV']:
                    if col in filtered_df.columns:
                        filtered_df[col] = filtered_df[col].apply(lambda x: np.mean(x) if isinstance(x, list) else x)
                metric_cols = [
                    'AP_max_rise_mV_ms', 'AP_height_mV', 'AP_latency_ms',
                    'AP_peaks_mV', 'AP_decay_mV_ms', 'AP_rise_mV_ms',
                    'AP_width_ms', 'IF_slope', 'max_firing_Hz',
                    'IF_rheobase_pA', 'I_step_max_firing_pA',
                    '%_sag', 'IF_voltage_threshold_mV'
                ]
                metric_cols = [col for col in metric_cols if col in filtered_df.columns]
                agg_dict = {col: 'mean' for col in metric_cols}
                agg_IF_IC_df = filtered_df.groupby(['cell_id', 'time']).agg(agg_dict).reset_index()
                folder_files = (
                    filtered_df
                    .groupby(['cell_id', 'time'])['folder_file']
                    .apply(list)
                    .reset_index(name='folder_files')
                )
                agg_IF_IC_df = agg_IF_IC_df.merge(folder_files, on=['cell_id', 'time'], how='left')
                agg_IF_IC_df = self.add_cell_mapping(agg_IF_IC_df, additional_cols=['I_set', 'treatment'])
                return agg_IF_IC_df

            elif self.data_type == 'APP_IC': #
                filtered_df = self.APP_IC_df[self.APP_IC_df['folder_file'].isin(self.valid_files)].copy()
                timepoints = ['PRE', 'APP', 'WASH']
                sweep_vars = [
                    'AP_count',
                    'RA_count',
                    'SAP_count',
                    'RMP_mV',
                    'inputR_MOhm',
                    'sEPSP_frequency_Hz',
                    'sEPSP_count',
                    'sEPSP_mean_amplitude_mV',
                ]
                reshaped_data = []

                for timepoint in timepoints:
                    current_data = pd.DataFrame()
                    current_data['cell_id'] = filtered_df['cell_id']
                    current_data['folder_file'] = filtered_df['folder_file']
                    current_data['time'] = timepoint
                    if 'sweep_duration_s' in filtered_df.columns:
                        current_data['sweep_duration_s'] = filtered_df['sweep_duration_s']

                    for base in sweep_vars:
                        sweep_col = f'sweep_{base}'
                        if sweep_col not in filtered_df.columns:
                            continue

                        # Filter out invalid types (not list/array)
                        mask_invalid = filtered_df[sweep_col].apply(lambda x: not isinstance(x, (list, np.ndarray)))
                        dropped_cells = filtered_df.loc[mask_invalid, 'cell_id'].unique()
                        # if len(dropped_cells) > 0:
                        #     print(f"Cells with invalid {sweep_col}: {dropped_cells} for {timepoint} set to NaN.")
                        valid_df = filtered_df.loc[~mask_invalid].copy()

                        # Align to time window
                        def extract_time_segment(row):
                            sweep_data = row[sweep_col]
                            drug_in = self.sweep_index(row, 'drug_in', 0)
                            drug_out = self.sweep_index(row, 'drug_out', len(sweep_data))
                            if timepoint == 'PRE':
                                return sweep_data[:drug_in]
                            elif timepoint == 'APP':
                                return sweep_data[drug_in:drug_out]
                            elif timepoint == 'WASH':
                                return sweep_data[drug_out:]
                            return []

                        # Apply extraction + mean aggregation
                        extracted = valid_df.apply(extract_time_segment, axis=1)
                        current_data[sweep_col] = extracted
                        current_data[base] = extracted.apply(self.mean_or_nan)

                    reshaped_data.append(current_data)

                agg_APP_IC_df = pd.concat(reshaped_data, ignore_index=True)
                agg_APP_IC_df = self.add_cell_mapping(agg_APP_IC_df, additional_cols=['I_set', 'treatment'])
                return agg_APP_IC_df

            else:
                filtered_df = data_type_df[data_type_df['folder_file'].isin(self.valid_files)].copy()
                if filtered_df.empty:
                    return self.add_cell_mapping(filtered_df, additional_cols=['I_set', 'treatment'])

                filtered_df['time'] = filtered_df.apply(self.application_time_label, axis=1)
                cols_to_keep = (
                    ['cell_id', 'folder_file', 'time'] +
                    self.data_type_selector_columns() +
                    valid_dvs_for_data_type
                )
                cols_to_keep = [
                    col for col in dict.fromkeys(cols_to_keep)
                    if col in filtered_df.columns and col != 'treatment'
                ]
                filtered_df = filtered_df[cols_to_keep]
                return self.add_cell_mapping(filtered_df, additional_cols=['I_set', 'treatment'])
            
    def data_type_selector_columns(self):
        """
        Columns needed to select a specific dependent-variable value.
        """
        selector_cols = {
            "PPR_VC": ["ISI_ms"],
            "IF_IC": ["I_steps_pA"],
        }
        return selector_cols.get(self.data_type, [])

    def add_cell_mapping(self, df, additional_cols: list = None):
        """
        Adds cell feature columns based off cell_id in cell_df.

        Parameters:
            df (pd.DataFrame): DataFrame to merge with cell_df features.
            additional_cols (list, optional): List of extra columns to include from cell_df.

        Returns:
            pd.DataFrame: Merged DataFrame with added features.
        """
        if 'cell_id' not in df.columns or 'cell_id' not in self.cell_df.columns:
            raise ValueError("Both DataFrames must have 'cell_id' column.")

        # Always include cell_id and project-specific subject/cell factors
        columns_to_map = self.project_obj.subject_cell_factor_columns(include_cell_id=True)

        # Add any extra columns specified by the caller
        if additional_cols:
            for col in additional_cols:
                if col in self.cell_df.columns and col not in columns_to_map:
                    columns_to_map.append(col)

        missing_columns = [
            column for column in columns_to_map
            if column not in self.cell_df.columns
        ]
        if missing_columns:
            print(f"Warning: columns missing from cell_df and not mapped: {missing_columns}")
            columns_to_map = [
                column for column in columns_to_map
                if column in self.cell_df.columns
            ]

        return df.merge(self.cell_df[columns_to_map].drop_duplicates(), on='cell_id', how='left')



    def generate_treatment_count_df(self) -> pd.DataFrame:
        '''
        Calculates the n for each treatment x cell_type given the threshold_access_change, saved as excel in cache.
        TODO: 
        '''

        def is_valid_app(app_valid):
            return isinstance(app_valid, str) and len(app_valid) > 0
        def is_valid_fp(fp_valid):
            return isinstance(fp_valid, list) and all(isinstance(x, str) for x in fp_valid)
        def process_group(group_df):
            valid_fp = group_df[group_df['IF_IC_folder_files'].apply(is_valid_fp)]
            valid_app = group_df[group_df['APP_IC_folder_files'].apply(is_valid_app)]

            fp_count = valid_fp['cell_id'].nunique()
            app_count = valid_app['cell_id'].nunique()
            both_valid_count = group_df[
                group_df['IF_IC_folder_files'].apply(is_valid_fp) & group_df['APP_IC_folder_files'].apply(is_valid_app)
            ]['cell_id'].nunique()

            cell_id_fp = valid_fp['cell_id'].unique().tolist()
            cell_id_app = valid_app['cell_id'].unique().tolist()

            return pd.Series({
                'FP_valid_count': fp_count,
                'APP_valid_count': app_count,
                'both_valid_count': both_valid_count,
                'cell_id_FP': cell_id_fp,
                'cell_id_APP': cell_id_app
            })
        if self.threshold_access_change is not None and "Rs_pct_change" in self.cell_df.columns:
            rs_pct = pd.to_numeric(self.cell_df['Rs_pct_change'], errors="coerce")
            if rs_pct.notna().any():
                access_filtered_df = self.cell_df[rs_pct.abs() <= self.threshold_access_change]
            elif self.project_obj.project_type == "application":
                access_df = self.project_obj.load_access_df()
                if not access_df.empty and "Rs_pct_change" in access_df.columns:
                    access_df = access_df.copy()
                    access_df["Rs_pct_change"] = pd.to_numeric(access_df["Rs_pct_change"], errors="coerce")
                    if access_df["Rs_pct_change"].notna().any():
                        access_df = access_df[access_df["Rs_pct_change"].abs() <= self.threshold_access_change]
                        valid_access_cells = access_df["cell_id"].dropna().unique()
                        access_filtered_df = self.cell_df[self.cell_df["cell_id"].isin(valid_access_cells)]
                    else:
                        print("Treatment count access filter skipped: no numeric Rs_pct_change values.")
                        access_filtered_df = self.cell_df
                else:
                    print("Treatment count access filter skipped: no numeric Rs_pct_change values.")
                    access_filtered_df = self.cell_df
            else:
                print("Treatment count access filter skipped: no numeric Rs_pct_change values.")
                access_filtered_df = self.cell_df
        else:
            access_filtered_df = self.cell_df

        treatment_count_df = access_filtered_df.groupby(['treatment', 'cell_type']).apply(process_group).reset_index()
        self.cache(f'treatment_count_df_{self.threshold_access_change}', treatment_count_df)
        self.save_excel( f'treatment_count_df_{self.threshold_access_change}', treatment_count_df)
        return treatment_count_df
        
    def fetch_data_type (self, folder_file):
        '''
        Finds relevant info for folder_file and returns truple of:  string lable , the row from the df
        '''
        data_type_df = getattr(self, f"{self.data_type}_df")
        if folder_file in self.IF_IC_df['folder_file'].values:
            row = self.IF_IC_df[self.IF_IC_df['folder_file'] == folder_file]
            return 'IF_IC', row
        elif folder_file in self.APP_IC_df['folder_file'].values:
            row = self.APP_IC_df[self.APP_IC_df['folder_file'] == folder_file]
            return 'APP_IC', row
        else:
            return 'Unknown', None

        
    def folder_file_AP_df(self, cell_id, folder_file,  V_array=None, I_array=None, command_array=None, sampling_rate=None):
        '''builds action potential pd.DataFrame 'AP_df' for single folder_file.'''
        if V_array is None:
            V_array, I_array, command_array, stim_array, V_list = self.project_obj.load_data(folder_file)
            sampling_rate = self.project_obj.sampling_rate
        elif sampling_rate is None:
            sampling_rate = getattr(self.project_obj, "sampling_rate", 2e4)
        
        V_array, protocol_array, _ = select_protocol_array(
            V_array,
            command_array=command_array,
            I_array=I_array,
            clean_I_fallback=True,
        )

        def protocol_value_at(loc, sweep):
            if protocol_array is None:
                return np.nan
            sweep = min(int(sweep), protocol_array.shape[1] - 1)
            loc = min(int(loc), protocol_array.shape[0] - 1)
            return protocol_array[loc, sweep]
        
        # Extract AP characteristics
        peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_rise_all  , peak_max_dvdt_all,  peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all   , peak_indices_all , sweep_indices_all , peak_decay_all = ap_characteristics_extractor_main(folder_file, V_array, sampling_rate=sampling_rate)
        
        # Early return if no APs found
        if np.all(np.isnan(peak_latencies_all)):
            print(f"No APs detected in voltage trace {folder_file}.")
            return pd.DataFrame(columns=['folder_file', 'peak_location', 'upshoot_location', 'voltage_threshold',
                                        'slope', 'latency', 'peak_voltage', 'height', 'width', 'sweep',
                                        'I_injected', 'AP_type'])
        
        # file_data_type, row_info = self.fetch_data_type(folder_file) #check it always wors
        file_data_type = self.data_type
        row_info = self.project_obj.feature_df[self.project_obj.feature_df['folder_file'] == folder_file]
        if row_info.empty:
            row_info = self.project_obj.feature_df[self.project_obj.feature_df['cell_id'] == cell_id]
        if row_info.empty:
            raise ValueError(f"No features row found for folder_file {folder_file}")
        row_info = row_info.iloc[0]

        drug_used = row_info['treatment']

        if file_data_type == 'APP_IC':
            drug_in = self.sweep_index(row_info, 'drug_in', 0)
            drug_out = self.sweep_index(row_info, 'drug_out', V_array.shape[1])
            drug_labels = ['PRE' if _ < drug_in else 'APP' if drug_in <= _ <= drug_out else 'WASH' for _ in sweep_indices_all]
            current_injected = [
                protocol_value_at(loc, sweep)
                for loc, sweep in zip(peak_locs_corr_all, sweep_indices_all)
            ]

        elif file_data_type == 'IF_IC':
            time_label = self.project_obj.application_time_label(row_info)
            drug_labels = [time_label for _ in sweep_indices_all]
            current_injected = [
                protocol_value_at(loc, sweep)
                for loc, sweep in zip(peak_locs_corr_all, sweep_indices_all)
            ]

        else:
            print(f"JASMINE GENERALISE THIS FOR ALL DATA TYEPS :) !")
            #TODO generalise for all data_types also

        AP_df = pd.DataFrame({
        'folder_file': folder_file,
        'cell_id': cell_id,
        'data_type': file_data_type,
            'cell_treatment': drug_used,
        'treatment': drug_labels,
        'peak_location': peak_locs_corr_all,
        'upshoot_location': upshoot_locs_all,
        'voltage_threshold': v_thresholds_all,
        'peak_rise': peak_rise_all,
        'peak_decay': peak_decay_all,
        'peak_max_rise': peak_max_dvdt_all,
        'latency': peak_latencies_all,
        'peak_voltage': peak_voltages_all,
        'height': peak_heights_all,
        'width': peak_fw_all,
        'sweep': sweep_indices_all,
        'I_injected': current_injected,  # Sweep index is 0 as I_array is identical
        'AP_type': 'somatic'  
        })
        try: # should try ramp_IC for ramp_voltage_threshold_mV first #TODO
            IF_IC_local = IF_IC(self.project_obj.project).df
            pre_mask = IF_IC_local.apply(self.project_obj.application_is_pre, axis=1)
            FP_cell_id_PRE = IF_IC_local[(IF_IC_local['cell_id'] == cell_id) & pre_mask]
            cell_threshold = (FP_cell_id_PRE['IF_voltage_threshold_mV'].apply(lambda x: sum(x) / len(x) if isinstance(x, list) else x)).mean()

            # valid_IF_IC_folder_files = self.cell_df[self.cell_df['cell_id']==cell_id]['FP_valid'].values[0][:2]
            # mean_voltage_threshold = IF_IC_local[IF_IC_local['folder_file'].isin(valid_IF_IC_folder_files)]['voltage_threshold'].explode().astype(float).mean()

            AP_df.loc[(AP_df['voltage_threshold'] < cell_threshold-20 ), 'AP_type'] = 'RA'
        except(IndexError, TypeError):
            print(f" Cell {cell_id} has no valid FP to assess voltage threshold, setting RMP<-65mV")
            AP_df.loc[(AP_df['voltage_threshold'] < -65) , 'AP_type'] = 'RA'

        if file_data_type == 'FP':
            AP_df_positive = AP_df[AP_df['I_injected'] > 0].iloc[:20] #HARD CODE keeping first 20 APs on + I step
            AP_df_non_positive = AP_df[AP_df['I_injected'] <= 0]
            AP_df = pd.concat([AP_df_non_positive, AP_df_positive]).sort_index().reset_index(drop=True)

        return AP_df
