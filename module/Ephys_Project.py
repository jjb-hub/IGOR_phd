import os
import warnings
import re
import hashlib
import json
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from pyabf import ABF 
from module.Cachable import Cachable
import traceback
from scipy.optimize import curve_fit
from tqdm import tqdm
from IPython.display import display
from itertools import combinations
import igor2 as igor
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from module.action_potential_functions import calculate_max_firing, sweep_mean_RMP_calculator, sweep_mean_inputR_calculator, ap_characteristics_extractor_main, extract_FI_x_y, sag_current_analyser, mean_RMP_APP_calculator, spike_remover_nan, correct_I_offset_IF, denoise_steps, FI_slope_and_rheobase, _step_indices_from_command_trace, command_array_to_match_V, has_protocol_steps, select_protocol_array, EPSP_detector
from scipy.stats import ttest_ind
from scipy.signal import savgol_filter
tqdm.pandas()

# Root directory for projects
ROOT = f"{os.getcwd()}/PROJECTS"
if not os.path.exists(ROOT):
    os.mkdir(ROOT)

DEFAULT_SAMPLING_RATE_HZ = 2e4


@dataclass
class Project(Cachable):
    '''
    Class for handeling file system for a project. 
    '''
    project: str
    input_dir: str = field(init=False)
    output_dir: str = field(init=False)
    figure_output_dir: str = field(init=False)
    accepted_extensions = ['.ibw', '.abf']
    project_type: str = None # user can pass either 'application' or 'intrinsic_properties'

    def __post_init__(self):
        super().__init__(cache_dir=f"{ROOT}/{self.project}/cache")
        self._sampling_rate_warning_files = set()
        self.location = f"{ROOT}/{self.project}"
        self.input_dir = self._checkFileSystem("input")
        self.output_dir = self._checkFileSystem("output")
        self.figure_output_dir = self._checkFileSystem("figures")
        self.feature_df = self.load_feature_xlsx('features')
        self.validate_unique_folder_files()
        self.check_project_type()
        self.classify_independant_variables()
            

    def check_project_type(self):
        if self.project_type is None:
            if "data_type" in self.feature_df.columns:
                unique_types = set(self.feature_df["data_type"].dropna().unique())
                if "APP_IC" in unique_types:
                    self.project_type = "application"
                    self.validate_application_time_column()
                elif unique_types & {"st_VC", "ramp_IC", "IV_VC", "spont_IC", "IF_IC"}:
                    self.project_type = "intrinsic_properties"
                else:
                    raise ValueError("Unrecognized data_type values in features.xlsx.")
            else:
                raise ValueError("features.xlsx must contain 'data_type' column.")
        # print(f"Project type set to: {self.project_type}")

    def validate_unique_folder_files(self):
        """
        Require one features.xlsx row per raw recording.
        """
        if "folder_file" not in self.feature_df.columns:
            return

        folder_files = self.feature_df["folder_file"]
        duplicate_rows = self.feature_df[folder_files.notna() & folder_files.duplicated(keep=False)]
        if duplicate_rows.empty:
            return

        display_cols = [
            col for col in [
                "folder_file",
                "cell_id",
                "data_type",
                "time",
                "treatment",
                "R_series",
                "I_set",
            ]
            if col in duplicate_rows.columns
        ]
        examples = duplicate_rows[display_cols].head(12).to_dict("records")
        message = (
            f"features.xlsx contains duplicate folder_file values in project {self.project}. "
            f"Each raw recording should have one row. Please check: {examples}"
        )
        print(f"[WARNING] {message}")
        raise ValueError(message)

    def validate_application_time_column(self):
        """
        Application feature sheets must explicitly label PRE/POST in ``time``.
        """
        if "time" not in self.feature_df.columns:
            message = (
                f"Application project {self.project} requires a 'time' column "
                "in features.xlsx with PRE/POST values."
            )
            print(f"[WARNING] {message}")
            raise ValueError(message)

        time_values = self.feature_df["time"]
        blank_rows = time_values.isna() | (time_values.astype(str).str.strip() == "")
        if blank_rows.any():
            folder_files = self.feature_df.loc[blank_rows, "folder_file"].dropna().head(8).tolist()
            message = (
                f"Application project {self.project} has blank 'time' values in features.xlsx. "
                f"Expected PRE or POST. Example folder_files: {folder_files}"
            )
            print(f"[WARNING] {message}")
            raise ValueError(message)

        labels = time_values.astype(str).str.strip().str.upper()
        invalid_rows = ~labels.isin(["PRE", "POST"])
        if invalid_rows.any():
            examples = (
                self.feature_df.loc[invalid_rows, ["folder_file", "time"]]
                .head(8)
                .to_dict("records")
            )
            message = (
                f"Application project {self.project} has invalid 'time' values in features.xlsx. "
                f"Expected PRE or POST. Examples: {examples}"
            )
            print(f"[WARNING] {message}")
            raise ValueError(message)

    def classify_independant_variables(self):
        """
        Classify feature columns as subject-level or cell-level factors.

        Subject-level columns have one value per subject_id across features.xlsx.
        Cell-level columns have one value per cell_id, but are not already
        subject-level. File/data bookkeeping columns are ignored.
        """
        ignored_cols = {
            "folder_file",
            "subject_id",
            "cell_id",
            "data_type",
            "drug_in",
            "drug_out",
        }

        self.subject_independant_vairables = []
        self.cell_independant_vairables = []

        if "subject_id" not in self.feature_df.columns:
            self.subject_independant_variables = self.subject_independant_vairables
            self.cell_independant_variables = self.cell_independant_vairables
            return

        candidate_cols = [
            col for col in self.feature_df.columns
            if col not in ignored_cols
        ]

        for col in candidate_cols:
            if self._is_unique_within_group("subject_id", col):
                self.subject_independant_vairables.append(col)

        if "cell_id" in self.feature_df.columns:
            for col in candidate_cols:
                if col in self.subject_independant_vairables:
                    continue
                if self._is_unique_within_group("cell_id", col):
                    self.cell_independant_vairables.append(col)

        self.subject_independant_variables = self.subject_independant_vairables
        self.cell_independant_variables = self.cell_independant_vairables

    def _is_unique_within_group(self, group_col: str, value_col: str) -> bool:
        if group_col not in self.feature_df.columns or value_col not in self.feature_df.columns:
            return False

        grouped_nunique = (
            self.feature_df
            .dropna(subset=[group_col])
            .groupby(group_col)[value_col]
            .nunique(dropna=True)
        )

        if grouped_nunique.empty:
            return False

        return grouped_nunique.max() <= 1

    def feature_metadata_columns(self, explicit_columns: list | None = None) -> list:
        """
        Build metadata columns to carry through data-type extraction.

        Keeps each extractor's explicit columns for backwards compatibility, then
        appends project-specific subject/cell factors discovered from features.xlsx.
        Missing optional columns are ignored so project-specific feature schemas can
        differ.
        """
        columns = []

        def add(col):
            if col in self.feature_df.columns and col not in columns:
                columns.append(col)

        for col in (explicit_columns or ["folder_file", "cell_id", "data_type"]):
            add(col)

        for col in ["folder_file", "cell_id", "data_type"]:
            add(col)

        for col in self.subject_independant_vairables + self.cell_independant_vairables:
            add(col)

        for col in self.feature_df.columns:
            add(col)

        return columns

    def subject_cell_factor_columns(self, include_cell_id: bool = False) -> list:
        """
        Return project-specific factors that can be mapped onto cell-level tables.
        """
        columns = []
        if include_cell_id:
            columns.append("cell_id")

        if "subject_id" in self.feature_df.columns and "subject_id" not in columns:
            columns.append("subject_id")

        for col in self.subject_independant_vairables + self.cell_independant_vairables:
            if col in self.feature_df.columns and col not in columns:
                columns.append(col)

        return columns

    def data_independant_columns(self, extra_columns: list | None = None) -> list:
        """
        Columns that should not be treated as dependent variables in extracted dfs.
        """
        columns = [
            "folder_file",
            "folder_files",
            "cell_id",
            "subject_id",
            "data_type",
            "error",
            "traceback",
            "_feature_signature",
            "_extractor_version",
            "time",
            "valid",
            "I_set",
            "drug_in",
            "drug_out",
            "sweep_duration_s",
            "ISI_ms",
        ]

        columns.extend(self.subject_independant_vairables)
        columns.extend(self.cell_independant_vairables)
        columns.extend(self.feature_df.columns)

        if extra_columns:
            columns.extend(extra_columns)

        return list(dict.fromkeys(columns))

    def application_time_label(self, row: pd.Series) -> str:
        """
        Return explicit PRE/POST phase from an application feature row.

        Application projects require a ``time`` column in features.xlsx so
        ``treatment`` can remain the drug/group label.
        """
        folder_file = row.get("folder_file", "missing") if isinstance(row, pd.Series) else "missing"
        if not isinstance(row, pd.Series) or "time" not in row.index:
            message = (
                f"Application project {self.project} requires a 'time' column "
                f"in features.xlsx with PRE/POST values. Missing for folder_file: {folder_file}"
            )
            print(f"[WARNING] {message}")
            raise ValueError(message)

        value = row.get("time", np.nan)
        if pd.isna(value) or str(value).strip() == "":
            message = (
                f"Application project {self.project} has blank 'time' in features.xlsx. "
                f"Expected PRE or POST for folder_file: {folder_file}"
            )
            print(f"[WARNING] {message}")
            raise ValueError(message)

        label = str(value).strip().upper()
        if label not in {"PRE", "POST"}:
            message = (
                f"Application project {self.project} has invalid time '{value}'. "
                f"Expected PRE or POST for folder_file: {folder_file}"
            )
            print(f"[WARNING] {message}")
            raise ValueError(message)
        return label

    def application_is_pre(self, row: pd.Series) -> bool:
        return self.application_time_label(row) == "PRE"

    def add_missing_feature_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Refresh feature metadata columns on a cached data-type dataframe.
        """
        if "folder_file" not in df.columns:
            return df

        metadata_cols = [
            col for col in self.feature_metadata_columns()
            if col != "folder_file" and col in self.feature_df.columns
        ]

        if not metadata_cols:
            return df

        metadata = (
            self.feature_df[["folder_file"] + metadata_cols]
            .drop_duplicates(subset=["folder_file"])
        )

        refresh_cols = [col for col in metadata_cols if col in df.columns]
        df = df.drop(columns=refresh_cols)
        return df.merge(metadata, on="folder_file", how="left")

    def add_missing_cell_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Refresh project-specific cell/subject factors on a cached cell_df.
        """
        if "cell_id" not in df.columns:
            return df

        factor_cols = self.subject_cell_factor_columns()
        if not factor_cols:
            return df

        def unique_or_nan(series):
            values = series.dropna().unique()
            if len(values) == 0:
                return np.nan
            if len(values) == 1:
                return values[0]
            return values.tolist()

        factor_df = (
            self.feature_df[["cell_id"] + factor_cols]
            .dropna(subset=["cell_id"])
            .groupby("cell_id", as_index=False)
            .agg(unique_or_nan)
        )

        refresh_cols = [col for col in factor_cols if col in df.columns]
        df = df.drop(columns=refresh_cols)
        return df.merge(factor_df, on="cell_id", how="left")

    def _get_extension(self, folder_file: str) -> str:
        """
        Parameteres:
            folder_file: str - identifier of unique file e.g.  folder1/file_no_extension
        Returns:
            str: Either '.ibw' or '.abf'
        Raises:
            FileNotFoundError: If no accepted extensions are found in the folder.
        """
        if '/' in folder_file:
            folder, _ = folder_file.split('/', 1)
            base_dir = os.path.join(self.input_dir, 'PatchData', folder)
        else:
            base_dir = os.path.join(self.input_dir, 'PatchData')
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Folder not found: {base_dir}")
        files = os.listdir(base_dir)
        for ext in self.accepted_extensions:
            if any(f.endswith(ext) for f in files):
                return ext
        raise FileNotFoundError(f"No known file extensions {self.accepted_extensions} found in {base_dir}")
    
    def load_data(self, folder_file: str):
        """
        Parameters:
            folder_file (str): The folder_file identifier (can include a subfolder).
        Returns:
            tuple: (V_array, I_array, command_array, stim_array, V_list)
        """
        extension = self._get_extension(folder_file)
        if extension == '.ibw':
            return self.IGOR_load(folder_file)
        elif extension == '.abf':
            return self.ABF_load(folder_file)
        else:
            raise ValueError(f"Unsupported extension type for folder_file {folder_file}: {extension}")

    def _set_sampling_rate(self, sampling_rate_hz, folder_file: str, source: str) -> float:
        """
        Set sampling rate for the file currently being loaded.
        """
        try:
            sampling_rate_hz = float(sampling_rate_hz)
        except (TypeError, ValueError):
            sampling_rate_hz = np.nan

        if np.isfinite(sampling_rate_hz) and sampling_rate_hz > 0:
            self.sampling_rate = sampling_rate_hz
            return sampling_rate_hz

        self.sampling_rate = DEFAULT_SAMPLING_RATE_HZ
        if (
            getattr(self, "print_warnings", True)
            and folder_file not in self._sampling_rate_warning_files
        ):
            print(
                f"[WARNING] Unverified sampling rate | folder_file: {folder_file} | "
                f"source: {source} | defaulting to {DEFAULT_SAMPLING_RATE_HZ:g} Hz"
            )
            self._sampling_rate_warning_files.add(folder_file)
        return self.sampling_rate

    @staticmethod
    def folder_file_sort_key(folder_file: str):
        """
        Natural sort key for folder_file names.

        The trailing recording number gives chronological order, and the full
        natural key also behaves well when dates/folders are present.
        """
        parts = re.split(r'(\d+)', str(folder_file))
        return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)

    def build_access_df(self, st_vc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build per-folder_file series-resistance access checks from st_VC recordings.

        For each folder_file in features.xlsx, the access check uses the nearest
        valid st_VC recording before/at that file and after/at that file for the
        same cell_id. Missing bracketing st_VC recordings are retained with a
        status and NaN change values.
        """
        base_cols = [col for col in ["folder_file", "cell_id", "data_type"] if col in self.feature_df.columns]
        feature_rows = self.feature_df[base_cols].dropna(subset=["folder_file", "cell_id"]).copy()

        output_cols = [
            "folder_file",
            "cell_id",
            "data_type",
            "previous_st_VC_folder_file",
            "next_st_VC_folder_file",
            "previous_Rs_MOhm",
            "next_Rs_MOhm",
            "Rs_abs_change",
            "Rs_pct_change",
            "Rs_access_status",
        ]

        if st_vc_df is None or st_vc_df.empty or "Rs_MOhm" not in st_vc_df.columns:
            access_df = feature_rows.copy()
            for col in output_cols:
                if col not in access_df.columns:
                    access_df[col] = np.nan
            access_df["Rs_access_status"] = "missing_st_VC"
            return access_df[output_cols]

        st_access = st_vc_df[["folder_file", "cell_id", "Rs_MOhm"]].copy()
        st_access["Rs_MOhm"] = pd.to_numeric(st_access["Rs_MOhm"], errors="coerce")
        st_access = st_access.dropna(subset=["folder_file", "cell_id", "Rs_MOhm"])

        rows = []
        for cell_id, cell_features in feature_rows.groupby("cell_id", sort=False):
            ordered_features = sorted(
                cell_features.to_dict("records"),
                key=lambda row: self.folder_file_sort_key(row["folder_file"])
            )
            position_by_file = {
                row["folder_file"]: idx
                for idx, row in enumerate(ordered_features)
            }

            cell_st_access = st_access[
                (st_access["cell_id"] == cell_id)
                & (st_access["folder_file"].isin(position_by_file))
            ].copy()
            cell_st_access["position"] = cell_st_access["folder_file"].map(position_by_file)
            st_records = cell_st_access.sort_values("position").to_dict("records")

            for feature_row in ordered_features:
                folder_file = feature_row["folder_file"]
                position = position_by_file[folder_file]
                previous_st = next(
                    (row for row in reversed(st_records) if row["position"] <= position),
                    None
                )
                next_st = next(
                    (row for row in st_records if row["position"] >= position),
                    None
                )
                if (
                    feature_row.get("data_type") == "st_VC"
                    and previous_st is not None
                    and next_st is not None
                    and previous_st["folder_file"] == next_st["folder_file"]
                ):
                    previous_strict = next(
                        (row for row in reversed(st_records) if row["position"] < position),
                        None
                    )
                    next_strict = next(
                        (row for row in st_records if row["position"] > position),
                        None
                    )
                    if next_strict is not None:
                        next_st = next_strict
                    elif previous_strict is not None:
                        previous_st = previous_strict

                if previous_st is None and next_st is None:
                    status = "missing_st_VC"
                    abs_change = np.nan
                    pct_change = np.nan
                elif previous_st is None:
                    status = "missing_previous_st_VC"
                    abs_change = np.nan
                    pct_change = np.nan
                elif next_st is None:
                    status = "missing_next_st_VC"
                    abs_change = np.nan
                    pct_change = np.nan
                elif previous_st["folder_file"] == next_st["folder_file"]:
                    status = "missing_adjacent_st_VC"
                    abs_change = np.nan
                    pct_change = np.nan
                else:
                    previous_rs = previous_st["Rs_MOhm"]
                    next_rs = next_st["Rs_MOhm"]
                    abs_change = abs(next_rs - previous_rs)
                    pct_change = ((next_rs - previous_rs) / previous_rs) * 100 if previous_rs != 0 else np.nan
                    status = "ran"

                rows.append({
                    "folder_file": folder_file,
                    "cell_id": cell_id,
                    "data_type": feature_row.get("data_type", np.nan),
                    "previous_st_VC_folder_file": previous_st["folder_file"] if previous_st is not None else np.nan,
                    "next_st_VC_folder_file": next_st["folder_file"] if next_st is not None else np.nan,
                    "previous_Rs_MOhm": previous_st["Rs_MOhm"] if previous_st is not None else np.nan,
                    "next_Rs_MOhm": next_st["Rs_MOhm"] if next_st is not None else np.nan,
                    "Rs_abs_change": abs_change,
                    "Rs_pct_change": pct_change,
                    "Rs_access_status": status,
                })

        return pd.DataFrame(rows, columns=output_cols)

    def load_access_df(self, st_vc_df: pd.DataFrame | None = None, cache: bool = True) -> pd.DataFrame:
        """Load or build the per-folder_file access table."""
        cache_key = "access_by_file"
        if cache and self.isCached(cache_key):
            return self.getCache(cache_key)
        if st_vc_df is None:
            st_vc_df = st_VC(self.project, print_warnings=getattr(self, "print_warnings", False)).df
        access_df = self.build_access_df(st_vc_df)
        if cache:
            self.cache(cache_key, access_df)
            self.save_excel(cache_key, access_df)
        return access_df
    
    def load_feature_xlsx(self, filename: str):
        """Loads data from cache or an Excel file."""
        filepath = os.path.join(self.input_dir, f"{filename}.xlsx")
        cache_path = os.path.join(self.cache_dir, f"{self.sanitize_filename(filename)}.pkl")

        if self.isCached(filename):
            if not os.path.exists(filepath) or os.path.getmtime(cache_path) >= os.path.getmtime(filepath):
                return self.getCache(filename)
            print(f"[INFO] {filepath} is newer than cache; reloading {filename}.")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Excel file {filename} not found in {self.input_dir}")

        required_columns = ['folder_file', 'cell_id', 'data_type', 'treatment']
        df = pd.read_excel(filepath)
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required column(s) in features.xlsx: {missing_cols}")

        for col in ["drug_in", "drug_out"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in df.columns: # detect 1/0 True/False columns as boolian
            if col in {"drug_in", "drug_out"}:
                continue
            unique_vals = df[col].dropna().unique()

            if len(unique_vals) == 0: #skip empty columns
                continue
            if set(unique_vals).issubset({0, 1}):
                # Only convert non-null values to boolean
                df[col] = df[col].where(df[col].isna(), df[col].astype(bool))
                print(f"[INFO] Column '{col}' inferred as boolean (True/False).")


        self.cache(filename, df)
        return df

    def load_subject_data(self, filepath_or_name: str, cache: bool = True) -> pd.DataFrame:
        """
        Load animal-level non-ephys data and map subject-level factors from features.xlsx.

        Expected input format:
            one row per animal, with a required subject_id column.

        Mapped factors:
            all columns classified as subject_independant_vairables when the
            project was initialised.

        filepath_or_name can be an absolute path, a path relative to the project input
        directory, or a bare filename in the project input directory.
        """
        path = Path(filepath_or_name)
        if not path.is_absolute():
            path = Path(self.input_dir) / filepath_or_name

        if not path.exists():
            raise FileNotFoundError(f"Subject data file not found: {path}")

        factor_key = "_".join(self.subject_independant_vairables)
        cache_key = f"subject_data_{path.stem}_{factor_key}"
        if cache and self.isCached(cache_key):
            return self.getCache(cache_key)

        suffix = path.suffix.lower()
        if suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError("Subject data must be a .xlsx, .xls, or .csv file.")

        if "subject_id" not in df.columns:
            raise ValueError("Subject data must contain a 'subject_id' column.")

        df = self.map_subject_factors(df)

        if cache:
            self.cache(cache_key, df)

        return df

    def map_subject_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add project-level subject factors to a subject-level dataframe.
        """
        if "subject_id" not in df.columns:
            raise ValueError("Subject data must contain a 'subject_id' column.")

        df = df.copy()
        df["subject_id"] = df["subject_id"].astype(str)

        available_factor_cols = [
            col for col in self.subject_independant_vairables
            if col in self.feature_df.columns and col not in df.columns
        ]

        if available_factor_cols:
            subject_factors = (
                self.feature_df[["subject_id"] + available_factor_cols]
                .dropna(subset=["subject_id"])
                .copy()
            )
            subject_factors["subject_id"] = subject_factors["subject_id"].astype(str)
            subject_factors = (
                subject_factors
                .groupby("subject_id", as_index=False)
                .agg(lambda s: self._unique_subject_factor(s, s.name))
            )
            df = df.merge(subject_factors, on="subject_id", how="left")

        if "cell_id" not in df.columns:
            df["cell_id"] = df["subject_id"]

        return df

    @staticmethod
    def _unique_subject_factor(series: pd.Series, column_name: str):
        values = series.dropna().unique()
        if len(values) == 0:
            return np.nan
        if len(values) == 1:
            return values[0]
        raise ValueError(
            f"subject_id maps to multiple values for '{column_name}': {list(values)}"
        )

            
    def ABF_load(self, folder_file: str):
        """
        Loads data from .abf (Axon) files using pyabf.

        Returns:
            V_array: 2D numpy array (time x sweeps) of voltage
            I_array: 2D numpy array (time x sweeps) of current
            command_array: 2D numpy array of the clamp command waveform from sweepC
            stim_array: 2D numpy array of stimulus channel, or None if not present
            V_list: 1D flattened array (column-major sweep order)
        """
        
        path = os.path.join(self.input_dir, 'PatchData', folder_file + '.abf')
        if not os.path.exists(path):
            raise FileNotFoundError(f"ABF file not found for folder_file {folder_file}: {path}")

        abf = ABF(path)
        num_sweeps = abf.sweepCount
        num_points = abf.sweepPointCount
        self._set_sampling_rate(abf.dataRate, folder_file, "abf.dataRate")

        # Identify voltage and current channels by unit
        unit_map = {i: unit for i, unit in enumerate(abf.adcUnits)}
        voltage_ch = next((i for i, unit in unit_map.items() if 'V' in unit.upper()), None)
        current_ch = next((i for i, unit in unit_map.items() if 'A' in unit.upper()), None)

        # check for external stimulation channel ie PPR
        all_ch = set(range(abf.channelCount))
        used_ch = {ch for ch in [voltage_ch, current_ch] if ch is not None}
        remaining_ch = all_ch - used_ch
        if len(remaining_ch) == 1:
            stim_ch = remaining_ch.pop()
            stim_array = np.zeros((num_points, num_sweeps))
        elif len(remaining_ch) > 1:
            channel_names = [abf.adcNames[ch] for ch in remaining_ch]
            if getattr(self, "print_warnings", True):
                print(f"Multiple additional channels found for {folder_file}: {channel_names}. No stim channel assigned.")
            stim_ch = None
            stim_array = None
        else:
            stim_ch = None
            stim_array = None

        if voltage_ch is None or current_ch is None:
            raise ValueError(f"Couldn't identify voltage/current channels for folder_file {folder_file} from units: {abf.adcUnits}")

        command_ch = self._abf_command_channel(
            abf,
            [ch for ch in [voltage_ch, current_ch] if ch is not None]
        )
        V_array = np.zeros((num_points, num_sweeps))
        I_array = np.zeros((num_points, num_sweeps))
        command_array = np.zeros((num_points, num_sweeps))

        for i in range(num_sweeps):
            abf.setSweep(i, channel=voltage_ch)
            V_array[:, i] = abf.sweepY
            abf.setSweep(i, channel=current_ch)
            I_array[:, i] = abf.sweepY
            abf.setSweep(i, channel=command_ch)
            command_array[:, i] = abf.sweepC
            if stim_ch is not None:
                abf.setSweep(i, channel=stim_ch)
                stim_array[:, i] = abf.sweepY

        V_list = V_array.ravel(order='F')  # Column-major, like IGOR

        return V_array, I_array, command_array, stim_array, V_list

    def _abf_command_channel(self, abf: ABF, channels: list[int]) -> int:
        """
        Select the ADC channel associated with the largest reconstructed command
        waveform. Suitable for single-cell recordings with one active DAC command.
        """
        command_ranges = {}
        sweeps_to_check = abf.sweepList[:min(3, len(abf.sweepList))]
        for channel in channels:
            ranges = []
            for sweep in sweeps_to_check:
                abf.setSweep(sweep, channel=channel)
                command = np.asarray(abf.sweepC, dtype=float)
                ranges.append(np.nanmax(command) - np.nanmin(command))
            command_ranges[channel] = np.nanmax(ranges) if ranges else 0
        return max(command_ranges, key=command_ranges.get)
            

    def IGOR_load(self, folder_file):
        # S1 exports cn/cs/ds/es/ws/Time as metadata/settings waves; Soma_outwave is the command trace.
        path_V, path_command = self.make_path(folder_file)
        V_list, V_array, V_header = self.igor_exporter(path_V)
        sfA = V_header.get("sfA")
        dt = float(sfA[0]) if sfA is not None and len(sfA) > 0 else np.nan
        sampling_rate_hz = 1 / dt if np.isfinite(dt) and dt > 0 else np.nan
        self._set_sampling_rate(sampling_rate_hz, folder_file, "igor.wave_header.sfA[0]")
        I_array = None
        try:
            _, command_array, _ = self.igor_exporter(path_command)
        except FileNotFoundError:
            command_array = None
        stim_array = None
        return V_array, I_array, command_array, stim_array, V_list

    def make_path(self, folder_file): 
        """Generates file paths for voltage and IGOR command data."""
        if not isinstance(folder_file, str) or pd.isna(folder_file):
            raise ValueError(f"Invalid folder_file: {folder_file}")
        extension_V = "Soma.ibw"  # Voltage data file extension
        extension_I = "Soma_outwave.ibw"  # IGOR command waveform export

        path_V = os.path.join(self.input_dir, 'PatchData',  folder_file + extension_V)
        path_I = os.path.join(self.input_dir, 'PatchData', folder_file + extension_I)
        return path_V, path_I

    def igor_exporter(self, path):
        """Loads and processes .ibw files using igor binarywave."""
        igor_file = igor.binarywave.load(path)
        wave = igor_file["wave"]["wData"]
        wave_header = igor_file["wave"].get("wave_header", {})
        igor_df = pd.DataFrame(wave)
        V_array_2d = igor_df.to_numpy()
        point_list = V_array_2d.ravel(order='F') 
        return point_list, V_array_2d, wave_header
    

    def inspect_folder_file(self, folder_file, stacked=False, n_sweeps=None, filename=None):
        '''
        Plots any waveform based off folder_file.
        Stacked will plot each column on top of each other, defaults to False.
        '''
        feature_df = self.load_feature_xlsx('features')
        display(feature_df[feature_df['folder_file'] == folder_file])  # Show file info

        V_array , I_array, command_array, stim_array, V_list = self.load_data(folder_file)
        sampling_rate_hz = self.sampling_rate
        # self.quick_line_plot(V_array, f'Voltage trace for {folder_file}', 'Voltage (mV)', n_sweeps=n_sweeps, stacked=stacked )
        fig_v = self.quick_line_plot(
            V_array,
            f'Voltage trace for {folder_file}',
            'Voltage (mV)',
            n_sweeps=n_sweeps,
            stacked=stacked,
            sampling_rate=sampling_rate_hz
        )

        if filename: # should actualy be able to use save function from Figure class need to build interface #TODO
            safe_name = self.sanitize_filename(f"{filename}_voltage")
            fig_v.savefig(os.path.join(self.figure_output_dir, f"{safe_name}.svg"))
            fig_v.savefig(os.path.join(self.figure_output_dir, f"{safe_name}.png"))

        try:
            # self.quick_line_plot(I_array, f'Current (I) trace for {folder_file}', 'Current (pA)', n_sweeps=n_sweeps,  stacked=stacked) #TODO add if check shape hwen no I 
            if I_array is not None:
                fig_I = self.quick_line_plot(
                    I_array,
                    f'Current (I) trace for {folder_file}',
                    'Current (pA)',
                    n_sweeps=n_sweeps,
                    stacked=stacked,
                    sampling_rate=sampling_rate_hz
                )
                if filename:
                    safe_name = self.sanitize_filename(f"{filename}_current")
                    fig_I.savefig(os.path.join(self.figure_output_dir, f"{safe_name}.svg"))
                    fig_I.savefig(os.path.join(self.figure_output_dir, f"{safe_name}.png"))
            else:
                print(f'No I file found for {folder_file}')
                
        except FileNotFoundError:
            print(f'No I file found for {folder_file}')

        if command_array is not None:
            fig_command = self.quick_line_plot(
                command_array,
                f'Command trace for {folder_file}',
                'Command (pA)',
                n_sweeps=n_sweeps,
                stacked=stacked,
                sampling_rate=sampling_rate_hz
            )
            if filename:
                safe_name = self.sanitize_filename(f"{filename}_command")
                fig_command.savefig(os.path.join(self.figure_output_dir, f"{safe_name}.svg"))
                fig_command.savefig(os.path.join(self.figure_output_dir, f"{safe_name}.png"))

    def quick_line_plot(self, plot_array, plottitle, y_label,  n_sweeps=None, stacked=False, sampling_rate=None):
        '''
        Plots line plot for given array without adding a legend for stacked plots.
        
        Parameters:
            plot_array (numpy.ndarray): 2D array to plot, where each column is a sweep.
            plottitle (str): Title for the plot.
            stacked (bool): If True, plots each sweep stacked. If False, concatenates sweeps.
            sampling_rate (float): Sampling rate in Hz. If provided, x-axis is time in seconds.
        '''
        fig, ax = plt.subplots()
        num_sweeps = plot_array.shape[1]
        if n_sweeps is None or n_sweeps > num_sweeps:
            n_sweeps = num_sweeps 
        
        if stacked:
            x = (
                np.arange(plot_array.shape[0]) / sampling_rate
                if sampling_rate
                else np.arange(plot_array.shape[0])
            )
            for i in range(n_sweeps):
                ax.plot(x, plot_array[:, i])  # Plot each sweep
        else:
            # Concatenate sweeps for continuous plotting
            cropped_array = plot_array[:, :n_sweeps] 
            continuous_plot = cropped_array.ravel(order='F')  # Flatten array in column-major order
            x = (
                np.arange(len(continuous_plot)) / sampling_rate
                if sampling_rate
                else np.arange(len(continuous_plot))
            )
            ax.plot(x, continuous_plot)  # Plot continuous
        
        ax.set_title(plottitle)
        ax.set_xlabel('Time (s)' if sampling_rate else 'Time (samples)')
        ax.set_ylabel(y_label)
        fig.tight_layout() 
        plt.show()
        return fig




@dataclass
class EphysData (Project):

    '''Generic data_type extractor, child classes process the data nd make aggregate dfs'''
    
    # project: str #name of the excel_filename project_filename in notebook
    initial_columns: list = None #defined by child classes
    sampling_rate: float = 2e4
    data_type: str = None #defined by child class
    filename: str = None # defined by child class
    print_warnings: bool = False
    extractor_version: int = 1
    extraction_feature_columns: list = None
    cache_internal_columns: tuple = ("_feature_signature", "_extractor_version")

    def __post_init__(self):
        super().__post_init__()
        self.initial_columns = self.feature_metadata_columns(self.initial_columns)
        self.df = self.update()
   
    def update(self):
        """
        Bring the cached extractor dataframe up to date.

        Existing cached rows are reused when their ``folder_file``,
        feature-signature and extractor-version still match the current
        features.xlsx state. New or changed rows are extracted and merged back
        into the dataframe in features.xlsx order.
        """
        return self._update_cache(rerun_all=False)

    def regenerate(self):
        """Rerun all rows for this extractor and overwrite its cache."""
        return self._update_cache(rerun_all=True)

    def generate(self, force: bool = False, update: bool = True):
        """
        Backwards-compatible wrapper.

        Prefer ``update()`` for normal use and ``regenerate()`` when all rows
        should be rerun.
        """
        if force or not update:
            return self.regenerate()
        return self.update()

    def _update_cache(self, rerun_all: bool = False):
        """Internal extractor cache updater."""
        input_df = self._feature_rows_for_extraction()
        if rerun_all or not self.isCached(self.filename):
            df = self._extract_rows(input_df)
            self.cache(self.filename, df)
            return df

        cached_df = self.getCache(self.filename)
        if self._cannot_incrementally_update(cached_df, input_df):
            df = self._extract_rows(input_df)
            self.cache(self.filename, df)
            return df

        cached_df, tracking_backfilled = self._backfill_cache_tracking_columns(cached_df, input_df)
        rows_to_extract, reason_counts = self._rows_needing_extraction(input_df, cached_df)

        if rows_to_extract.empty:
            df = self._refresh_cached_rows(cached_df, input_df)
            df = self._order_extractor_columns(df)
            if tracking_backfilled or self._features_newer_than_cache():
                self.cache(self.filename, df)
            return df

        reason_text = ", ".join(
            f"{reason}: {count}"
            for reason, count in reason_counts.items()
            if count
        )
        print(
            f"[INFO] Updating {self.filename}: extracting "
            f"{len(rows_to_extract)}/{len(input_df)} rows ({reason_text})."
        )

        extracted_df = self._extract_rows(rows_to_extract)
        rerun_files = set(rows_to_extract["folder_file"])
        refreshed_cache = self._refresh_cached_rows(cached_df, input_df)
        reused_df = refreshed_cache[~refreshed_cache["folder_file"].isin(rerun_files)]
        df = pd.concat([reused_df, extracted_df], ignore_index=True, sort=False)
        df = self._sort_like_features(df, input_df)
        df = self._order_extractor_columns(df)
        self.cache(self.filename, df)
        return df

    def _feature_rows_for_extraction(self) -> pd.DataFrame:
        """Build current feature rows for this extractor and add cache metadata."""
        df = self.feature_df[self.feature_df['data_type'] == self.data_type][self.initial_columns].copy()
        if df.empty:
            for col in self.cache_internal_columns:
                df[col] = pd.Series(dtype="object")
            return df

        df["_feature_signature"] = df.apply(self._feature_signature, axis=1)
        df["_extractor_version"] = self.extractor_version
        return df

    def _extract_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run extraction for the provided feature rows."""
        if df.empty:
            return self._order_extractor_columns(df.copy())

        df = df.progress_apply(lambda row: self._handle_extraction(row, self.process), axis=1) # log errors
        df["_feature_signature"] = df.apply(self._feature_signature, axis=1)
        df["_extractor_version"] = self.extractor_version
        return self._order_extractor_columns(df)

    def _order_extractor_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep feature columns first, extracted columns next, cache columns last."""
        if df.empty:
            return df
        internal_columns = [col for col in self.cache_internal_columns if col in df.columns]
        additional_columns = [col for col in df.columns if col not in self.initial_columns]
        additional_columns = [col for col in additional_columns if col not in internal_columns]
        ordered_columns = [
            col for col in self.initial_columns + additional_columns + internal_columns
            if col in df.columns
        ]
        return df[ordered_columns]

    def _cannot_incrementally_update(self, cached_df: pd.DataFrame, input_df: pd.DataFrame) -> bool:
        """Return True when the cache shape cannot safely be row-updated."""
        if "folder_file" not in cached_df.columns or "folder_file" not in input_df.columns:
            return True
        if input_df["folder_file"].duplicated().any():
            examples = input_df[input_df["folder_file"].duplicated(keep=False)].head(8).to_dict("records")
            raise ValueError(
                f"Duplicate folder_file values in {self.data_type}. "
                f"Each raw recording should have one features.xlsx row. Examples: {examples}"
            )
        return False

    def _backfill_cache_tracking_columns(
        self,
        cached_df: pd.DataFrame,
        input_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, bool]:
        """Add cache tracking columns to older cached dataframes without rerunning them."""
        cached_df = cached_df.copy()
        changed = False
        signature_map = input_df.set_index("folder_file")["_feature_signature"].to_dict()

        if "_feature_signature" not in cached_df.columns:
            cached_df["_feature_signature"] = cached_df["folder_file"].map(signature_map)
            changed = True
        else:
            missing_signature = cached_df["_feature_signature"].isna()
            if missing_signature.any():
                cached_df.loc[missing_signature, "_feature_signature"] = (
                    cached_df.loc[missing_signature, "folder_file"].map(signature_map)
                )
                changed = True

        if "_extractor_version" not in cached_df.columns:
            cached_df["_extractor_version"] = self.extractor_version
            changed = True
        else:
            missing_version = cached_df["_extractor_version"].isna()
            if missing_version.any():
                cached_df.loc[missing_version, "_extractor_version"] = self.extractor_version
                changed = True

        return cached_df, changed

    def _rows_needing_extraction(
        self,
        input_df: pd.DataFrame,
        cached_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict]:
        """Return current feature rows absent from cache or invalidated by tracking columns."""
        cached_lookup = (
            cached_df
            .drop_duplicates(subset=["folder_file"], keep="last")
            .set_index("folder_file")
        )

        missing_cache = ~input_df["folder_file"].isin(cached_lookup.index)
        cached_signature = input_df["folder_file"].map(cached_lookup["_feature_signature"])
        cached_version = input_df["folder_file"].map(cached_lookup["_extractor_version"])

        feature_changed = (~missing_cache) & (cached_signature != input_df["_feature_signature"])
        version_changed = (~missing_cache) & (cached_version.astype(str) != str(self.extractor_version))
        update_mask = missing_cache | feature_changed | version_changed

        reason_counts = {
            "new": int(missing_cache.sum()),
            "feature_changed": int(feature_changed.sum()),
            "version_changed": int(version_changed.sum()),
        }
        return input_df[update_mask].copy(), reason_counts

    def _refresh_cached_rows(self, cached_df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
        """Refresh feature metadata on cached rows while keeping extracted values."""
        cached_payload = cached_df.drop_duplicates(subset=["folder_file"], keep="last").copy()
        drop_cols = [
            col for col in input_df.columns
            if col != "folder_file" and col in cached_payload.columns
        ]
        cached_payload = cached_payload.drop(columns=drop_cols)
        refreshed = input_df.merge(cached_payload, on="folder_file", how="left")
        return self._sort_like_features(refreshed, input_df)

    def _sort_like_features(self, df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
        """Sort extracted rows in the same order as features.xlsx."""
        if df.empty or "folder_file" not in df.columns:
            return df
        order = {folder_file: idx for idx, folder_file in enumerate(input_df["folder_file"])}
        df = df.copy()
        df["_feature_order"] = df["folder_file"].map(order)
        df = df.sort_values("_feature_order").drop(columns=["_feature_order"])
        return df.reset_index(drop=True)

    def _features_newer_than_cache(self) -> bool:
        """Return True when features.xlsx is newer than this extractor cache."""
        feature_path = os.path.join(self.input_dir, "features.xlsx")
        cache_path = os.path.join(self.cache_dir, f"{self.sanitize_filename(self.filename)}.pkl")
        return (
            os.path.exists(feature_path)
            and os.path.exists(cache_path)
            and os.path.getmtime(feature_path) > os.path.getmtime(cache_path)
        )

    def _feature_signature(self, row: pd.Series) -> str:
        """Hash the feature values that determine extraction for one row."""
        payload = {
            col: self._normalise_signature_value(row.get(col, np.nan))
            for col in self._signature_columns()
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _signature_columns(self) -> list:
        """Columns from features.xlsx that should invalidate extracted rows."""
        columns = self.extraction_feature_columns or ["folder_file", "data_type"]
        return [col for col in columns if col in self.feature_df.columns]

    def _normalise_signature_value(self, value):
        """Convert pandas/numpy values into stable JSON-friendly objects."""
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            return [self._normalise_signature_value(item) for item in list(value)]
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.generic):
            value = value.item()
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value
    
    def process(self):
        raise NotImplementedError
    
    def _debug_extraction(self, row: pd.Series, process_function) -> pd.Series:
        '''Direct processing without error catching — use during debugging.'''
        row = row.copy()
        previous_print_warnings = self.print_warnings
        self.print_warnings = True
        try:
            row = process_function(row)  # Let any exception raise naturally
        except Exception as e:
            print(
                f"[ERROR] {row.get('data_type', self.data_type)} extraction failed | "
                f"folder_file: {row.get('folder_file', 'missing')} | "
                f"{type(e).__name__}: {e}"
            )
            raise
        finally:
            self.print_warnings = previous_print_warnings
        if 'warning' not in row:
            row['warning'] = None
        row['error'] = 'ran'
        row['traceback'] = None
        return row

    def _append_warning(self, row: pd.Series, message: str) -> pd.Series:
        existing = row.get('warning', None)
        if existing is None or (isinstance(existing, float) and np.isnan(existing)):
            row['warning'] = message
        else:
            row['warning'] = f"{existing}; {message}"
        return row

    def _print_warning(self, row: pd.Series, message: str) -> None:
        if not self.print_warnings:
            return
        folder_file = row.get('folder_file', 'missing')
        data_type = row.get('data_type', self.data_type)
        print(
            f"[WARNING] {message} | folder_file: {folder_file} | "
            f"data_type: {data_type}"
        )

    def _warn_protocol_source(self, row: pd.Series, purpose: str, source: str) -> pd.Series:
        if source not in {"I_array_fallback", "missing"}:
            return row

        if source == "I_array_fallback":
            message = f"{purpose}: command_array missing, measured I_array used as protocol fallback"
        else:
            message = f"{purpose}: no command_array or measured-I protocol steps detected"
        self._print_warning(row, message)
        return self._append_warning(row, message)

    def _safe_nanmean(self, values, row: pd.Series, metric: str):
        arr = np.asarray(values, dtype=float)
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            folder_file = row.get('folder_file', 'unknown file')
            self._append_warning(row, f"{metric}: no valid values in {folder_file}")
            return np.nan
        return float(np.mean(valid))
    
    def _handle_extraction(self, row: pd.Series, process_function) -> pd.Series:
        '''Error handeling to add rows 'error' and 'ran' to each df made. '''
        row = row.copy()
        error_msg = None
        error_traceback = None

        def log_error(msg, tb):
            nonlocal error_msg
            nonlocal error_traceback
            error_msg = msg
            error_traceback = tb

        try:
            row = process_function(row)
        except Exception as e:
            error_type = type(e).__name__
            error_tb = traceback.format_exc()
            lines = error_tb.split('\n')
            relevant_tb = [lines[idx - 1].strip() for idx, line in enumerate(lines) if f'{error_type}: {str(e)}' in line]

            log_error(f'{error_type}: {str(e)}', relevant_tb)
            error_traceback = relevant_tb
        
        if error_msg:
            row['error'] = error_msg
            row['traceback'] = error_traceback
        else:
            row['error'] = 'ran'
            row['traceback'] = None
        if 'warning' not in row:
            row['warning'] = None
        return row
    

@dataclass
class st_VC(EphysData):
    filename: str = "st_VC_df"
    data_type: str = 'st_VC'
    
    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'treatment', 'region', 'hemisphere', 'cell_type', 'cell_subtype', 'sex', 'behaviour', 'subject_id']
        super().__post_init__()
    
    def process(self, row: pd.Series) -> pd.Series:
        """Extract Rs, Rm, Cm, tau from each voltage step in the st_VC protocol.""" 
        V_array, I_array, command_array, stim_array, V_list = self.load_data(row['folder_file'])
        if I_array is None:
            raise ValueError(
                f"st_VC expected measured current but I_array is missing. "
                f"Check features.xlsx data_type for folder_file: {row['folder_file']}"
            )
        V = V_array[:, 0]  # mV
        I = I_array[:, 0]  # pA
        command_array_adj = command_array_to_match_V(V_array, command_array)
        protocol_V = command_array_adj[:, 0] if has_protocol_steps(command_array_adj) else V
        dt = 1 / self.sampling_rate
        t = np.arange(len(I)) * dt
        min_step_separation = int(0.005 / dt)  # 5 ms

        # Detect voltage steps 
        dV = np.diff(protocol_V)
        step_indices = np.where(np.abs(dV) > 0.4)[0]  # threshold in mV #adj 0.5-->0.4 to capture high resistance neurons
        
        # Merge detections < min_step_separation 
        if len(step_indices) > 0:
            keep = np.concatenate(([True],
                                np.diff(step_indices) > min_step_separation))
            step_indices = step_indices[keep]

        if len(step_indices) < 1:
            raise ValueError(
                f"No voltage steps detected for folder_file {row['folder_file']} "
                f"(data_type {row['data_type']}). "
                "Check features.xlsx: this may be the wrong data_type for this file."
            )

        #  time between steps to define analysis window
        step_durations = np.diff(step_indices) * dt
        avg_step_duration = np.median(step_durations)
        window_post = int(min(0.02, 0.5 * avg_step_duration) / dt)
        window_pre = int(min(0.005, 0.2 * avg_step_duration) / dt)
        steady_window = int(0.01 / dt)

        # voltage step size 
        unique_Vs, counts = np.unique(np.round(protocol_V, 1), return_counts=True)
        if len(unique_Vs) < 2:
            raise ValueError(
                f"Not enough voltage levels to define delta_V for folder_file {row['folder_file']} "
                f"(data_type {row['data_type']}). "
                "Check features.xlsx: this may be the wrong data_type for this file."
            )
        sorted_Vs = unique_Vs[np.argsort(-counts)]
        V_baseline_mode, V_step_mode = sorted_Vs[:2]
        global_delta_V = abs(V_step_mode - V_baseline_mode)

        Rs_list, Rm_list, tau_list, Cm_list = [], [], [], []

        for idx in step_indices:
            start = max(0, idx + 1)
            end = min(len(t), len(protocol_V), idx + 1 + window_post)
            baseline = max(0, idx - window_pre)
            baseline_I = np.mean(I[baseline:start])

            if end - start < 5:
                continue  # too short to analyze

            local_delta_V = protocol_V[start] - protocol_V[baseline] #less accurate than global assuming global is consistent
            if np.abs(local_delta_V) < 1e-3:
                continue  # ignore tiny steps

            I_step = I[start:end]
            t_step = t[start:end] - t[start]

            # get dI and steady state I
            I_peak = np.max(I_step) if local_delta_V > 0 else np.min(I_step)
            I_deflection = abs(I_peak - baseline_I)
            I_steady = np.mean(I_step[-steady_window:])

            try:
                Rs = global_delta_V * 1e-3 / (I_deflection * 1e-12)
                Rm = global_delta_V * 1e-3 / (abs(I_steady - baseline_I) * 1e-12)
            except ZeroDivisionError:
                continue

            # Tau: exponential decay fit
            def exp_decay(t, A, tau, C):
                return A * np.exp(-t / tau) + C

            try:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always", RuntimeWarning)
                    popt, _ = curve_fit(exp_decay, t_step, I_step, p0=[I_deflection, 0.01, I_steady])
                for warning in caught_warnings:
                    if issubclass(warning.category, RuntimeWarning):
                        self._print_warning(
                            row,
                            f"tau fit warning | step: {idx} | {warning.message}"
                        )
                tau = popt[1]
            except Exception:
                tau = np.nan

            Cm = tau / Rm if Rm != 0 else np.nan

            # BOUNDS CHECK with debug
            if not (0 < Rs < 1e9):
                self._print_warning(row, f"Rs out of bounds: {Rs:.2e} ohm ({Rs/1e6:.2f} MOhm) | step: {idx}")
                Rs = np.nan
            if not (1e6 < Rm < 1e9):
                self._print_warning(row, f"Rm out of bounds: {Rm:.2e} ohm ({Rm/1e6:.2f} MOhm) | step: {idx}")
                Rm = np.nan
            if not (0 < Cm < 500e-12):
                self._print_warning(row, f"Cm out of bounds: {Cm:.2e} F ({Cm*1e12:.2f} pF) | step: {idx}")
                Cm = np.nan
            if not (0 < tau < 1):
                self._print_warning(row, f"tau out of bounds: {tau:.2e} s ({tau*1e3:.2f} ms) | step: {idx}")
                tau = np.nan

            # # # plot to check
            # plt.figure()
            # plt.plot(t_step * 1e3, I_step, label="I_step")
            # if not np.isnan(tau):
            #     plt.plot(t_step * 1e3, exp_decay(t_step, *popt), 'r--', label=f"fit τ = {popt[1]*1e3:.2f} ms")
            # plt.axhline(I_peak, color='purple', linestyle=':', label=f"I_peak (amp = {I_deflection:.1f} pA)")
            # plt.axhline(I_steady, color='green', linestyle='--', label="I_steady")
            # plt.xlabel("Time (ms)")
            # plt.ylabel("Current (pA)")
            # plt.legend()
            # plt.title(f"{row.cell_id} step {idx}")
            # plt.show()

            Rs_list.append(Rs / 1e6)      # to MOhm
            Rm_list.append(Rm / 1e6)
            tau_list.append(tau * 1e3 if not np.isnan(tau) else np.nan)  # ms
            Cm_list.append(Cm * 1e12 if not np.isnan(Cm) else np.nan)    # pF

        # Final averaged values
        row['Rs_MOhm'] = self._safe_nanmean(Rs_list, row, 'Rs_MOhm') #Ra same
        row['Rm_MOhm'] = self._safe_nanmean(Rm_list, row, 'Rm_MOhm')
        row['tau_ms'] = self._safe_nanmean(tau_list, row, 'tau_ms')
        row['Cm_pF'] = self._safe_nanmean(Cm_list, row, 'Cm_pF')

        # Baseline RMP: before the first step
        baseline_end = step_indices[0]  # first step index
        baseline_window = int(0.01 / dt)  # 10 ms window
        row['RMP_mV'] = np.mean(V[max(0, baseline_end - baseline_window):baseline_end])
        row['holding_I'] = np.mean(I[max(0, baseline_end - baseline_window):baseline_end])

        return row


@dataclass
class ramp_IC(EphysData):
    filename: str = "ramp_IC_df"
    data_type: str = 'ramp_IC'
    
    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'treatment', 'region', 'hemisphere', 'cell_type', 'cell_subtype', 'sex', 'behaviour', 'subject_id']
        super().__post_init__()

    @staticmethod
    def _trace_missing_steps(trace, min_range=5):
        if trace is None:
            return True
        trace = np.asarray(trace, dtype=float)
        if trace.size < 2 or np.all(np.isnan(trace)):
            return True
        trace_range = np.nanmax(trace) - np.nanmin(trace)
        return not np.isfinite(trace_range) or trace_range < min_range

    @staticmethod
    def _measured_holding_current(trace, start_idx):
        if trace is None:
            return np.nan
        trace = np.asarray(trace, dtype=float)
        start_idx = min(max(int(start_idx), 0), len(trace) - 1)
        values = trace[start_idx:]
        if values.size == 0 or np.all(np.isnan(values)):
            return np.nan
        return 5 * round(np.nanmean(values) / 5)
    
    def process(self, row: pd.Series) -> pd.Series:
        """Extract rheobase (pA), voltage_threshold (mV) and AP_charecteristics of the first AP."""  #HERE TO CHECK AND WORK FOR HFD
        V_array, I_array, command_array, stim_array, V_list = self.load_data(row['folder_file'])
        dt = 1 / self.sampling_rate
        t = np.arange(V_array.shape[0]) * dt

        rheobase_list = []
        holding_I_list = []
        threshold_list = []
        height_list = []
        rise_list = []
        decay_list = []
        fwhm_list = []
        sweep_RMP_mV = []
        V_protocol_array, protocol_array, protocol_source = select_protocol_array(
            V_array,
            command_array=command_array,
            I_array=I_array,
            clean_I_fallback=True,
        )

        row = self._warn_protocol_source(row, "ramp detection/current", protocol_source)
        num_sweeps = V_protocol_array.shape[1]

        for sweep_idx in range(num_sweeps):
            V_sweep = V_protocol_array[:, sweep_idx]
            I_sweep = I_array[:, sweep_idx] if I_array is not None and sweep_idx < I_array.shape[1] else None

            # Run AP detection on this sweep only
            (
                peak_voltages_all, peak_latencies_all, v_thresholds_all, peak_rise_all,
                peak_max_dvdt_all, peak_locs_corr_all, upshoot_locs_all, peak_heights_all,
                peak_fw_all, peak_indices_all, sweep_indices_all, peak_decay_all
            ) =  ap_characteristics_extractor_main(row['folder_file'], V_sweep, sampling_rate=self.sampling_rate)

            if peak_voltages_all is None or len(peak_voltages_all) == 0: 
                continue  # No AP detected

            firt_AP_peak_loc = peak_locs_corr_all[0]

            try:
                source_trace = protocol_array[:, sweep_idx] if protocol_array is not None else None
                if self._trace_missing_steps(source_trace):
                    continue

                d_source = np.diff(source_trace)
                ramp_end_idx = np.argmax(np.abs(d_source))
                post_ramp_start = min(ramp_end_idx + 50, len(source_trace) - 1)
                source_offset = 5 * round(np.nanmean(source_trace[post_ramp_start:]) / 5)
                AP_idx = min(firt_AP_peak_loc, len(source_trace) - 1)
                rheobase = source_trace[AP_idx] - source_offset  # pA
                offset = self._measured_holding_current(I_sweep, post_ramp_start)
                rmp_start = min(post_ramp_start, len(V_sweep) - 1)
                rmp = np.nanmean(V_sweep[rmp_start:])
            except IndexError:
                continue  # Skip corrupted index
            
            sweep_RMP_mV.append(rmp)
            rheobase_list.append(rheobase)
            holding_I_list.append(offset)
            threshold_list.append(v_thresholds_all[0])
            height_list.append(peak_heights_all[0])
            rise_list.append(peak_rise_all[0])
            decay_list.append(peak_decay_all[0])
            fwhm_list.append(peak_fw_all[0])


        row['holding_I'] = self._safe_nanmean(holding_I_list, row, 'holding_I')
        row['RMP_mV'] = self._safe_nanmean(sweep_RMP_mV, row, 'RMP_mV')
        row['ramp_rheobase_pA'] = self._safe_nanmean(rheobase_list, row, 'ramp_rheobase_pA')
        row['ramp_voltage_threshold_mV'] = self._safe_nanmean(threshold_list, row, 'ramp_voltage_threshold_mV')
        row['AP_height_mV'] = self._safe_nanmean(height_list, row, 'AP_height_mV')
        row['AP_rise_mV_ms'] = self._safe_nanmean(rise_list, row, 'AP_rise_mV_ms')
        row['AP_decay_mV_ms'] = self._safe_nanmean(decay_list, row, 'AP_decay_mV_ms')
        row['AP_width_ms'] = self._safe_nanmean(fwhm_list, row, 'AP_width_ms')

       

        return row


@dataclass
class IV_VC(EphysData):
    filename: str = "IV_VC_df"
    data_type: str = 'IV_VC'
    
    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'treatment', 'region', 'hemisphere', 'cell_type', 'cell_subtype', 'sex', 'behaviour', 'subject_id']
        super().__post_init__()
    
    def process(self, row: pd.Series) -> pd.Series:
        """Extract steady-state current (I_steady) for each voltage step (V_inj).""" 
        V_array, I_array, command_array, stim_array, V_list = self.load_data(row['folder_file'])
        if I_array is None:
            raise ValueError(
                f"IV_VC expected current channel but I_array is missing. "
                f"Check features.xlsx data_type for folder_file: {row['folder_file']}"
            )
        dt = 1 / self.sampling_rate

        n_sweeps = V_array.shape[1]
        I_steady_list = []
        V_inj_list = []
        last_rmp = np.nan
        last_holding_I = np.nan
        command_array_adj = command_array_to_match_V(V_array, command_array)

        for sweep in range(n_sweeps):
            V = V_array[:, sweep]
            I = I_array[:, sweep]
            command_sweep = (
                command_array_adj[:, sweep]
                if command_array_adj is not None and sweep < command_array_adj.shape[1]
                else None
            )

            command_step_indices, _, _ = _step_indices_from_command_trace(command_sweep)
            if command_step_indices is not None:
                source_V = command_sweep
                start = int(command_step_indices[0])
                end = int(command_step_indices[-1])
            else:
                source_V = V
                #detect step start and finish of V step using dvdt
                dV = np.gradient(V)
                thresh = np.std(dV) * 3
                step_start_candidates = np.where(np.abs(dV) > thresh)[0]

                if len(step_start_candidates) < 2:
                    continue  # skip if no clear step
                start = int(step_start_candidates[0])
                end = int(step_start_candidates[-1])

            end = min(end, len(V), len(I), len(source_V))
            if end - start < 2:
                continue
            step_len = end-start
            steady_start = int(start + 0.75 * step_len)

            V_steady = np.mean(source_V[steady_start:end])
            I_clean_step = spike_remover_nan(I[steady_start:end], threshold_sd=0.5) # remove APs / spikes
            I_steady = np.nanmean(I_clean_step)

            V_inj_list.append(V_steady)
            I_steady_list.append(I_steady)

            post_start = min(end + 500, len(V))
            if post_start < len(V):
                last_rmp = np.mean(V[post_start:])
                last_holding_I = np.mean(I[post_start:])

        if len(V_inj_list) == 0:
            self._print_warning(row, "IV step detection failed")
            row = self._append_warning(row, "IV step detection failed")

        row['V_steps_mV'] = V_inj_list
        row['I_step_steady_mV'] = I_steady_list
        row['RMP_mV'] = last_rmp
        row['holding_I'] = last_holding_I
        return row


@dataclass
class spont_IC(EphysData):    #TODO BUILD EXCLUSION - traces with high vairability of baseline and remove APs
    filename: str = "spont_IC_df"
    data_type: str = 'spont_IC'
    amplitude_threshold: float = None
    noise_multiplier: float = 4 # multiplier for noise SD to set amplitude threshold
    min_amplitude_threshold: float = 0.1
    max_amplitude_threshold: float = 1.0
    baseline_method: str = 'rolling_median'
    baseline_window_s: float = 0.1
    rise_time_range: tuple = (0.5e-3, 5e-3) # 0.5 - 5 ms
    decay_time_range: tuple = (2e-3, 20e-3) # 2 - 20 ms
    peak_window_s: float = 0.001
    upshoot_baseline_window_s: float = 0.001
    onset_search_window_s: float = 0.080
    debug_local_baseline_plot: bool = False
    debug_plot: bool = False
    
    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'treatment', 'region', 'hemisphere', 'cell_type', 'cell_subtype', 'sex', 'behaviour', 'subject_id']
        super().__post_init__()
    
    def process(self, row: pd.Series) -> pd.Series:
        """
        Extract AP, sEPSP and sIPSP frequency.
        Designed for a gap free recording. 
        """  
        V_array, I_array, command_array, stim_array, V_list = self.load_data(row['folder_file'])
        epsp = EPSP_detector(
            V_array,
            sampling_rate=self.sampling_rate,
            folder_file=row['folder_file'],
            amplitude_threshold=self.amplitude_threshold,
            noise_multiplier=self.noise_multiplier,
            min_amplitude_threshold=self.min_amplitude_threshold,
            max_amplitude_threshold=self.max_amplitude_threshold,
            baseline_method=self.baseline_method,
            baseline_window_s=self.baseline_window_s,
            rise_time_range=self.rise_time_range,
            peak_window_s=self.peak_window_s,
            upshoot_baseline_window_s=self.upshoot_baseline_window_s,
            onset_search_window_s=self.onset_search_window_s,
            debug_local_baseline_plot=self.debug_local_baseline_plot,
            debug_plot=self.debug_plot,
            print_warnings=self.print_warnings,
        )

        row["RMP_mV"] = epsp['RMP_mV']  # median raw voltage after AP/valid masking
        row['holding_I'] = float(np.nanmean(I_array)) if I_array is not None else np.nan  # mean measured holding current
        row['baseline_drift'] = epsp['baseline_drift_mV']  # slow baseline range in mV
        row['sEPSP_frequency_Hz'] = epsp['frequency_Hz']  # accepted EPSP count / valid recording time
        row['sEPSP_amplitudes_mV'] = epsp['amplitudes_mV']  # raw peak minus raw upshoot value
        row['sEPSP_locs'] = epsp['peak_locs']  # accepted peak locations from processed detection trace, samples
        row['sEPSP_raw_peak_locs'] = epsp['raw_peak_locs']  # locally refined raw peak locations, samples
        row['sEPSP_peak_values_mV'] = epsp['peak_values_mV']  # raw voltage at refined peaks
        row['sEPSP_upshoot_values_mV'] = epsp['upshoot_values_mV']  # raw voltage at selected upshoot/baseline
        row['sEPSP_upshoot_locs'] = epsp['upshoot_locs']  # selected raw upshoot/baseline locations, samples
        row['sEPSP_upshoot_status'] = epsp['upshoot_status']  # QC label for each amplitude measurement
        row['sEPSP_amplitude_threshold_mV'] = epsp['amplitude_threshold_mV']  # processed-trace detection threshold
        row['sEPSP_noise_sd_mV'] = epsp['noise_sd_mV']  # robust processed-trace noise estimate
        row['sEPSP_valid_time_s'] = epsp['valid_time_s']  # seconds used for frequency after masks

        for warning in epsp.get('warnings', []):
            row = self._append_warning(row, warning)

        return row


@dataclass
class PPR_VC(EphysData):                    
    filename: str = "PPR_VC_df"
    data_type: str = 'PPR_VC'

    pulse_search_window_ms: float = 16  # ms to search after pulse offset
    max_holding_I_pA: float = 250

    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'treatment', 'region', 'hemisphere', 'cell_type', 'cell_subtype', 'sex', 'behaviour', 'subject_id'] # some wont exist in all projects check functionality
        super().__post_init__()

    def process(self, row: pd.Series) -> pd.Series:
        """
        Detects two pulses per sweep from stim channel, extracts peak currents,
        computes ISI and paired-pulse ratio (PPR).
        Assumes inward (negative) current.
        """
        V_array, I_array, command_array, stim_array, V_list = self.load_data(row['folder_file'])
        dt = 1 / self.sampling_rate
        w_samples = int(self.pulse_search_window_ms / 1000 / dt)
        folder_file = row.get('folder_file', 'missing')
        data_type = row.get('data_type', self.data_type)

        # Check stim_array exists; pulse count is detected per sweep below.
        if stim_array is None or stim_array.ndim != 2:
            raise ValueError(
                f"PPR_VC expected a stim protocol but stim_array is invalid. "
                f"Check features.xlsx data_type for folder_file: {folder_file}"
            )

        ISIs, pulse1_amp, pulse2_amp, PPRs, baseline_V, baseline_I = [], [], [], [], [], []
        skipped_sweeps = []

        def warn_skip(sweep, reason):
            self._print_warning(row, f"PPR_VC sweep skipped | sweep: {sweep} | reason: {reason}")
            skipped_sweeps.append(f"sweep {sweep}: {reason}")

        for sweep in range(stim_array.shape[1]):  # axis 1 = sweeps
            stim = stim_array[:, sweep]
            I = I_array[:, sweep]
            I_smooth = savgol_filter(I, window_length=11, polyorder=2) # could change for Butterworth / Bessel / Chebyshev filter los pass filters
            V = V_array[:, sweep] # for holding voltage
            
            #detect stim
            d_stim = np.diff(stim)
            stim_threshold = 0.5 * np.max(d_stim)  # 50% of the max slope
            pulse_onsets = np.where(d_stim > stim_threshold)[0] + 1
            pulse_offsets = np.where(d_stim < -stim_threshold)[0] + 1 

            if len(pulse_onsets) != len(pulse_offsets):
                min_len = min(len(pulse_onsets), len(pulse_offsets))
                pulse_onsets = pulse_onsets[:min_len]
                pulse_offsets = pulse_offsets[:min_len]

            pulses = list(zip(pulse_onsets, pulse_offsets)) # list of tuples [(on1, off1), (on2, off2), ...]

            if len(pulses) != 2:
                reason = f"expected 2 stim pulses, detected {len(pulses)}"
                warn_skip(sweep, reason)
                continue

            #  CHECK FOR APs  #
            I_AP_rate_threshold = 200  # pA/ms, rapid deflection in current (AP in F4693/2025_11_07_0009 5MeO7j 684 pA/ms, F4832/2025_12_12_0050 236pA/ms, M5098/2026_04_29_0023 114 pA/ms)
            buffer_samples = int(0.001 / dt)  # 1 ms buffer after each pulse
            min_peak_latency_samples = int(1 / 1000 / dt) # 1 ms minimum physiological latency
            baseline = I_smooth[:pulse_onsets[0] - buffer_samples]
            if len(baseline) == 0:
                reason = "empty baseline window before first pulse"
                warn_skip(sweep, reason)
                continue
            baseline_mean = np.mean(baseline)
            baseline_sd = np.std(baseline)
            if not np.isfinite(baseline_mean):
                reason = "holding_I could not be calculated from baseline"
                warn_skip(sweep, reason)
                continue
            if abs(baseline_mean) > self.max_holding_I_pA:
                reason = f"holding_I {baseline_mean:.1f} pA exceeds +/-{self.max_holding_I_pA:g} pA"
                warn_skip(sweep, reason)
                continue
            inward_noise_threshold = baseline_mean - 3 * baseline_sd  # HARD CODE threshold 3SD
            valid_sweep = True
            skip_reason = None

            # LOOP ON PULSES #
            peak_amplitudes = []
            peak_values =[]
            peak_sweep_idxs = [] # peak index in sweep
            for pulse_num, offset in enumerate(pulse_offsets[:2], start=1):
                start_window = offset + buffer_samples
                end_window = min(len(I), offset + w_samples)

                I_window = I[start_window:end_window]
                I_window_smooth = I_smooth[start_window:end_window]
                if len(I_window) < 2:
                    skip_reason = f"pulse {pulse_num}: empty search window after stim offset"
                    valid_sweep = False
                    break

                # FIND PEAK #
                peak_prominence = 3 * baseline_sd
                peaks, props = find_peaks(-I_window_smooth, prominence=peak_prominence)
                if len(peaks) == 0:
                    skip_reason = f"pulse {pulse_num}: no inward peak above prominence threshold ({peak_prominence:.2f} pA)"
                    valid_sweep = False
                    break
                # strongest negative peak
                peak_idx = peaks[np.argmax(-I_window_smooth[peaks])]  
                peak_val = I_window_smooth[peak_idx]
                peak_sweep_idx = start_window + peak_idx 
                peak_amplitude = peak_val - baseline_mean

                # max rate of change to detect APs /l atency occurance of peak from end of stim physiological threshold
                max_rate = np.max(np.abs(np.diff(I_window)))
                peak_latency_ms = peak_idx * dt * 1000
                reject_reasons = []
                if max_rate > I_AP_rate_threshold:
                    reject_reasons.append(f"rapid current deflection {max_rate:.1f} > {I_AP_rate_threshold:g}")
                if peak_idx < min_peak_latency_samples:
                    reject_reasons.append(f"peak latency {peak_latency_ms:.2f} ms < {min_peak_latency_samples * dt * 1000:.2f} ms")
                if peak_amplitude < -1000:
                    reject_reasons.append(f"peak amplitude {peak_amplitude:.1f} pA < -1000 pA")
                if reject_reasons:
                    skip_reason = f"pulse {pulse_num}: potential AP/artefact ({'; '.join(reject_reasons)})"
                    valid_sweep = False
                    # DEBUG PLOT 
                    # plt.figure(figsize=(6, 2))
                    # x = np.arange(len(I_window))
                    # plt.plot(x, I_window, color='lightgrey', label='I window')
                    # plt.plot(x, I_window_smooth, color='black', label='I smooth')
                    # # show detected peak if it exists
                    # if 'peak_idx' in locals():
                    #     plt.plot(peak_idx, I_window_smooth[peak_idx], 'ro', label='peak')
                    # plt.title(f"AP reject - {row['folder_file']} sweep {sweep}")
                    # plt.legend()
                    # plt.tight_layout()
                    # plt.show()
                    break  

                valid_peak = peak_val < inward_noise_threshold
                if valid_peak:
                    peak_amplitudes.append(peak_amplitude)
                    peak_values.append(peak_val)
                    peak_sweep_idxs.append(peak_sweep_idx)
                else:
                    skip_reason = (
                        f"pulse {pulse_num}: peak did not pass noise threshold "
                        f"(peak {peak_val:.1f} pA, threshold {inward_noise_threshold:.1f} pA, "
                        f"amp {peak_amplitude:.1f} pA)"
                    )
                    valid_sweep = False
                    break

            if not valid_sweep:
                reason = skip_reason or "unknown PPR quality check failed"
                warn_skip(sweep, reason)
                continue

            p1, p2 = pulse_offsets[:2]
            amp1=peak_amplitudes[0]
            amp2=peak_amplitudes[1]
            peak_val1= peak_values[0]
            peak_val2=peak_values[1]
            peak1_sweep_idx = peak_sweep_idxs[0]
            peak2_sweep_idx = peak_sweep_idxs[1]

            if (amp2 / amp1) < 0 or (amp2 / amp1) > 4.2: #physiological catch often polysynaptic or slow AP ie 'F5104/2026_05_05_0007' <10 bad peak detection
                reason = f"PPR {amp2/amp1:.2f} outside expected range 0-4.2"
                warn_skip(sweep, reason)
                continue
                # PPRs.append(np.nan) #TODO revisit if this shouldbe done
            
            PPRs.append(amp2 / amp1 if amp1 != 0 else np.nan)
            baseline_V.append(np.median(V[:p1-5]))
            baseline_I.append(np.mean(I[:p1-5]))
            ISIs.append(int((pulse_onsets[1] - p1) * dt * 1000)) #beginning of second to end of first
            pulse1_amp.append(amp1)
            pulse2_amp.append(amp2)

        # DEBUG PLOT
        # plt.figure(figsize=(9, 3))
        # plt.plot(I, color='lightgrey', label='I raw')
        # plt.plot(I_smooth, color='black', label='I smooth')
        # plt.plot(stim * 10, color='red', alpha=0.6, label='stim x10')
        # # pulses + windows
        # for i, offset in enumerate(pulse_offsets[:2]):
        #     c = 'blue' if i == 0 else 'green'
        #     start = offset + buffer_samples
        #     end = min(len(I), offset + w_samples)
        #     plt.axvline(offset, color=c, linestyle='--')
        #     plt.axvspan(start, end, color=c, alpha=0.12)
        # # PEAKS (THIS is the correct way using your computed indices)
        # p1 = pulse_offsets[0] + buffer_samples + np.where(I_smooth[pulse_offsets[0]+buffer_samples : pulse_offsets[0]+w_samples] == np.min(I_smooth[pulse_offsets[0]+buffer_samples : pulse_offsets[0]+w_samples]))[0][0]
        # p2 = pulse_offsets[1] + buffer_samples + np.where(I_smooth[pulse_offsets[1]+buffer_samples : pulse_offsets[1]+w_samples] == np.min(I_smooth[pulse_offsets[1]+buffer_samples : pulse_offsets[1]+w_samples]))[0][0]
        # plt.plot(peak1_sweep_idx, peak_val1, 'bo', ms=8, label='Peak 1')
        # plt.plot(peak2_sweep_idx, peak_val2, 'go', ms=8, label='Peak 2')
        # plt.axhline(inward_noise_threshold, color='grey', linestyle='--', alpha=0.6, label='noise')
        # plt.title(f"Sweep {sweep} - {row['folder_file']}")
        # plt.xlabel("Samples")
        # plt.ylabel("Current (pA)")
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        # path = f"/Users/jasminebutler/Desktop/PPR_exampleplot_{row['treatment']}_{row['cell_id']}.svg"
        # plt.savefig(path, format='svg', dpi=300)
        
        if len(np.unique(ISIs)) > 1:
            most_common = np.bincount(ISIs).argmax()
            self._print_warning(
                row,
                f"ISIs vary across sweeps: {np.unique(ISIs)}, using most common {most_common} ms"
            )
            row['ISI_ms'] = most_common
        else:
            try:
                row['ISI_ms'] = ISIs[0]
            except IndexError:
                row['ISI_ms'] = np.nan  

        if len(PPRs) == 0:
            message = "PPR_VC no valid sweeps"
            if skipped_sweeps:
                shown = "; ".join(skipped_sweeps[:5])
                if len(skipped_sweeps) > 5:
                    shown = f"{shown}; ... (+{len(skipped_sweeps) - 5} more)"
                message = f"{message}: {shown}"
            row = self._append_warning(row, message)
        elif skipped_sweeps and self.print_warnings:
            shown = "; ".join(skipped_sweeps[:5])
            if len(skipped_sweeps) > 5:
                shown = f"{shown}; ... (+{len(skipped_sweeps) - 5} more)"
            row = self._append_warning(row, f"PPR_VC skipped sweeps: {shown}")

        row['pulse1_amplitude_pA'] = pulse1_amp
        row['pulse2_amplitude_pA'] = pulse2_amp
        row['PPR'] = PPRs
        row['RMP_mV'] = np.median(baseline_V) if baseline_V else np.nan
        row['holding_I'] = np.mean(baseline_I) if baseline_I else np.nan

        return row


@dataclass
class IF_IC(EphysData):    
    filename: str = "IF_IC_df"
    data_type: str = 'IF_IC'
    
    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'treatment', 'region', 'cell_subtype', 'cell_type', 'sex', 'hemisphere', 'behaviour', 'subject_id'] #, 'R_series'] # R_series is redundant for pCLAMP data #TODO
        super().__post_init__()
    
    def process(self, row: pd.Series) -> pd.Series:
        """
        Processes current-clamp step protocols (I–F curves).

        Takes: 
            Voltage (V) and current (I) traces from a single cell recording.

        Returns:
            The input row (pd.Series) with extracted properties added. Columns added include:
                '%_sag'
                'IF_rheobase_pA', 'IF_slope'
                'valid_APs'
                'AP_peaks_mV', 'IF_voltage_threshold_mV', 'AP_height_mV', 
                'AP_width_ms', 'AP_rise_mV_ms', 'AP_decay_mV_ms', 
                'AP_latency_ms', 'AP_max_rise_mV_ms'
                'I_steps_pA', 'AP_frequencies_Hz', 'max_firing_Hz',
                'I_step_max_firing_pA'
                'IF_fit_status', 'IF_rheobase_method'
                'off_step_peak_locs'
                'holding_I', 'RMP_mV'
        """
        V_array, I_array, command_array, stim_array, V_list = self.load_data(row['folder_file'])
        measured_I_available = I_array is not None

        peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_rise_all  , peak_max_dvdt_all,  peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all   , peak_indices_all , sweep_indices_all , peak_decay_all = ap_characteristics_extractor_main(row['folder_file'], V_array, sampling_rate=self.sampling_rate)
        if len(peak_voltages_all)==0: #returns is no APs are detected
            row["IF_fit_status"] = "no_APs"
            row["IF_rheobase_method"] = None
            row['I_step_max_firing_pA'] = np.nan
            row['valid_APs'] = False
            return row
    
        
        if measured_I_available:
            I_array_offset, offset = correct_I_offset_IF(I_array)
            I_array_adj_clean = denoise_steps(I_array_offset)
        else:
            offset = np.nan
            I_array_adj_clean = None

        I_steps_pA , AP_frequencies_Hz, V_rest , off_step_peak_locs, fi_step_details = extract_FI_x_y(
            row['folder_file'],
            V_array,
            I_array_adj_clean,
            self.sampling_rate,
            peak_locs_corr_all=peak_locs_corr_all,
            sweep_indices_all=sweep_indices_all,
            command_array=command_array,
            return_details=True
        )
        row = self._warn_protocol_source(
            row,
            "IF step detection",
            fi_step_details.get("step_source")
        )

        if not isinstance(I_steps_pA, (list, tuple, np.ndarray)):
            self._print_warning(row, "IF step detection failed")
            row = self._append_warning(row, "IF step detection failed")
            row["IF_fit_status"] = "no_step_detected"
            row["IF_rheobase_method"] = None
            row['valid_APs'] = False
            return row
        
        # unirom step size correction
        I_steps_pA = np.asarray(I_steps_pA, dtype=float)
        unique_steps = np.unique(I_steps_pA)
        step_size = np.nan
        if len(unique_steps) > 1:
            step_size = np.round(np.median(np.diff(unique_steps)))
            if np.isfinite(step_size) and step_size != 0:
                I_steps_pA = (np.round(I_steps_pA / step_size) * step_size).astype(int).tolist()
        if isinstance(I_steps_pA, np.ndarray):
            I_steps_pA = I_steps_pA.astype(int).tolist()

        FI_slope, rheobase_threshold, valid_APs, fit_details = FI_slope_and_rheobase(
            row['folder_file'],
            I_steps_pA,
            AP_frequencies_Hz,
            return_details=True,
            verbose=False
        )

        sag_step_source = command_array if fi_step_details.get("step_source") == "command_array" else I_array_adj_clean
        row["%_sag"] = sag_current_analyser(row['folder_file'], V_array, sag_step_source, I_steps_pA, AP_frequencies_Hz)
        row["IF_rheobase_pA"] = rheobase_threshold
        row["IF_slope"] = FI_slope
        row['valid_APs'] = valid_APs
        row["IF_fit_status"] = fit_details.get("status")
        row["IF_rheobase_method"] = fit_details.get("method")
        row["IF_fit_quality"] = fit_details.get("fit_quality")
        row["IF_last_no_AP_pA"] = fit_details.get("last_I")
        row["IF_first_AP_pA"] = fit_details.get("first_I")
        if row["IF_fit_status"] not in [None, "ran"]:
            self._append_warning(
                row,
                f"IF fit status: {row['IF_fit_status']} ({row['IF_rheobase_method']})"
            )

        row['AP_peaks_mV'] = peak_voltages_all[:10] #turn around on APs
        row["IF_voltage_threshold_mV"] = v_thresholds_all[:10] #mV at upshoot
        row["AP_height_mV"] = peak_heights_all[:10]
        row["AP_width_ms"] = peak_fw_all[:10]
        row["AP_rise_mV_ms"] = peak_rise_all[:10]
        row["AP_decay_mV_ms"] = peak_decay_all[:10]
        row["AP_latency_ms"] = peak_latencies_all[:10]
        row["AP_max_rise_mV_ms"] = peak_max_dvdt_all[:10]

        row['I_steps_pA'] = I_steps_pA
        row['AP_frequencies_Hz'] = AP_frequencies_Hz
        max_firing_locs = []
        max_firing_sweep = None
        if len(sweep_indices_all) > 0:
            sweep_counts = pd.Series(sweep_indices_all).value_counts()
            max_firing_sweep = sweep_counts.idxmax()
            max_firing_locs = sorted(
                peak_loc
                for peak_loc, sweep_idx in zip(peak_locs_corr_all, sweep_indices_all)
                if sweep_idx == max_firing_sweep
            )
        if len(max_firing_locs) >= 2:
            row['max_firing_Hz'] = float(np.nanmean(self.sampling_rate / np.diff(max_firing_locs)))
        else:
            row['max_firing_Hz'] = np.nan
        try:
            row['I_step_max_firing_pA'] = I_steps_pA[int(max_firing_sweep)]
        except (TypeError, ValueError, IndexError):
            row['I_step_max_firing_pA'] = np.nan
        row['IF_step_size_pA'] = step_size

        row['off_step_peak_locs']=off_step_peak_locs
        row["RMP_mV"]=V_rest
        row['holding_I'] = offset if measured_I_available else np.nan

        # retro axonal action potential detection RA APs
        try:
            cell_threshold = np.mean(row['IF_voltage_threshold_mV'])
        except:
            cell_threshold = -45 #so when you -20 is 65 for cells without FP
        RA_condition = lambda peak_voltage, threshold: threshold <= (cell_threshold - 20) and peak_voltage > 0 
        if any(RA_condition(peak_voltage, threshold) for peak_voltage, threshold in zip(peak_voltages_all, v_thresholds_all)):
            row['RA'] = True
            row['RA_locs'] = [peak_locs_corr_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if threshold <= -65 and peak_voltage > 20]
            total_minutes = (V_array.shape[0] * V_array.shape[1]) / self.sampling_rate / 60
            row['RA_per_min'] = len(row['RA_locs']) / total_minutes if total_minutes > 0 else np.nan #RA/minute

        # FILE VALIDATOR 
        if np.mean(np.array(peak_voltages_all[:10])[~np.isnan(peak_voltages_all[:10])]) < 15: #mean of first 11 AP peaks is less than 15mV the file is marked invalid
            row['valid'] = False 

        freq = np.array(AP_frequencies_Hz) # at least 4 consecutive non-zero firing frequencies
        if freq.size == 0 or np.max(np.diff(np.flatnonzero(np.concatenate(([0], freq > 0, [0])) == 0)) - 1) < 4:
            row['valid'] = False

        return row


@dataclass
class APP_IC(EphysData):
    
    filename: str = "APP_IC_df"
    data_type: str = 'APP_IC'
    extraction_feature_columns: list = field(default_factory=lambda: ["folder_file", "data_type", "drug_in", "drug_out", "I_set"])
    amplitude_threshold: float = None
    noise_multiplier: float = 4
    min_amplitude_threshold: float = 0.1
    max_amplitude_threshold: float = 1.0
    baseline_method: str = 'rolling_median'
    baseline_window_s: float = 0.1
    rise_time_range: tuple = (0.5e-3, 5e-3)
    peak_window_s: float = 0.001
    upshoot_baseline_window_s: float = 0.001
    onset_search_window_s: float = 0.080
    debug_local_baseline_plot: bool = False
    debug_plot: bool = False
    validation_pre_sweep_window: int = 4
    baseline_variability_threshold: float = 0.30
    AP_burst_window_s: float = 1
    AP_height_drift_threshold_mV: float = 20
    AP_height_collapse_threshold_mV: float = 25
    RMP_drift_threshold_mV: float = 20
    terminal_depolarization_threshold_mV: float = -20
    terminal_depolarization_min_s: float = 60
    linear_trend_min_r2: float = 0.35
    trend_edge_fraction: float = 0.20

    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'I_set', 'treatment', 'drug_in', 'drug_out', 'cell_type', 'cell_subtype', 'region', 'hemisphere','sex']
        super().__post_init__()

    def _detect_sweep_sEPSPs(self, V_array, protocol_array, row: pd.Series) -> dict:
        """Detect sEPSPs per APP sweep, using off-step command periods when available."""
        V_array = np.asarray(V_array, dtype=float)
        if V_array.ndim == 1:
            V_array = V_array.reshape(-1, 1)

        if protocol_array is not None:
            protocol_array = np.asarray(protocol_array, dtype=float)
            if protocol_array.ndim == 1:
                protocol_array = protocol_array.reshape(-1, 1)

        sweep_frequency = []
        sweep_count = []
        sweep_mean_amplitude = []
        sweep_amplitudes = []
        sweep_locs = []
        sweep_raw_peak_locs = []
        sweep_upshoot_locs = []
        sweep_upshoot_status = []
        sweep_valid_time = []
        sweep_warnings = []

        for sweep in range(V_array.shape[1]):
            V_sweep = V_array[:, sweep]
            valid_mask = None
            if protocol_array is not None and sweep < protocol_array.shape[1]:
                _, _, rest_indices = _step_indices_from_command_trace(protocol_array[:, sweep])
                if rest_indices is not None and len(rest_indices) > 0:
                    valid_mask = np.zeros(V_sweep.shape, dtype=bool)
                    rest_indices = rest_indices[rest_indices < len(V_sweep)]
                    valid_mask[rest_indices] = True

            epsp = EPSP_detector(
                V_sweep,
                sampling_rate=self.sampling_rate,
                folder_file=f"{row['folder_file']} sweep {sweep}",
                amplitude_threshold=self.amplitude_threshold,
                noise_multiplier=self.noise_multiplier,
                min_amplitude_threshold=self.min_amplitude_threshold,
                max_amplitude_threshold=self.max_amplitude_threshold,
                baseline_method=self.baseline_method,
                baseline_window_s=self.baseline_window_s,
                rise_time_range=self.rise_time_range,
                peak_window_s=self.peak_window_s,
                upshoot_baseline_window_s=self.upshoot_baseline_window_s,
                onset_search_window_s=self.onset_search_window_s,
                valid_mask=valid_mask,
                debug_local_baseline_plot=self.debug_local_baseline_plot,
                debug_plot=self.debug_plot,
                print_warnings=self.print_warnings,
            )

            amplitudes = epsp['amplitudes_mV']
            finite_amplitudes = amplitudes[np.isfinite(amplitudes)]
            sweep_frequency.append(epsp['frequency_Hz'])
            sweep_count.append(len(epsp['peak_locs']))
            sweep_mean_amplitude.append(np.nanmean(finite_amplitudes) if len(finite_amplitudes) > 0 else np.nan)
            sweep_amplitudes.append(amplitudes)
            sweep_locs.append(epsp['peak_locs'])
            sweep_raw_peak_locs.append(epsp['raw_peak_locs'])
            sweep_upshoot_locs.append(epsp['upshoot_locs'])
            sweep_upshoot_status.append(epsp['upshoot_status'])
            sweep_valid_time.append(epsp['valid_time_s'])
            sweep_warnings.append(epsp.get('warnings', []))

        return {
            'frequency_Hz': np.asarray(sweep_frequency, dtype=float),
            'count': np.asarray(sweep_count, dtype=int),
            'mean_amplitude_mV': np.asarray(sweep_mean_amplitude, dtype=float),
            'amplitudes_mV': sweep_amplitudes,
            'locs': sweep_locs,
            'raw_peak_locs': sweep_raw_peak_locs,
            'upshoot_locs': sweep_upshoot_locs,
            'upshoot_status': sweep_upshoot_status,
            'valid_time_s': np.asarray(sweep_valid_time, dtype=float),
            'warnings': sweep_warnings,
        }

    def process(self, row: pd.Series) -> pd.Series:
        """Generate APP_IC_df from scratch, 
        Processing logic specific to APP data type."""
        V_array, I_array, command_array, stim_array, V_list = self.load_data(row['folder_file'])
        drug_in = 0 if pd.isna(row.get('drug_in', np.nan)) else int(row['drug_in'])
        row['sweep_duration_s'] = V_array.shape[0] / self.sampling_rate

        V_protocol, protocol_array, protocol_source = select_protocol_array(
            V_array,
            command_array=command_array,
            I_array=I_array,
            clean_I_fallback=True,
        )
        if protocol_source == "I_array_fallback":
            row = self._warn_protocol_source(row, "APP step detection", protocol_source)
        elif protocol_source == "missing":
            message = "APP protocol missing: RMP calculated from whole sweep; inputR set to NaN"
            self._print_warning(row, message)
            row = self._append_warning(row, message)

        row['sweep_inputR_MOhm'] = sweep_mean_inputR_calculator(
            V_array,
            command_array=command_array,
            I_array=I_array,
        )

        epsp = self._detect_sweep_sEPSPs(V_protocol, protocol_array, row)
        row['sweep_sEPSP_frequency_Hz'] = epsp['frequency_Hz']  # per-sweep sEPSP frequency from off-step valid time
        row['sweep_sEPSP_count'] = epsp['count']  # accepted sEPSP count per sweep
        row['sweep_sEPSP_mean_amplitude_mV'] = epsp['mean_amplitude_mV']  # mean raw amplitude per sweep
        row['sweep_sEPSP_amplitudes_mV'] = epsp['amplitudes_mV']  # accepted raw amplitudes per sweep
        row['sweep_sEPSP_locs'] = epsp['locs']  # accepted peak locations in processed detection trace, samples within sweep
        row['sweep_sEPSP_raw_peak_locs'] = epsp['raw_peak_locs']  # refined raw peak locations, samples within sweep
        row['sweep_sEPSP_upshoot_locs'] = epsp['upshoot_locs']  # selected raw upshoot/baseline locations, samples within sweep
        row['sweep_sEPSP_upshoot_status'] = epsp['upshoot_status']  # QC label for each sweep/event amplitude
        row['sweep_sEPSP_valid_time_s'] = epsp['valid_time_s']  # valid off-step seconds per sweep
        row['sweep_sEPSP_warnings'] = epsp['warnings']  # detector warning strings per sweep


        #TODO REMOVE AFTER CHECHING USE #UPDATE HIST
        # mean_RMP_PRE, mean_RMP_APP, mean_RMP_WASH = mean_RMP_APP_calculator(V_array, row.drug_in, row.drug_out, I_array=pass_I_array) #mean per sweep
        # row['RMP_PRE'] = mean_RMP_PRE
        # row['RMP_APP'] = mean_RMP_APP
        # row['RMP_WASH'] = mean_RMP_WASH

        row['sweep_RMP_mV'] = sweep_mean_RMP_calculator(
            V_array,
            command_array=command_array,
            I_array=I_array,
        )

        peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_rise_all  , peak_max_dvdt_all,  peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all   , peak_indices_all , sweep_indices_all , peak_decay_all = ap_characteristics_extractor_main(row.folder_file, V_array, sampling_rate=self.sampling_rate)

        
        # sweep_AP_count
        if len(v_thresholds_all)>0:
            if all (x > drug_in for x in sweep_indices_all):
                row['induced_APs'] = True #unused TODO
            APs_per_sweep = np.zeros(V_array.shape[1], dtype=int)
            unique, counts = np.unique(sweep_indices_all, return_counts=True)
            APs_per_sweep[unique] = counts
            row['sweep_AP_count']=APs_per_sweep 

        else:
            row['sweep_AP_count']=np.zeros(V_array.shape[1], dtype=int)
            
        def mean_threshold_value(value):
            if isinstance(value, (list, np.ndarray, pd.Series)):
                values = pd.to_numeric(pd.Series(list(value), dtype="object"), errors="coerce").dropna()
                return values.mean() if len(values) > 0 else np.nan
            return pd.to_numeric(pd.Series([value], dtype="object"), errors="coerce").iloc[0]

        try:
            IF_IC_df = self.getCache("IF_IC_df")
            pre_mask = IF_IC_df.apply(self.application_is_pre, axis=1)
            FP_cell_id_PRE = IF_IC_df[(IF_IC_df['cell_id'] == row['cell_id']) & pre_mask]
            threshold_col = (
                'IF_voltage_threshold_mV'
                if 'IF_voltage_threshold_mV' in FP_cell_id_PRE.columns
                else 'voltage_threshold'
                if 'voltage_threshold' in FP_cell_id_PRE.columns
                else None
            )
            cell_threshold = (
                FP_cell_id_PRE[threshold_col].apply(mean_threshold_value).mean()
                if threshold_col is not None
                else np.nan
            )
            if not np.isfinite(cell_threshold):
                cell_threshold = -45
        except ValueError:
            raise
        except Exception:
            cell_threshold = -45 #so when you -20 is 65 for cells without FP

        RA_condition = lambda peak_voltage, threshold: threshold <= (cell_threshold - 20) and peak_voltage > 0

        if any(RA_condition(peak_voltage, threshold) for peak_voltage, threshold in zip(peak_voltages_all, v_thresholds_all)):
            row['RA'] = True
            row['RA_locs'] = [peak_locs_corr_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if RA_condition(peak_voltage, threshold)]
            row['RA_sweep_locs'] = [sweep_indices_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if RA_condition(peak_voltage, threshold)]
            total_minutes = (V_array.shape[0] * V_array.shape[1]) / self.sampling_rate / 60
            row['RA_per_min'] = len(row['RA_locs']) / total_minutes if total_minutes > 0 else np.nan

            # sweep_RA_count
            RA_sweep_locs = row['RA_sweep_locs']
            RA_per_sweep = np.zeros(V_array.shape[1], dtype=int)
            unique, counts = np.unique(RA_sweep_locs, return_counts=True)
            RA_per_sweep[unique] = counts
            row['sweep_RA_count'] = RA_per_sweep
            row['sweep_SAP_count'] = row['sweep_AP_count'] - row['sweep_RA_count']

        else:
            row['RA_locs'] = []
            row['sweep_RA_count'] = np.zeros(V_array.shape[1], dtype=int)
            row['sweep_SAP_count'] = row['sweep_AP_count']

        row['AP_locs'] = peak_locs_corr_all
        row['AP_sweep_locs'] = sweep_indices_all
        row['peak_voltages_all'] = peak_voltages_all

        # GENERIC functions
        invalid_reasons = []

        def add_invalid_reason(reason):
            if reason not in invalid_reasons:
                invalid_reasons.append(reason)

        def check_variability(values, variability_threshold):
            """Return whether values stay within a max-min fractional range."""
            values = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").dropna().to_numpy()
            if len(values) <= 1:
                return True, np.nan, len(values)
            min_val = np.min(values)
            max_val = np.max(values)
            if min_val == 0:
                variability = 0 if max_val == 0 else np.inf
            else:
                variability = abs(max_val - min_val) / abs(min_val)
            return variability <= variability_threshold, variability, len(values)
        
        def group_AP_burst_values(peak_locs_corr_all, sweep_indices_all, event_values_all, burst_window_seconds=0.5):
            """
            Groups APs into bursts and returns the max value in each burst.

            This prevents APs inside the same burst from looking like a
            progressive AP-height drop across the recording.

            - peak_locs_corr_all: AP peak locations within sweep
            - sweep_indices_all: sweep of each AP
            - event_values_all: AP peak height or other AP-level value
            - burst_window_seconds: The time window (in seconds) to consider APs as part of the same burst. Default is 0.5 seconds.
            """
            burst_window_samples = int(burst_window_seconds * self.sampling_rate)
            bursts = []
            current_burst = []
            for peak_loc, sweep_index, value in zip(peak_locs_corr_all, sweep_indices_all, event_values_all):
                curr_time = (sweep_index * V_array.shape[0] + peak_loc) / self.sampling_rate
                if not current_burst:
                    current_burst.append((value, curr_time))
                    continue
                _, prev_time = current_burst[-1]
                time_diff = curr_time - prev_time
                time_diff_samples = time_diff * self.sampling_rate
                if time_diff_samples <= burst_window_samples:
                    current_burst.append((value, curr_time))
                else:
                    finite_values = [
                        value for value, _ in current_burst
                        if np.isfinite(pd.to_numeric(value, errors="coerce"))
                    ]
                    if finite_values:
                        bursts.append(max(finite_values))
                    current_burst = [(value, curr_time)]
            if current_burst:
                finite_values = [
                    value for value, _ in current_burst
                    if np.isfinite(pd.to_numeric(value, errors="coerce"))
                ]
                if finite_values:
                    bursts.append(max(finite_values))
            return bursts

        def trend_or_shift(values, threshold, direction="either"):
            """Flag robust whole-recording trends or early/late shifts."""
            values = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").dropna().to_numpy()
            n_values = len(values)
            if n_values < 3:
                return True, np.nan, np.nan, np.nan, n_values

            x = np.arange(n_values, dtype=float)
            slope, intercept = np.polyfit(x, values, 1)
            fitted = slope * x + intercept
            fitted_change = fitted[-1] - fitted[0]
            ss_res = np.sum((values - fitted) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot)

            edge_n = max(2, int(np.ceil(n_values * self.trend_edge_fraction)))
            edge_n = max(1, min(edge_n, n_values // 2))
            early_late_shift = np.nanmedian(values[-edge_n:]) - np.nanmedian(values[:edge_n])

            def passes_threshold(change):
                if not np.isfinite(change):
                    return False
                if direction == "decrease":
                    return change <= -threshold
                if direction == "increase":
                    return change >= threshold
                return abs(change) >= threshold

            trend_failed = (
                passes_threshold(fitted_change)
                and np.isfinite(r_squared)
                and r_squared >= self.linear_trend_min_r2
            )
            shift_failed = passes_threshold(early_late_shift)
            return not (trend_failed or shift_failed), fitted_change, early_late_shift, r_squared, n_values

        def terminal_collapse(values, threshold):
            """Flag an early large AP height that does not recover by the end."""
            values = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").dropna().to_numpy()
            n_values = len(values)
            if n_values < 3:
                return True, np.nan, np.nan, np.nan, n_values, np.nan

            edge_n = max(2, int(np.ceil(n_values * self.trend_edge_fraction)))
            edge_n = max(1, min(edge_n, n_values // 2))
            early_max = np.nanmax(values[:edge_n])
            terminal_median = np.nanmedian(values[-edge_n:])
            collapse = early_max - terminal_median
            return collapse < threshold, collapse, early_max, terminal_median, n_values, edge_n

        def terminal_true_run(mask):
            """Return length and start index of a True run ending at the final sweep."""
            count = 0
            for value in mask[::-1]:
                if bool(value):
                    count += 1
                else:
                    break
            start = len(mask) - count if count else None
            return count, start
        
        # APP FILE INVALIDATORS 
        validation_window = (
            int(self.validation_pre_sweep_window)
            if self.validation_pre_sweep_window is not None
            else None
        )
        baseline_start = max(0, drug_in - validation_window) if validation_window is not None else 1
        baseline = row['sweep_RMP_mV'][baseline_start:drug_in]
        baseline_valid, baseline_variability, baseline_n = check_variability(
            baseline,
            variability_threshold=self.baseline_variability_threshold,
        )
        if validation_window is not None and baseline_n < validation_window:
            add_invalid_reason(
                f"baseline_window: {baseline_n} finite PRE sweeps before drug_in "
                f"< required {validation_window}"
            )
        if baseline_valid == False: #assigns True if < variability threshold
            add_invalid_reason(
                f"baseline_variability: RMP baseline variability "
                f"{baseline_variability:.3f} > {self.baseline_variability_threshold:.3f} "
                f"using sweeps {baseline_start}:{drug_in}"
            )

        if len(peak_heights_all)>0: # if APs
            somatic_peak_locs = []
            somatic_sweeps = []
            somatic_heights = []
            for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)):
                if (
                    i < len(peak_locs_corr_all)
                    and i < len(sweep_indices_all)
                    and i < len(peak_heights_all)
                    and not RA_condition(peak_voltage, threshold)
                ):
                    somatic_peak_locs.append(peak_locs_corr_all[i])
                    somatic_sweeps.append(sweep_indices_all[i])
                    somatic_heights.append(peak_heights_all[i])
            AP_height_burst_max = group_AP_burst_values(
                somatic_peak_locs,
                somatic_sweeps,
                somatic_heights,
                burst_window_seconds=self.AP_burst_window_s,
            )
            ap_height_valid, ap_height_fit_change, ap_height_shift, ap_height_r2, ap_height_n = trend_or_shift(
                AP_height_burst_max,
                threshold=self.AP_height_drift_threshold_mV,
                direction="decrease",
            )
            if ap_height_valid == False:
                add_invalid_reason(
                    f"AP_height_drift: somatic AP burst height decrease detected "
                    f"(fit_change {ap_height_fit_change:.1f} mV, "
                    f"early_late_shift {ap_height_shift:.1f} mV, "
                    f"r2 {ap_height_r2:.2f}, n {ap_height_n}) "
                    f">= {self.AP_height_drift_threshold_mV:g} mV"
                )
            ap_collapse_valid, ap_collapse, ap_early_max, ap_terminal_median, ap_collapse_n, ap_collapse_edge_n = terminal_collapse(
                AP_height_burst_max,
                threshold=self.AP_height_collapse_threshold_mV,
            )
            if ap_collapse_valid == False:
                add_invalid_reason(
                    f"AP_height_collapse: early max somatic AP burst height "
                    f"{ap_early_max:.1f} mV - terminal median "
                    f"{ap_terminal_median:.1f} mV = {ap_collapse:.1f} mV "
                    f">= {self.AP_height_collapse_threshold_mV:g} mV "
                    f"(edge_n {ap_collapse_edge_n}, n {ap_collapse_n})"
                )

        pre_rmp = row['sweep_RMP_mV'][:drug_in]
        rmp_valid, rmp_fit_change, rmp_shift, rmp_r2, rmp_n = trend_or_shift(
            pre_rmp,
            threshold=self.RMP_drift_threshold_mV,
            direction="either",
        ) #assigns True if # REFACTOR as not used in plotter
        if  rmp_valid == False:
            add_invalid_reason(
                f"RMP_drift: PRE RMP trend/shift detected "
                f"(fit_change {rmp_fit_change:.1f} mV, "
                f"early_late_shift {rmp_shift:.1f} mV, "
                f"r2 {rmp_r2:.2f}, n {rmp_n}) "
                f">= {self.RMP_drift_threshold_mV:g} mV"
            )

        rmp_values = pd.to_numeric(pd.Series(row['sweep_RMP_mV'], dtype="object"), errors="coerce").to_numpy(dtype=float)
        depol_mask = np.isfinite(rmp_values) & (rmp_values > self.terminal_depolarization_threshold_mV)
        terminal_depol_sweeps, terminal_depol_start = terminal_true_run(depol_mask)
        terminal_depol_s = terminal_depol_sweeps * row['sweep_duration_s']
        if terminal_depol_s >= self.terminal_depolarization_min_s:
            add_invalid_reason(
                f"terminal_depolarization: RMP > {self.terminal_depolarization_threshold_mV:g} mV "
                f"for terminal {terminal_depol_s:.1f} s "
                f"(sweeps {terminal_depol_start}:{len(rmp_values)}) "
                f">= {self.terminal_depolarization_min_s:g} s"
            )

        if invalid_reasons:
            reason_text = "; ".join(dict.fromkeys(invalid_reasons))
            self._print_warning(row, f"APP_IC file marked invalid | reason: {reason_text}")
            row['valid'] = False
            row['invalid_reason'] = reason_text
            row = self._append_warning(row, f"APP_IC invalid: {reason_text}")
        else:
            row['valid'] = None #could be True
            row['invalid_reason'] = None
        return row
        
class Hunter(EphysData):
    '''Handels data type Hunter currently just fetching the RA locations.'''
    filename: str = "RA_hunter_df"
    data_type: str = 'Hunter'

    def __post_init__(self):
        self.initial_columns = ['folder_file', 'cell_id', 'data_type', 'treatment', 'cell_type', 'cell_subtype']
        super().__post_init__()

    def process(self, row: pd.Series) -> pd.Series:
        V_array, I_array, command_array, stim_array, V_list = self.load_data(row['folder_file'])


        peak_voltages_all, peak_latencies_all  , v_thresholds_all  , peak_rise_all  , peak_max_dvdt_all,  peak_locs_corr_all , upshoot_locs_all  , peak_heights_all  , peak_fw_all   , peak_indices_all , sweep_indices_all , peak_decay_all = ap_characteristics_extractor_main(row.folder_file, V_array, sampling_rate=self.sampling_rate)

        
        if any(threshold <= -65 and peak_voltage > 20 for peak_voltage, threshold in zip(peak_voltages_all, v_thresholds_all)):
            row['RA'] = True
            row['RA_locs'] = [peak_locs_corr_all[i] for i, (peak_voltage, threshold) in enumerate(zip(peak_voltages_all, v_thresholds_all)) if threshold <= -65 and peak_voltage > 20]
        return row



@dataclass
class Ephys(EphysData):
    ''' 
    Buiilding aggregate df with cell info based off extracted data from each data type in either : 
        application  ['APP_IC', 'IF_IC']         or       intrinsic_properties ['st_VC', 'ramp_IC', 'IV_VC', 'spont_IC', 'IF_IC', 'PPR' ] +AMPA/NMDA #TODO
        
        feature_df: excel input mapping folder_files to features
        
        ~ application = multiple timepoints                         
        IF_IC_df: extraction of firing property data (IF_IC) 
        APP_IC_df: extraction of applications data (APP_IC) 
        
        ~ intrinsic_properties = one timepoint
        st_VC: .. ect 


    Generates:
        cell_df: mapping of cells to features including change in access and IF_IC_valid and APP_IC_valid columns with valid folder_files
          '''
    filename: str = 'cell_df'
    sampling_rate: float = 2e4

    def _ensure_project_dataframes(self):
        """Load extractor dataframes needed to build a fresh cell_df."""
        if self.project_type == 'application': #TODO change to loop for data types in project
            if not hasattr(self, "IF_IC_df"):
                self.IF_IC_df = IF_IC(self.project, print_warnings=self.print_warnings).df
            if not hasattr(self, "APP_IC_df"):
                self.APP_IC_df = APP_IC(self.project, print_warnings=self.print_warnings).df
            feature_data_types = set(self.feature_df["data_type"].dropna()) if "data_type" in self.feature_df.columns else set()
            if "st_VC" in feature_data_types and not hasattr(self, "st_VC_df"):
                self.st_VC_df = st_VC(self.project, print_warnings=self.print_warnings).df

        elif self.project_type == 'intrinsic_properties':
            if not hasattr(self, "st_VC_df"):
                self.st_VC_df = st_VC(self.project, print_warnings=self.print_warnings).df
            if not hasattr(self, "IV_VC_df"):
                self.IV_VC_df = IV_VC(self.project, print_warnings=self.print_warnings).df
            if not hasattr(self, "ramp_IC_df"):
                self.ramp_IC_df = ramp_IC(self.project, print_warnings=self.print_warnings).df
            if not hasattr(self, "IF_IC_df"):
                self.IF_IC_df = IF_IC(self.project, print_warnings=self.print_warnings).df
            if not hasattr(self, "spont_IC_df"):
                self.spont_IC_df = spont_IC(self.project, print_warnings=self.print_warnings).df
            if not hasattr(self, "PPR_VC_df"):
                self.PPR_VC_df = PPR_VC(self.project, print_warnings=self.print_warnings).df

    def __post_init__(self):
        Project.__post_init__(self) # initates project only to get self.project_type
        self.df = self.update()
        
    
    def update(self) -> pd.DataFrame:
        """
        Bring cell_df up to date using current features.xlsx and extractor caches.
        """
        return self._update_cell_df(rerun_all=False)

    def regenerate(self) -> pd.DataFrame:
        """Rebuild cell_df from the current extractor dataframes."""
        return self._update_cell_df(rerun_all=True)

    def generate(self, force: bool = False) -> pd.DataFrame:
        """
        Backwards-compatible wrapper.

        Prefer ``update()`` for normal use and ``regenerate()`` to rebuild
        cell_df. This does not force raw extractor reruns.
        """
        if force:
            return self.regenerate()
        return self.update()

    def _update_cell_df(self, rerun_all: bool = False) -> pd.DataFrame:
        """Internal cell_df cache updater."""
        self._ensure_project_dataframes()

        if (
            not rerun_all
            and self.isCached(self.filename)
            and not self._features_newer_than_cache()
            and not self._extractor_cache_newer_than_cell_df()
        ):
            return self.add_missing_cell_factors(self.getCache(self.filename))

        if self.project_type == 'application':
            return self.generate_application_cell_df()
        elif self.project_type == 'intrinsic_properties':
            return self.generate_intrinsic_cell_df()

    def _extractor_cache_newer_than_cell_df(self) -> bool:
        """Return True when extractor caches have changed since cell_df was built."""
        cell_cache_path = os.path.join(self.cache_dir, f"{self.sanitize_filename(self.filename)}.pkl")
        if not os.path.exists(cell_cache_path):
            return True

        cell_cache_mtime = os.path.getmtime(cell_cache_path)
        for cache_name in self._extractor_cache_names():
            cache_path = os.path.join(self.cache_dir, f"{self.sanitize_filename(cache_name)}.pkl")
            if os.path.exists(cache_path) and os.path.getmtime(cache_path) > cell_cache_mtime:
                return True
        return False

    def _extractor_cache_names(self) -> list:
        """Extractor cache names used by this project type."""
        feature_data_types = (
            set(self.feature_df["data_type"].dropna())
            if "data_type" in self.feature_df.columns
            else set()
        )
        cache_names_by_type = {
            "st_VC": "st_VC_df",
            "IV_VC": "IV_VC_df",
            "ramp_IC": "ramp_IC_df",
            "IF_IC": "IF_IC_df",
            "spont_IC": "spont_IC_df",
            "PPR_VC": "PPR_VC_df",
            "APP_IC": "APP_IC_df",
        }

        if self.project_type == "application":
            data_types = ["IF_IC", "APP_IC"]
            if "st_VC" in feature_data_types:
                data_types.append("st_VC")
        else:
            data_types = ["st_VC", "IV_VC", "ramp_IC", "IF_IC", "spont_IC", "PPR_VC"]

        return [cache_names_by_type[data_type] for data_type in data_types]

    def generate_intrinsic_cell_df(self):
        df = self.feature_df.copy()
        cell_wise_columns = self.subject_cell_factor_columns()
        cell_df = (df.groupby('cell_id')
                    .apply(lambda g: self.apply_check_unique(g, unique_cols=cell_wise_columns))
                    .reset_index()
                )
        
        # Build file-level access checks, then keep a cell-level summary for backwards compatibility.
        access_df = self.build_access_df(self.st_VC_df)
        self.cache("access_by_file", access_df)
        self.save_excel("access_by_file", access_df)

        def max_abs_signed(series):
            values = series.dropna()
            if values.empty:
                return np.nan
            return values.loc[values.abs().idxmax()]

        rs_summary = (
            access_df
            .groupby("cell_id", as_index=False)
            .agg(
                Rs_abs_change=("Rs_abs_change", "max"),
                Rs_pct_change=("Rs_pct_change", max_abs_signed),
            )
        )
        st_vc_files = (
            self.st_VC_df
            .sort_values(by="folder_file", key=lambda col: col.map(self.folder_file_sort_key))
            .groupby("cell_id")["folder_file"]
            .apply(list)
            .reset_index(name=self.folder_files_col("Rs_MOhm"))
        )
        rs_df = rs_summary.merge(st_vc_files, on="cell_id", how="left")
        cell_df = cell_df.merge(rs_df, on="cell_id", how="left")
    
        # df , columns to reduce, data_type, average, n_files, sub_grouping
        reductions_spec = [
            (self.st_VC_df, ['Rm_MOhm', 'tau_ms', 'Cm_pF'], "st_VC", True, 1, None),
            (self.IF_IC_df, ["I_steps_pA", "AP_frequencies_Hz", "I_step_max_firing_pA"], "IF_IC", False, 1, None),
            (self.ramp_IC_df, ["ramp_rheobase_pA", 
                               "ramp_voltage_threshold_mV", 
                               "AP_height_mV", "AP_rise_mV_ms", 
                               "AP_decay_mV_ms", "AP_width_ms"],  "ramp_IC", True, 1, None),
            (self.IV_VC_df, ["I_step_steady_mV", "V_steps_mV"], "IV_VC", False, 1, None),
            (self.PPR_VC_df, ["PPR"], "PPR_VC", True, 2, ["ISI_ms"]),
            (self.spont_IC_df, ["sEPSP_frequency_Hz", 
                                "sEPSP_amplitudes_mV"],  "spont_IC", True, 1, None)
        ]

        reductions = [
            dict(zip(["df", "cols", "data_type", "avg", "n_files", "sub_grouping"], spec))
            for spec in reductions_spec
        ]

        for spec in reductions:
            merged = self.reduce_cellwise(
                spec['data_type'],
                spec["df"],
                spec["cols"],
                spec["avg"],
                spec["n_files"],
                spec["sub_grouping"]
            )

            cell_df = cell_df.merge(merged, on="cell_id", how="left")

        self.cache("cell_df", cell_df)
        self.save_excel("cell_df", cell_df)
        return cell_df

    @staticmethod
    def max_abs_signed(series):
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return np.nan
        return values.loc[values.abs().idxmax()]

    def add_st_vc_access_summary(self, cell_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cell-level access summary from st_VC recordings when available.

        This supports application projects that use regular st_VC files instead
        of the older per-IF-file R_series feature column.
        """
        st_vc_df = getattr(self, "st_VC_df", None)
        if st_vc_df is None or st_vc_df.empty or "Rs_MOhm" not in st_vc_df.columns:
            return cell_df

        access_df = self.build_access_df(st_vc_df)
        self.cache("access_by_file", access_df)
        self.save_excel("access_by_file", access_df)

        access_df["Rs_pct_change"] = pd.to_numeric(access_df["Rs_pct_change"], errors="coerce")
        access_df["Rs_abs_change"] = pd.to_numeric(access_df["Rs_abs_change"], errors="coerce")
        summary_source = access_df[access_df["data_type"] != "st_VC"].copy()
        if not summary_source["Rs_pct_change"].notna().any():
            return cell_df

        rs_summary = (
            summary_source
            .groupby("cell_id", as_index=False)
            .agg(
                Rs_abs_change=("Rs_abs_change", "max"),
                Rs_pct_change=("Rs_pct_change", self.max_abs_signed),
            )
        )

        merged = cell_df.merge(rs_summary, on="cell_id", how="left", suffixes=("", "_st_VC"))
        for col in ["Rs_abs_change", "Rs_pct_change"]:
            st_col = f"{col}_st_VC"
            if st_col not in merged.columns:
                continue
            existing = (
                pd.to_numeric(merged[col], errors="coerce")
                if col in merged.columns
                else pd.Series(np.nan, index=merged.index)
            )
            merged[col] = merged[st_col].combine_first(existing)
            merged = merged.drop(columns=[st_col])

        return merged


    def generate_application_cell_df(self):
        df = self.feature_df.copy()
        cell_wise_columns = [
            col for col in self.subject_cell_factor_columns()
            if col not in ["treatment", "I_set"]
        ] # treatment and I_set added later based off APP_IC
        
        def _extract_APP_attributes(group):
            app_rows = group[group['data_type'] == 'APP_IC']
            if app_rows.empty:
                treatments = group['treatment'].dropna()
                return pd.Series({
                    'I_set': np.nan,
                    'treatment': treatments.iloc[0] if not treatments.empty else np.nan,
                })

            app_row = app_rows.iloc[0]
            return pd.Series({
                'I_set': app_row['I_set'] if 'I_set' in app_row else np.nan,
                'treatment': app_row['treatment']
            })

        
        def calculate_percentage_diff(group):  
            #there are two ways access could be handeled 
            # 1 there is a column R_series for given folder files or 2 there are st_VC folder_files
            # 2 there is no column R_series and the change should be calculated from first to last st_VC recording 
            """
            USING R_series column for foler_files (data_type == 'IF_IC')
            
            Selects the two PRE and two non-PRE FP files with the most similar R_series values to compute access change. 
            If several have the same access chose the filder_files that have the least mising values."""
            cell_id = group.name

            #FIRING PROPERTY 
            cell_fp_df = self.IF_IC_df[self.IF_IC_df['cell_id'] == cell_id]
            if "R_series" not in cell_fp_df.columns:
                return pd.Series({'Rs_pct_change': None, self.folder_files_col("IF_IC"): None})

            pre_mask = cell_fp_df.apply(self.application_is_pre, axis=1)
            pre_values = cell_fp_df[pre_mask][['R_series', 'folder_file']].copy()
            non_pre_values = cell_fp_df[~pre_mask][['R_series', 'folder_file']].copy()
            pre_values['R_series'] = pd.to_numeric(pre_values['R_series'], errors='coerce')
            non_pre_values['R_series'] = pd.to_numeric(non_pre_values['R_series'], errors='coerce')
            
            # Extract R_series and folder_file
            pre_series = pre_values['R_series'].dropna().values
            non_pre_series = non_pre_values['R_series'].dropna().values
            
            # Check if there are enough values
            if len(pre_series) < 2 or len(non_pre_series) < 2:
                return pd.Series({'Rs_pct_change': None, self.folder_files_col("IF_IC"): None}) # mayher here files without pairs or not used should be dropped?
            
            # Generate all combinations of two values
            pre_combinations = list(combinations(pre_series, 2))
            non_pre_combinations = list(combinations(non_pre_series, 2))
            
            min_abs_diff = float('inf')
            min_diff = np.nan
            best_pre_pair = None
            best_non_pre_pair = None
            
            # Calculate percentage difference for all combinations
            for pre_pair in pre_combinations:
                pre_mean = np.mean(pre_pair)
                for non_pre_pair in non_pre_combinations:
                    non_pre_mean = np.mean(non_pre_pair)
                    if pre_mean == 0:
                        continue
                    percentage_change = (non_pre_mean - pre_mean) / pre_mean * 100
                    
                    if abs(percentage_change) < min_abs_diff:
                        min_abs_diff = abs(percentage_change)
                        min_diff = percentage_change
                        best_pre_pair = pre_pair
                        best_non_pre_pair = non_pre_pair
            
            if best_pre_pair is None or best_non_pre_pair is None:
                return pd.Series({
                    'Rs_pct_change': None,
                    self.folder_files_col("IF_IC"): None
                })

            # folder_file filtered on access
            pre_folder_files = pre_values[pre_values['R_series'].isin(best_pre_pair)]['folder_file'].tolist() 
            non_pre_folder_files = non_pre_values[non_pre_values['R_series'].isin(best_non_pre_pair)]['folder_file'].tolist()

            # filter folder_files on extracted features and absence of RA
            if len(pre_folder_files) > 2 or len(non_pre_folder_files) > 2:
                IF_IC_feature_cols = [
                    'AP_peaks_mV', 'AP_rise_mV_ms', 'AP_width_ms', 'IF_slope',
                    'max_firing_Hz', 'IF_rheobase_pA', '%_sag',  'IF_voltage_threshold_mV', 'AP_decay_mV_ms'
                ]
                IF_IC_feature_cols = [col for col in IF_IC_feature_cols if col in cell_fp_df.columns]
                
                pre_df = cell_fp_df[pre_mask].copy()
                non_pre_df = cell_fp_df[~pre_mask].copy()
                
                # Only keep rows that match the selected best R_series
                pre_df = pre_df[pre_df['R_series'].isin(best_pre_pair)]
                non_pre_df = non_pre_df[non_pre_df['R_series'].isin(best_non_pre_pair)]
                
                # Count missing values in relevant columns
                pre_df['missing_count'] = pre_df[IF_IC_feature_cols].isna().sum(axis=1)
                non_pre_df['missing_count'] = non_pre_df[IF_IC_feature_cols].isna().sum(axis=1)

                # Sort and select top 2
                pre_folder_files = pre_df.sort_values(by='missing_count')['folder_file'].iloc[:2].tolist()
                non_pre_folder_files = non_pre_df.sort_values(by='missing_count')['folder_file'].iloc[:2].tolist()
            else:
                # Safe fallback if only 1–2 values are returned, keep them directly
                pre_folder_files = pre_folder_files[:2]
                non_pre_folder_files = non_pre_folder_files[:2]

            return pd.Series({
                'Rs_pct_change': min_diff,
                self.folder_files_col("IF_IC"): pre_folder_files + non_pre_folder_files
            })

        #initalise off feature df
        cell_df = (
            df.groupby('cell_id')
            .apply(lambda g: self.apply_check_unique(g, unique_cols=cell_wise_columns, extra_logic=_extract_APP_attributes))
            .reset_index())
        
        diff_df = ( self.IF_IC_df 
                   .groupby('cell_id', group_keys=False)
                   .apply(calculate_percentage_diff))

        # diff_df = self.IF_IC_df.groupby('cell_id').apply(calculate_percentage_diff).reset_index()
        cell_df = cell_df.merge(diff_df, on='cell_id', how='left')
        cell_df = self.add_st_vc_access_summary(cell_df)

        cell_df = self.add_application_feature_folder_files(cell_df)


        # APPLICATION FILES
        filtered_app_df = self.APP_IC_df.copy()
        if "valid" in filtered_app_df.columns:
            filtered_app_df = filtered_app_df[filtered_app_df['valid'] != False]
        valid_files_dict = filtered_app_df.set_index('cell_id')['folder_file'].to_dict()
        cell_df[self.folder_files_col("APP_IC")] = cell_df['cell_id'].map(valid_files_dict)


        # Check RA status in IF_IC_df and APP_IC_df
        def ra_rows(df):
            required_cols = {'RA', 'cell_id', 'folder_file', 'RA_per_min'}
            if df is None or not required_cols.issubset(df.columns):
                return pd.DataFrame(columns=['cell_id', 'folder_file', 'RA_per_min'])
            return df[df['RA'] == True][['cell_id', 'folder_file', 'RA_per_min']]

        fp_ra_df = ra_rows(self.IF_IC_df) #FP and APP dataframes where RA is True
        app_ra_df = ra_rows(self.APP_IC_df)
        combined_ra_df = pd.concat([fp_ra_df, app_ra_df])

        ra_folder_files = combined_ra_df.groupby('cell_id')['folder_file'].apply(list).to_dict()

        ra_avg_per_min = combined_ra_df.groupby('cell_id')['RA_per_min'].mean().to_dict() #average RA_per_min per cell_id

        cell_df['RA'] = cell_df['cell_id'].isin(ra_folder_files)
        cell_df['RA_folder_file'] = cell_df['cell_id'].map(ra_folder_files)
        cell_df['RA_per_min'] = cell_df['cell_id'].map(ra_avg_per_min)

        self.cache("cell_df", cell_df)
        self.save_excel("cell_df", cell_df)
        return cell_df

    def add_application_feature_folder_files(self, cell_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add one folder-file column per data_type from features.xlsx for application projects.

        Existing columns are kept and only filled where missing, so access-selected
        IF_IC files and valid APP_IC files can still override the generic mapping.
        """
        if "data_type" not in self.feature_df.columns:
            return cell_df

        feature_files = self.feature_df.dropna(subset=["cell_id", "folder_file", "data_type"]).copy()
        if feature_files.empty:
            return cell_df

        for data_type, sub in feature_files.groupby("data_type", sort=False):
            folder_col = self.folder_files_col(data_type)
            file_map = (
                sub
                .sort_values(by="folder_file", key=lambda col: col.map(self.folder_file_sort_key))
                .groupby("cell_id")["folder_file"]
                .apply(list)
            )

            mapped_files = cell_df["cell_id"].map(file_map)
            if data_type == "APP_IC":
                mapped_files = mapped_files.apply(
                    lambda files: files[0]
                    if isinstance(files, list) and len(files) == 1
                    else files
                )

            if folder_col in cell_df.columns:
                cell_df[folder_col] = cell_df[folder_col].combine_first(mapped_files)
            else:
                cell_df[folder_col] = mapped_files

        return cell_df

    def apply_check_unique(self, group: pd.DataFrame, unique_cols: list, extra_logic=None):
        cell_id = group.name

        def check_unique(series, cell_id):
            unique_values = series.dropna().unique()
            if len(unique_values) == 0:
                return None
            if len(unique_values) == 1:
                return unique_values[0]
            else:
                raise ValueError(f"Non-unique values found for cell_id: {cell_id} with values: {unique_values}")
            
        aggregated_data = group.agg({
            col: lambda series: check_unique(series, cell_id) for col in unique_cols
        })

        if extra_logic is not None:
            aggregated_data = pd.concat([aggregated_data, extra_logic(group)])

        return aggregated_data
    
    def folder_files_col(self, data_type: str, group_value: str | int | None = None) -> str:
        """Return standardized column name for valid folder files of a given data_type and optional group."""
        if group_value is not None:
            return f"{group_value}_{data_type}_folder_files"
        return f"{data_type}_folder_files"

    def select_folder_files(self, df: pd.DataFrame, n_files=1, subgroup_key=None) -> list:
        """
        Select best folder_file(s) based on RMP near -70mV & minimal holding current.

        Returns:
            selected_folder_files: list of folder files 
        """
        df = df.copy() 
        #expand feature df to include scoring columns 
        df["rmp_score"] = -abs(df["RMP_mV"] + 70)
        df["holding_score"] = -df["holding_I"]
        df["total_score"] = df[["rmp_score", "holding_score"]].mean(axis=1)

        #sort based on total score and select top n_files
        df = df.sort_values(by="total_score", ascending=False)
        if n_files == "all":
            selected_folder_files = df["folder_file"].tolist()
        else:
            selected_folder_files = df["folder_file"].head(min(n_files, len(df))).tolist()
        return selected_folder_files


    def reduce_cellwise(
        self, 
        data_type: str,
        df: pd.DataFrame, 
        cols: list, 
        avg: bool = False, 
        n_files: int | str = 1, 
        subgroup_key: list | None = None
    ) -> pd.DataFrame:
        
        if df is None or df.empty:
            folder_col = self.folder_files_col(data_type, group_value=subgroup_key)
            expected_output_cols = ["cell_id"] + ([subgroup_key] if subgroup_key else []) + cols + [folder_col]
            return pd.DataFrame(columns=expected_output_cols) # return empty dummy df with expected columns
        
        if not subgroup_key:
            rows = []
            for cell_id, sub in df.groupby("cell_id"):
                folder_files_list = self.select_folder_files(sub, n_files=n_files)
                sub_filtered = sub[sub["folder_file"].isin(folder_files_list)]
                # vals = sub_filtered[cols].mean().to_dict() if avg else sub_filtered.iloc[0][cols].to_dict()
                if avg:
                    numeric_df = sub_filtered[cols].select_dtypes(include="number")
                    vals = numeric_df.mean().to_dict()
                else:
                    vals = sub_filtered.iloc[0][cols].to_dict()

                row = {"cell_id": cell_id}
                row.update(vals)
                row[self.folder_files_col(data_type)] = folder_files_list
                rows.append(row)
            return pd.DataFrame(rows)
        
        cell_rows = {}
        for cell_id, sub in df.groupby("cell_id"):
            cell_rows[cell_id] = {}

            #ensure string to avoid warning?
            if isinstance(subgroup_key, list) and len(subgroup_key) == 1:
                subgroup_key = subgroup_key[0]

            for subgroup_value, grp in sub.groupby(subgroup_key):
                folder_files_list = self.select_folder_files(grp, n_files=n_files)
                sub_filtered = grp[grp["folder_file"].isin(folder_files_list)]
                if avg:
                    vals = {}
                    for col in cols:
                        lists = sub_filtered[col].dropna().tolist()   # get lists in col, drop NaN
                        flat = [v for l in lists for v in (l if isinstance(l, list) else [l])]
                        vals[col] = np.mean(flat) if flat else np.nan
                else:
                    vals = {}
                    for col in cols:
                        first_row = sub_filtered.iloc[0][col]
                        if isinstance(first_row, list):
                            vals[col] = np.nan if len(first_row) == 0 else first_row[0]
                        else:
                            vals[col] = first_row

                for col, val in vals.items():
                    flat_col = f"{subgroup_value}_{col}"
                    cell_rows[cell_id][flat_col] = val

                folder_col = self.folder_files_col(data_type, group_value=subgroup_value)
                cell_rows[cell_id][folder_col] = folder_files_list

        df_rows = []
        for cell_id, values in cell_rows.items():
            row = {"cell_id": cell_id}
            row.update(values)
            df_rows.append(row)

        return pd.DataFrame(df_rows)
        
        # gcols = ["cell_id"]
        # if subgroup_key:
        #     if isinstance(subgroup_key, list):
        #         gcols.extend(subgroup_key)
        #     else:
        #         gcols.append(subgroup_key)

        # rows = []
        # for keys, sub in df.groupby(gcols):
        #     if not isinstance(keys, tuple):
        #         keys = (keys,)
                
        #     folder_files_list = self.select_folder_files(sub, n_files=n_files, subgroup_key=subgroup_key)
        #     sub_filtered = sub[sub["folder_file"].isin(folder_files_list)]

        #     if avg:
        #         vals = sub_filtered[cols].mean().to_dict()
        #     else:
        #         vals = sub_filtered.iloc[0][cols].to_dict()

        #     row = {col: key for col, key in zip(gcols, keys)}
        #     row.update(vals)
        #     folder_col_name = self.folder_files_col(data_type, group_value=keys[1] if subgroup_key else None)
        #     row[folder_col_name] = folder_files_list
        #     rows.append(row)

        # return pd.DataFrame(rows)
