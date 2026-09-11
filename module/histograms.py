from module.figure_common import *
from module.figure_base import Figure
from module.selection import DataSelection
from module.stats import MixedLMStatsMixin

class Histogram:
    """
    Shared histogram plotting base.

    Concrete subclasses provide the data-loading and statistics:
    - CellHistogram: ephys/cell-level data and MixedLM stats.
    - SubjectHistogram: animal-level external data and subject stats.
    """

    def get_plot_param(self, key, default=None):
        return getattr(self, "plot_params", {}).get(key, default)

    def safe_str(self, x):
        if x is None:
            return None
        if isinstance(x, (list, tuple, set)):
            return ", ".join(map(str, x))
        return str(x)

    def build_name(self, *args, sep="_", titlecase=False):
        def flatten(items):
            for item in items:
                if item is None:
                    continue
                if isinstance(item, (list, tuple, set)):
                    yield from flatten(item)
                else:
                    yield str(item).strip()

        parts = [p for p in flatten(args) if p]
        if titlecase:
            parts = [p.title() for p in parts]
        return sep.join(parts)

    def rgba_color(self, color, alpha=1.0):
        return mcolors.to_rgba(color, alpha=alpha)

    def p_to_star(self, p_val):
        if pd.isna(p_val):
            return "ns"
        if p_val < 0.001:
            return "***"
        if p_val < 0.01:
            return "**"
        if p_val < 0.05:
            return "*"
        return "ns"

    def save_plot(self, fig, filename: str, formats=('png', 'svg')):
        for fmt in formats:
            filepath = os.path.join(self.figure_output_dir, f"{filename}.{fmt}")
            fig.savefig(filepath, format=fmt, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved figure: {filename} in formats: {formats}")

    def check_valid_dependant_var(self):
        if self.dependant_var not in self.agg_df.columns:
            metadata_cols = []
            if hasattr(self, "project_obj") and hasattr(self.project_obj, "data_independant_columns"):
                metadata_cols = self.project_obj.data_independant_columns()
            dvs = [col for col in self.agg_df.columns if col not in metadata_cols]
            raise ValueError(f"Invalid dependant variable: {self.dependant_var}. Valid dv's : {dvs}")

    def available_histogram_factors(self):
        factor_cols = []
        if hasattr(self, "project_obj"):
            factor_cols.extend(getattr(self.project_obj, "subject_independant_vairables", []))
            factor_cols.extend(getattr(self.project_obj, "cell_independant_vairables", []))

        for col in ["subject_id", "cell_id", "time"]:
            if hasattr(self, "agg_df") and col in self.agg_df.columns:
                factor_cols.append(col)

        return list(dict.fromkeys(factor_cols))

    def validate_histogram_factors(self):
        requested_factors = [
            self.first_factor,
            getattr(self, "second_factor", None),
        ]
        available_factors = self.available_histogram_factors()
        missing_factors = [
            factor for factor in requested_factors
            if factor is not None and factor not in available_factors
        ]

        if missing_factors:
            missing_label = (
                missing_factors[0]
                if len(missing_factors) == 1
                else missing_factors
            )
            verb = "is" if len(missing_factors) == 1 else "are"
            print(
                f"Warning: {missing_label} {verb} not an existing factor in "
                f"project {self.project}."
            )
            print(f"Available factors are: {available_factors}")
            raise ValueError(
                f"Histogram factor(s) not found for project {self.project}: "
                f"{missing_factors}. Available factors are: {available_factors}"
            )

    def filter_n_minimum(self, df):
        df = df.dropna(subset=[self.dependant_var]).reset_index(drop=True)

        group_cols = [self.first_factor]
        if getattr(self, "second_factor", None) is not None:
            group_cols.append(self.second_factor)
        if "time" in df.columns:
            group_cols.append("time")

        group_cols = [col for col in group_cols if col in df.columns]
        group_sizes = df.groupby(group_cols).size()
        insufficient_groups = group_sizes[group_sizes < self.n_minimum]

        if not insufficient_groups.empty:
            print(f"Warning: The following groups have less than {self.n_minimum} samples and will be excluded:")
            print(insufficient_groups)
            df = df[~df[group_cols].apply(tuple, axis=1).isin(insufficient_groups.index)].reset_index(drop=True)

        if df.empty:
            print("No groups meet the minimum sample size requirement. Statistical analysis will not be performed.")
            return None
        return df

    def prepare_histogram_df(self, df=None):
        plot_df = self.data.copy() if df is None else df.copy()

        group_cols = [self.first_factor]
        if getattr(self, "second_factor", None) is not None:
            group_cols.append(self.second_factor)

        plot_df = plot_df.dropna(subset=group_cols).copy()
        plot_df[self.dependant_var] = pd.to_numeric(
            plot_df[self.dependant_var],
            errors="coerce",
        )
        plot_df = plot_df.dropna(subset=[self.dependant_var]).copy()

        if getattr(self, "second_factor", None) is not None:
            plot_df["plot_group"] = (
                plot_df[self.first_factor].astype(str) + "_" +
                plot_df[self.second_factor].astype(str)
            )

        return plot_df

    def plot_histogram(self, df=None, subgroup_name=None):
        if df is None and self.data is None:
            return None

        plot_df = self.prepare_histogram_df(df)
        self.plot_df = plot_df.copy()
        self.filename = self.build_histogram_filename(subgroup_name=subgroup_name)

        self.configure_plot_groups(plot_df)

        fig, ax = self.draw_histogram(plot_df)

        stats_results = self.run_histogram_stats(plot_df)
        self.annotate_stats(ax, plot_df, stats_results)

        self.finalize_histogram(ax, fig, subgroup_name=subgroup_name)

        return fig

    def configure_plot_groups(self, df):
        is_application_pre_post = (
            getattr(self.project_obj, "project_type", None) == "application"
            and "time" in df.columns
            and getattr(self, "second_factor", None) is None
        )

        self.order = [
            value for value in color_dict.keys()
            if value in df[self.first_factor].unique()
        ]
        self.order += [
            value for value in df[self.first_factor].dropna().unique()
            if value not in self.order
        ]

        if getattr(self, "second_factor", None) is None:
            self.x_axis = self.first_factor
            if is_application_pre_post:
                self.hue = "time"
                self.hue_order = [
                    value for value in ["PRE", "POST", "APP", "WASH"]
                    if value in df["time"].dropna().unique()
                ]
                self.stats_group_col = "plot_group"
                self.stats_group_order = [
                    f"{first_value}_{time_value}"
                    for first_value in self.order
                    for time_value in self.hue_order
                    if "plot_group" in df.columns
                    and f"{first_value}_{time_value}" in set(df["plot_group"])
                ]
            else:
                self.hue_order = None
                self.hue = self.first_factor
                self.stats_group_col = self.first_factor
                self.stats_group_order = self.order
        else:
            self.hue_order = [
                value for value in color_dict.keys()
                if value in df[self.second_factor].unique()
            ]
            self.hue_order += [
                value for value in df[self.second_factor].dropna().unique()
                if value not in self.hue_order
            ]
            self.x_axis = self.first_factor
            self.hue = self.second_factor
            self.stats_group_col = "plot_group"
            self.stats_group_order = [
                f"{a}_{b}"
                for a in self.order
                for b in self.hue_order
                if f"{a}_{b}" in set(df["plot_group"])
            ]

        self.palette = self.build_palette(df)

    def build_palette(self, df):
        color_values = []
        for col in [self.first_factor, getattr(self, "second_factor", None), "time"]:
            if col is not None and col in df.columns:
                color_values.extend(df[col].dropna().unique().tolist())

        unique_values = []
        for value in color_values:
            if value not in unique_values:
                unique_values.append(value)

        fallback_colors = cycle(sns.color_palette("tab10", n_colors=10))
        palette = {}
        for value in unique_values:
            palette[value] = color_dict[value] if value in color_dict else next(fallback_colors)

        return palette

    def draw_histogram(self, df):
        fig, ax = plt.subplots(
            figsize=(
                self.get_plot_param("figwidth", 15),
                self.get_plot_param("figheight", 10),
            )
        )

        sns.barplot(
            x=self.x_axis,
            y=self.dependant_var,
            hue=self.hue,
            hue_order=self.hue_order,
            order=self.order,
            data=df,
            errorbar=self.get_plot_param("errorbar", "se"),
            palette=self.palette,
            edgecolor=self.get_plot_param("bar_edgecolor", "k"),
            ax=ax,
            alpha=self.get_plot_param("bar_alpha", 1.0),
        )

        sns.swarmplot(
            x=self.x_axis,
            y=self.dependant_var,
            hue=self.hue,
            hue_order=self.hue_order,
            order=self.order,
            data=df,
            palette=self.palette,
            edgecolor="k",
            linewidth=0.5,
            ax=ax,
            legend=False,
            marker="o",
            size=0.05,
            alpha=0.7,
            dodge=True,
        )

        legend_handles_labels = self.specify_markers(df, ax, self.x_axis, self.hue)

        current_handles, current_labels = ax.get_legend_handles_labels()
        if not current_handles:
            current_handles, current_labels = self.default_color_legend()

        combined_handles = current_handles + list(legend_handles_labels.values())
        combined_labels = current_labels + [
            handle.get_label()
            for handle in legend_handles_labels.values()
        ]

        if combined_handles:
            ax.legend(
                handles=combined_handles,
                labels=combined_labels,
                loc=self.get_plot_param("legend_loc", "best"),
                title="Legend",
            )

        self.add_sample_size_labels(ax, df)

        return fig, ax

    def default_color_legend(self):
        if self.hue is None:
            return [], []

        if self.hue_order is not None:
            values = self.hue_order
        elif self.hue == self.first_factor:
            values = self.order
        else:
            values = []

        handles = []
        labels = []
        for value in values:
            if value not in self.palette:
                continue
            handles.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=self.palette[value],
                    edgecolor=self.get_plot_param("bar_edgecolor", "k"),
                    alpha=self.get_plot_param("bar_alpha", 1.0),
                )
            )
            labels.append(value)

        return handles, labels

    def specify_markers(self, df, ax, x_axis, hue):
        marker_df = df.copy()
        marker_col = getattr(self, "specify", None)
        legend_handles_labels = {}

        if marker_col is None:
            before_collections = len(ax.collections)
            sns.stripplot(
                x=x_axis,
                y=self.dependant_var,
                hue=hue,
                hue_order=self.hue_order,
                order=self.order,
                data=marker_df,
                palette=self.get_plot_param("marker_color", self.palette),
                edgecolor=self.get_plot_param("marker_edgecolor", "k"),
                linewidth=self.get_plot_param("marker_linewidth", 1),
                dodge=True,
                jitter=self.get_plot_param("marker_jitter", 0.15),
                ax=ax,
                legend=False,
                marker="o",
                size=self.get_plot_param("markersize", 7),
                alpha=self.get_plot_param("marker_alpha", 1.0),
            )
            self.style_marker_collections(ax.collections[before_collections:])
            return legend_handles_labels

        if marker_col not in marker_df.columns:
            return legend_handles_labels

        marker_df[marker_col] = marker_df[marker_col].fillna("none")

        markers = cycle(["o", "D", "s", "^", "v", "<", ">"]) if self.get_plot_param("marker_by_specify", True) else cycle(["o"])

        for value in marker_df[marker_col].unique():
            marker = next(markers)
            subset_to_plot = marker_df[marker_df[marker_col] == value]
            before_collections = len(ax.collections)

            sns.stripplot(
                x=x_axis,
                y=self.dependant_var,
                hue=hue,
                hue_order=self.hue_order,
                order=self.order,
                data=subset_to_plot,
                palette=self.get_plot_param("marker_color", self.palette),
                edgecolor=self.get_plot_param("marker_edgecolor", "k"),
                linewidth=self.get_plot_param("marker_linewidth", 1),
                linestyle="-",
                dodge=True,
                jitter=self.get_plot_param("marker_jitter", 0.15),
                ax=ax,
                legend=False,
                marker=marker,
                size=self.get_plot_param("markersize", 7),
                alpha=self.get_plot_param("marker_alpha", 1.0),
            )

            self.style_marker_collections(ax.collections[before_collections:])

            if self.get_plot_param("marker_by_specify", True):
                legend_facecolor = self.get_plot_param("marker_facecolor", None)
                if legend_facecolor == "none":
                    markerfacecolor = "none"
                elif legend_facecolor is not None:
                    markerfacecolor = self.rgba_color(
                        legend_facecolor,
                        self.get_plot_param("marker_facealpha", 1.0),
                    )
                else:
                    markerfacecolor = "white"

                legend_handles_labels[value] = plt.Line2D(
                    [0],
                    [0],
                    marker=marker,
                    label=value,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor="black",
                    color="black",
                    linestyle="None",
                )

        return legend_handles_labels

    def style_marker_collections(self, collections):
        marker_facecolor = self.get_plot_param("marker_facecolor", None)
        marker_edgecolor = self.get_plot_param("marker_edgecolor", None)

        for collection in collections:
            if marker_edgecolor is None:
                collection.set_edgecolors(collection.get_facecolors())
            else:
                collection.set_edgecolors(marker_edgecolor)

            if marker_facecolor == "none":
                collection.set_facecolors("none")
                collection.set_alpha(self.get_plot_param("marker_alpha", 1.0))
            elif marker_facecolor is not None:
                face_color = self.rgba_color(
                    marker_facecolor,
                    self.get_plot_param("marker_facealpha", 1.0),
                )
                n_points = len(collection.get_offsets())
                collection.set_facecolors(np.tile(face_color, (n_points, 1)))
                collection.set_alpha(None)
            else:
                collection.set_alpha(self.get_plot_param("marker_alpha", 1.0))

    def add_sample_size_labels(self, ax, df):
        fontsize = self.get_plot_param("n_label_fontsize", 22)

        if getattr(self, "is_subject_level_data", False):
            animal_counts = df.groupby(self.first_factor)["subject_id"].nunique()
            for tick, factor_value in enumerate(self.order):
                ax.text(
                    tick,
                    -0.1,
                    f"n (animals) = {animal_counts.get(factor_value, 0)}",
                    ha="center",
                    va="top",
                    fontsize=fontsize,
                    color="black",
                    transform=ax.get_xaxis_transform(),
                    linespacing=1.2,
                )
            return

        cell_counts = df.groupby(self.first_factor)["cell_id"].nunique()
        animal_counts = (
            df.groupby(self.first_factor)["subject_id"].nunique()
            if "subject_id" in df.columns
            else None
        )

        for tick, factor_value in enumerate(self.order):
            n_cells = cell_counts.get(factor_value, 0)
            n_animals = animal_counts.get(factor_value, 0) if animal_counts is not None else None
            text_label = (
                f"n (cells) = {n_cells}\n n (animals) = {n_animals}"
                if n_animals is not None
                else f"n = {n_cells}"
            )

            ax.text(
                tick,
                -0.1,
                text_label,
                ha="center",
                va="top",
                fontsize=fontsize,
                color="black",
                transform=ax.get_xaxis_transform(),
                linespacing=1.2,
            )

    def run_histogram_stats(self, df):
        raise NotImplementedError("Histogram subclasses must implement run_histogram_stats().")

    def add_two_way_stats_box(self, ax):
        return

    def get_selector_for_dependant_var(self):
        return None, None

    def histogram_optional_labels(self, subgroup_name=None, for_filename=False):
        parts = []

        selector = self.selector_label(sep="_" if for_filename else " = ")
        if selector is not None:
            parts.append(selector)

        if subgroup_name is not None:
            parts.append(f"subgroup_{subgroup_name}" if for_filename else str(subgroup_name))

        if getattr(self, "pre_sweep_window", None) is not None:
            parts.append(
                f"pre_sweep_window_{self.pre_sweep_window}"
                if for_filename
                else f"pre sweep window = {self.pre_sweep_window}"
            )

        if getattr(self, "post_sweep_window", None) is not None:
            parts.append(
                f"post_sweep_window_{self.post_sweep_window}"
                if for_filename
                else f"post sweep window = {self.post_sweep_window}"
            )

        return parts

    def selector_label(self, sep=" = "):
        selector_col, selector_value = self.get_selector_for_dependant_var()
        if selector_col is None:
            return None
        return f"{selector_col}{sep}{selector_value}"

    def build_histogram_filename(self, subgroup_name=None):
        return self.build_name(
            *self.histogram_name_parts(subgroup_name=subgroup_name, for_filename=True),
            sep="_",
        )

    def histogram_name_parts(self, subgroup_name=None, for_filename=False):
        factor_label = self.first_factor
        if getattr(self, "second_factor", None) is not None:
            factor_label = f"{self.first_factor}_by_{self.second_factor}"

        selection_parts = (
            self.selection_label_parts(for_filename=for_filename)
            if hasattr(self, "selection_label_parts")
            else []
        )

        parts = [
            self.dependant_var,
            getattr(self, "data_type", None),
            factor_label,
            selection_parts,
        ]

        parts.extend(
            self.histogram_optional_labels(
                subgroup_name=subgroup_name,
                for_filename=for_filename,
            )
        )

        if getattr(self, "specify", None) is not None:
            parts.append(f"markers_{self.specify}")

        return [part for part in parts if part is not None]

    def build_histogram_title(self, subgroup_name=None):
        y_label = unit_dict.get(self.dependant_var, self.dependant_var)
        selection_parts = (
            self.selection_label_parts(for_filename=False)
            if hasattr(self, "selection_label_parts")
            else []
        )
        parts = [
            y_label,
            selection_parts,
        ]

        parts.extend(
            self.histogram_optional_labels(
                subgroup_name=subgroup_name,
                for_filename=False,
            )
        )

        return self.build_name(*parts, sep=" ")

    def histogram_y_label(self):
        return unit_dict.get(self.dependant_var, self.dependant_var)

    def group_x_position(self, df, group):
        if (
            getattr(self, "stats_group_col", None) == "plot_group"
            and "plot_group" in df.columns
            and getattr(self, "hue", None) is not None
        ):
            row = df[df["plot_group"] == group].iloc[0]
            first_value = row[self.x_axis]
            hue_value = row[self.hue]

            x_index = self.order.index(first_value)
            hue_index = self.hue_order.index(hue_value)

            n_hue = len(self.hue_order)
            total_width = 0.8
            hue_width = total_width / n_hue

            return x_index - total_width / 2 + hue_width * (hue_index + 0.5)

        if getattr(self, "second_factor", None) is None:
            return self.order.index(group)

        row = df[df[self.stats_group_col] == group].iloc[0]
        first_value = row[self.first_factor]
        second_value = row[self.second_factor]

        x_index = self.order.index(first_value)
        hue_index = self.hue_order.index(second_value)

        n_hue = len(self.hue_order)
        total_width = 0.8
        hue_width = total_width / n_hue

        return x_index - total_width / 2 + hue_width * (hue_index + 0.5)

    def annotate_stats(self, ax, df, stats_results, alpha=None):
        if not stats_results:
            return
        if alpha is None:
            alpha = self.alpha

        y_min = df[self.dependant_var].min()
        y_max = df[self.dependant_var].max()
        y_range = y_max - y_min

        if y_range == 0:
            y_range = abs(y_max) * 0.1 if y_max != 0 else 1

        visible_results = []
        for res in stats_results:
            if getattr(self, "significant_only", True) and not res["significant"]:
                continue
            visible_results.append(res)

        if not visible_results:
            return

        base_y = y_max + y_range * 0.08
        step_y = y_range * 0.08
        tick_y = y_range * 0.02

        for i, res in enumerate(visible_results):
            group1 = res["group1"]
            group2 = res["group2"]

            try:
                x1 = self.group_x_position(df, group1)
                x2 = self.group_x_position(df, group2)
            except (ValueError, IndexError):
                print(f"[annotate_stats] could not place {group1} vs {group2}")
                continue

            y = base_y + i * step_y

            ax.plot(
                [x1, x1, x2, x2],
                [y, y + tick_y, y + tick_y, y],
                lw=1.5,
                color="black",
            )

            label = self.p_to_star(res["p_val"])
            if not res["significant"]:
                label = f"p={res['p_val']:.3f}"

            ax.text(
                (x1 + x2) / 2,
                y + tick_y,
                label,
                ha="center",
                va="bottom",
                fontsize=16,
                color="black",
            )

            p_adj = res.get("p_val", np.nan)
            p_unc = res.get("p_uncorrected", np.nan)
            p_unc_label = "NA" if pd.isna(p_unc) else f"{p_unc:.4f}"
            p_adj_label = "NA" if pd.isna(p_adj) else f"{p_adj:.4f}"
            print(f"{group1} vs {group2}: p_unc={p_unc_label}, p_adj={p_adj_label}")

        ax.set_ylim(top=base_y + len(visible_results) * step_y + y_range * 0.12)

    def finalize_histogram(self, ax, fig, subgroup_name=None):
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_ylabel(self.histogram_y_label(), fontsize=24)
        ax.set_xlabel("")
        ax.set_title(
            self.build_histogram_title(subgroup_name=subgroup_name),
            fontsize=28,
        )
        ax.tick_params(axis="x", labelsize=24)
        ax.tick_params(axis="y", labelsize=24)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self.add_two_way_stats_box(ax)
        self.apply_y_axis_range(ax)
        plt.tight_layout()
        plt.show()
        self.save_plot(fig, self.filename)

    def apply_y_axis_range(self, ax):
        y_axis_range = self.get_plot_param("y_axis_range", None)
        if y_axis_range is not None:
            ax.set_ylim(y_axis_range)


@dataclass
class CellHistogram(MixedLMStatsMixin, Histogram, Figure):
    '''
    Generic histogram class for plotting histograms of a specified dependant variable across treatments (and timepoints if project == application).
    '''
    filename: str = None
    dependant_var: str = field(kw_only=True)
    first_factor: str = field(kw_only=True, default='treatment') #bars to compare on x-axis
    second_factor: str | None = None
    plot_params: dict = field(default_factory=dict)
    alpha: float = 0.05 # p value threshold
    specify: str = field(kw_only = True, default = 'treatment') # specify marker to see subsets e.g. I_set or cell_id
    n_minimum: float = field(kw_only = True, default = 3)
    significant_only: bool = field(kw_only=True, default=True)

    pre_sweep_window: int = None # window before and after drug_in
    post_sweep_window: int = None
    I_steps_pA: int = None  # only for plotting IF_IC AP_frequencies_Hz
    ISI_ms: int = None # only for plotting PPR_VC PPR


    def __post_init__(self):
        self.filename = self.build_histogram_filename()
        self.is_subject_level_data = False
        super().__post_init__()
        self.validate_histogram_factors()
        self.check_valid_dependant_var()
        self.raw_time_df = self.agg_df.copy()
        if self.data_type == "APP_IC":
            self.data = self.build_pre_post_plot_df(self.raw_time_df, slice=True)
        else:
            self.data = self.raw_time_df.copy()

        self.data = self.filter_n_minimum(self.data) # TODO NOW here there is a col RMP_mV averaged dont know why or what it is / and there is the sweep_RMP_mV CHECK WHATS HAPPENING

            # REDUNDANT?
            # If dependant_var contains lists or arrays, average them to a single numeric value
            # if self.data[self.dependant_var].apply(lambda x: isinstance(x, (list, np.ndarray, pd.Series))).any(): #phasing this out
            #     print(f"[Histogram DEBUG] collapsing lists in {self.dependant_var}")
            # self.data[self.dependant_var] = self.data[self.dependant_var].apply(
            #     lambda x: np.mean(x) if isinstance(x, (list, np.ndarray, pd.Series)) else x
            # )

        self.fig = self.plot_histogram()

    def run_histogram_stats(self, df):
        """
        Run the appropriate mixed-model stats and return pairwise results.

        One-factor:
            pairwise MixedLM over first_factor

        Two-factor:
            two-way MixedLM for main effects/interaction
            plus pairwise MixedLM over plot_group
        """
        if (
            self.second_factor is None
            and getattr(self.project_obj, "project_type", None) == "application"
            and "time" in df.columns
            and set(df["time"].dropna().unique()).issubset({"PRE", "POST"})
        ):
            return self.run_application_pre_post_stats(df)

        if self.second_factor is None:
            return self.mixedlm_pairwise_stats(
                df,
                group_col=self.stats_group_col,
                value_col=self.dependant_var,
                group_order=self.stats_group_order,
            )

        print("\n=== TWO-WAY MIXED MODEL ===")
        self.two_way_mixed_model(df)

        return self.mixedlm_pairwise_stats(
            df,
            group_col=self.stats_group_col,
            value_col=self.dependant_var,
            group_order=self.stats_group_order,
        )

    def run_application_pre_post_stats(self, df):
        """
        Run paired PRE vs POST MixedLM separately within each first_factor group.
        """
        results = []
        for first_value in self.order:
            sub_df = df[df[self.first_factor] == first_value].copy()
            if sub_df.empty or sub_df["time"].nunique() < 2:
                continue

            try:
                group_results = self.mixedlm_pairwise_stats(
                    sub_df,
                    group_col="time",
                    value_col=self.dependant_var,
                    group_order=["PRE", "POST"],
                )
            except Exception as exc:
                print(f"MixedLM skipped for {first_value}: {exc}")
                continue

            for res in group_results:
                res = res.copy()
                res["group1"] = f"{first_value}_{res['group1']}"
                res["group2"] = f"{first_value}_{res['group2']}"
                res["first_factor"] = first_value
                results.append(res)

        self.posthoc_results = results
        self.application_delta_df = self.build_application_delta_df(df)
        self.application_delta_results = self.run_application_delta_stats(self.application_delta_df)
        self.posthoc_results = results
        return results

    def build_application_delta_df(self, df):
        """
        Build one POST - PRE change score per cell for between-treatment stats.
        """
        required_cols = ["cell_id", "time", self.first_factor, self.dependant_var]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Delta stats skipped: missing columns {missing_cols}")
            return pd.DataFrame()

        rows = []
        for cell_id, sub_df in df.groupby("cell_id"):
            pre_values = sub_df.loc[sub_df["time"] == "PRE", self.dependant_var].dropna()
            post_values = sub_df.loc[sub_df["time"] == "POST", self.dependant_var].dropna()
            if pre_values.empty or post_values.empty:
                continue

            row = {
                "cell_id": cell_id,
                self.first_factor: sub_df[self.first_factor].dropna().iloc[0],
                "application_delta": float(post_values.iloc[0] - pre_values.iloc[0]),
            }

            for col in ["subject_id", "cell_type", "cell_subtype", "sex", "region", "behaviour", "I_set"]:
                if col in sub_df.columns:
                    vals = sub_df[col].dropna()
                    if len(vals) > 0:
                        row[col] = vals.iloc[0]

            rows.append(row)

        return pd.DataFrame(rows)

    def run_application_delta_stats(self, delta_df):
        """
        Compare POST - PRE changes between first_factor groups.
        """
        if delta_df is None or delta_df.empty:
            return []
        if delta_df[self.first_factor].dropna().nunique() < 2:
            return []

        try:
            results = self.mixedlm_pairwise_stats(
                delta_df,
                group_col=self.first_factor,
                value_col="application_delta",
                group_order=self.order,
            )
        except Exception as exc:
            print(f"Delta MixedLM skipped: {exc}")
            return []

        self.print_application_delta_stats(results)
        return results

    def print_application_delta_stats(self, results):
        if not results:
            return

        print("\n=== APPLICATION DELTA MIXEDLM ===")
        print(f"Delta variable: POST - PRE {self.dependant_var}")
        for res in results:
            sig = "SIGNIFICANT" if res["significant"] else "ns"
            print(
                f"{res['group1']} vs {res['group2']} | "
                f"delta effect = {res['effect']:.4g} | "
                f"p_adj = {res['p_val']:.4g} | {sig}"
            )
        print("=" * 35)

    def get_selector_for_dependant_var(self):
        """
        Return the selector column/value needed for special dependent variables.

        Examples:
        - IF_IC AP_frequencies_Hz needs I_steps_pA
        - PPR_VC PPR needs ISI_ms

        Returns:
            (selector_col, selector_value) or (None, None)
        """
        selector_map = {
            ("IF_IC", "AP_frequencies_Hz"): ("I_steps_pA", self.I_steps_pA),
            ("PPR_VC", "PPR"): ("ISI_ms", self.ISI_ms),
        }

        return selector_map.get(
            (self.data_type, self.dependant_var),
            (None, None)
        )

    def is_list_like_value(self, value):
        """
        True for row values that store multiple measurements.
        """
        return isinstance(value, (list, np.ndarray, pd.Series))

    def apply_selector_filter(self, df, selector_col, selector_value):
        """
        Filter/extract rows for selector-specific variables.

        Handles two cases:

        1. selector_col is scalar per row:
            keep rows where df[selector_col] == selector_value

        2. selector_col is list-like per row:
            find selector_value inside that list and extract the matching item from
            self.dependant_var.

        Returns a copy of df.
        """
        if selector_col is None:
            return df.copy()

        if selector_value is None:
            raise ValueError(
                f"{self.dependant_var} requires {selector_col}. "
                f"Please pass {selector_col}=..."
            )

        if selector_col not in df.columns:
            raise ValueError(f"Selector column {selector_col} not found in dataframe.")

        if self.dependant_var not in df.columns:
            raise ValueError(f"Dependent variable {self.dependant_var} not found in dataframe.")

        df = df.copy()

        selector_is_list = df[selector_col].apply(self.is_list_like_value).any()

        if not selector_is_list:
            df = df[df[selector_col] == selector_value].copy()
            return df

        def extract_selected_value(row):
            selectors = row[selector_col]
            values = row[self.dependant_var]

            if not self.is_list_like_value(selectors):
                return pd.Series({
                    self.dependant_var: np.nan,
                    selector_col: np.nan,
                })

            if not self.is_list_like_value(values):
                return pd.Series({
                    self.dependant_var: np.nan,
                    selector_col: np.nan,
                })

            selectors = list(selectors)
            values = list(values)

            try:
                idx = selectors.index(selector_value)
            except ValueError:
                return pd.Series({
                    self.dependant_var: np.nan,
                    selector_col: np.nan,
                })

            if idx >= len(values):
                return pd.Series({
                    self.dependant_var: np.nan,
                    selector_col: np.nan,
                })

            return pd.Series({
                self.dependant_var: values[idx],
                selector_col: selector_value,
            })

        df[[self.dependant_var, selector_col]] = df.apply(extract_selected_value, axis=1)
        df = df.dropna(subset=[self.dependant_var, selector_col]).copy()

        return df

    def collapse_unselected_lists(self, df):
        """
        Collapse list-like dependent variable values only when no selector is needed.

        This preserves your old behavior for variables where a list should simply
        become its mean, but avoids averaging variables like AP_frequencies_Hz before
        selecting I_steps_pA.
        """
        df = df.copy()

        has_lists = df[self.dependant_var].apply(self.is_list_like_value).any()
        if has_lists:
            print(f"[Histogram DEBUG] collapsing lists in {self.dependant_var}")
            df[self.dependant_var] = df[self.dependant_var].apply(
                lambda x: np.mean(x) if self.is_list_like_value(x) else x
            )

        return df

    def prepare_histogram_df(self, df=None):
        """
        Build the dataframe used for plotting and stats.

        self.data remains the base data.
        This method returns a transformed plot_df with:
        - missing factor rows removed
        - selector-specific variables extracted/filtered
        - remaining list-like dependent values collapsed
        - plot_group added for two-factor designs
        """
        if df is None:
            plot_df = self.data.copy()
        else:
            plot_df = df.copy()

        group_cols = [self.first_factor]
        if self.second_factor is not None:
            group_cols.append(self.second_factor)

        plot_df = plot_df.dropna(subset=group_cols).copy()

        selector_col, selector_value = self.get_selector_for_dependant_var()
        plot_df = self.apply_selector_filter(plot_df, selector_col, selector_value)

        if selector_col is None:
            plot_df = self.collapse_unselected_lists(plot_df)

        plot_df[self.dependant_var] = pd.to_numeric(
            plot_df[self.dependant_var],
            errors="coerce"
        )
        plot_df = plot_df.dropna(subset=[self.dependant_var]).copy()

        if (
            self.second_factor is None
            and getattr(self.project_obj, "project_type", None) == "application"
            and "time" in plot_df.columns
        ):
            plot_df["plot_group"] = (
                plot_df[self.first_factor].astype(str) + "_" +
                plot_df["time"].astype(str)
            )
        elif self.second_factor is not None:
            plot_df["plot_group"] = (
                plot_df[self.first_factor].astype(str) + "_" +
                plot_df[self.second_factor].astype(str)
            )

        return plot_df

    def two_way_mixed_model(self, df):
        """
        Fit the main two-factor mixed-effects model.

        Model:
            dependent_variable ~ first_factor * second_factor + (1 | subject_id)

        This tests main effects and interaction while accounting for multiple cells
        from the same animal.
        """
        mixedlm_group_col = self.get_mixedlm_group_col(df)
        model_df = self.clean_mixedlm_df(
            df,
            group_col=self.stats_group_col,
            value_col=self.dependant_var,
            mixedlm_group_col=mixedlm_group_col,
        )

        combo_table = pd.crosstab(model_df[self.first_factor], model_df[self.second_factor])
        has_empty_combinations = (combo_table == 0).any().any()

        if has_empty_combinations:
            print(
                "\nWARNING: Some first_factor x second_factor combinations are missing. "
                "Cannot fit a full two-way interaction model. "
                "Fitting combined plot_group model instead.\n"
            )

            formula = f"{self.dependant_var} ~ C({self.stats_group_col})"

        else:
            formula = (
                f"{self.dependant_var} ~ "
                f"C({self.first_factor}, Treatment(reference='{self.order[0]}')) * "
                f"C({self.second_factor}, Treatment(reference='{self.hue_order[0]}'))"
            )

        self.mixedlm_result = self.fit_mixedlm(
            formula,
            model_df,
            groups=model_df[mixedlm_group_col],
            label="two-way MixedLM",
        )
        
        pvals = self.mixedlm_result.pvalues
        p_first = [pvals[k] for k in pvals.index if f"C({self.first_factor}" in k and ":" not in k]
        p_second = [pvals[k] for k in pvals.index if f"C({self.second_factor}" in k and ":" not in k]
        p_interaction = [pvals[k] for k in pvals.index if ":" in k]
        self.two_way_pvals = {
            self.first_factor: p_first[0] if p_first else np.nan,
            self.second_factor: p_second[0] if p_second else np.nan,
            "interaction": p_interaction[0] if p_interaction else np.nan,
        }
        for label, p_val in self.two_way_pvals.items():
            print(f"{label:15s}: p = {p_val:.4g}")
        print("=" * 35)

        return self.mixedlm_result

    def add_two_way_stats_box(self, ax):
        """
        Add main-effect/intervention p-values from the two-way MixedLM to the plot.
        """
        if self.second_factor is None:
            if not getattr(self, "application_delta_results", None):
                return

            lines = ["Delta MixedLM", "POST - PRE"]
            for res in self.application_delta_results:
                p_val = res["p_val"]
                p_label = "NA" if pd.isna(p_val) else f"{p_val:.3g}"
                lines.append(f"{res['group1']} vs {res['group2']}: p = {p_label}")

            ax.text(
                0.98,
                0.98,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=self.get_plot_param("stats_box_fontsize", 14),
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "alpha": 0.8,
                },
            )
            return

        if not hasattr(self, "two_way_pvals"):
            return

        lines = ["MixedLM"]
        for label, p_val in self.two_way_pvals.items():
            if np.isnan(p_val):
                lines.append(f"{label}: p = NA")
            else:
                lines.append(f"{label}: p = {p_val:.3g}")

        ax.text(
            0.98,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=self.get_plot_param("stats_box_fontsize", 14),           
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.8,
            },
        )
    
@dataclass(init=False)
class SubjectHistogram(Histogram, Cachable):
    """
    Histogram for animal-level non-ephys data.

    Rows are independent subjects, so stats use ordinary independent-subject
    models/tests rather than mixed-effects models.
    """
    def __init__(
        self,
        *,
        project: str,
        subject_data: str | pd.DataFrame,
        dependant_var: str,
        first_factor: str = "treatment",
        second_factor: str | None = None,
        plot_params: dict | None = None,
        alpha: float = 0.05,
        specify: str | None = None,
        n_minimum: float = 3,
        significant_only: bool = True,
        subject_data_cache: bool = True,
        filename: str | None = None,
    ):
        self.project = project
        self.subject_data = subject_data
        self.dependant_var = dependant_var
        self.first_factor = first_factor
        self.second_factor = second_factor
        self.plot_params = {} if plot_params is None else plot_params.copy()
        if specify is not None and self.plot_params.get("marker_by_specify") is False:
            print(
                "Warning: SubjectHistogram received specify but "
                "marker_by_specify=False. Using marker_by_specify=True so "
                f"{specify} is shown with separate markers and a legend."
            )
            self.plot_params["marker_by_specify"] = True
        self.alpha = alpha
        self.specify = specify
        self.n_minimum = n_minimum
        self.significant_only = significant_only
        self.subject_data_cache = subject_data_cache
        self.filename = filename

        self.data_type = None
        self.threshold_access_change = None
        self.pre_sweep_window = None
        self.post_sweep_window = None
        self.I_steps_pA = None
        self.ISI_ms = None

        self.__post_init__()

    def __post_init__(self):
        if self.filename is None:
            self.filename = self.build_histogram_filename()
        self.is_subject_level_data = True
        Cachable.__init__(self, cache_dir=f"{ROOT}/{self.project}/cache")
        self.location = f"{ROOT}/{self.project}"
        self.input_dir = self._checkFileSystem("input")
        self.output_dir = self._checkFileSystem("output")
        self.figure_output_dir = self._checkFileSystem("figures")
        self.project_obj = Project(self.project)
        self.agg_df = self.load_subject_histogram_df()
        self.check_valid_dependant_var()
        self.warn_missing_subject_factors()
        self.data = self.filter_n_minimum(self.agg_df)

        self.fig = self.plot_histogram()

    def load_subject_histogram_df(self):
        if isinstance(self.subject_data, pd.DataFrame):
            return self.project_obj.map_subject_factors(self.subject_data)

        return self.project_obj.load_subject_data(
            self.subject_data,
            cache=self.subject_data_cache,
        )

    def get_selector_for_dependant_var(self):
        return None, None

    def warn_missing_subject_factors(self):
        """
        Warn when subject rows will be dropped because requested factors failed to map.
        """
        factor_cols = [self.first_factor]
        if self.second_factor is not None:
            factor_cols.append(self.second_factor)
        if self.specify is not None:
            factor_cols.append(self.specify)

        factor_cols = [
            col for col in dict.fromkeys(factor_cols)
            if col is not None and col in self.agg_df.columns
        ]

        if not factor_cols or "subject_id" not in self.agg_df.columns:
            return

        for col in factor_cols:
            missing_df = self.agg_df[self.agg_df[col].isna()]
            if missing_df.empty:
                continue

            missing_ids = missing_df["subject_id"].dropna().astype(str).unique().tolist()
            print(
                f"Warning: {len(missing_ids)} subject_id(s) have no mapped "
                f"'{col}' value and will be excluded when plotting/statistics "
                f"use that factor: {missing_ids}"
            )

    def run_histogram_stats(self, df):
        if self.second_factor is None:
            self.subject_one_way_stats(df)
            return self.subject_pairwise_stats(
                df,
                group_col=self.stats_group_col,
                group_order=self.stats_group_order,
            )

        self.subject_two_way_stats(df)
        return self.subject_pairwise_stats(
            df,
            group_col=self.stats_group_col,
            group_order=self.stats_group_order,
        )

    def clean_subject_stats_df(self, df, group_col):
        needed = [group_col, self.dependant_var, "subject_id"]
        missing = [col for col in needed if col not in df.columns]
        if missing:
            raise ValueError(f"SubjectHistogram stats requires missing columns: {missing}")

        stats_df = df.dropna(subset=needed).copy()
        stats_df[self.dependant_var] = pd.to_numeric(
            stats_df[self.dependant_var],
            errors="coerce",
        )
        stats_df = stats_df.dropna(subset=[self.dependant_var])
        stats_df = stats_df.drop_duplicates(subset=["subject_id", group_col])

        if stats_df["subject_id"].nunique() < 2:
            raise ValueError("SubjectHistogram stats needs at least 2 animals.")

        if stats_df[group_col].nunique() < 2:
            raise ValueError(f"SubjectHistogram stats needs at least 2 groups in {group_col}.")

        return stats_df

    def subject_pairwise_stats(self, df, group_col, group_order, p_adjust="holm"):
        stats_df = self.clean_subject_stats_df(df, group_col)
        available_groups = set(stats_df[group_col].dropna().unique())
        group_order = [g for g in group_order if g in available_groups]

        if len(group_order) > 2:
            return self.subject_tukey_stats(stats_df, group_col, group_order)

        raw_results = []
        for group1, group2 in itertools.combinations(group_order, 2):
            values1 = stats_df.loc[
                stats_df[group_col] == group1,
                self.dependant_var,
            ].dropna()
            values2 = stats_df.loc[
                stats_df[group_col] == group2,
                self.dependant_var,
            ].dropna()

            if len(values1) < 2 or len(values2) < 2:
                p_val = np.nan
            else:
                _, p_val = ttest_ind(values1, values2, equal_var=False, nan_policy="omit")

            raw_results.append({
                "group1": group1,
                "group2": group2,
                "p_uncorrected": p_val,
                "effect": float(values1.mean() - values2.mean()),
            })

        valid_pvals = [
            res["p_uncorrected"]
            for res in raw_results
            if not pd.isna(res["p_uncorrected"])
        ]

        if valid_pvals:
            reject, pvals_adj, _, _ = multipletests(
                valid_pvals,
                alpha=self.alpha,
                method=p_adjust,
            )
        else:
            reject, pvals_adj = [], []

        adjusted_iter = iter(zip(reject, pvals_adj))
        results = []
        for res in raw_results:
            if pd.isna(res["p_uncorrected"]):
                p_adj = np.nan
                is_sig = False
            else:
                is_sig, p_adj = next(adjusted_iter)

            results.append({
                "group1": res["group1"],
                "group2": res["group2"],
                "p_val": float(p_adj) if not pd.isna(p_adj) else np.nan,
                "p_uncorrected": res["p_uncorrected"],
                "effect": res["effect"],
                "significant": bool(is_sig),
            })

        self.posthoc_results = results
        return results

    def subject_tukey_stats(self, stats_df, group_col, group_order):
        tukey_df = stats_df[stats_df[group_col].isin(group_order)].copy()
        tukey = pairwise_tukeyhsd(
            endog=tukey_df[self.dependant_var],
            groups=tukey_df[group_col],
            alpha=self.alpha,
        )

        results = []
        for row in tukey._results_table.data[1:]:
            group1, group2, meandiff, p_adj, _lower, _upper, reject = row
            results.append({
                "group1": group1,
                "group2": group2,
                "p_val": float(p_adj),
                "p_uncorrected": float(p_adj),
                "effect": float(meandiff),
                "significant": bool(reject),
            })

        self.posthoc_results = results
        self.posthoc_tukey_result = tukey
        return results

    def subject_one_way_stats(self, df):
        model_df = self.clean_subject_stats_df(df, self.stats_group_col)
        formula = f'Q("{self.dependant_var}") ~ C(Q("{self.stats_group_col}"))'
        self.subject_ols_result = ols(formula, data=model_df).fit()
        self.subject_anova = anova_lm(self.subject_ols_result, typ=2)
        self.subject_anova_pvals = {
            self.stats_group_col: self._anova_table_p_value(
                self.subject_anova,
                f'C(Q("{self.stats_group_col}"))',
            )
        }

        print("\n=== ONE-WAY SUBJECT ANOVA ===")
        print(self.subject_anova)
        print("=" * 35)

        return self.subject_ols_result

    def subject_two_way_stats(self, df):
        model_df = self.clean_subject_stats_df(df, self.stats_group_col)
        first_levels = model_df[self.first_factor].dropna().nunique()
        second_levels = model_df[self.second_factor].dropna().nunique()

        if first_levels < 2 and second_levels < 2:
            print(
                "\nWARNING: Two-way subject ANOVA was not run because only one "
                "level is present for both requested factors in the data.\n"
            )
            self.two_way_anova = None
            self.two_way_pvals = {
                self.first_factor: np.nan,
                self.second_factor: np.nan,
                "interaction": np.nan,
            }
            return None

        if first_levels < 2 or second_levels < 2:
            varying_factor = self.first_factor if first_levels >= 2 else self.second_factor
            skipped_factor = self.second_factor if varying_factor == self.first_factor else self.first_factor
            formula = f'Q("{self.dependant_var}") ~ C(Q("{varying_factor}"))'
            self.subject_ols_result = ols(formula, data=model_df).fit()
            self.two_way_anova = anova_lm(self.subject_ols_result, typ=2)
            self.two_way_pvals = {
                self.first_factor: np.nan,
                self.second_factor: np.nan,
                "interaction": np.nan,
            }
            self.two_way_pvals[varying_factor] = self._anova_p_value(
                f'C(Q("{varying_factor}"))'
            )

            print(
                "\nWARNING: Two-way subject ANOVA was not run because only one "
                f"factor is present in the data. {skipped_factor} has fewer "
                f"than 2 levels, so running one-way subject ANOVA for "
                f"{varying_factor}.\n"
            )
            print(self.two_way_anova)
            print("=" * 35)
            return self.subject_ols_result

        formula = (
            f'Q("{self.dependant_var}") ~ '
            f'C(Q("{self.first_factor}")) * C(Q("{self.second_factor}"))'
        )
        self.subject_ols_result = ols(formula, data=model_df).fit()
        try:
            self.two_way_anova = anova_lm(self.subject_ols_result, typ=2)
        except ValueError as exc:
            print(
                "\nWARNING: Type-II subject ANOVA failed. "
                f"Trying Type-III ANOVA instead. Original error: {exc}\n"
            )
            self.two_way_anova = anova_lm(self.subject_ols_result, typ=3)
        self.two_way_pvals = {
            self.first_factor: self._anova_p_value(f'C(Q("{self.first_factor}"))'),
            self.second_factor: self._anova_p_value(f'C(Q("{self.second_factor}"))'),
            "interaction": self._anova_p_value(
                f'C(Q("{self.first_factor}")):C(Q("{self.second_factor}"))'
            ),
        }

        print("\n=== TWO-WAY SUBJECT ANOVA ===")
        print(self.two_way_anova)
        print("=" * 35)

        return self.subject_ols_result

    def _anova_p_value(self, row_name):
        if row_name not in self.two_way_anova.index:
            return np.nan
        return self.two_way_anova.loc[row_name, "PR(>F)"]

    def _anova_table_p_value(self, anova_table, row_name):
        if anova_table is None or row_name not in anova_table.index:
            return np.nan
        return anova_table.loc[row_name, "PR(>F)"]

    def add_two_way_stats_box(self, ax):
        if self.second_factor is None:
            if not hasattr(self, "subject_anova_pvals"):
                return

            lines = ["Subject ANOVA"]
            for label, p_val in self.subject_anova_pvals.items():
                if pd.isna(p_val):
                    lines.append(f"{label}: p = NA")
                else:
                    lines.append(f"{label}: p = {p_val:.3g}")

            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=self.get_plot_param("stats_box_fontsize", 14),
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "alpha": 0.8,
                },
            )
            return

        if not hasattr(self, "two_way_pvals"):
            return

        lines = ["Subject ANOVA"]
        for label, p_val in self.two_way_pvals.items():
            if pd.isna(p_val):
                lines.append(f"{label}: p = NA")
            else:
                lines.append(f"{label}: p = {p_val:.3g}")

        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=self.get_plot_param("stats_box_fontsize", 14),
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.8,
            },
        )

@dataclass(init=False)
class ResponseHistogram(MixedLMStatsMixin, Histogram, Cachable):
    """
    Histogram for ResponseCharecterisation.response_cell_df metrics.
    """

    def __init__(
        self,
        *,
        project: str,
        response_data,
        dependant_var: str,
        first_factor: str = "treatment",
        second_factor: str | None = None,
        filters: dict | None = None,
        plot_params: dict | None = None,
        alpha: float = 0.05,
        specify: str | None = None,
        n_minimum: float = 3,
        significant_only: bool = True,
        filename: str | None = None,
    ):
        self.project = project
        self.response_data = response_data
        self.dependant_var = dependant_var
        self.first_factor = first_factor
        self.second_factor = second_factor
        self.filters = {} if filters is None else filters.copy()
        self.plot_params = {} if plot_params is None else plot_params.copy()
        self.alpha = alpha
        self.specify = specify
        self.n_minimum = n_minimum
        self.significant_only = significant_only
        self.filename = filename

        self.data_type = "response"
        self.threshold_access_change = None
        self.pre_sweep_window = None
        self.post_sweep_window = None
        self.is_subject_level_data = False

        self.__post_init__()

    def __post_init__(self):
        Cachable.__init__(self, cache_dir=f"{ROOT}/{self.project}/cache")
        self.location = f"{ROOT}/{self.project}"
        self.input_dir = self._checkFileSystem("input")
        self.output_dir = self._checkFileSystem("output")
        self.figure_output_dir = self._checkFileSystem("figures")
        self.project_obj = Project(self.project)

        self.agg_df = self.load_response_histogram_df()
        self.validate_response_histogram_inputs()
        self.agg_df = self.apply_response_histogram_filters(self.agg_df)
        self.check_valid_dependant_var()
        self.validate_histogram_factors()
        self.data = self.filter_n_minimum(self.agg_df)
        if self.filename is None:
            self.filename = self.build_histogram_filename()
        self.fig = self.plot_histogram()

    def selection_filters(self):
        return DataSelection.selection_filters(self)

    def filter_values(self, value):
        return DataSelection.filter_values(self, value)

    def format_selection_value(self, value, for_filename=False):
        return DataSelection.format_selection_value(self, value, for_filename=for_filename)

    def selection_label_parts(self, keys=None, for_filename=False):
        return DataSelection.selection_label_parts(self, keys=keys, for_filename=for_filename)

    def available_selection_filters(self, df=None):
        return DataSelection.available_selection_filters(self, df=df)

    def load_response_histogram_df(self):
        if isinstance(self.response_data, (list, tuple)):
            dfs = [self.response_input_to_cell_df(item) for item in self.response_data]
            dfs = [df for df in dfs if df is not None and not df.empty]
            if not dfs:
                raise ValueError("ResponseHistogram received no response_cell_df rows.")
            return pd.concat(dfs, ignore_index=True)

        return self.response_input_to_cell_df(self.response_data)

    def response_input_to_cell_df(self, response_input):
        if response_input is None:
            return pd.DataFrame()

        if isinstance(response_input, pd.DataFrame):
            return response_input.copy()

        if isinstance(response_input, dict):
            for key in ["cells", "response_cell_df"]:
                value = response_input.get(key)
                if isinstance(value, pd.DataFrame):
                    return value.copy()
            raise ValueError(
                "ResponseHistogram needs a response_cell_df table. "
                "Pass response_data=resp, response_data=resp.response_cell_df, "
                "or a cached bundle containing key 'cells'."
            )

        if hasattr(response_input, "response_cell_df"):
            value = getattr(response_input, "response_cell_df")
            if isinstance(value, pd.DataFrame):
                return value.copy()

        raise ValueError(
            "ResponseHistogram needs ResponseCharecterisation output or response_cell_df."
        )

    def validate_response_histogram_inputs(self):
        if self.agg_df is None or self.agg_df.empty:
            raise ValueError("ResponseHistogram received an empty response_cell_df table.")

        required = ["cell_id", self.first_factor, self.dependant_var]
        if self.second_factor is not None:
            required.append(self.second_factor)
        if self.specify is not None:
            required.append(self.specify)

        missing = [col for col in required if col not in self.agg_df.columns]
        if missing:
            raise ValueError(
                f"ResponseHistogram missing column(s): {missing}. "
                f"Available columns are: {self.agg_df.columns.tolist()}"
            )

        if "subject_id" not in self.agg_df.columns:
            print(
                "Warning: ResponseHistogram has no subject_id column. "
                "MixedLM stats will be skipped."
            )

    def available_histogram_factors(self):
        return self.agg_df.columns.tolist()

    def apply_response_histogram_filters(self, df):
        filtered_df = df.copy()
        for column_name, attribute in self.selection_filters().items():
            if column_name not in filtered_df.columns:
                raise ValueError(
                    f"Cannot filter ResponseHistogram on '{column_name}'. "
                    f"Available filters are: {self.available_selection_filters(filtered_df)}"
                )
            filtered_df = filtered_df[
                filtered_df[column_name].isin(self.filter_values(attribute))
            ].copy()
        return filtered_df

    def histogram_name_parts(self, subgroup_name=None, for_filename=False):
        factor_label = self.first_factor
        if self.second_factor is not None:
            factor_label = f"{self.first_factor}_by_{self.second_factor}"

        parts = [
            self.dependant_var,
            "response",
            factor_label,
            self.selection_label_parts(for_filename=for_filename),
        ]

        if self.specify is not None:
            parts.append(f"markers_{self.specify}")

        return [part for part in parts if part is not None]

    def response_metric_label(self):
        return {
            "response_latency_s": "Response latency (s)",
            "first_response_duration_s": "First response duration (s)",
            "first_response_magnitude": "First response magnitude (threshold units)",
            "second_response_latency_s": "Second response latency (s)",
            "second_response_duration_s": "Second response duration (s)",
            "second_response_magnitude": "Second response magnitude (threshold units)",
            "response_episode_count": "Response episode count",
            "total_response_duration_s": "Total response duration (s)",
            "total_response_magnitude": "Total response magnitude (threshold units)",
            "peak_response_magnitude": "Peak response magnitude (threshold units)",
            "washout_time_s": "Washout time (s)",
            "latency": "Response latency (s)",
            "latency_sweeps": "Response latency (sweeps)",
        }.get(self.dependant_var, self.dependant_var)

    def histogram_y_label(self):
        return self.response_metric_label()

    def build_histogram_title(self, subgroup_name=None):
        return self.build_name(
            self.response_metric_label(),
            self.selection_label_parts(for_filename=False),
            sep=" ",
        )

    def run_histogram_stats(self, df):
        if "subject_id" not in df.columns or df["subject_id"].nunique() < 2:
            print("ResponseHistogram MixedLM skipped: needs at least 2 subject_id groups.")
            return []

        if self.second_factor is None:
            try:
                return self.mixedlm_pairwise_stats(
                    df,
                    group_col=self.stats_group_col,
                    value_col=self.dependant_var,
                    group_order=self.stats_group_order,
                )
            except Exception as exc:
                print(f"ResponseHistogram MixedLM skipped: {exc}")
                return []

        print("\n=== TWO-WAY RESPONSE MIXED MODEL ===")
        try:
            self.two_way_mixed_model(df)
            return self.mixedlm_pairwise_stats(
                df,
                group_col=self.stats_group_col,
                value_col=self.dependant_var,
                group_order=self.stats_group_order,
            )
        except Exception as exc:
            print(f"ResponseHistogram MixedLM skipped: {exc}")
            return []

    def add_two_way_stats_box(self, ax):
        if self.second_factor is None or not hasattr(self, "two_way_pvals"):
            return

        lines = ["Response MixedLM"]
        for label, p_val in self.two_way_pvals.items():
            p_label = "NA" if pd.isna(p_val) else f"{p_val:.3g}"
            lines.append(f"{label}: p = {p_label}")

        ax.text(
            0.98,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=self.get_plot_param("stats_box_fontsize", 14),
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.8,
            },
        )
