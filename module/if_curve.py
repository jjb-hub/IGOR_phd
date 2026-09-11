from module.figure_common import *
from module.figure_base import Figure
from module.stats import MixedLMStatsMixin

@dataclass
class IF_curve(MixedLMStatsMixin, Figure):
    '''
    Plotting IF_IC data between compare which defaults to treatment. 
    Dependant vairables will be I_steps_pA AND AP_frequencies_Hz, not an input pram.
    '''
    filename: str = None
    alpha: float = 0.05
    plot_params: dict = field(default_factory=dict)
    first_factor: str = field(kw_only = True, default = 'treatment') 

    I_range_pA: str = field(kw_only = True, default = 'all_cells')
    
    specify: str = field(kw_only = True, default = 'treatment') # specify marker to see subsets e.g. I_set or cell_id if set to None single cells will not be plotted 
    n_minimum: float = field(kw_only = True, default = 3)
    show_values: bool = field(kw_only=True, default=False)
    dependant_var: str = 'AP_frequencies_Hz' 
    significant_only: bool = field(kw_only=True, default=True)

    def __post_init__(self):
        if self.data_type != 'IF_IC':
            print(f"IF curve is only possible with data_type IF_IC. ")
            return
        
        self.filename = self.build_IF_filename()

        super().__post_init__()
        if self.is_application_IF_curve():
            self.data = self.build_application_IF_curve_data()
        else:
            self.data = self.filter_n_minimum(self.agg_df)
        if self.data is None or self.data.empty:
            print("No IF curve data available after filtering.")
            return
        self.df_long =  self.preprocess_IF_data(n_min=self.n_minimum, I_range_pA=self.I_range_pA)
        if self.df_long is None or self.df_long.empty:
            print("No IF curve data available after IF preprocessing.")
            return
        self.fig = self.plot_IF_curve()
        self.cell_fig = self.plot_cell_id_curves()

    def is_application_IF_curve(self):
        return (
            getattr(self.project_obj, "project_type", None) == "application"
            and self.data_type == "IF_IC"
        )

    def build_application_IF_curve_data(self):
        """
        Preserve IF curve arrays for application projects.

        build_agg_df() intentionally averages scalar IF_IC variables per cell/time
        for CellHistogram. IF_curve needs the original I_steps_pA and
        AP_frequencies_Hz arrays so PRE and POST curves can be compared.
        """
        filtered_df = self.IF_IC_df[self.IF_IC_df['folder_file'].isin(self.valid_files)].copy()
        if filtered_df.empty:
            return filtered_df

        filtered_df['time'] = filtered_df.apply(self.application_time_label, axis=1)
        keep_cols = [
            'cell_id',
            'folder_file',
            'time',
            'I_steps_pA',
            'AP_frequencies_Hz',
        ]
        keep_cols = [col for col in keep_cols if col in filtered_df.columns]
        filtered_df = filtered_df[keep_cols]
        return self.add_cell_mapping(filtered_df, additional_cols=['I_set', 'treatment'])

    def IF_curve_group_cols(self):
        group_cols = [self.first_factor]
        if self.is_application_IF_curve():
            group_cols.append('time')
        return group_cols

    def IF_curve_ordered_values(self, df, col):
        values = [
            value for value in color_dict.keys()
            if col in df.columns and value in df[col].dropna().unique()
        ]
        if col in df.columns:
            values.extend([
                value for value in df[col].dropna().unique()
                if value not in values
            ])
        return values

    def IF_curve_time_order(self, df):
        return [
            value for value in ['PRE', 'POST', 'APP', 'WASH']
            if 'time' in df.columns and value in df['time'].dropna().unique()
        ]
    
    def preprocess_IF_data(self, n_min=None, I_range_pA=None):
        """
        Build a long-format DataFrame for IF plotting, with optional binning,
        filtering for minimum number of cells per group, and optional I-step range.
        
        Parameters
        ----------
        n_min : int
            Minimum number of cells per compare group to include a bin.
        bin_size : int
            Size of I-step binning in pA.
        I_range_pA : None, tuple, or 'all_cells'
            - None: include all bins
            - (min_I, max_I): include only bins within this current range
            - 'all_cells': include only bins where all cells are represented
        """

        df = self.data.copy()
        keep_cols = ['cell_id', 'subject_id', self.first_factor, 'I_steps_pA', 'AP_frequencies_Hz']        
        if self.is_application_IF_curve() and 'time' in df.columns:
            keep_cols.append('time')
        if self.specify is not None and self.specify != self.first_factor:
            keep_cols.append(self.specify)
        required_cols = ['cell_id', self.first_factor, 'I_steps_pA', 'AP_frequencies_Hz']
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            raise ValueError(f"IF_curve requires missing columns: {missing_required}")
        keep_cols = [col for col in dict.fromkeys(keep_cols) if col in df.columns]
        df = df[keep_cols]

        valid_curve_rows = df.apply(
            lambda row: (
                isinstance(row['I_steps_pA'], (list, tuple, np.ndarray))
                and isinstance(row['AP_frequencies_Hz'], (list, tuple, np.ndarray))
                and len(row['I_steps_pA']) == len(row['AP_frequencies_Hz'])
                and len(row['I_steps_pA']) > 0
            ),
            axis=1
        )
        dropped_rows = int((~valid_curve_rows).sum())
        if dropped_rows > 0:
            print(f"IF_curve dropped {dropped_rows} rows with invalid or mismatched IF arrays.")
        df = df[valid_curve_rows].copy()
        if df.empty:
            return df

        df = df.explode(['I_steps_pA', 'AP_frequencies_Hz'])

        # Convert to numeric
        df['I_steps_pA'] = pd.to_numeric(df['I_steps_pA'], errors='coerce')
        df['AP_frequencies_Hz'] = pd.to_numeric(df['AP_frequencies_Hz'], errors='coerce')
        df = df.dropna(subset=['I_steps_pA', 'AP_frequencies_Hz']).copy()

        # Create binned current steps
        df['I_step_bin'] =  df['I_steps_pA']

        if self.is_application_IF_curve():
            group_cols = [
                'cell_id',
                'subject_id',
                self.first_factor,
                'time',
                'I_step_bin',
            ]
            if self.specify is not None and self.specify not in group_cols and self.specify in df.columns:
                group_cols.append(self.specify)
            group_cols = [col for col in dict.fromkeys(group_cols) if col in df.columns]
            df = (
                df.groupby(group_cols, dropna=False)[self.dependant_var]
                .mean()
                .reset_index()
            )
            df['I_steps_pA'] = df['I_step_bin']

        # Filter bins with fewer than n_min cells per compare group
        count_group_cols = ['I_step_bin'] + self.IF_curve_group_cols()
        count_group_cols = [col for col in count_group_cols if col in df.columns]
        counts = df.groupby(count_group_cols)['cell_id'].nunique().reset_index(name='n_cells')
        valid_bins = counts.groupby('I_step_bin').filter(lambda g: (g['n_cells'] >= n_min).all())['I_step_bin'].unique()
        df = df[df['I_step_bin'].isin(valid_bins)]

        # filter by I_range_pA
        if I_range_pA is not None:
            if I_range_pA == 'all_cells':   #  only bins where all cells are represented
                cells_per_bin = df.groupby('I_step_bin')['cell_id'].nunique()
                max_cells = df['cell_id'].nunique()
                valid_bins = cells_per_bin[cells_per_bin == max_cells].index
                df = df[df['I_step_bin'].isin(valid_bins)]
            elif isinstance(I_range_pA, (tuple, list)) and len(I_range_pA) == 2: # set I step range
                df = df[(df['I_step_bin'] >= I_range_pA[0]) & (df['I_step_bin'] <= I_range_pA[1])]
            else:
                raise ValueError("I_range_pA must be None, a tuple/list (min,max), or 'all_cells'")
        return df
    
    def plot_IF_curve(self):
        df = self.df_long.copy()
        agg = self.aggregate_IF_data(df)

        if self.is_application_IF_curve():
            bin_stats = self.run_application_IF_bin_stats(df)
            fig = self.plot_application_IF_curve(df, agg, bin_stats)
            return fig

        fig, ax = self.draw_IF_curve(df)
        legend_handles_labels = self.draw_IF_points(ax, df)

        bin_stats = self.run_IF_bin_stats(df)
        self.annotate_IF_bin_stats(ax, df, bin_stats)

        self.finalize_IF_curve(ax, fig, agg, legend_handles_labels)

        return fig
    
    def aggregate_IF_data(self, df):
        return (
            df.groupby(["I_step_bin"] + self.IF_curve_group_cols())[self.dependant_var]
            .agg(mean="mean", sd="std", n="count")
            .reset_index()
        )

    def plot_application_IF_curve(self, df, agg, bin_stats=None):
        """
        Plot application IF_IC data as PRE vs POST curves, faceted by first_factor.
        """
        if bin_stats is None:
            bin_stats = {}

        facet_values = self.IF_curve_ordered_values(df, self.first_factor)
        if not facet_values:
            facet_values = [None]

        n_facets = len(facet_values)
        n_cols = min(self.get_plot_param("facet_cols", 2), n_facets)
        n_rows = int(np.ceil(n_facets / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                self.get_plot_param("figwidth", 6 * n_cols),
                self.get_plot_param("figheight", 4.5 * n_rows),
            ),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axes_flat = axes.ravel()
        time_order = self.IF_curve_time_order(df)
        time_palette = {
            value: color_dict.get(value, color)
            for value, color in zip(time_order, sns.color_palette("tab10", len(time_order)))
        }

        for ax, facet_value in zip(axes_flat, facet_values):
            sub_df = df if facet_value is None else df[df[self.first_factor] == facet_value].copy()
            if sub_df.empty:
                ax.set_axis_off()
                continue

            sns.lineplot(
                data=sub_df,
                x="I_step_bin",
                y=self.dependant_var,
                hue="time",
                hue_order=time_order,
                style="time",
                style_order=time_order,
                errorbar=self.get_plot_param("errorbar", "se"),
                ax=ax,
                palette=time_palette,
                linewidth=self.get_plot_param("line_width", 2.5),
                marker=self.get_plot_param("line_marker", "o"),
                markersize=self.get_plot_param("line_markersize", 6),
            )

            n_mapping = sub_df.groupby("time")["cell_id"].nunique().to_dict()
            handles, labels = ax.get_legend_handles_labels()
            labels = [
                f"{label} (n={n_mapping[label]})"
                if label in n_mapping
                else label
                for label in labels
            ]
            ax.legend(
                handles,
                labels,
                title="time",
                loc=self.get_plot_param("legend_loc", "best"),
                fontsize=self.get_plot_param("legend_fontsize", None),
            )
            title = str(facet_value) if facet_value is not None else "IF curve"
            ax.set_title(title, fontsize=self.get_plot_param("subplot_title_fontsize", 14))
            ax.set_xlabel("Current injection (pA)")
            ax.set_ylabel("Firing frequency (Hz)")
            self.annotate_application_IF_bin_stats(
                ax,
                sub_df,
                bin_stats.get(facet_value, {}),
                facet_value,
            )
            sns.despine(ax=ax)

        for ax in axes_flat[n_facets:]:
            ax.set_axis_off()

        fig.suptitle(
            f"{self.build_IF_title()} PRE vs POST",
            fontsize=self.get_plot_param("title_fontsize", 18),
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        self.save_plot(fig, self.filename)
        return fig

    def paired_application_IF_bin_df(self, df_bin):
        """
        Keep only cells with both PRE and POST values in one current bin.
        """
        if df_bin.empty or "time" not in df_bin.columns:
            return pd.DataFrame()

        paired_cells = (
            df_bin
            .dropna(subset=["cell_id", "time"])
            .groupby("cell_id")["time"]
            .apply(lambda values: {"PRE", "POST"}.issubset(set(values)))
        )
        paired_cells = paired_cells[paired_cells].index
        if len(paired_cells) == 0:
            return pd.DataFrame()

        return df_bin[
            df_bin["cell_id"].isin(paired_cells)
            & df_bin["time"].isin(["PRE", "POST"])
        ].copy()

    def application_IF_bin_is_testable(self, paired_df):
        """
        Decide whether a PRE/POST IF bin has enough paired cells for MixedLM.
        """
        if paired_df.empty:
            return False
        if paired_df["cell_id"].nunique() < 2:
            return False
        if paired_df["time"].nunique() < 2:
            return False

        values = pd.to_numeric(paired_df[self.dependant_var], errors="coerce").dropna()
        if values.nunique() < 2:
            return False
        if np.isclose(values.var(ddof=0), 0):
            return False
        return True

    def run_application_IF_bin_stats(self, df):
        """
        Run paired PRE vs POST MixedLM separately for each treatment/current step.
        """
        all_results = {}
        skipped_bins = {}
        flat_results = []

        for facet_value, facet_df in df.groupby(self.first_factor, dropna=False):
            facet_results = {}
            facet_skipped = []

            for i_step, df_bin in facet_df.groupby("I_step_bin"):
                paired_df = self.paired_application_IF_bin_df(df_bin)
                if not self.application_IF_bin_is_testable(paired_df):
                    facet_skipped.append(i_step)
                    continue

                try:
                    results = self.mixedlm_pairwise_stats(
                        paired_df,
                        group_col="time",
                        value_col=self.dependant_var,
                        group_order=["PRE", "POST"],
                        alpha=self.alpha,
                    )
                except Exception as exc:
                    facet_skipped.append(i_step)
                    print(f"[IF_curve PRE/POST stats skipped] {facet_value} I={i_step}: {exc}")
                    continue

                results = [
                    res for res in results
                    if np.isfinite(res["p_val"])
                ]
                if not results:
                    facet_skipped.append(i_step)
                    continue

                annotated_results = []
                for res in results:
                    res = res.copy()
                    res[self.first_factor] = facet_value
                    res["I_step_bin"] = i_step
                    annotated_results.append(res)
                    flat_results.append(res)

                facet_results[i_step] = annotated_results

            if facet_results:
                all_results[facet_value] = facet_results
            if facet_skipped:
                skipped_bins[facet_value] = facet_skipped

        if skipped_bins:
            summary = {
                str(facet): len(bins)
                for facet, bins in skipped_bins.items()
            }
            print(f"[IF_curve PRE/POST stats skipped bins] {summary}")

        self.application_IF_bin_stats = all_results
        self.application_IF_skipped_bins = skipped_bins
        self.posthoc_results = flat_results
        return all_results

    def annotate_application_IF_bin_stats(self, ax, df, bin_stats, facet_value):
        """
        Annotate PRE vs POST stats above each current step in one treatment axis.
        """
        if not bin_stats:
            return

        y0, y1 = ax.get_ylim()
        axis_range = y1 - y0
        offset = axis_range * self.get_plot_param("stats_offset_frac", 0.015)
        used_labels = []

        for i_step, results in sorted(bin_stats.items(), key=lambda item: item[0]):
            visible = [
                res for res in results
                if res["significant"] or not self.significant_only
            ]
            if not visible:
                continue

            df_bin = df[df["I_step_bin"] == i_step]
            y_base = (
                df_bin
                .groupby("time")[self.dependant_var]
                .mean()
                .max()
            )
            y = y_base + offset
            best_p = min(res["p_val"] for res in visible)
            label = self.p_to_star(best_p)
            if not any(res["significant"] for res in visible):
                label = f"p={best_p:.2f}"

            ax.text(
                i_step,
                y,
                label,
                ha="center",
                va="bottom",
                fontsize=self.get_plot_param("stats_fontsize", 10),
                color="black",
            )
            used_labels.append(y)

            for res in visible:
                print(
                    f"{facet_value} I={i_step}: PRE vs POST "
                    f"p_unc={res['p_uncorrected']:.4g}, "
                    f"p_adj={res['p_val']:.4g}"
                )

        if used_labels:
            current_top = ax.get_ylim()[1]
            needed_top = max(used_labels) + axis_range * 0.04
            if needed_top > current_top:
                ax.set_ylim(top=needed_top)

    def draw_IF_curve(self, df):
        """
        Draw the main mean IF curve.

        This creates the figure/axis and plots the group-level IF curve with SE error.
        Optional individual points are handled separately by draw_IF_points().
        """
        fig, ax = plt.subplots(
            figsize=(
                self.get_plot_param("figwidth", 12),
                self.get_plot_param("figheight", 8),
            )
        )

        sns.lineplot(
            data=df,
            x="I_step_bin",
            y=self.dependant_var,
            hue=self.first_factor,
            errorbar=self.get_plot_param("errorbar", "se"),
            ax=ax,
            palette=color_dict,
            linewidth=self.get_plot_param("line_width", 2.5),
            marker=self.get_plot_param("line_marker", "o"),
            markersize=self.get_plot_param("line_markersize", 6),
        )

        return fig, ax
    
    def draw_IF_points(self, ax, df):
        """
        Optionally draw individual cell/value points on top of the mean IF curve.

        Points are grouped by self.specify using different marker shapes, while
        edge color follows self.first_factor.
        """
        legend_handles_labels = {}

        if not self.show_values:
            return legend_handles_labels

        if self.specify is None:
            return legend_handles_labels

        if self.specify not in df.columns:
            return legend_handles_labels

        marker_df = df.copy()
        marker_df[self.specify] = marker_df[self.specify].fillna("none")

        markers = cycle(["o", "s", "^", "D", "v", "<", ">"])

        for spec_value in marker_df[self.specify].unique():
            marker = next(markers)
            sub_df = marker_df[marker_df[self.specify] == spec_value].copy()

            for comp in sub_df[self.first_factor].unique():
                comp_df = sub_df[sub_df[self.first_factor] == comp]

                ax.scatter(
                    comp_df["I_step_bin"],
                    comp_df[self.dependant_var],
                    marker=marker,
                    s=self.get_plot_param("raw_point_size", 10),
                    facecolors="none",
                    edgecolors=color_dict.get(comp, "k"),
                    linewidths=self.get_plot_param("raw_point_linewidth", 1.2),
                    alpha=self.get_plot_param("raw_point_alpha", 0.9),
                    zorder=10,
                )

            legend_handles_labels[spec_value] = plt.Line2D(
                [0],
                [0],
                marker=marker,
                label=str(spec_value),
                color="black",
                linestyle="",
                markersize=self.get_plot_param("raw_marker_legend_size", 6),
            )

        return legend_handles_labels

    def finalize_IF_curve(self, ax, fig, agg, legend_handles_labels):
        """
        Apply legend, labels, title, layout, and save the main IF curve figure.
        """
        n_mapping = agg.groupby(self.first_factor)["n"].max().to_dict()

        handles, labels = ax.get_legend_handles_labels()

        new_labels = [
            f"{label} (n={n_mapping[label]})"
            if label in n_mapping
            else label
            for label in labels
        ]

        scatter_handles = list(legend_handles_labels.values())
        scatter_labels = [handle.get_label() for handle in scatter_handles]

        combined_handles = handles + scatter_handles
        combined_labels = new_labels + scatter_labels

        ax.legend(
            combined_handles,
            combined_labels,
            loc=self.get_plot_param("legend_loc", "best"),
            title="Legend",
            fontsize=self.get_plot_param("legend_fontsize", None),
        )

        ax.set_xlabel(
            "Current injection (pA)",
            fontsize=self.get_plot_param("xlabel_fontsize", 16),
        )
        ax.set_ylabel(
            "Firing frequency (Hz)",
            fontsize=self.get_plot_param("ylabel_fontsize", 16),
        )

        ax.set_title(
            self.build_IF_title(),
            fontsize=self.get_plot_param("title_fontsize", 18),
        )
        sns.despine(ax=ax)
        plt.tight_layout()
        plt.show()

        self.save_plot(fig, self.filename)

    def plot_cell_id_curves(self):
        """
        Plot each cell_id's I–F curve with a unique color, separated by self.compare (e.g., treatment group).
        Each compare group is shown in its own subplot for easier visual inspection.
        """
        df = self.df_long.copy()

        if self.first_factor not in df.columns:
            raise ValueError(f"'{self.first_factor}' not found in DataFrame columns: {df.columns.tolist()}")

        compare_groups = self.IF_curve_ordered_values(df, self.first_factor)
        n_groups = len(compare_groups)

        fig, axes = plt.subplots(
            n_groups, 1,
            figsize=(12, 6 * n_groups),
            sharex=True,
            sharey=True
        )

        if n_groups == 1:
            axes = [axes]  # ensure iterable if single axis

        for ax, comp in zip(axes, compare_groups):
            sub_df = df[df[self.first_factor] == comp].copy()

            # Reset palette per group so colors don't repeat across groups
            cell_ids = sub_df["cell_id"].unique()
            palette = sns.color_palette("husl", len(cell_ids))
            cell_color = dict(zip(cell_ids, palette))
            time_styles = {"PRE": "--", "POST": "-", "APP": "-", "WASH": ":"}

            for cell_id in cell_ids:
                cell_df = sub_df[sub_df["cell_id"] == cell_id]
                if self.is_application_IF_curve() and "time" in cell_df.columns:
                    for time_label in self.IF_curve_time_order(cell_df):
                        time_df = cell_df[cell_df["time"] == time_label]
                        if time_df.empty:
                            continue
                        ax.plot(
                            time_df["I_step_bin"],
                            time_df[self.dependant_var],
                            color=cell_color[cell_id],
                            linestyle=time_styles.get(time_label, "-"),
                            linewidth=1.5,
                            alpha=0.9,
                            label=f"{cell_id} {time_label}"
                        )
                else:
                    ax.plot(
                        cell_df["I_step_bin"],
                        cell_df[self.dependant_var],
                        color=cell_color[cell_id],
                        linewidth=1.5,
                        alpha=0.9,
                        label=str(cell_id)
                    )

            # Legend (always includes n= count)
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=color, lw=2, label=str(cid))
                for color, cid in zip(palette, cell_ids)
            ]
            if self.is_application_IF_curve() and "time" in sub_df.columns:
                legend_elements.extend([
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        lw=2,
                        linestyle=time_styles.get(time_label, "-"),
                        label=time_label,
                    )
                    for time_label in self.IF_curve_time_order(sub_df)
                ])
            leg = ax.legend(
                handles=legend_elements,
                title=f"{comp} (n={len(cell_ids)})",
                fontsize=8,
                ncol=2 if len(cell_ids) > 20 else 1,
                frameon=True,
                loc='best'
            )
            leg.get_frame().set_alpha(0.8)

            # Labels (no subplot titles)
            ax.set_xlabel("Current injection (pA)", fontsize=12)
            ax.set_ylabel("Firing frequency (Hz)", fontsize=12)
            sns.despine(ax=ax)

        title = "Per-cell I–F curves by group"
        if self.is_application_IF_curve():
            title = "Per-cell I–F curves by group and time"
        fig.suptitle(title, fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
        self.save_plot(fig, self.build_IF_filename(suffix="cell_traces"))
        return fig

    def IF_bin_is_testable(self, df_bin):
        """
        Decide whether one I_step_bin has enough information for MixedLM.

        Skip bins where:
        - fewer than 2 compare groups exist
        - fewer than 2 animals exist
        - the dependent variable has no variation, e.g. all AP frequencies are 0

        If some values are 0 and some are not, the bin is still testable.
        """
        values = pd.to_numeric(df_bin[self.dependant_var], errors="coerce").dropna()

        if df_bin[self.first_factor].nunique() < 2:
            return False

        if df_bin["subject_id"].nunique() < 2:
            return False

        if values.nunique() < 2:
            return False

        if np.isclose(values.var(ddof=0), 0):
            return False

        return True
    
    def run_IF_bin_stats(self, df):
        """
        Run mixed-model pairwise stats separately at each current step.

        Skips untestable bins, especially early IF steps where every cell is 0 Hz.
        """
        all_results = {}
        skipped_bins = []

        for i_step, df_bin in df.groupby("I_step_bin"):
            if not self.IF_bin_is_testable(df_bin):
                skipped_bins.append(i_step)
                continue

            try:
                results = self.mixedlm_pairwise_stats(
                    df_bin,
                    group_col=self.first_factor,
                    value_col=self.dependant_var,
                    alpha=self.alpha,
                )

                results = [
                    res for res in results
                    if np.isfinite(res["p_val"])
                ]

                if results:
                    all_results[i_step] = results
                else:
                    skipped_bins.append(i_step)

            except Exception as e:
                skipped_bins.append(i_step)
                print(f"[IF_curve stats skipped] I_step_bin={i_step}: {e}")

        if skipped_bins:
            print(f"[IF_curve stats skipped bins] {skipped_bins}")

        self.IF_bin_stats = all_results
        self.IF_skipped_bins = skipped_bins

        return all_results

    def annotate_IF_bin_stats(self, ax, df, bin_stats):
        """
        Add significance annotation above each current step.

        Places labels close to the data using the current axis range, not the full
        data range. This keeps stars from floating too high above the curve.
        """
        if not bin_stats:
            return

        y0, y1 = ax.get_ylim()
        axis_range = y1 - y0
        offset = axis_range * self.get_plot_param("stats_offset_frac", 0.015)

        used_labels = []

        for i_step, results in bin_stats.items():
            visible = [
                res for res in results
                if res["significant"] or not self.significant_only
            ]

            if not visible:
                continue

            df_bin = df[df["I_step_bin"] == i_step]
            # y = df_bin[self.dependant_var].max() + offset # based off the highest value
            y_base = (
                df_bin
                .groupby(self.first_factor)[self.dependant_var]
                .mean()
                .max()
            )
            y = y_base + offset # based off the mean

            best_p = min(res["p_val"] for res in visible)
            label = self.p_to_star(best_p)

            if not any(res["significant"] for res in visible):
                label = f"p={best_p:.2f}"

            ax.text(
                i_step,
                y,
                label,
                ha="center",
                va="bottom",
                fontsize=self.get_plot_param("stats_fontsize", 10),
                color="black",
            )

            used_labels.append(y)

            for res in visible:
                print(
                    f"I={i_step}: {res['group1']} vs {res['group2']} "
                    f"p={res['p_val']:.4g}"
                )

        if used_labels:
            current_top = ax.get_ylim()[1]
            needed_top = max(used_labels) + axis_range * 0.04
            if needed_top > current_top:
                ax.set_ylim(top=needed_top)

    def IF_optional_params(self, for_filename=False):
        """
        Optional IF-curve parameters to show in filenames/titles.
        """
        params = {
            "I_range_pA": self.I_range_pA,
            "n_minimum": self.n_minimum,
        }

        labels = self.optional_param_labels(params, for_filename=for_filename)

        if self.show_values:
            labels.append("show_values" if for_filename else "show values")

        if self.specify is not None:
            labels.append(
                f"markers_{self.specify}" if for_filename else f"markers = {self.specify}"
            )

        return labels

    def build_IF_filename(self, suffix=None):
        """
        Build saved filename for IF curve figures.
        """
        parts = [
            "IF_curve",
            self.data_type,
            self.first_factor,
            self.selection_label_parts(keys=["region", "cell_type"], for_filename=True),
        ]

        parts.extend(self.IF_optional_params(for_filename=True))

        if suffix is not None:
            parts.append(suffix)

        return self.sanitize_filename(
            self.build_name(*parts, sep="_")
        )

    def build_IF_title(self):
        """
        Build visible IF curve title.
        """
        parts = [
            "IF curve",
            self.selection_label_parts(keys=["region", "cell_type"], for_filename=False),
        ]

        parts.extend(self.IF_optional_params(for_filename=False))

        return self.build_name(*parts, sep=" ")
