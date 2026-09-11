from module.figure_common import *

class MixedLMStatsMixin:
    def fit_mixedlm(self, formula, model_df, groups, label="MixedLM"):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            result = mixedlm(
                formula,
                data=model_df,
                groups=groups,
            ).fit(reml=True, method="powell")

        for warning in caught_warnings:
            if issubclass(warning.category, ConvergenceWarning):
                print(f"[WARNING] {label}: {warning.message}")
            else:
                warnings.warn(warning.message, warning.category)

        return result

    def get_mixedlm_group_col(self, df=None):
        """Use cell_id for paired application PRE/POST data, otherwise subject_id."""
        explicit_group_col = getattr(self, "mixedlm_group_col", None)
        if explicit_group_col is not None:
            return explicit_group_col

        if (
            getattr(getattr(self, "project_obj", None), "project_type", None) == "application"
            and df is not None
            and "cell_id" in df.columns
            and "time" in df.columns
            and set(df["time"].dropna().unique()).issubset({"PRE", "POST"})
        ):
            return "cell_id"

        return "subject_id"

    def clean_mixedlm_df(self, df, group_col, value_col, mixedlm_group_col=None):
        if mixedlm_group_col is None:
            mixedlm_group_col = self.get_mixedlm_group_col(df)

        needed = [group_col, value_col, mixedlm_group_col]
        missing = [col for col in needed if col not in df.columns]
        if missing:
            raise ValueError(f"MixedLM requires missing columns: {missing}")

        model_df = df.dropna(subset=needed).copy()
        model_df[value_col] = pd.to_numeric(model_df[value_col], errors="coerce")
        model_df = model_df.dropna(subset=[value_col])

        if model_df[mixedlm_group_col].nunique() < 2:
            raise ValueError(f"MixedLM needs at least 2 groups in {mixedlm_group_col}.")

        if model_df[group_col].nunique() < 2:
            raise ValueError(f"MixedLM needs at least 2 groups in {group_col}.")

        return model_df

    def fixed_effect_row(self, mixedlm_result, group_col, group_value):
        design_info = mixedlm_result.model.data.design_info
        new_df = pd.DataFrame({group_col: [group_value]})
        row = build_design_matrices([design_info], new_df)[0]
        return np.asarray(row)[0]

    def p_to_star(self, p_val):
        if p_val < 0.001:
            return "***"
        if p_val < 0.01:
            return "**"
        if p_val < 0.05:
            return "*"
        return "ns"

    def mixedlm_pairwise_stats(
        self,
        df,
        group_col=None,
        value_col=None,
        alpha=None,
        group_order=None,
        p_adjust="holm",
    ):
        if alpha is None:
            alpha = getattr(self, "alpha", 0.05)

        if group_col is None:
            group_col = self.stats_group_col

        if value_col is None:
            value_col = self.dependant_var

        mixedlm_group_col = self.get_mixedlm_group_col(df)
        model_df = self.clean_mixedlm_df(
            df,
            group_col,
            value_col,
            mixedlm_group_col=mixedlm_group_col,
        )

        if group_order is None:
            group_order = list(model_df[group_col].dropna().unique())

        available_groups = set(model_df[group_col].dropna().unique())
        group_order = [g for g in group_order if g in available_groups]

        if len(group_order) < 2:
            raise ValueError(f"Need at least 2 valid groups for pairwise MixedLM: {group_order}")

        reference = group_order[0]
        formula = f'Q("{value_col}") ~ C(Q("{group_col}"), Treatment(reference="{reference}"))'

        posthoc_model = self.fit_mixedlm(
            formula,
            model_df,
            groups=model_df[mixedlm_group_col],
            label=f"MixedLM pairwise {group_col}",
        )

        raw_results = []

        for g1, g2 in itertools.combinations(group_order, 2):
            row1 = self.fixed_effect_row(posthoc_model, group_col, g1)
            row2 = self.fixed_effect_row(posthoc_model, group_col, g2)

            contrast = np.asarray(row1 - row2, dtype=float)[None, :]
            test = posthoc_model.t_test(contrast)

            raw_results.append({
                "group1": g1,
                "group2": g2,
                "p_uncorrected": float(np.ravel(test.pvalue)[0]),
                "effect": float(np.ravel(test.effect)[0]),
            })

        reject, pvals_adj, _, _ = multipletests(
            [res["p_uncorrected"] for res in raw_results],
            alpha=alpha,
            method=p_adjust,
        )

        results = []
        for res, p_adj, is_sig in zip(raw_results, pvals_adj, reject):
            results.append({
                "group1": res["group1"],
                "group2": res["group2"],
                "p_val": float(p_adj),
                "p_uncorrected": res["p_uncorrected"],
                "effect": res["effect"],
                "significant": bool(is_sig),
            })
        
        # print("\n" + "=" * 60)
        # print(f"POSTHOC MIXEDLM - group_col = {group_col}, p_adjust = {p_adjust}")
        # print("=" * 60)
        # for res in results:
        #     sig = "SIGNIFICANT" if res["significant"] else "ns"
        #     print(
        #         f"{res['group1']:20s} vs {res['group2']:20s} | "
        #         f"effect = {res['effect']:.4g} | "
        #         f"p_unc = {res['p_uncorrected']:.4g} | "
        #         f"p_adj = {res['p_val']:.4g} | {sig}"
        #     )
        # print("=" * 60 + "\n")

        self.posthoc_results = results
        self.posthoc_mixedlm_result = posthoc_model

        return results

