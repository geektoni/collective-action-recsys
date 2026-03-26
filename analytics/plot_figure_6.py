import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines

sns.set_theme(font_scale=1.0,
        style="ticks",
        rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        "font.family": "serif",
    })

METRICS = [
    ("empirical_reduction", "Average empirical reduction (%)", "harm_comparison"),
    ("nDCG @ k", "Average nDCG @ k", "ndcg_comparison"),
    ("Recall @ k", "Average Recall @ k", "recall_comparison"),
    ("|S|", "Average |S|", "set_size_comparison"),
]


# Group by run_id to apply rescaling per cross-validation run
def rescale_group(group, rescale=True):

    group = group.copy()

    allowed_reductions = np.linspace(0, 100, num=25)

    # Get the baseline value for this run_id (alpha == -1)
    value_baseline = group.loc[group['alpha'] == -1, 'H(S,X)'].values[0]
    
    # Drop the row where alpha == value_baseline (originally incorrect condition)
    group = group[group.alpha != value_baseline]
    
    # Remap alpha values based on the baseline
    group.loc[group['alpha'] == -1, 'alpha'] = value_baseline
    group.loc[:, 'reduction'] = 100 * (1 - (group['alpha'] / value_baseline))
    
    # Map to closest allowed reduction
    group.loc[:, 'alpha'] = group['reduction'].apply(
        lambda x: allowed_reductions[np.argmin(np.abs(allowed_reductions - x))]
    )
    
    if rescale:
        max_harm = group['H(S,X)'].max()
        min_harm = group['H(S,X)'].min()
        group.loc[group['alpha'] == 0, 'H(S,X)'] = max_harm
        group['H(S,X)'] = ((group['H(S,X)']-min_harm)/(max_harm-min_harm))*value_baseline
    
    # Compute empirical harmfulness
    group.loc[:, 'empr_harmfulness'] = 100 * (1 - (group['H(S,X)'] / value_baseline))

    return group


def _prepare_metric_columns(df):
    numeric_cols = [
        "reduction_fraction",
        "H(S,X)",
        "nDCG @ k",
        "Recall @ k",
        "|S|",
        "base_harmfulness",
        "Collective",
        "Report Fraction",
        "run_id",
        "user_id",
        "alpha",
        "random_items"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_main_data(input_files):
    dfs = []
    for file_path in input_files:
        df = pd.read_csv(file_path).copy()
        df["source_file"] = Path(file_path).name
        dfs.append(df)

    if not dfs:
        raise ValueError("No input files were provided.")

    df = pd.concat(dfs, ignore_index=True)
    df = _prepare_metric_columns(df)

    if "Collective" not in df.columns:
        df["Collective"] = "NA"
    if "Report Fraction" not in df.columns:
        df["Report Fraction"] = "NA"
    if "run_id" not in df.columns:
        df["run_id"] = 0
    if "user_id" not in df.columns:
        df["user_id"] = 0

    return df


def build_main_methodology(df):
    # Keep all rows for ranking metrics and set size.
    # Only harmfulness reduction needs a valid harmfulness baseline.
    df = df.copy()

    print(df.columns)

    df["desired_reduction"] = 100.0 * (1.0 - df["reduction_fraction"])

    baseline_df = (
        df[df["reduction_fraction"] == 1][
            [
                "source_file",
                "run_id",
                "user_id",
                "Collective",
                "Report Fraction",
                "H(S,X)",
            ]
        ]
        .rename(columns={"H(S,X)": "baseline"})
        .copy()
    )

    duplicated = baseline_df.duplicated(
        ["source_file", "run_id", "user_id", "Collective", "Report Fraction"]
    )
    if duplicated.any():
        dup_rows = baseline_df.loc[
            duplicated,
            ["source_file", "run_id", "user_id", "Collective", "Report Fraction"],
        ]
        raise ValueError(
            "Found multiple baseline rows for some "
            "(source_file, run_id, user_id, Collective, Report Fraction) combinations.\n"
            f"Examples:\n{dup_rows.head()}"
        )

    df = df.merge(
        baseline_df,
        on=["source_file", "run_id", "user_id", "Collective", "Report Fraction"],
        how="left",
    )

    df["empirical_reduction"] = (
        100.0 * (1.0 - df["H(S,X)"] / df["baseline"])
    )

    df["Methodology"] = "Per-user reduction_fraction"

    run_avg = (
        df.groupby(
            [
                "source_file",
                "run_id",
                "Collective",
                "Report Fraction",
                "desired_reduction",
                "Methodology",
            ],
            as_index=False,
        )[["empirical_reduction", "nDCG @ k", "Recall @ k", "|S|", "random_items"]]
        .mean()
    )
    return run_avg


def load_alternative_data(alternative_files, split_alternatives_by_file=False, rescale=True):
    if not alternative_files:
        return None

    dfs = []

    for file_path in alternative_files:
        tmp = pd.read_csv(file_path, float_precision="high").copy()
        tmp["source_file"] = Path(file_path).name
        tmp = _prepare_metric_columns(tmp)

        if "run_id" not in tmp.columns:
            tmp["run_id"] = 0
        if "Collective" not in tmp.columns:
            tmp["Collective"] = "NA"
        if "Report Fraction" not in tmp.columns:
            tmp["Report Fraction"] = "NA"

        tmp = tmp.groupby('run_id', group_keys=False).apply(rescale_group, rescale=rescale, include_groups=True)
        tmp = tmp.rename(columns={"alpha": "desired_reduction"})
        tmp = tmp.rename(columns={"empr_harmfulness": "empirical_reduction"})

        if tmp.empty:
            continue

        if "Method" in tmp.columns:
            tmp = tmp[tmp["Method"] == "Conformal"].copy()

        if split_alternatives_by_file:
            tmp["Methodology"] = f"Alternative: {Path(file_path).stem}"
        else:
            tmp["Methodology"] = "Alternative alpha-rescaled"

        dfs.append(tmp)

    if not dfs:
        raise ValueError(
            "Alternative results were loaded, but no rows remained after preprocessing. "
            "Check the alpha == -1 baseline rows and available desired-reduction levels."
        )

    alt_df = pd.concat(dfs, ignore_index=True)
    run_avg = (
        alt_df.groupby(
            [
                "source_file",
                "run_id",
                "Collective",
                "Report Fraction",
                "desired_reduction",
                "Methodology",
            ],
            as_index=False,
        )[["empirical_reduction", "nDCG @ k", "Recall @ k", "|S|", "random_items"]]
        .mean()
    )
    return run_avg


def build_comparison_dataframe(input_files, alternative_files=None, split_alternatives_by_file=False):
    main_df = build_main_methodology(load_main_data(input_files))

    if alternative_files is None:
        return main_df

    alt_df = load_alternative_data(
        alternative_files,
        split_alternatives_by_file=split_alternatives_by_file,
        rescale=True,
    )
    if alt_df is None or alt_df.empty:
        return main_df

    main_df["fraction_replaced_items"] = 100*main_df["random_items"]/main_df["|S|"]
    alt_df["fraction_replaced_items"] = 100*alt_df["random_items"]/alt_df["|S|"]

    # Keep only one row per comparison key for each side, then compute main - alternative
    join_keys = [
        "run_id",
        "Collective",
        "Report Fraction",
        "desired_reduction",
    ]
    metric_cols = ["empirical_reduction", "nDCG @ k", "Recall @ k", "|S|", "fraction_replaced_items"]

    main_cmp = (
        main_df[join_keys + metric_cols]
        .groupby(join_keys, as_index=False)
        .mean()
    )
    alt_cmp = (
        alt_df[join_keys + metric_cols]
        .groupby(join_keys, as_index=False)
        .mean()
    )

    print(main_cmp["fraction_replaced_items"])

    out = main_cmp.merge(
        alt_cmp,
        on=join_keys,
        suffixes=("_input", "_alternative"),
        how="inner",
    )

    for col in metric_cols:
        out[col] = out[f"{col}_input"] - out[f"{col}_alternative"]

    out["Methodology"] = "Input - Alternative"
    return out


def plot_metric_barplot(run_avg, metric_col, ylabel, title, output_pdf, ylim=None):

    plot_df = run_avg.dropna(subset=[metric_col]).copy()
    if plot_df.empty:
        raise ValueError(f"No rows available for plotting metric '{metric_col}'.")
    
    fig, ax = plt.subplots(1,1, figsize=(3.2,2))
    g = sns.lineplot(
        data=plot_df,
        x="desired_reduction",
        y=metric_col,
        hue=r'$\gamma$',
        style=r'$\gamma$',
        markers=True,
        dashes=False,
        errorbar="ci",
        legend=True,
        linewidth=1.5,
        ax=ax,
        palette=sns.color_palette("crest", as_cmap=True)
    )

    ax.grid(axis='y')
    ax.legend(
        title=r'$\gamma$',
        fontsize="small",
        title_fontsize="small"
    )

    ax.axhline(0, linestyle="--", linewidth=1, color='k')

    ax.set_ylabel(ylabel, fontsize="small", loc="center")
    ax.set_xlabel(r"Desired reduction in unwanted content (\%)", fontsize="small", loc="center")

    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main(input_files, output_prefix, alternative_files=None, split_alternatives_by_file=False):
    run_avg = build_comparison_dataframe(
        input_files,
        alternative_files=alternative_files,
        split_alternatives_by_file=split_alternatives_by_file,
    )

    print("Rows after preprocessing:", len(run_avg))
    if len(run_avg) > 0:
        print("Rows with valid empirical reduction:", run_avg["empirical_reduction"].notna().sum())

    run_avg = run_avg[run_avg["Report Fraction"].isin([0.001, 0.01, 0.1])]
    run_avg = run_avg[run_avg["Collective"].isin([0.01])]
    run_avg.rename(
        columns={"Report Fraction": r'$\gamma$'},
        inplace=True
    )
    run_avg = run_avg[run_avg["desired_reduction"].isin([0, 12.5, 25, 50, 75, 100])]

    plot_metric_barplot(
        run_avg=run_avg,
        metric_col="empirical_reduction",
        ylabel=r"$\Delta H(S,X)$",
        title="Empirical harmfulness reduction by methodology, collective, and report fraction",
        output_pdf=f"{output_prefix}_harm_comparison.pdf",
        ylim=(0, 105),
    )

    plot_metric_barplot(
        run_avg=run_avg,
        metric_col="nDCG @ k",
        ylabel=r"$\Delta$ nDCG @ 20",
        title="nDCG @ k by methodology, collective, and report fraction",
        output_pdf=f"{output_prefix}_ndcg_comparison.pdf",
    )

    plot_metric_barplot(
        run_avg=run_avg,
        metric_col="Recall @ k",
        ylabel=r"$\Delta$ Recall @ 20",
        title="Recall @ k by methodology, collective, and report fraction",
        output_pdf=f"{output_prefix}_recall_comparison.pdf",
    )

    plot_metric_barplot(
        run_avg=run_avg,
        metric_col="fraction_replaced_items",
        ylabel=r"$\Delta$ Prev. seen items (\%)",
        title="Set size |S| by methodology, collective, and report fraction",
        output_pdf=f"{output_prefix}_set_size_comparison.pdf",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_csvs",
        nargs="+",
        help="One or more input CSV files for the current methodology",
    )
    parser.add_argument(
        "--alternative-files",
        nargs="+",
        default=None,
        help="One or more CSV files for the alternative methodology results",
    )
    parser.add_argument(
        "--split-alternatives-by-file",
        action="store_true",
        help="Keep each alternative file as its own methodology label",
    )
    parser.add_argument(
        "--output-prefix",
        default="metrics_collective_report_fraction",
        help="Prefix for output PDF files",
    )

    args = parser.parse_args()
    main(
        args.input_csvs,
        args.output_prefix,
        alternative_files=args.alternative_files,
        split_alternatives_by_file=args.split_alternatives_by_file,
    )