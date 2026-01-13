import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

import matplotlib.lines as mlines
import os

from analytics.utils import METHOD_DICTIONARY, HUE_ORDER, HUE_ORDER_SHORT

sns.set_theme(font_scale=1.0,
        style="ticks",
        rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        "font.family": "serif",
    })

def load_csv_files_to_dataframe_adv(directory, rescale=True):
    # List to store dataframes
    dataframes = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            # Extract model name
            model_name = os.path.basename(filepath).split("_")[3]
            users_targeted = os.path.basename(filepath).split("_")[-1].replace(".csv", "")
            print(f"Loading file: {filepath}")
            # Read the CSV file and append to the list
            tmp = pd.read_csv(filepath, float_precision='high')
            tmp["score_method"] = model_name
            tmp["target_tag"] = users_targeted.capitalize()

            dataframes.append(tmp)


    # Concatenate all dataframes into one
    combined_dataframe = pd.concat(dataframes, ignore_index=True)
    return combined_dataframe

def bucketize_gamma_by_rank(
    df: pd.DataFrame,
    run_col: str = "run_id",
    gamma_col: str = "gamma",
    n_buckets: int | None = None,
    bucket_col: str = "gamma_bucket",
    gamma_bucket_center_col: str = "gamma_bucket_center",
    center_mode: str = "global_quantile"  # "global_quantile" or "bucket_midpoint"
) -> pd.DataFrame:
    """
    Adds:
      - gamma_bucket: int in [0, n_buckets-1], comparable across runs (rank-based)
      - gamma_bucket_center: a representative gamma for plotting on x-axis

    n_buckets:
      - if None, uses the minimum number of unique gammas across runs
    center_mode:
      - "global_quantile": bucket centers are global quantiles of ALL gammas
      - "bucket_midpoint": bucket centers are (b+0.5)/B in normalized space (good if you plot vs bucket)
    """
    out = df.copy()

    per = out.groupby(run_col)[gamma_col].nunique()
    if n_buckets is None:
        n_buckets = int(per.min())
    if n_buckets < 2:
        raise ValueError("n_buckets must be >= 2")

    # rank within run -> bucket id
    def _assign_bucket(g):
        # stable ordering; if duplicates exist, rank(method="first") keeps a consistent sequence
        r = g[gamma_col].rank(method="first").to_numpy()  # 1..N
        N = len(r)
        # map to 0..B-1 by relative position
        b = np.floor((r - 1) * n_buckets / N).astype(int)
        b = np.clip(b, 0, n_buckets - 1)
        return pd.Series(b, index=g.index)

    out[bucket_col] = out.groupby(run_col, group_keys=False).apply(_assign_bucket)

    # choose bucket "x-axis" values
    if center_mode == "global_quantile":
        qs = (np.arange(n_buckets) + 0.5) / n_buckets
        centers = np.quantile(out[gamma_col].to_numpy(), qs)
        center_map = dict(enumerate(centers))
        out[gamma_bucket_center_col] = out[bucket_col].map(center_map).astype(float)
    elif center_mode == "bucket_midpoint":
        out[gamma_bucket_center_col] = (out[bucket_col] + 0.5) / n_buckets
    else:
        raise ValueError("center_mode must be 'global_quantile' or 'bucket_midpoint'")

    return out

def improve_legend(ax, custom_handles=[], additional_labels_to_remove=[], save_legend=False):
    # Get the current legend and remove it
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()

    additional_labels_to_remove += ["Risk Control", "Strategy", "epoch"]

    # Keep only unique legend items (avoid redundant hue/style elements)
    new_labels = [label for label in labels if label not in additional_labels_to_remove]  # Remove duplicates while preserving order
    new_handles = [handles[labels.index(label)] for label in new_labels]

    if len(custom_handles) > 0:
        for cstmlabel, cstmhandle in custom_handles:
            new_labels += [cstmlabel]
            new_handles += [cstmhandle]
    
    # Save legend separately to disk if neede
    if save_legend:
        legend_fig = plt.figure(figsize=(len(new_labels) * 1, 0.4))  # Width based on number of labels
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis("off")
        _ = legend_ax.legend(
            new_handles,
            new_labels,
            loc="center",
            ncol=len(labels),  # Put all items in one row
            frameon=True,
        )
        legend_fig.savefig("legend_only.pdf", bbox_inches='tight', pad_inches=0, format="pdf")
    
    ax.grid(axis='y')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("--collective", type=float)
    parser.add_argument("--fraction", type=float)
    args = parser.parse_args()

    df_adv = load_csv_files_to_dataframe_adv(args.file, rescale=True)

    df_adv["fraction_flagged"] = 100*df_adv.total_flagged_items / df_adv.total_items_seen_by_adv_users
    df_adv["avg_item_per_users"] = df_adv.total_items_seen_by_adv_users / df_adv.num_adv_users

    dfb = bucketize_gamma_by_rank(df_adv, run_col="run_id", gamma_col="gamma", center_mode="bucket_midpoint")

    dfb["parsed_strategy"] = dfb["Report Strategy"].apply(lambda x: METHOD_DICTIONARY.get(x))
    dfb["parsed_strategy"] = dfb.apply(lambda x: x.parsed_strategy+f" ($g={x.target_tag}$)" if x.parsed_strategy == r'\texttt{Tag}' else x.parsed_strategy, axis=1)

    # Condition 1: picks only elements of one collective
    condition_1 = (dfb['Report Fraction'].isin([args.fraction]))

    # Condition 2: picks tags
    condition_2 = (dfb['Report Fraction'] == 0.25) & (dfb['Report Strategy'] == "tag")

    # Condition 3: picks the collective
    condition_3 = dfb.Collective.isin([args.collective])

    # Set the figure size and create subplots
    FIGURE_SIZE = (3.2,2)
    fig, ax = plt.subplots(1,1, figsize=FIGURE_SIZE)

    #g = sns.relplot(
    g = sns.lineplot(
        data=dfb[ (condition_1 | condition_2) & condition_3 ],
        x="gamma_bucket_center",
        y="avg_harm_adv_raw",
        hue="parsed_strategy",
        style="parsed_strategy",
        markers=True,
        dashes=False,
        errorbar="sd",
        hue_order = HUE_ORDER,
        ax=ax
    )
    # Show one marker every N points
    N = 10
    for line in ax.lines:
        line.set_markevery(N)
    improve_legend(ax, save_legend=True)

    g.set(yscale="log")
    ax.set_xlabel(r'$\lambda$', fontsize="small", loc="center")
    ax.set_ylabel(r'Empirical average risk', fontsize="small", loc="center")
    fig.savefig(f"01_strategies_{args.collective}_{args.fraction}.pdf", format="pdf", bbox_inches='tight')
    plt.clf()

    fig, ax = plt.subplots(1,1, figsize=(4.2, 2))
    g = sns.barplot(
        data=dfb[ (condition_1 | condition_2) & condition_3 ],
        x='parsed_strategy',
        y="fraction_flagged",
        hue="parsed_strategy",
        errorbar="sd",
        err_kws={"linewidth": 1.2},
        capsize=.4,
        order=HUE_ORDER_SHORT,
        hue_order = HUE_ORDER,
        ax=ax
    )
    improve_legend(ax, save_legend=False)
    ax.tick_params(axis='x', labelsize='x-small')
    ax.set_xlabel('')
    ax.set_ylabel(r'\% of reported items', fontsize="small", loc="center")

    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    labels[0] = r'\texttt{LowRisk}/\texttt{Likes}'+'\n'+r'\texttt{TopRanker}/\texttt{Random}'  # change first label
    ax.set_xticklabels(labels)

    fig.savefig(f"01_reported_{args.collective}_{args.fraction}.pdf", format="pdf", bbox_inches='tight')