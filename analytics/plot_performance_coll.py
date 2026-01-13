import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

import matplotlib.lines as mlines
import os

from analytics.utils import METHOD_DICTIONARY, HUE_ORDER, HUE_ORDER_ONLY_SAMPLING
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(font_scale=1.0,
        style="ticks",
        rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        "font.family": "serif",
    },
    palette=sns.color_palette('colorblind')
)

# Group by run_id to apply rescaling per cross-validation run
def rescale_group(group, rescale=True, metrics=("nDCG @ k", "Recall @ k", "empr_harmfulness")):

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

def load_csv_files_to_dataframe(directory, rescale=True):
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
            tmp["users"] = users_targeted.capitalize()
            tmp["target_tag"] = users_targeted.capitalize()

            tmp = tmp.groupby('run_id', group_keys=False).apply(rescale_group, rescale=rescale, include_groups=True)

            dataframes.append(tmp)


    # Concatenate all dataframes into one
    combined_dataframe = pd.concat(dataframes, ignore_index=True)
    return combined_dataframe

def filter_name(row):
    if row.conformal_score == "Naive" or row.conformal_score == "Global Harm":
        return row.score_method
    else:
        return r"\textsc{"+f"{row.score_method.capitalize()}"+r"}"

def improve_legend(ax, custom_handles=[], additional_labels_to_remove=[], save_legend=False, legend_title="classic_legend"):
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
            #fontsize="large"
        )
        legend_fig.savefig(f"{legend_title}.pdf", bbox_inches='tight', pad_inches=0, format="pdf")
    
    ax.grid(axis='y')

X_AXIS_TEXT = "Desired reduction in unwanted content (\%)"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("--collective", type=float)
    parser.add_argument("--fraction", type=float)
    args = parser.parse_args()

    df = load_csv_files_to_dataframe(args.file, rescale=True)

    df["conformal_score"] = df["conformal_score"].fillna("None")
    df["conformal_score"] = df.conformal_score.apply(lambda x : "Similarity" if x=="weights" else x)
    df["conformal_method"] = df["conformal_method"].fillna("None")

    df["Method"] = df["Method"].apply(lambda x: "None" if x == "Classic" else x)
    df["Method"] = df["Method"].apply(lambda x: "Pre-train" if x == "Classic (masked)" else x)
    df["Strategy"] = df.apply(lambda x: f"{x.conformal_method.capitalize()}" if x.Method == "Conformal" else x.Method, axis=1)
    df["Strategy"] = df["Strategy"].apply(lambda x: r"\textsc{Remove}" if x=="Remove" else r"\textsc{Replace} (Ours)")
    df["Risk Control"] = df.apply(lambda x: filter_name(x) if x.Method == "Conformal" else "None", axis=1)

    df = df[df["Risk Control"] != r'\textsc{Ncf}']

    max_alpha = df["alpha"].max()

    df.rename(
        columns={"alpha": X_AXIS_TEXT,
                "|S|": r"$|S_\lambda(U)|$",
                "random_items": r"\# of replaced items"},
        inplace=True
    )

    df["Report Strategy"].fillna("None", inplace=True)
    df["parsed_strategy"] = df["Report Strategy"].apply(lambda x: METHOD_DICTIONARY.get(x))
    df["parsed_strategy"] = df.apply(lambda x: x.parsed_strategy+f"($g={x.target_tag}$)" if x.parsed_strategy == r'\texttt{Tag}' else x.parsed_strategy, axis=1)

    # Condition 1: picks only elements of one collective
    condition_1 = (df['Report Fraction'].isin([args.fraction]))

    # Condition 2: picks tags
    condition_2_1 = ((df['Report Fraction'] == 0.25) & (df['Report Strategy'] == "tag"))
    condition_2_2 = ((df['Report Fraction'] == 0.25) & (df['Report Strategy'] == "None") & df.Collective.isin([0]))
    condition_2 = (condition_2_1 | condition_2_2)

    # Condition 3: picks alpha
    condition_3 = (df[X_AXIS_TEXT] == 25)

    # Condition 4: picks a collective
    condition_4 = df.Collective.isin([0.01])
    
    metrics = ["nDCG @ k", "Recall @ k"]
    keys = ["run_id", X_AXIS_TEXT]  # add other columns if needed (e.g., your x-axis reduction column)

    # baseline condition ("None" strategy rows)
    baseline_cond = (
        (df["Report Fraction"] == 0.25)
        & (df["Report Strategy"] == "None")
        & (df["Collective"].isin([0]))
    )

    baseline = (
        df.loc[baseline_cond, keys + metrics]
        .groupby(keys, as_index=False)
        .mean(numeric_only=True)
        .rename(columns={m: f"{m}__none" for m in metrics})
    )

    df = df.merge(baseline, on=keys, how="left")

    # 3) Compute per-metric differences for strategies != "None"
    mask = df["Report Strategy"].ne("None")
    for m in metrics:
        base_col = f"{m}__none"
        diff_col = f"difference__{m}"
        df[diff_col] = np.where(
            mask & df[base_col].notna(),
            (1/df["Collective"])*(df[base_col]-df[m]), # we invert wrt the original formulation just for plotting ease
            np.nan
        )
    
    print(df.Collective.unique())

    # Set the figure size and create subplots
    FIGURE_SIZE = (3.2,2)
    fig, ax = plt.subplots(1,1, figsize=FIGURE_SIZE)

    g = sns.lineplot(
        data=df[ ((condition_1 | condition_2) & condition_4) | condition_2_2 ],
        x=X_AXIS_TEXT,
        y="empr_harmfulness",
        hue="parsed_strategy",
        style="parsed_strategy",
        markers=True,
        dashes=False,
        errorbar="sd",
        hue_order = HUE_ORDER + ['None'],
        ax=ax,
        legend=False
    )
    ax.grid(axis="y")
    ax.plot([0, 100], [0, 100], linestyle=':', color='black', label="Optim.")
    ax.set(ylim=(105, -5.0))
    ax.set_ylabel(r"Empirical risk reduction (\%)", fontsize="small")
    ax.set_xlabel(X_AXIS_TEXT, fontsize="small", loc="center")
    fig.savefig(f"00_performance_{args.collective}_{args.fraction}.pdf", format="pdf", bbox_inches='tight')
    plt.clf()

    fig, ax = plt.subplots(1,1, figsize=FIGURE_SIZE)
    g = sns.lineplot(
        data=df[ (condition_1 | condition_2) & condition_3],
        x="Collective",
        y="difference__nDCG @ k",
        hue="parsed_strategy",
        style="parsed_strategy",
        markers=True,
        dashes=False,
        errorbar="ci",
        ax = ax,
        hue_order = HUE_ORDER,
    )
    improve_legend(ax, save_legend=True, legend_title="02_legend", additional_labels_to_remove=['None'])
    ax.invert_yaxis()
    ax.set_ylabel(r"Reduction", fontsize="small")
    ax.set_xlabel(r"$\beta$", fontsize="small", loc="center")
    fig.savefig(f"00_ndcg_{args.collective}_{args.fraction}.pdf", format="pdf", bbox_inches='tight')
    plt.clf()

    fig, ax = plt.subplots(1,1, figsize=FIGURE_SIZE)
    g = sns.lineplot(
        data=df[ (condition_1 | condition_2) & condition_3 ],
        x="Collective",
        y="difference__Recall @ k",
        hue="parsed_strategy",
        style="parsed_strategy",
        markers=True,
        legend=True,
        dashes=False,
        errorbar="ci",
        ax=ax, hue_order=HUE_ORDER
    )
    improve_legend(ax)
    ax.invert_yaxis()
    ax.set_ylabel(r"Reduction", fontsize="small")
    ax.set_xlabel(r"$\beta$", fontsize="small", loc="center")
    plt.savefig(f"00_recall_{args.collective}_{args.fraction}.pdf", format="pdf", bbox_inches='tight')
    plt.clf()

    fig, ax = plt.subplots(1,1, figsize=FIGURE_SIZE)#(4.2,2))
    tmp = df[ (condition_1 | condition_2) & ( condition_4 | condition_2_2) ]
    tmp = tmp[tmp[X_AXIS_TEXT].isin([12.5, 25, 50, 75])]
    tmp["fraction"] = 100*tmp[r"\# of replaced items"]/tmp[r"$|S_\lambda(U)|$"]
    g = sns.barplot(
        data=tmp,
        x=X_AXIS_TEXT,
        y="fraction",
        hue="parsed_strategy",
        legend=True,
        errorbar="sd",
        err_kws={"linewidth": 1.2},
        capsize=.4,
        ax=ax, hue_order=HUE_ORDER+['None']
    )
    improve_legend(ax, save_legend=False)
    ax.set_xlabel(X_AXIS_TEXT, fontsize="small", loc="center")
    ax.set_ylabel(r"\% of previously seen items", fontsize="small")
    fig.savefig(f"00_replacement_{args.collective}_{args.fraction}.pdf", format="pdf", bbox_inches='tight')
    plt.clf()