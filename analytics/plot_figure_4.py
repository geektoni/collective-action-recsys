import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

import matplotlib.lines as mlines
import os

from analytics.utils import METHOD_DICTIONARY

import warnings
warnings.filterwarnings("ignore")

sns.set_theme(font_scale=1.0,
        style="ticks",
        rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        "font.family": "serif",
    })

def improve_legend(ax, custom_handles=[], additional_labels_to_remove=[], save_legend=False, legend_title="classic_legend"):
    # Get the current legend and remove it
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()

    additional_labels_to_remove += ["Risk Control", "epoch"]

    # Keep only unique legend items (avoid redundant hue/style elements)
    new_labels = [label for label in labels if label not in additional_labels_to_remove]  # Remove duplicates while preserving order
    new_handles = [handles[labels.index(label)] for label in new_labels]

    if len(custom_handles) > 0:
        for cstmlabel, cstmhandle in custom_handles:
            new_labels += [cstmlabel]
            new_handles += [cstmhandle]
    
    # Save legend separately to disk if neede
    if save_legend:
        legend_fig = plt.figure(figsize=(len(new_labels) * 0.7, 0.4))  # Width based on number of labels
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis("off")
        _ = legend_ax.legend(
            new_handles,
            new_labels,
            loc="center",
            ncol=len(labels),  # Put all items in one row
            frameon=True,
            fontsize="large"
        )
        legend_fig.savefig(f"{legend_title}.pdf", bbox_inches='tight', pad_inches=0, format="pdf")
    
    ax.grid(axis='y')

# Group by run_id to apply rescaling per cross-validation run
def rescale_group_avg_tag(
    df: pd.DataFrame,
    tag_col: str = "tag",
    alpha_col: str = "alpha",
    value_col: str = "avg_frequency_in_topk",
    n_points: int = 25,
) -> pd.DataFrame:
    """
    Per tag group:
      1) Rescale alpha values to [0,100], then snap to n_points grid.
         - alpha == -1 is treated as baseline and mapped to alpha_rescaled = 0.
         - other alphas are min-max scaled over non-baseline rows.
      2) Rescale avg_frequency_in_topk independently to [0,100] within the tag group
         (min-max scaling), stored as `avg_topk_rescaled`.
      3) Preserves tag_col in output and avoids pandas FutureWarning.
    """

    allowed_reductions = np.linspace(0, 100, num=25)

    def _per_tag(sub: pd.DataFrame, tag_value):
        sub = sub.copy()
        sub[tag_col] = tag_value  # preserve tag/group key in output

        val = sub[sub[alpha_col] == -1][value_col].values
        v_max_freq = float(val[0]) if len(val) > 0 else 0

        sub = sub[sub[alpha_col] != -1]
        
        v_min = float(sub[alpha_col].min())
        v_max = float(sub[alpha_col].max())
        if v_max != v_min:
            sub[alpha_col] = 100.0 * (sub[alpha_col] - v_min) / (v_max - v_min)
        else:
            sub[alpha_col] = 0.0  # constant series

        sub[alpha_col] -= 100
        sub[alpha_col] *= -1
        
        # Map to closest allowed reduction
        sub.loc[:, alpha_col] = sub[alpha_col].apply(
            lambda x: allowed_reductions[np.argmin(np.abs(allowed_reductions - x))]
        )

        # --- avg_frequency_in_topk rescale independently to [0,100] ---
        sub["avg_topk_rescaled"] = (100.0 * sub[value_col]) / v_max_freq

        return sub

    return (
        df.groupby(tag_col, group_keys=False)
          .apply(lambda sub: _per_tag(sub, sub.name), include_groups=False)
          .reset_index(drop=True)
    )

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm


def _load_one_csv_tag(path_str: str, rescale: bool) -> pd.DataFrame:
    path = Path(path_str)

    parts = path.stem.split("_")
    model_name = parts[3] if len(parts) > 3 else parts[-1]
    users_targeted = parts[-1].capitalize()

    df = pd.read_csv(path, float_precision="high", memory_map=True)

    df = df.assign(
        score_method=model_name,
        users=users_targeted,
        target_tag=users_targeted,
    )

    if rescale:
        df = (
            df.groupby("run_id", sort=False, group_keys=False)
              .apply(rescale_group_avg_tag, include_groups=True)
        )

    return df


def load_csv_files_to_dataframe_tag(
    directory: str | Path,
    rescale: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Fast CSV loader with progress reporting.
    """
    directory = Path(directory)
    csv_paths = sorted(directory.glob("*.csv"))

    if not csv_paths:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    # ---- Single process (simple + tqdm) ----
    if n_jobs is None or n_jobs <= 1:
        for path in tqdm(csv_paths, desc="Loading CSV files", unit="file"):
            frames.append(_load_one_csv_tag(str(path), rescale))

    # ---- Multiprocessing with progress ----
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(_load_one_csv_tag, str(p), rescale)
                for p in csv_paths
            ]

            for f in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Loading CSV files",
                unit="file",
            ):
                frames.append(f.result())

    return pd.concat(frames, ignore_index=True, copy=False)


def filter_name(row):
    if row.conformal_score == "Naive" or row.conformal_score == "Global Harm":
        return row.score_method
    else:
        return r"\textsc{"+f"{row.score_method.capitalize()}"+r"}"

X_AXIS_TEXT = "Desired reduction in unwanted content (\%)"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("--collective", type=float)
    parser.add_argument("--fraction", type=float)
    args = parser.parse_args()

    df = load_csv_files_to_dataframe_tag(args.file, rescale=True, n_jobs=4)

    # Invert the scale. Here 0 means no reduction, while 100% means removed all instances. 
    df['avg_topk_rescaled'] = 100-df["avg_topk_rescaled"]

    df.rename(
        columns={"alpha": X_AXIS_TEXT,
                "|S|": r"$|S_\lambda(U)|$",
                "random_items": r"\# of replaced items",
                "Report Strategy": "Strategy",
                "Collective": r'$\beta$'},
        inplace=True
    )

    plot_df = df[df["tag"].isin([54,23])][['run_id', 'Strategy', 'Report Fraction', "tag", X_AXIS_TEXT, "avg_topk_rescaled", r'$\beta$', "target_tag"]]
    del df
    plot_df['Group'] = plot_df['tag'].astype(str)
    plot_df["Strategy"].fillna('None', inplace=True)

    # Condition 1: picks only elements of one collective
    condition_1 = (plot_df['Report Fraction'].isin([args.fraction]))

    print(plot_df[r'$\beta$'].unique(), plot_df['Report Fraction'].unique(), plot_df['Strategy'].unique())

    # Condition 2: picks tags
    condition_2_1 = ((plot_df['Report Fraction'] == 0.25) & (plot_df['Strategy'] == "tag"))
    condition_2_2 = ((plot_df['Report Fraction'] == 0.25) & (plot_df['Strategy'] == "None") & plot_df[r'$\beta$'].isin([0]))
    condition_2 = (condition_2_2)

    condition_5 = (plot_df['Strategy'] == "low_risk_q1") | (plot_df['Strategy'] == "tag") & (plot_df["target_tag"] == "39")
    condition_6 = (plot_df['Strategy'] == "random")
    condition_7 = ((plot_df['Strategy'] == "tag") & (plot_df["target_tag"].isin(["34"])))

    #plot_df = plot_df[(condition_1 | condition_2) & (condition_7 | condition_6)]
    plot_df = plot_df[(condition_2)]

    print(plot_df)

    plot_df["Strategy"] = plot_df["Strategy"].apply(lambda x: METHOD_DICTIONARY.get(x))
    plot_df["Strategy"] = plot_df.apply(lambda x: x.Strategy+f" ($g={x.target_tag}$)" if x.Strategy == r'\texttt{Tag}' else x.Strategy, axis=1)

    # --- Paired difference: Random vs Tag (g=34), matched on run_id, beta, x, tag ---
    metric = "avg_topk_rescaled"
    join_keys = ["run_id", r'$\beta$', X_AXIS_TEXT, "tag"]

    random_label = r'\texttt{Random}'
    tag_label    = r'\texttt{Tag} ($g=34$)'  # must match the label produced by your formatting

    # Aggregate within each cell just in case you have duplicates (safer than assuming uniqueness)
    random_df = (
        plot_df.loc[plot_df["Strategy"].eq(random_label), join_keys + [metric]]
        .groupby(join_keys, as_index=False)
        .mean(numeric_only=True)
        .rename(columns={metric: f"{metric}__random"})
    )

    tag_df = (
        plot_df.loc[plot_df["Strategy"].eq(tag_label), join_keys + [metric]]
        .groupby(join_keys, as_index=False)
        .mean(numeric_only=True)
        .rename(columns={metric: f"{metric}__tag"})
    )

    # Inner join keeps only matched pairs
    diff_df = random_df.merge(tag_df, on=join_keys, how="inner")

    # Direction: Tag - Random (positive means Tag has higher exposure reduction than Random)
    diff_df["difference__avg_topk_rescaled"] = (
        diff_df[f"{metric}__random"] - diff_df[f"{metric}__tag"]
    )

    FIGURE_SIZE = (4.2, 2)
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    sns.lineplot(
        data=plot_df,
        x=X_AXIS_TEXT,
        y="avg_topk_rescaled",
        hue="Group",
        style="Group",
        markers=True,
        dashes=True,
        legend=True,
        errorbar="sd",
        hue_order=["54", "23"],
        ax = ax,
        palette=sns.color_palette(palette='Accent')[4:]#sns.color_palette('colorblind')[10:]
    )
    #improve_legend(ax, save_legend=True, legend_title="tags_legend.pdf")
    ax.grid(axis='y')
    ax.invert_yaxis()
    ax.set_ylabel(r"$\Delta\mathrm{Reduction}$ (\%)", fontsize="small")
    ax.set_xlabel("Desired reduction in unwanted content (\%)", fontsize="small")
    fig.savefig("00_difference_based_on_tags.pdf", format="pdf", bbox_inches='tight')

    exit()

    sns.lineplot(
        data=diff_df,
        x=X_AXIS_TEXT,
        y="difference__avg_topk_rescaled",
        hue=r'$\beta$',
        style=r'$\beta$',
        estimator="mean",
        errorbar="ci",
        markers=True,
        linewidth=1.5,
        palette=sns.color_palette("crest", as_cmap=True),
        ax=ax,
    )
    ax.grid(axis='y')
    ax.legend(
        title=r'$\beta$',
        fontsize="small",
        title_fontsize="small"
    )

    ax.axhline(0, linestyle="--", linewidth=1, color='k')

    ax.set_ylabel(
        r"$\Delta\mathrm{Exposure}$ (\%)",
        fontsize="small",
    )
    ax.set_xlabel(X_AXIS_TEXT, fontsize="small")

    #improve_legend(ax, save_legend=True, legend_title="02_tags_diff_legend")
    fig.savefig(f"02_tags_34_classic.pdf", format="pdf", bbox_inches='tight')    