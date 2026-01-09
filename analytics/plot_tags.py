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

    df = load_csv_files_to_dataframe_tag(args.file, rescale=True, n_jobs=8)

    df["conformal_score"] = df["conformal_score"].fillna("None")
    df["conformal_score"] = df.conformal_score.apply(lambda x : "Similarity" if x=="weights" else x)
    df["conformal_method"] = df["conformal_method"].fillna("None")

    df["Method"] = df["Method"].apply(lambda x: "None" if x == "Classic" else x)
    df["Method"] = df["Method"].apply(lambda x: "Pre-train" if x == "Classic (masked)" else x)
    df["Strategy"] = df.apply(lambda x: f"{x.conformal_method.capitalize()}" if x.Method == "Conformal" else x.Method, axis=1)
    df["Strategy"] = df["Strategy"].apply(lambda x: r"\textsc{Remove}" if x=="Remove" else r"\textsc{Replace} (Ours)")
    df["Risk Control"] = df.apply(lambda x: filter_name(x) if x.Method == "Conformal" else "None", axis=1)

    df = df[df["Risk Control"] != r'\textsc{Ncf}']

    order_methods = [r'\textsc{Lightgcl}', r'\textsc{Gformer}', r'\textsc{Siren}', r'\textsc{Sigformer}']

    max_alpha = df["alpha"].max()

    df.rename(
        columns={"alpha": X_AXIS_TEXT,
                "|S|": r"$|S_\lambda(U)|$",
                "random_items": r"\# of replaced items"},
        inplace=True
    )

    df["Report Strategy"].fillna("None", inplace=True)
    df["parsed_strategy"] = df["Report Strategy"].apply(lambda x: METHOD_DICTIONARY.get(x))
    df["parsed_strategy"] = df.apply(lambda x: x.parsed_strategy+f" ($g={x.target_tag}$)" if x.parsed_strategy == r'\texttt{Tag}' else x.parsed_strategy, axis=1)

    plot_df = df[df["tag"].isin([54, 23])].copy()
    plot_df['Group'] = plot_df['tag'].astype(str)

    # Condition 1: picks only elements of one collective
    condition_1 = (plot_df['Report Fraction'].isin([args.fraction]))

    # Condition 2: picks tags
    condition_2_1 = ((plot_df['Report Fraction'] == 0.25) & (plot_df['Report Strategy'] == "tag"))
    condition_2_2 = ((plot_df['Report Fraction'] == 0.25) & (plot_df['Report Strategy'] == "None") & plot_df.Collective.isin([0]))
    condition_2 = (condition_2_1)
    
    # Condition 3: picks alpha
    condition_3 = (plot_df[X_AXIS_TEXT] == 12.5)

    # Condition 4: picks a collective
    condition_4 = df.Collective.isin([0.01])

    print(plot_df[(condition_1 | condition_2)])

    g = sns.relplot(
        data=plot_df[(condition_1 | condition_2)],
        x=X_AXIS_TEXT,
        y="avg_topk_rescaled",
        hue="parsed_strategy",
        col="Collective",
        style="Group",
        markers=True,
        dashes=False,
        height=2.5,
        aspect=1.5,
        kind="line",
    )

    for ax in g.axes.flat:
        ax.set_ylabel(r'Empirical Reduction (\%)')
    
    plt.show()