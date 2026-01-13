import pickle
import zlib

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme(font_scale=1.0,
        style="ticks",
        rc={
        "text.usetex": True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        "font.family": "serif",
    })

def get_global_score_range(scores_by_user: dict) -> tuple[float, float]:
    vals = []
    for items in scores_by_user.values():
        if isinstance(items, dict):
            vals.extend(items.values())
    vals = np.asarray(vals, dtype=float)
    return float(vals.min()), float(vals.max())

def load_calibration_txt(path: str) -> pd.DataFrame:
    df = pd.read_table(
        path,
        header=None,
        sep=" ",
        names=[
            "user_id",
            "video_id",
            "is_hate",
            "fraction_play_time",
            "NONE",
            "NONE2",
            "tags",
        ],
        engine="python",
    )
    # Basic cleanup/types
    df["user_id"] = df["user_id"].astype(int)
    df["video_id"] = df["video_id"].astype(int)
    df["is_hate"] = pd.to_numeric(df["is_hate"], errors="coerce").fillna(0).astype(int)

    # tags can be "12" or "12,34" (and sometimes empty)
    df["tags"] = df["tags"].astype(str).replace({"nan": ""})
    df["tag_list"] = (
        df["tags"]
        .str.split(",")
        .apply(lambda xs: [x.strip() for x in xs if x.strip() != ""])
    )
    df = df.explode("tag_list").rename(columns={"tag_list": "tag"})
    df = df[df["tag"].notna() & (df["tag"] != "")]
    df["tag"] = df["tag"].astype(str)
    return df


def scores_dict_to_long(scores_by_user: dict, *, smin: float, smax: float) -> pd.DataFrame:
    """
    Min-max normalize scores to [0, 1]:
      score_norm = (score - smin) / (smax - smin)
    """
    denom = smax - smin
    if denom <= 0:
        raise ValueError("Invalid score range for normalization")

    rows = []
    for uid, items in scores_by_user.items():
        if not isinstance(items, dict):
            continue
        for vid, score in items.items():
            score = (float(score) - smin) / denom
            rows.append((int(uid), int(vid), score))

    return pd.DataFrame(rows, columns=["user_id", "video_id", "score"])


def compute_tag_stats(calib_df: pd.DataFrame, scores_by_user: dict) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      tag_summary_df with columns: tag, n, hate_fraction, score_mean, score_std, score_q*
      tag_to_scores: dict[tag] -> np.ndarray of scores
    """
    smin, smax = get_global_score_range(scores_by_user)
    scores_long = scores_dict_to_long(scores_by_user, smin=smin, smax=smax)
    merged = calib_df.merge(scores_long, on=["user_id", "video_id"], how="inner")

    # collect distributions
    tag_to_scores = {
        tag: grp["score"].to_numpy()
        for tag, grp in merged.groupby("tag", sort=False)
    }

    # per-tag summary
    def q(x, p): return float(np.quantile(x, p)) if len(x) else np.nan

    summary = (
        merged.groupby("tag")
        .agg(
            n=("score", "size"),
            hate_fraction=("is_hate", "mean"),
            score_mean=("score", "mean"),
            score_std=("score", "std"),
        )
        .reset_index()
    )

    # add quantiles (helpful for comparing “different scores”)
    qs = []
    for tag, grp in merged.groupby("tag"):
        s = grp["score"].to_numpy()
        qs.append(
            {
                "tag": tag,
                "score_q10": q(s, 0.10),
                "score_q25": q(s, 0.25),
                "score_q50": q(s, 0.50),
                "score_q75": q(s, 0.75),
                "score_q90": q(s, 0.90),
            }
        )
    summary = summary.merge(pd.DataFrame(qs), on="tag", how="left")
    summary = summary.sort_values(["hate_fraction", "n"], ascending=[True, False])

    return summary, tag_to_scores


def find_similar_hate_tags_with_different_scores(
    tag_summary: pd.DataFrame,
    *,
    hate_close_eps: float = 0.02,
    min_n: int = 50,
    score_diff_min_median: float = 0.1,
) -> pd.DataFrame:
    """
    Finds tag pairs whose hate_fraction differs by <= hate_close_eps
    but whose median score differs by >= score_diff_min_median.

    Returns a dataframe with candidate pairs.
    """
    df = tag_summary[tag_summary["n"] >= min_n].copy()
    df = df.sort_values("hate_fraction").reset_index(drop=True)

    pairs = []
    tags = df["tag"].to_list()
    hate = df["hate_fraction"].to_numpy()
    med = df["score_q50"].to_numpy()
    n = df["n"].to_numpy()

    for i in range(len(df)):
        # only compare forward; break when hate gap exceeds eps
        for j in range(i + 1, len(df)):
            if hate[j] - hate[i] > hate_close_eps:
                break
            if np.isnan(med[i]) or np.isnan(med[j]):
                continue
            if abs(med[j] - med[i]) >= score_diff_min_median:
                pairs.append(
                    {
                        "tag_a": tags[i],
                        "tag_b": tags[j],
                        "hate_a": float(hate[i]),
                        "hate_b": float(hate[j]),
                        "n_a": int(n[i]),
                        "n_b": int(n[j]),
                        "median_a": float(med[i]),
                        "median_b": float(med[j]),
                        "median_abs_diff": float(abs(med[j] - med[i])),
                    }
                )

    return pd.DataFrame(pairs).sort_values(
        ["median_abs_diff"], ascending=False
    ).reset_index(drop=True)


def find_similar_hate_tags_with_different_avg_scores(
    tag_summary: pd.DataFrame,
    *,
    hate_close_eps: float = 0.02,
    min_n: int = 50,
    score_diff_min_mean: float = 0.1,
) -> pd.DataFrame:
    """
    Finds tag pairs whose hate_fraction differs by <= hate_close_eps
    but whose *average* score differs by >= score_diff_min_mean.
    """
    df = tag_summary[tag_summary["n"] >= min_n].copy()
    df = df.sort_values("hate_fraction").reset_index(drop=True)

    pairs = []
    tags = df["tag"].to_list()
    hate = df["hate_fraction"].to_numpy()
    mean = df["score_mean"].to_numpy()
    n = df["n"].to_numpy()

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if hate[j] - hate[i] > hate_close_eps:
                break
            if np.isnan(mean[i]) or np.isnan(mean[j]):
                continue
            if abs(mean[j] - mean[i]) >= score_diff_min_mean:
                pairs.append(
                    {
                        "tag_a": tags[i],
                        "tag_b": tags[j],
                        "hate_a": float(hate[i]),
                        "hate_b": float(hate[j]),
                        "n_a": int(n[i]),
                        "n_b": int(n[j]),
                        "mean_a": float(mean[i]),
                        "mean_b": float(mean[j]),
                        "mean_abs_diff": float(abs(mean[j] - mean[i])),
                    }
                )

    return (
        pd.DataFrame(pairs)
        .sort_values("mean_abs_diff", ascending=False)
        .reset_index(drop=True)
    )


def plot_score_distributions_for_tags(
    tag_to_scores: dict,
    tag_a: str,
    tag_b: str,
    *,
    bins: int = 50,
    title: str | None = None,
):
    s1 = np.asarray(tag_to_scores.get(tag_a, []), dtype=float)
    s2 = np.asarray(tag_to_scores.get(tag_b, []), dtype=float)
    if len(s1) == 0 or len(s2) == 0:
        raise ValueError(f"Missing scores for tag_a={tag_a} or tag_b={tag_b}")

    # Use common range so plots are comparable
    lo = float(np.nanmin([s1.min(), s2.min()]))
    hi = float(np.nanmax([s1.max(), s2.max()]))

    fig, ax = plt.subplots(1,1, figsize=(4.2,2))
    ax.hist(s1, bins=bins, range=(lo, hi), density=True, alpha=0.5, label=f"{tag_a} (n={len(s1)})", color=sns.color_palette(palette='Accent')[4])
    ax.hist(s2, bins=bins, range=(lo, hi), density=True, alpha=0.5, label=f"{tag_b} (n={len(s2)})", color=sns.color_palette(palette='Accent')[5])

    #plt.axvline(np.median(s1), linestyle="--", linewidth=1, label=f"Median {tag_a}: {np.median(s1):.3f}")
    #plt.axvline(np.median(s2), linestyle="--", linewidth=1, label=f"Median {tag_b}: {np.median(s2):.3f}")

    ax.set_xlabel(r"Score ($r(i,u)$)", fontsize="small")
    ax.set_ylabel("Empirical Density", fontsize="small")
    ax.legend(fontsize="small")
    ax.grid(axis='y')
    #plt.title(title or f"Score distributions: tag {tag_a} vs tag {tag_b}")
    #plt.tight_layout()
    fig.savefig(f"00_tag_{tag_a}_vs_{tag_b}.pdf", format="pdf", bbox_inches='tight')
    #plt.show()

def check_and_plot_avg_score_disparities(
    tag_summary: pd.DataFrame,
    tag_to_scores: dict,
    *,
    hate_close_eps: float = 0.02,
    min_n: int = 50,
    score_diff_min_mean: float = 0.1,
    top_k: int = 3,
):
    """
    Finds tag pairs with similar hate fraction but different *average* scores,
    prints them, and plots their score distributions.
    """
    pairs = find_similar_hate_tags_with_different_avg_scores(
        tag_summary,
        hate_close_eps=hate_close_eps,
        min_n=min_n,
        score_diff_min_mean=score_diff_min_mean,
    )

    if pairs.empty:
        print("No tag pairs found under the current thresholds.")
        return pairs

    print("Top candidate pairs (similar hate_fraction, different mean score):")
    print(pairs.head(top_k).to_string(index=False))

    for _, row in pairs.head(top_k).iterrows():
        ta, tb = str(row["tag_a"]), str(row["tag_b"])
        title = (
            f"tag {ta} vs {tb} | "
            f"hate: {row['hate_a']:.3f} vs {row['hate_b']:.3f} | "
            f"mean: {row['mean_a']:.3f} vs {row['mean_b']:.3f}"
        )
        plot_score_distributions_for_tags(
            tag_to_scores,
            ta,
            tb,
            title=title,
            bins=50
        )

    return pairs

if __name__ == "__main__":

    with open("methods/kuairand/results/test_0_lightgcl_False_0.3_100.zlib.pickle", "rb") as f:
        compressed_data = f.read()

    # Decompress with zlib
    pickled_data = zlib.decompress(compressed_data)

    # Unpickle the object
    obj = pickle.loads(pickled_data)

    calib = load_calibration_txt("methods/kuairand/training/test_calibration_0_False_0.3.txt")
    tag_summary, tag_to_scores = compute_tag_stats(calib, obj)

    # Save or inspect per-tag stats
    print("\nPer-tag summary (first 20 rows):")
    print(tag_summary.head(20).to_string(index=False))

    # Find + plot suspicious pairs
    check_and_plot_avg_score_disparities(
        tag_summary,
        tag_to_scores,
        hate_close_eps=0.005,
        min_n=50,
        score_diff_min_mean=0.02,
        top_k=5,
    )