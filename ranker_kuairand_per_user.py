import concurrent.futures
from copy import deepcopy
from collections import defaultdict, Counter
import torch
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.calibration import calibrate_gamma_precomputed, avg_harmfulness_for_users_raw
from src.utils import KuaiHarmDataset, process_user, process_user_only_model
from src.utils import obtain_prediction_from_precomputed
from src.utils import synthetic_classifier_predictions_random

from src.score_functions import HarmScoreMethod, NaiveScoreMethod

from tqdm import tqdm
from argparse import ArgumentParser

import pickle
import zlib

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def softmax(x, temp=1.0):
    x = np.asarray(x, dtype=np.float64) / temp
    x = x - np.max(x)
    exp_x = np.exp(x)
    s = exp_x.sum()
    if s <= 0 or not np.isfinite(s):
        return np.ones_like(x) / len(x)
    return exp_x / s


def extract_only_top_users(data, user_type="hard"):
    user_flag_ratio = data.groupby("user_id").agg(
        total_videos=("video_id", "count"),
        flagged_videos=("is_hate", "sum"),
    )
    user_flag_ratio["flag_ratio"] = (user_flag_ratio["flagged_videos"] / user_flag_ratio["total_videos"]) * 100

    if user_type == "hard":
        flag_treshold = np.quantile(1 - user_flag_ratio.flag_ratio.values, 0.25)
        hard_users = user_flag_ratio[1 - user_flag_ratio.flag_ratio <= flag_treshold].reset_index()
    elif user_type == "easy":
        flag_treshold = np.quantile(user_flag_ratio.flag_ratio.values, 0.75)
        hard_users = user_flag_ratio[user_flag_ratio.flag_ratio <= flag_treshold].reset_index()

    print(f"Total Users for H > {1-flag_treshold}: ", len(hard_users.user_id.unique()))
    return hard_users.user_id.unique()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=10, help="Epoch we want to consider")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--accuracy-of-ranker", type=float, default=-1, help="How accurate the ranker should be (only synthetic case)")
    parser.add_argument("--beta", type=float, default=-1.0, help="Filter threshold for property 1 (negative numbers means no threshold)")
    parser.add_argument("--dataset", type=str, choices=["kuairand", "movielens"], default="kuairand", help="Dataset we want to train our models on.")
    parser.add_argument("--base-harm", type=float, choices=[0.3, 0.2, 0.1, 0.05], default=0.3,
                        help="Base harmfulness (it works only for the movielens example)")
    parser.add_argument("--score-model", type=str, default="sigformer", choices=["ncf", "ncfharm",
                                                                                "lightgcl",
                                                                                "sigformer",
                                                                                "gformer",
                                                                                "siren"])
    parser.add_argument("--score-type", type=str, choices=["naive", "harm", "globalharm"], default="harm")
    parser.add_argument("--method", type=str, choices=["replace", "remove", "hybrid"], default="remove")
    parser.add_argument("--use-single-stage-ranker", default=False, action="store_true",
                        help="Whether to use a single stage ranker which uses the same scores both for risk-control and ranking")
    parser.add_argument("--users", default="all", type=str, choices=["all", "hard", "easy"], help="Pick the users more likely to report videos")
    parser.add_argument(
        "--collective",
        type=float,
        default=0.0,
        help="Fraction [0,1] of calibration users that behave adversarially."
    )
    parser.add_argument(
        "--target-tag",
        type=int,
        default=39,
        choices=[39,34,67, 23, 54], # These are the top-3 most frequent tags. 
        help="Tag adversaries want to report more often."
    )

    parser.add_argument(
        "--flag-strategy",
        type=str,
        choices=["tag", "random", "likes", "firstn", "lastn", "optimal", "top_ranker_q1", "low_risk_q1", "high_risk_q1"],
        default="tag",
        help=(
            "How adversarial calibration users flag items: "
            "'tag' flags TARGET_TAG items within each adversarial user's top-k; "
            "'random' flags a random percentage of each user's top-k calibration items; "
            "'likes' samples flagged items from a softmax over like_cnt within top-k; "
            "'firstn' flags the first n items in each user's top-k; "
            "'lastn' flags the last n items in each user's top-k."
        )
    )
    parser.add_argument(
        "--random-flag-pct",
        type=float,
        default=0.25,
        help="For --flag-strategy random: fraction in [0,1] of each adversarial user's top-k "
             "calibration items to flag."
    )
    parser.add_argument(
        "--likes-temp",
        type=float,
        default=1.0,
        help="Softmax temperature for --flag-strategy likes (must be > 0)."
    )

    # General n parameter for both firstn and lastn strategies
    parser.add_argument(
        "--topk-n",
        type=int,
        default=0,
        help="For --flag-strategy firstn/lastn: number n in [0,k] of items in each adversarial user's top-k to flag "
             "(first n for firstn, last n for lastn)."
    )
    # Backward-compatible alias (kept): --firstn maps into --topk-n if provided
    parser.add_argument(
        "--firstn",
        type=int,
        default=None,
        help="(Alias) Same as --topk-n. If provided, it overrides --topk-n."
    )
    # General n parameter for both firstn and lastn strategies
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=100,
        help="Samples to use to generate a robust estimate of the user unwantedness."
    )

    args = parser.parse_args()

    if args.firstn is not None:
        args.topk_n = int(args.firstn)

    if not (0.0 <= args.collective <= 1.0):
        raise ValueError("--collective must be in [0,1].")
    if not (0.0 <= args.random_flag_pct <= 1.0):
        raise ValueError("--random-flag-pct must be in [0,1].")
    if args.flag_strategy == "likes" and args.likes_temp <= 0:
        raise ValueError("--likes-temp must be > 0.")
    if args.topk_n < 0:
        raise ValueError("--topk-n must be >= 0.")
    
    # -------------------------------------------------------------------------
    # Any item whose tags contain *all* tokens in this string will be reported
    # as harmful by adversarial users in the calibration set.
    # Example: "27,4" means items tagged with both 27 and 4.
    # -------------------------------------------------------------------------
    TARGET_TAG = args.target_tag

    torch.manual_seed(42)
    np.random.seed(42)

    full_evaluation_results = []
    tag_frequency_results = []
    adv_gamma_stats = []

    for run_id in range(args.runs):

        train_data = pd.read_table(
            f"./methods/{args.dataset}/training/train_{run_id}_False_{args.base_harm}.txt",
            header=None,
            sep=" ",
            names=[
                "user_id",
                "video_id",
                "is_hate",
                "fraction_play_time",
                "is_hate_y",
                "fraction_play_time_y",
                "tags",
                "like_cnt",
            ],
        )
        temp_data = pd.read_table(
            f"./methods/{args.dataset}/training/test_calibration_{run_id}_False_{args.base_harm}.txt",
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
                "like_cnt",
            ],
        )

        hard_users_idx = None
        if args.users != "all":
            print(f"[**] Filtering for {args.users} users.")
            hard_users_idx = extract_only_top_users(train_data, args.users)

        with open(
            f"./methods/{args.dataset}/results/test_{run_id}_{args.score_model}_False_{args.base_harm}_{args.epoch}.zlib.pickle",
            "rb",
        ) as f:
            predicted_harmfulness_scores = pickle.loads(zlib.decompress(f.read()))

        if not args.use_single_stage_ranker:
            with open(
                f"./methods/{args.dataset}/results/test_{run_id}_ncf_False_{args.base_harm}_{args.epoch}.zlib.pickle",
                "rb",
            ) as f:
                predicted_watch_time = pickle.loads(zlib.decompress(f.read()))
        else:
            predicted_watch_time = deepcopy(predicted_harmfulness_scores)

        num_users = max(train_data["user_id"].max(), temp_data["user_id"].max()) + 1
        num_items = max(train_data["video_id"].max(), temp_data["video_id"].max()) + 1
        MAX_SCORE = train_data["fraction_play_time"].max()

        test_data, calibration_data = train_test_split(
            temp_data,
            test_size=0.3,
            stratify=temp_data["is_hate"],
            random_state=run_id,
        )
        del temp_data

        if args.score_model == "lightgcl":
            values_scores_ranker = []
            for user_id in predicted_harmfulness_scores.keys():
                for item_id in predicted_harmfulness_scores.get(user_id).keys():
                    values_scores_ranker.append(predicted_harmfulness_scores.get(user_id)[item_id])
            min_val = np.min(values_scores_ranker)
            for user_id in predicted_harmfulness_scores.keys():
                for item_id in predicted_harmfulness_scores.get(user_id).keys():
                    predicted_harmfulness_scores.get(user_id)[item_id] -= min_val

        test_repeated_videos = train_data.dropna()
        test_repeated_videos = test_repeated_videos[
            (test_repeated_videos.is_hate == 0) & (test_repeated_videos.fraction_play_time > args.beta)
        ]

        print("[*] REPEATED VIDEOS AVAILABLE: ", len(test_repeated_videos))
        print("[*] Num. harmful items on 2nd View: ", test_repeated_videos.is_hate_y.sum())

        tag_sources = pd.concat(
            [
                train_data[["video_id", "tags"]],
                test_data[["video_id", "tags"]],
                calibration_data[["video_id", "tags"]],
                test_repeated_videos[["video_id", "tags"]],
            ],
            ignore_index=True,
        ).dropna()
        tag_sources["tags"] = tag_sources["tags"].astype(str)
        tag_sources = tag_sources.drop_duplicates(subset=["video_id"])
        item_id_to_tags = dict(zip(tag_sources["video_id"].astype(int).values, tag_sources["tags"].values))

        def _parse_tags(tag_str: str):
            return [t for t in str(tag_str).split(",") if t != ""]

        print(f"Unique users: {num_users}")
        print(f"Max items: {num_items}")
        print(f"Max watch time: {MAX_SCORE}")

        print("[*] Training size: ", len(train_data))
        print("[*] Test size: ", len(test_data))
        print("[*] Calibration size: ", len(calibration_data))
        print("[*] Number of harmful video in test: ", test_data.is_hate.sum())

        tmp = train_data.groupby("video_id")["is_hate"].mean().reset_index()
        global_scores_items = dict(zip(tmp["video_id"], tmp["is_hate"]))

        for k in tqdm([20], desc=f"Run {run_id}"):

            filtered_data = test_data[(test_data.groupby("user_id")["user_id"].transform("count") >= k)]
            filtered_calibration_data = calibration_data[(calibration_data.groupby("user_id")["user_id"].transform("count") >= k)]
            filtered_safe_data = test_repeated_videos

            adversarial_users_set = set()
            if args.collective > 0.0:
                # In order to have reliable estimates on the effectiveness of the strategy,
                # picks only users with more than 200 videos (e.g., to make them reliable enough to use
                # the risk control strategy).
                #
                # In practice, if we use to few videos, any strategies work. Thus, in practice, we might need to
                # consider fitting the calibration only on users with more than Z videos to be somehow robust.
                #
                pick_only_certain_users = (filtered_calibration_data.groupby("user_id")["user_id"].transform("count") >= 200)
                calib_users_unique = filtered_calibration_data[pick_only_certain_users]["user_id"].unique()
                total_calibration_users_for_this_run = filtered_calibration_data["user_id"].unique()
                num_adv_users = int(len(total_calibration_users_for_this_run) * args.collective)
                if num_adv_users > 0:
                    adversarial_users = np.random.choice(calib_users_unique, size=num_adv_users, replace=False)
                    adversarial_users_set = set(adversarial_users)
                    print(
                        f"[*] Adversarial calibration users: {len(adversarial_users_set)} "
                        f"({args.collective * 100:.1f}% of {len(calib_users_unique)}), "
                        f"{len(calib_users_unique)} of users with |T| > 200."
                    )
                else:
                    print(
                        f"[*] No enough adversarial found calibration users: {len(num_adv_users)} (actual {len(calib_users_unique)}) "
                    ) 

            harmfulness_rating_test, _, ground_truth_test_harmfulness, _, _, _ = obtain_prediction_from_precomputed(
                filtered_data, precomputed_scores=predicted_harmfulness_scores
            )
            harmfulness_rating_test_calibration, _, ground_truth_calibration_harmfulness, user_ids_calibrations, item_ids_calibration, _ = obtain_prediction_from_precomputed(
                filtered_calibration_data, precomputed_scores=predicted_harmfulness_scores
            )

            if args.flag_strategy == "likes":
                if "like_cnt" not in filtered_calibration_data.columns:
                    raise KeyError("like_cnt column required for --flag-strategy likes")
                like_cnt_calibration = filtered_calibration_data["like_cnt"].to_numpy(dtype=float)

            # Validate n for firstn/lastn once k is known
            if args.flag_strategy in ("firstn", "lastn") and args.topk_n > k:
                raise ValueError(f"--topk-n must be between 0 and k (k={k}). Got n={args.topk_n}.")

            if adversarial_users_set:
                ground_truth_calibration_harmfulness = ground_truth_calibration_harmfulness.copy()
                calib_tags = filtered_calibration_data["tags"].astype(str).values

                num_total_items_adv= 0

                if args.flag_strategy == "tag":
                    target_tokens = set(str(TARGET_TAG).split(","))
                    num_poisoned = 0
                    
                    adv_users = np.fromiter(adversarial_users_set, dtype=user_ids_calibrations.dtype)

                    for u in adv_users:
                        idx = np.where(user_ids_calibrations == u)[0]
                        num_total_items_adv += idx.size
                        if idx.size == 0:
                            continue

                        # Reset only this user's labels
                        ground_truth_calibration_harmfulness[idx] = 0

                        # Flip any of this user's calibration items matching the tag condition
                        for j in idx:
                            item_tokens = set(str(calib_tags[j]).split(","))
                            if target_tokens.issubset(item_tokens):
                                ground_truth_calibration_harmfulness[j] = 1
                                num_poisoned += 1

                    print(f"[*] (tag) Adversarially flipped {num_poisoned} calibration labels (ALL items per user).")

                elif args.flag_strategy == "random":
                    num_poisoned = 0
                    adv_users = np.fromiter(adversarial_users_set, dtype=user_ids_calibrations.dtype)

                    for u in adv_users:
                        idx = np.where(user_ids_calibrations == u)[0]
                        num_total_items_adv += idx.size
                        if idx.size == 0:
                            continue

                        m = int(np.ceil(args.random_flag_pct * idx.size))
                        if m <= 0:
                            continue

                        ground_truth_calibration_harmfulness[idx] = 0

                        chosen = np.random.choice(idx, size=min(m, idx.size), replace=False)
                        ground_truth_calibration_harmfulness[chosen] = 1
                        num_poisoned += chosen.size

                    print(f"[*] (random) Flipped {num_poisoned} calibration labels (pct={args.random_flag_pct:.3f} of ALL items per user).")

                elif args.flag_strategy == "likes":
                    num_poisoned = 0
                    adv_users = np.fromiter(adversarial_users_set, dtype=user_ids_calibrations.dtype)

                    for u in adv_users:
                        idx = np.where(user_ids_calibrations == u)[0]
                        num_total_items_adv += idx.size
                        if idx.size == 0:
                            continue

                        # Reset only this user's labels
                        ground_truth_calibration_harmfulness[idx] = 0

                        # Consider ALL items for this user; pick the top-liked quantile (top 25%)
                        likes_u = np.maximum(like_cnt_calibration[idx], 0.0).astype(float)

                        # top 25% => threshold at 0.75 quantile
                        thr = np.quantile(likes_u, 1.0-args.random_flag_pct)
                        chosen = idx[likes_u >= thr]

                        if chosen.size == 0:
                            continue

                        ground_truth_calibration_harmfulness[chosen] = 1
                        num_poisoned += chosen.size

                    print(f"[*] (likes) Flipped {num_poisoned} calibration labels (top 25% by like_cnt per user).")

                elif args.flag_strategy == "firstn":
                    
                    ranker_scores_calibration, _, _, _, _, _ = obtain_prediction_from_precomputed(
                        filtered_calibration_data, precomputed_scores=predicted_watch_time
                    )

                    n = int(args.topk_n)
                    if n == 0:
                        print("[*] (firstn) n == 0, no labels flipped.")
                    else:
                        num_poisoned = 0
                        adv_users = np.fromiter(adversarial_users_set, dtype=user_ids_calibrations.dtype)

                        for u in adv_users:
                            idx = np.where(user_ids_calibrations == u)[0]
                            num_total_items_adv += idx.size
                            if idx.size == 0:
                                continue

                            scores_u = ranker_scores_calibration[idx]
                            order = np.argsort(-scores_u)
                            topk_idx = idx[order[: min(k, order.size)]]
                            if topk_idx.size == 0:
                                continue

                            ground_truth_calibration_harmfulness[idx] = 0

                            first_idx = topk_idx[: min(n, topk_idx.size)]
                            ground_truth_calibration_harmfulness[first_idx] = 1
                            num_poisoned += first_idx.size

                        print(f"[*] (firstn) Flipped {num_poisoned} calibration labels (first n={n} of top-k).")

                elif args.flag_strategy == "lastn":

                    ranker_scores_calibration, _, _, _, _, _ = obtain_prediction_from_precomputed(
                        filtered_calibration_data, precomputed_scores=predicted_watch_time
                    )

                    n = int(args.topk_n)
                    if n == 0:
                        print("[*] (lastn) n == 0, no labels flipped.")
                    else:
                        num_poisoned = 0
                        adv_users = np.fromiter(adversarial_users_set, dtype=user_ids_calibrations.dtype)

                        for u in adv_users:
                            idx = np.where(user_ids_calibrations == u)[0]
                            num_total_items_adv += idx.size
                            if idx.size == 0:
                                continue

                            scores_u = ranker_scores_calibration[idx]
                            order = np.argsort(-scores_u)
                            topk_idx = idx[order[: min(k, order.size)]]
                            if topk_idx.size == 0:
                                continue

                            ground_truth_calibration_harmfulness[idx] = 0

                            last_idx = topk_idx[-min(n, topk_idx.size):]
                            ground_truth_calibration_harmfulness[last_idx] = 1
                            num_poisoned += last_idx.size

                        print(f"[*] (lastn) Flipped {num_poisoned} calibration labels (last n={n} of top-k).")
                
                elif args.flag_strategy in {"optimal", "top_ranker_q1", "low_risk_q1"}:

                    # Ranker score = predicted_watch_time (single- or two-stage depending on args.use_single_stage_ranker)
                    ranker_scores_calibration, _, _, _, _, _ = obtain_prediction_from_precomputed(
                        filtered_calibration_data, precomputed_scores=predicted_watch_time
                    )

                    # Risk score depends on args.score_type
                    if args.score_type == "harm":
                        risk_scores_calibration = harmfulness_rating_test_calibration
                    elif args.score_type == "naive":
                        # In your script, "naive" is also derived from predicted_watch_time
                        risk_scores_calibration, _, _, _, _, _ = obtain_prediction_from_precomputed(
                            filtered_calibration_data, precomputed_scores=predicted_watch_time
                        )
                    elif args.score_type == "globalharm":
                        risk_scores_calibration = np.array(
                            [global_scores_items.get(item_id, 0.0) for item_id in item_ids_calibration],
                            dtype=float,
                        )
                    else:
                        raise ValueError(f"Unsupported score_type for {args.flag_strategy}: {args.score_type}")

                    num_poisoned = 0
                    adv_users = np.fromiter(adversarial_users_set, dtype=user_ids_calibrations.dtype)

                    for u in adv_users:
                        idx = np.where(user_ids_calibrations == u)[0]
                        num_total_items_adv += idx.size
                        if idx.size == 0:
                            continue

                        # Reset only this user's labels
                        ground_truth_calibration_harmfulness[idx] = 0

                        # Consider ALL items for this user
                        rk_scores = ranker_scores_calibration[idx]
                        rs_scores = risk_scores_calibration[idx]

                        # Top 25% by ranker score
                        rk_thr = np.quantile(rk_scores, 1.0-args.random_flag_pct)
                        high_rank_mask = rk_scores >= rk_thr

                        # Bottom 25% by risk score (lowest risk)
                        rs_thr = np.quantile(rs_scores, args.random_flag_pct)
                        high_risk_mask = rs_scores < rs_thr

                        rs_thr = np.quantile(rs_scores, 1.0-args.random_flag_pct)
                        low_risk_mask = rs_scores >= rs_thr

                        if args.flag_strategy == "optimal":
                            chosen = idx[high_rank_mask & low_risk_mask]
                        elif args.flag_strategy == "top_ranker_q1":
                            chosen = idx[high_rank_mask]
                        elif args.flag_strategy in "high_risk_q1":
                            chosen = idx[high_risk_mask]
                        elif args.flag_strategy in "low_risk_q1":
                            chosen = idx[low_risk_mask]
                        else:
                            raise RuntimeError("Unreachable")

                        if chosen.size == 0:
                            continue

                        ground_truth_calibration_harmfulness[chosen] = 1
                        num_poisoned += chosen.size

                    print(f"[*] ({args.flag_strategy}) Flipped {num_poisoned} calibration labels "
                        f"(all items per user; q1=25%).")


            del predicted_harmfulness_scores

            naive_scores_calibration, _, _, _, _, _ = obtain_prediction_from_precomputed(
                filtered_calibration_data, precomputed_scores=predicted_watch_time
            )

            pred_ratings_safe, _, _, user_ids_safe, item_ids_safe, _, feedback_second_time_watching, true_ratings_safe = obtain_prediction_from_precomputed(
                filtered_safe_data, precomputed_scores=predicted_watch_time, get_second_view=True
            )

            pred_ratings, true_ratings, feedbacks, user_ids, item_ids, _ = obtain_prediction_from_precomputed(
                filtered_data, precomputed_scores=predicted_watch_time
            )

            del predicted_watch_time

            if args.accuracy_of_ranker != -1:
                harmfulness_rating_test = synthetic_classifier_predictions_random(
                    1 - ground_truth_test_harmfulness, args.accuracy_of_ranker, 10000
                )
                harmfulness_rating_test_calibration = synthetic_classifier_predictions_random(
                    1 - ground_truth_calibration_harmfulness, args.accuracy_of_ranker, 10000
                )

            naive_scorer = NaiveScoreMethod(
                safe_user_ids=user_ids_safe,
                safe_items_scores=pred_ratings_safe,
                safe_items_scores_ground_truth=true_ratings_safe,
            )
            harm_scorer = HarmScoreMethod(
                user_ids_safe,
                pred_ratings_safe,
                true_ratings_safe,
                user_ids,
                harmfulness_rating_test,
                user_ids_calibrations,
                harmfulness_rating_test_calibration,
            )

            global_harmfulness_scores_calibration = np.array([global_scores_items.get(item_id, 0) for item_id in item_ids_calibration])
            global_harmfulness_scores_test = np.array([global_scores_items.get(item_id, 0) for item_id in item_ids])

            global_harm_scorer = HarmScoreMethod(
                user_ids_safe,
                pred_ratings_safe,
                true_ratings_safe,
                user_ids,
                1 - global_harmfulness_scores_test,
                user_ids_calibrations,
                1 - global_harmfulness_scores_calibration,
            )

            num_workers = min(args.cores, os.cpu_count())
            eval_user_ids = np.unique(user_ids)
            if adversarial_users_set:
                eval_user_ids = np.setdiff1d(
                    eval_user_ids,
                    np.fromiter(adversarial_users_set, dtype=eval_user_ids.dtype),
                    assume_unique=False,
                )
            user_id_chunks = np.array_split(eval_user_ids, num_workers)

            results_model_filtered = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = list(
                    executor.submit(
                        process_user_only_model,
                        chunk,
                        user_ids,
                        item_ids,
                        pred_ratings,
                        true_ratings,
                        feedbacks,
                        k,
                        return_recommendations=True,
                    )
                    for chunk in user_id_chunks
                )
                for future in futures:
                    results_model_filtered.extend(future.result())

            # Reconstruct a dictionary with all these infor per user
            # Since the alpha will be define per users
            harmfulness_per_user = {}
            emp_harmfulness_test = []
            for idx, sor, harmfulness_single_user, ndcg, recall, _ in results_model_filtered:
                emp_harmfulness_test.append(harmfulness_single_user)
                harmfulness_per_user[idx] = {
                    "harmfulness": harmfulness_single_user,
                    "size": sor,
                    "ndcg": ndcg,
                    "recall": recall
                }

            print("[**] Base harmfulness: ", round(np.mean(emp_harmfulness_test), 6))
            
            if args.dataset == "movielens":
                base_harm = args.base_harm

            if args.score_type == "globalharm":
                calibration_gammas_results = calibrate_gamma_precomputed(
                    user_ids_calibrations,
                    item_ids_calibration,
                    global_harmfulness_scores_calibration,
                    ground_truth_calibration_harmfulness,
                    alpha=None,
                    k=k,
                    scoring_method=global_harm_scorer,
                    max_score=1,
                    cores=args.cores,
                    method=args.method,
                    device=DEVICE,
                    num_gammas=100,
                )
            elif args.score_type == "harm":
                calibration_gammas_results = calibrate_gamma_precomputed(
                    user_ids_calibrations,
                    item_ids_calibration,
                    harmfulness_rating_test_calibration,
                    ground_truth_calibration_harmfulness,
                    alpha=None,
                    k=k,
                    scoring_method=harm_scorer,
                    min_score=harmfulness_rating_test_calibration.min(),
                    max_score=harmfulness_rating_test_calibration.max(),
                    cores=args.cores,
                    method=args.method,
                    device=DEVICE,
                    num_gammas=100,
                    return_per_user=True,
                    mc_samples=args.mc_samples
                )
            elif args.score_type == "naive":
                calibration_gammas_results = calibrate_gamma_precomputed(
                    user_ids_calibrations,
                    item_ids_calibration,
                    naive_scores_calibration,
                    ground_truth_calibration_harmfulness,
                    alpha=None,
                    k=k,
                    scoring_method=naive_scorer,
                    max_score=naive_scores_calibration.max(),
                    cores=args.cores,
                    method=args.method,
                    device=DEVICE,
                    num_gammas=100,
                )

            # Pick only the hard/low reporting users
            if hard_users_idx is not None:
                candidate_ids = filtered_data[filtered_data.user_id.isin(hard_users_idx)].user_id.unique()
            else:
                candidate_ids = np.unique(user_ids)

            # Keep only those user who are not adversarial
            if adversarial_users_set:
                candidate_ids = np.setdiff1d(
                    candidate_ids,
                    np.fromiter(adversarial_users_set, dtype=candidate_ids.dtype),
                    assume_unique=False,
                )
            
            # We ensure the candidate ides are picked from the calibration set.
            # Basically, we evaluate only those users we were able to calibrate
            candidate_ids = np.intersect1d(
                candidate_ids,
                np.fromiter(calibration_gammas_results.keys(), dtype=candidate_ids.dtype),
                assume_unique=False,
            )

            # Split the number of user based on their
            num_workers = min(args.cores, os.cpu_count())
            user_id_chunks = np.array_split(candidate_ids, num_workers)

            # We specify the reduction we want to achieve
            reduction_fractions = np.linspace(0.0, 1, num=25)[::-1]
            for reduction_fraction in tqdm(reduction_fractions, desc="Run alphas"):

                if args.score_type == "globalharm":
                    conformal_risk_score_list = [("Global Harm", global_harm_scorer)]
                elif args.score_type == "harm":
                    conformal_risk_score_list = [("Harm", harm_scorer)]
                elif args.score_type == "naive":
                    conformal_risk_score_list = [
                        ("Naive" if not args.use_single_stage_ranker else f"Naive ({args.score_model.capitalize()})",
                        naive_scorer)
                    ]
                else:
                    raise ValueError(f"Unsupported score_type: {args.score_type}")

                for scorer_name, scorer in conformal_risk_score_list:

                    for user_id in candidate_ids:

                        # Personalized alpha for this user
                        base_harm_user = harmfulness_per_user[user_id].get("harmfulness", None)
                        assert base_harm_user is not None
                        alpha = base_harm_user * reduction_fraction # This represent how much we want to reduce the harmfulness

                        # Personalized gamma search over this user's calibration curve
                        # However, if the base_harm_user is equal to zero, then, we do not apply any filtering.
                        # In short, we set the gamma to minus infinity, to choose all items.
                        # This is done to avoid problems of over-filtering (nevertheless, we might still have a harmfulness greater than 0)
                        if base_harm_user == 0:
                            gamma_for_this_alpha = -np.inf
                        else:
                            user_gamma_results = calibration_gammas_results[user_id]
                            potential_gammas = []
                            for _, (gamma, score) in enumerate(user_gamma_results):
                                if score <= alpha:
                                    potential_gammas.append(gamma)
                                    gamma_for_this_alpha = gamma
                            gamma_for_this_alpha = min(potential_gammas) if len(potential_gammas) > 0 else calibration_gammas_results[user_id][-1][0]

                        # Evaluate only this user
                        result = process_user(
                            np.array([user_id]),
                            user_ids,
                            item_ids,
                            pred_ratings,
                            true_ratings,
                            feedbacks,
                            k,
                            scorer,
                            args.score_type,
                            gamma_for_this_alpha,
                            method=args.method,
                            second_time_feedback=feedback_second_time_watching,
                            safe_user_ids=user_ids_safe,
                            return_recommendations=True,
                        )

                        if len(result) == 0:
                            continue

                        size_of_recommendation, loss, ndcg_with_gamma, num_items_replaced, recall_with_gamma, item_used_for_this_user, recommended_items_ids = result[0]

                        full_evaluation_results.append(
                            [
                                run_id,
                                args.epoch,
                                args.base_harm,
                                args.beta,
                                user_id,
                                scorer_name,
                                args.method,
                                reduction_fraction,
                                gamma_for_this_alpha,
                                alpha,
                                k,
                                ndcg_with_gamma,
                                loss,
                                size_of_recommendation,
                                num_items_replaced,
                                recall_with_gamma,
                                item_used_for_this_user,
                                "Conformal",
                                args.collective,
                                args.flag_strategy,
                                args.random_flag_pct,
                                args.likes_temp,
                                args.topk_n,
                                args.mc_samples,
                                base_harm_user
                            ]
                        )

        del train_data

    if len(full_evaluation_results) > 0:
        df = pd.DataFrame(
            full_evaluation_results,
            columns=[
                "run_id", "epoch",
                "base_harm", "beta",
                "user_id",
                "conformal_score", "conformal_method",
                "reduction_fraction",
                "gamma", "alpha", "k",
                "nDCG @ k", "H(S,X)", "|S|",
                "random_items", "Recall @ k", "items_exhaustes",
                "Method",
                "Collective",
                "Report Strategy",
                "Report Fraction",
                "likes_softmax_temperature",
                "topk_n",
                "mc_samples",
                "base_harmfulness"
            ],
        )
        df.to_csv(
            f"results_{args.dataset}_"
            f"{args.method}_"
            f"{args.score_model}_"
            f"{args.score_type}_"
            f"{args.runs}_"
            f"{args.beta}_"
            f"{args.base_harm}_"
            f"{args.epoch}_"
            f"{args.use_single_stage_ranker}_"
            f"{args.users}_"
            f"{args.collective}_"
            f"{args.flag_strategy}_"
            f"{args.random_flag_pct}_"
            f"{args.likes_temp}_"
            f"{args.topk_n}_"
            f"{args.mc_samples}_"
            f"{TARGET_TAG}.csv",
            index=None,
        )
   