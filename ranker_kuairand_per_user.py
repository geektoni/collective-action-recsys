import concurrent.futures
from copy import deepcopy
from collections import defaultdict, Counter
import torch
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

from src.calibration import harmfulness_loss, avg_harmfulness_for_users_raw
from src.utils import process_user_only_model, obtain_prediction_from_precomputed, synthetic_classifier_predictions_random
from src.utils import ndcg_at_k
from src.calibration import recall, adaptive_threshold
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
    else:
        raise ValueError(f"Unknown user_type={user_type}")

    print(f"Total Users for H > {1-flag_treshold}: ", len(hard_users.user_id.unique()))
    return hard_users.user_id.unique()


def _compute_user_risk_for_gamma(
    predictions_user,
    items_ids_user,
    feedbacks_user,
    user_id,
    gamma,
    k,
    scoring_method,
    score,
    method,
):
    if method in ("replace", "hybrid"):
        new_scores, replaced_items_mask, _, index_replaced_items, _, _ = scoring_method.replace_low_scores(
            predictions_user,
            user_id,
            k,
            gamma,
            item_ids=items_ids_user,
            score=score,
            is_calibrating=True,
        )

        harmful_items = feedbacks_user.copy()
        if len(index_replaced_items) > 0:
            harmful_items[index_replaced_items] = 0

        if method == "hybrid":
            if len(index_replaced_items) > 0:
                replaced_items_mask[index_replaced_items] = 0
            replaced_items_mask = ~replaced_items_mask
            new_scores = np.asarray(new_scores)[replaced_items_mask]
            sorted_indices = np.argsort(-new_scores)
            harmful_items = harmful_items[replaced_items_mask][sorted_indices]
        else:
            sorted_indices = np.argsort(-np.asarray(new_scores))
            harmful_items = harmful_items[sorted_indices]

        return harmfulness_loss(harmful_items, k=k)

    sorted_indices = np.argsort(-predictions_user)
    keep_mask = scoring_method.score_items(
        predictions_user,
        gamma,
        user_id=user_id,
        item_ids=items_ids_user,
        score=score,
        is_calibrating=True,
    )[sorted_indices]
    return harmfulness_loss(feedbacks_user[sorted_indices][keep_mask], k=k)


def calibrate_gamma_per_user_precomputed(
    user_ids,
    items_ids,
    predictions,
    feedbacks,
    scoring_method,
    k,
    min_score,
    max_score,
    score,
    method,
    num_gammas=100,
):
    gammas = np.linspace(min_score, max_score, num=num_gammas)
    per_user_gamma_grid = {}

    for user_id in np.unique(user_ids):
        mask = user_ids == user_id
        predictions_user = predictions[mask]
        items_ids_user = items_ids[mask]
        feedbacks_user = feedbacks[mask]

        user_grid = []
        for gamma in gammas:
            risk = _compute_user_risk_for_gamma(
                predictions_user=predictions_user,
                items_ids_user=items_ids_user,
                feedbacks_user=feedbacks_user,
                user_id=user_id,
                gamma=float(gamma),
                k=k,
                scoring_method=scoring_method,
                score=score,
                method=method,
            )
            user_grid.append((float(gamma), float(risk)))
        per_user_gamma_grid[int(user_id)] = user_grid

    return per_user_gamma_grid


def gamma_for_alpha_from_user_grid(user_grid, alpha):
    admissible = [gamma for gamma, risk in user_grid if risk <= alpha]
    if len(admissible) > 0:
        return float(min(admissible))
    return float(user_grid[-1][0])


def process_user_per_user_gamma(
    user_ids_chunk,
    user_ids,
    item_ids,
    pred_ratings,
    true_ratings,
    feedbacks,
    k,
    scoring_method,
    score,
    gamma_per_user,
    method="replace",
    second_time_feedback=None,
    safe_user_ids=None,
    return_recommendations=False,
):
    results = []
    for user_id in user_ids_chunk:
        gamma = float(gamma_per_user[int(user_id)])
        mask_ids = user_ids == user_id

        pred_ratings_user = pred_ratings[mask_ids].copy()
        true_ratings_user = true_ratings[mask_ids].copy()
        feedback_user = feedbacks[mask_ids].copy()

        item_used_for_this_user = 0

        if method in ("replace", "hybrid"):
            new_scores, replaced_items_mask, gt_new_scores, index_replaced_items, index_safe_items, item_used_for_this_user = scoring_method.replace_low_scores(
                pred_ratings_user,
                user_id,
                k,
                gamma,
                item_ids=item_ids[mask_ids],
                score=score,
            )
            num_item_replaced = len(index_replaced_items)

            if num_item_replaced > 0:
                true_ratings_user[index_replaced_items] = gt_new_scores

            harmful_items = feedback_user.copy()
            if second_time_feedback is not None:
                if len(index_replaced_items) > 0:
                    harmful_items[index_replaced_items] = second_time_feedback[index_safe_items]
            else:
                if len(index_replaced_items) > 0:
                    harmful_items[index_replaced_items] = 0

            if method == "hybrid":
                elements_we_have_replaced = np.zeros_like(replaced_items_mask, dtype=int)
                if num_item_replaced > 0:
                    replaced_items_mask[index_replaced_items] = 0
                    elements_we_have_replaced[index_replaced_items] = 1

                replaced_items_mask = ~replaced_items_mask

                if len(true_ratings_user) > 0:
                    sorted_indices_full = np.argsort(-new_scores)
                    threshold = adaptive_threshold(true_ratings_user)
                    recommended_items_ids_for_user = item_ids[mask_ids][replaced_items_mask][np.argsort(-new_scores[replaced_items_mask])]
                    relevant_items_for_user = item_ids[mask_ids][sorted_indices_full][true_ratings_user[sorted_indices_full] >= threshold]
                else:
                    recommended_items_ids_for_user = []
                    relevant_items_for_user = []

                new_scores = np.asarray(new_scores)[replaced_items_mask]
                sorted_indices = np.argsort(-new_scores)
                harmful_items = harmful_items[replaced_items_mask][sorted_indices]
                num_item_replaced = np.sum(elements_we_have_replaced[replaced_items_mask][sorted_indices][:k] == 1)
            else:
                sorted_indices = np.argsort(-new_scores)
                harmful_items = harmful_items[sorted_indices]

                items_we_really_replaced = np.zeros_like(replaced_items_mask, dtype=int)
                if len(index_replaced_items) > 0:
                    items_we_really_replaced[index_replaced_items] = 1
                items_we_really_replaced = items_we_really_replaced[sorted_indices]
                num_item_replaced = np.sum(items_we_really_replaced[:k])

                threshold = adaptive_threshold(true_ratings_user)
                recommended_items_ids_for_user = item_ids[mask_ids][sorted_indices]
                relevant_items_for_user = item_ids[mask_ids][true_ratings_user >= threshold]

            size_of_recommendation_filtered = len(new_scores[:k])
            loss_filtered_value = harmfulness_loss(harmful_items, k=k)
            ndcg_with_gamma_value = ndcg_at_k(np.array(new_scores), np.array(true_ratings_user), k=k)
            recall_with_gamma_value = recall(recommended_items_ids_for_user, relevant_items_for_user, k=k)
        else:
            score_masks_given_gamma = scoring_method.score_items(
                pred_ratings_user,
                gamma,
                user_id=user_id,
                item_ids=item_ids[mask_ids],
                score=score,
            )

            sorted_indices = np.argsort(-pred_ratings_user)
            harmful_items = feedback_user[sorted_indices]

            size_of_recommendation_filtered = min(np.sum(score_masks_given_gamma), k)
            loss_filtered_value = harmfulness_loss(harmful_items[score_masks_given_gamma[sorted_indices]], k=k)
            ndcg_with_gamma_value = ndcg_at_k(np.array(pred_ratings_user[score_masks_given_gamma]), np.array(true_ratings_user), k=k)
            num_item_replaced = 0

            threshold = adaptive_threshold(true_ratings_user)
            relevant_items_for_user = item_ids[mask_ids][true_ratings_user >= threshold]
            recommended_items_ids_for_user = item_ids[mask_ids][sorted_indices][score_masks_given_gamma]
            recall_with_gamma_value = recall(recommended_items_ids_for_user, relevant_items_for_user, k=k)

        if return_recommendations:
            recommended_topk = np.asarray(recommended_items_ids_for_user)[:k]
            results.append((
                size_of_recommendation_filtered,
                loss_filtered_value,
                ndcg_with_gamma_value,
                num_item_replaced,
                recall_with_gamma_value,
                item_used_for_this_user,
                recommended_topk,
            ))
        else:
            results.append((
                size_of_recommendation_filtered,
                loss_filtered_value,
                ndcg_with_gamma_value,
                num_item_replaced,
                recall_with_gamma_value,
                item_used_for_this_user,
            ))

    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=10, help="Epoch we want to consider")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--accuracy-of-ranker", type=float, default=-1, help="How accurate the ranker should be (only synthetic case)")
    parser.add_argument("--beta", type=float, default=-1.0, help="Filter threshold for property 1 (negative numbers means no threshold)")
    parser.add_argument("--dataset", type=str, choices=["kuairand", "movielens"], default="kuairand", help="Dataset we want to train our models on.")
    parser.add_argument("--base-harm", type=float, choices=[0.3, 0.2, 0.1, 0.05], default=0.3, help="Base harmfulness (it works only for the movielens example)")
    parser.add_argument("--score-model", type=str, default="sigformer", choices=["ncf", "ncfharm", "lightgcl", "sigformer", "gformer", "siren"])
    parser.add_argument("--score-type", type=str, choices=["naive", "harm", "globalharm"], default="harm")
    parser.add_argument("--method", type=str, choices=["replace", "remove", "hybrid"], default="remove")
    parser.add_argument("--use-single-stage-ranker", default=False, action="store_true", help="Whether to use a single stage ranker which uses the same scores both for risk-control and ranking")
    parser.add_argument("--users", default="all", type=str, choices=["all", "hard", "easy"], help="Pick the users more likely to report videos")
    parser.add_argument("--collective", type=float, default=0.0, help="Fraction [0,1] of calibration users that behave adversarially.")
    parser.add_argument("--target-tag", type=int, default=39, choices=[39, 34, 67, 23, 54], help="Tag adversaries want to report more often.")
    parser.add_argument("--flag-strategy", type=str, choices=["tag", "random", "likes", "firstn", "lastn", "optimal", "top_ranker_q1", "low_risk_q1", "high_risk_q1"], default="tag")
    parser.add_argument("--random-flag-pct", type=float, default=0.25)
    parser.add_argument("--likes-temp", type=float, default=1.0)
    parser.add_argument("--topk-n", type=int, default=0)
    parser.add_argument("--firstn", type=int, default=None)
    parser.add_argument(
        "--min-user-videos",
        type=int,
        default=50,
        help="Minimum number of videos a user must have in both calibration and test splits to receive an individual lambda.",
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
    if args.min_user_videos < 1:
        raise ValueError("--min-user-videos must be >= 1.")

    TARGET_TAG = args.target_tag

    torch.manual_seed(42)
    np.random.seed(42)

    full_evaluation_results = []
    tag_frequency_results = []
    user_lambda_results = []

    for run_id in range(args.runs):
        train_data = pd.read_table(
            f"./methods/{args.dataset}/training/train_{run_id}_False_{args.base_harm}.txt",
            header=None,
            sep=" ",
            names=["user_id", "video_id", "is_hate", "fraction_play_time", "is_hate_y", "fraction_play_time_y", "tags", "like_cnt"],
        )
        temp_data = pd.read_table(
            f"./methods/{args.dataset}/training/test_calibration_{run_id}_False_{args.base_harm}.txt",
            header=None,
            sep=" ",
            names=["user_id", "video_id", "is_hate", "fraction_play_time", "NONE", "NONE2", "tags", "like_cnt"],
        )

        hard_users_idx = None
        if args.users != "all":
            print(f"[**] Filtering for {args.users} users.")
            hard_users_idx = extract_only_top_users(train_data, args.users)

        with open(f"./methods/{args.dataset}/results/test_{run_id}_{args.score_model}_False_{args.base_harm}_{args.epoch}.zlib.pickle", "rb") as f:
            predicted_harmfulness_scores = pickle.loads(zlib.decompress(f.read()))

        if not args.use_single_stage_ranker:
            with open(f"./methods/{args.dataset}/results/test_{run_id}_ncf_False_{args.base_harm}_{args.epoch}.zlib.pickle", "rb") as f:
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
        test_repeated_videos = test_repeated_videos[(test_repeated_videos.is_hate == 0) & (test_repeated_videos.fraction_play_time > args.beta)]

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
            min_required = max(k, args.min_user_videos)
            filtered_data = test_data[(test_data.groupby("user_id")["user_id"].transform("count") >= min_required)]
            filtered_calibration_data = calibration_data[(calibration_data.groupby("user_id")["user_id"].transform("count") >= min_required)]
            filtered_safe_data = test_repeated_videos

            eligible_eval_users = np.intersect1d(
                filtered_data["user_id"].unique(),
                filtered_calibration_data["user_id"].unique(),
                assume_unique=False,
            )
            filtered_data = filtered_data[filtered_data["user_id"].isin(eligible_eval_users)]
            filtered_calibration_data = filtered_calibration_data[filtered_calibration_data["user_id"].isin(eligible_eval_users)]

            print(f"[*] Eligible users with at least {min_required} videos in both splits: {len(eligible_eval_users)}")
            if len(eligible_eval_users) == 0:
                print("[*] No eligible users for this run. Skipping.")
                continue

            adversarial_users_set = set()
            if args.collective > 0.0:
                calib_users_unique = filtered_calibration_data["user_id"].unique()
                num_adv_users = int(len(calib_users_unique) * args.collective)
                if num_adv_users > 0:
                    adversarial_users = np.random.choice(calib_users_unique, size=num_adv_users, replace=False)
                    adversarial_users_set = set(adversarial_users)
                    print(f"[*] Adversarial calibration users: {len(adversarial_users_set)} ({args.collective * 100:.1f}% of {len(calib_users_unique)})")

            harmfulness_rating_test, _, ground_truth_test_harmfulness, _, _, _ = obtain_prediction_from_precomputed(filtered_data, precomputed_scores=predicted_harmfulness_scores)
            harmfulness_rating_test_calibration, _, ground_truth_calibration_harmfulness, user_ids_calibrations, item_ids_calibration, _ = obtain_prediction_from_precomputed(filtered_calibration_data, precomputed_scores=predicted_harmfulness_scores)

            if args.flag_strategy == "likes":
                if "like_cnt" not in filtered_calibration_data.columns:
                    raise KeyError("like_cnt column required for --flag-strategy likes")
                like_cnt_calibration = filtered_calibration_data["like_cnt"].to_numpy(dtype=float)

            if args.flag_strategy in ("firstn", "lastn") and args.topk_n > k:
                raise ValueError(f"--topk-n must be between 0 and k (k={k}). Got n={args.topk_n}.")

            if adversarial_users_set:
                ground_truth_calibration_harmfulness = ground_truth_calibration_harmfulness.copy()
                calib_tags = filtered_calibration_data["tags"].astype(str).values
                if args.flag_strategy == "tag":
                    target_tokens = set(str(TARGET_TAG).split(","))
                    adv_users = np.fromiter(adversarial_users_set, dtype=user_ids_calibrations.dtype)
                    for u in adv_users:
                        idx = np.where(user_ids_calibrations == u)[0]
                        ground_truth_calibration_harmfulness[idx] = 0
                        for j in idx:
                            item_tokens = set(str(calib_tags[j]).split(","))
                            if target_tokens.issubset(item_tokens):
                                ground_truth_calibration_harmfulness[j] = 1
                elif args.flag_strategy == "random":
                    adv_users = np.fromiter(adversarial_users_set, dtype=user_ids_calibrations.dtype)
                    for u in adv_users:
                        idx = np.where(user_ids_calibrations == u)[0]
                        m = int(np.ceil(args.random_flag_pct * idx.size))
                        ground_truth_calibration_harmfulness[idx] = 0
                        if m > 0:
                            chosen = np.random.choice(idx, size=min(m, idx.size), replace=False)
                            ground_truth_calibration_harmfulness[chosen] = 1
                elif args.flag_strategy == "likes":
                    adv_users = np.fromiter(adversarial_users_set, dtype=user_ids_calibrations.dtype)
                    for u in adv_users:
                        idx = np.where(user_ids_calibrations == u)[0]
                        ground_truth_calibration_harmfulness[idx] = 0
                        likes_u = np.maximum(like_cnt_calibration[idx], 0.0).astype(float)
                        thr = np.quantile(likes_u, 1.0 - args.random_flag_pct)
                        chosen = idx[likes_u >= thr]
                        ground_truth_calibration_harmfulness[chosen] = 1
                elif args.flag_strategy in {"firstn", "lastn", "optimal", "top_ranker_q1", "low_risk_q1", "high_risk_q1"}:
                    ranker_scores_calibration, _, _, _, _, _ = obtain_prediction_from_precomputed(filtered_calibration_data, precomputed_scores=predicted_watch_time)
                    if args.score_type == "harm":
                        risk_scores_calibration = harmfulness_rating_test_calibration
                    elif args.score_type == "naive":
                        risk_scores_calibration, _, _, _, _, _ = obtain_prediction_from_precomputed(filtered_calibration_data, precomputed_scores=predicted_watch_time)
                    elif args.score_type == "globalharm":
                        risk_scores_calibration = np.array([global_scores_items.get(item_id, 0.0) for item_id in item_ids_calibration], dtype=float)
                    else:
                        raise ValueError(f"Unsupported score_type for {args.flag_strategy}: {args.score_type}")

                    adv_users = np.fromiter(adversarial_users_set, dtype=user_ids_calibrations.dtype)
                    for u in adv_users:
                        idx = np.where(user_ids_calibrations == u)[0]
                        if idx.size == 0:
                            continue
                        ground_truth_calibration_harmfulness[idx] = 0
                        rk_scores = ranker_scores_calibration[idx]
                        rs_scores = risk_scores_calibration[idx]
                        order = np.argsort(-rk_scores)
                        topk_idx = idx[order[: min(k, order.size)]]
                        if args.flag_strategy == "firstn":
                            chosen = topk_idx[: min(int(args.topk_n), topk_idx.size)]
                        elif args.flag_strategy == "lastn":
                            chosen = topk_idx[-min(int(args.topk_n), topk_idx.size):]
                        else:
                            rk_thr = np.quantile(rk_scores, 1.0 - args.random_flag_pct)
                            high_rank_mask = rk_scores >= rk_thr
                            low_risk_thr = np.quantile(rs_scores, 1.0 - args.random_flag_pct)
                            low_risk_mask = rs_scores >= low_risk_thr
                            high_risk_thr = np.quantile(rs_scores, args.random_flag_pct)
                            high_risk_mask = rs_scores < high_risk_thr
                            if args.flag_strategy == "optimal":
                                chosen = idx[high_rank_mask & low_risk_mask]
                            elif args.flag_strategy == "top_ranker_q1":
                                chosen = idx[high_rank_mask]
                            elif args.flag_strategy == "low_risk_q1":
                                chosen = idx[low_risk_mask]
                            elif args.flag_strategy == "high_risk_q1":
                                chosen = idx[high_risk_mask]
                            else:
                                raise RuntimeError("Unreachable")
                        ground_truth_calibration_harmfulness[chosen] = 1

            naive_scores_calibration, _, _, _, _, _ = obtain_prediction_from_precomputed(filtered_calibration_data, precomputed_scores=predicted_watch_time)
            pred_ratings_safe, _, _, user_ids_safe, item_ids_safe, _, feedback_second_time_watching, true_ratings_safe = obtain_prediction_from_precomputed(filtered_safe_data, precomputed_scores=predicted_watch_time, get_second_view=True)
            pred_ratings, true_ratings, feedbacks, user_ids, item_ids, _ = obtain_prediction_from_precomputed(filtered_data, precomputed_scores=predicted_watch_time)

            if args.accuracy_of_ranker != -1:
                harmfulness_rating_test = synthetic_classifier_predictions_random(1 - ground_truth_test_harmfulness, args.accuracy_of_ranker, 10000)
                harmfulness_rating_test_calibration = synthetic_classifier_predictions_random(1 - ground_truth_calibration_harmfulness, args.accuracy_of_ranker, 10000)

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
                eval_user_ids = np.setdiff1d(eval_user_ids, np.fromiter(adversarial_users_set, dtype=eval_user_ids.dtype), assume_unique=False)
            user_id_chunks = np.array_split(eval_user_ids, num_workers)

            results_model_filtered = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        process_user_only_model,
                        chunk,
                        user_ids,
                        item_ids,
                        pred_ratings,
                        true_ratings,
                        feedbacks,
                        k,
                        True,
                    )
                    for chunk in user_id_chunks
                ]
                for future in futures:
                    results_model_filtered.extend(future.result())

            size_of_recommendation, loss, ndcg_no_gamma, recall_no_gamma, recommended_items_ids_model = zip(*results_model_filtered)
            size_of_recommendation_model = np.mean(size_of_recommendation)
            loss_model = np.mean(loss)
            ndcg_model = np.mean(ndcg_no_gamma)
            recall_model = np.mean(recall_no_gamma)

            model_tag_avg = {}
            num_eval_users_model = len(recommended_items_ids_model) if "recommended_items_ids_model" in locals() else 0
            if num_eval_users_model > 0:
                tag_sum_freq_model = defaultdict(float)
                for rec_items in recommended_items_ids_model:
                    per_user_counts = Counter()
                    for _item in rec_items:
                        tag_str = item_id_to_tags.get(int(_item), "")
                        for _t in _parse_tags(tag_str):
                            per_user_counts[_t] += 1
                    for _t, _cnt in per_user_counts.items():
                        tag_sum_freq_model[_t] += _cnt / float(k)
                model_tag_avg = {_t: _sum / float(num_eval_users_model) for _t, _sum in tag_sum_freq_model.items()}

            base_harm = loss_model
            print("[**] Base harmfulness: ", base_harm)
            if args.dataset == "movielens":
                base_harm = args.base_harm

            if args.score_type == "globalharm":
                predictions_calibration = global_harmfulness_scores_calibration
                scorer = global_harm_scorer
                min_score = 0.0
                max_score = 1.0
            elif args.score_type == "harm":
                predictions_calibration = harmfulness_rating_test_calibration
                scorer = harm_scorer
                min_score = harmfulness_rating_test_calibration.min()
                max_score = harmfulness_rating_test_calibration.max()
            elif args.score_type == "naive":
                predictions_calibration = naive_scores_calibration
                scorer = naive_scorer
                min_score = 0.0
                max_score = float(naive_scores_calibration.max())
            else:
                raise ValueError(f"Unknown score_type={args.score_type}")

            calibration_per_user = calibrate_gamma_per_user_precomputed(
                user_ids=user_ids_calibrations,
                items_ids=item_ids_calibration,
                predictions=predictions_calibration,
                feedbacks=ground_truth_calibration_harmfulness,
                scoring_method=scorer,
                k=k,
                min_score=min_score,
                max_score=max_score,
                score=args.score_type,
                method=args.method,
                num_gammas=100,
            )

            if hard_users_idx is not None:
                candidate_ids = filtered_data[filtered_data.user_id.isin(hard_users_idx)].user_id.unique()
            else:
                candidate_ids = np.unique(user_ids)

            if adversarial_users_set:
                candidate_ids = np.setdiff1d(candidate_ids, np.fromiter(adversarial_users_set, dtype=candidate_ids.dtype), assume_unique=False)

            candidate_ids = np.array([u for u in candidate_ids if int(u) in calibration_per_user], dtype=int)
            if len(candidate_ids) == 0:
                print("[*] No candidate users left after per-user calibration filtering.")
                continue

            num_workers = min(args.cores, os.cpu_count())
            user_id_chunks = np.array_split(candidate_ids, num_workers)

            for alpha in tqdm(np.linspace(0.0, base_harm, num=25)[::-1], desc="Run alphas"):
                gamma_for_alpha = {int(u): gamma_for_alpha_from_user_grid(calibration_per_user[int(u)], alpha) for u in candidate_ids}
                gamma_values = np.array(list(gamma_for_alpha.values()), dtype=float)

                for u in candidate_ids:
                    user_lambda_results.append([
                        run_id,
                        args.epoch,
                        args.base_harm,
                        args.beta,
                        args.score_type,
                        args.method,
                        int(u),
                        alpha,
                        float(gamma_for_alpha[int(u)]),
                        k,
                        args.min_user_videos,
                        len(calibration_per_user[int(u)]),
                    ])

                if alpha == 0.0:
                    full_evaluation_results.append([
                        run_id, args.epoch, args.base_harm, args.beta,
                        args.score_type, args.method,
                        -1.0, alpha, k,
                        ndcg_model, loss_model, size_of_recommendation_model,
                        0, recall_model, 0,
                        "ConformalPerUser", args.collective,
                        args.flag_strategy, args.random_flag_pct, args.likes_temp, args.topk_n,
                        args.min_user_videos, float(np.mean(gamma_values)), float(np.min(gamma_values)), float(np.max(gamma_values)), len(candidate_ids)
                    ])
                    if model_tag_avg:
                        for _t, _avg in model_tag_avg.items():
                            tag_frequency_results.append([
                                run_id, args.epoch, args.base_harm, args.beta,
                                args.score_type, args.method,
                                -1.0, alpha, k,
                                "ConformalPerUser", args.collective,
                                args.flag_strategy, args.random_flag_pct, args.likes_temp, args.topk_n,
                                args.min_user_videos, _t, _avg
                            ])

                results = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(
                            process_user_per_user_gamma,
                            chunk,
                            user_ids,
                            item_ids,
                            pred_ratings,
                            true_ratings,
                            feedbacks,
                            k,
                            scorer,
                            args.score_type,
                            gamma_for_alpha,
                            args.method,
                            feedback_second_time_watching,
                            user_ids_safe,
                            True,
                        )
                        for chunk in user_id_chunks
                    ]
                    for future in futures:
                        results.extend(future.result())

                size_of_recommendation, loss, ndcg_with_gamma, num_items_replaced, recall_with_gamma, item_used_for_this_user, recommended_items_ids = zip(*results)
                loss = np.mean(loss)
                ndcg_with_gamma = np.mean(ndcg_with_gamma)
                size_of_recommendation = np.mean(size_of_recommendation)
                num_items_replaced = np.mean(num_items_replaced)
                recall_with_gamma = np.mean(recall_with_gamma)
                item_used_for_this_user = np.mean(item_used_for_this_user)

                num_eval_users = len(recommended_items_ids)
                if num_eval_users > 0:
                    tag_sum_freq = defaultdict(float)
                    for rec_items in recommended_items_ids:
                        per_user_counts = Counter()
                        for _item in rec_items:
                            tag_str = item_id_to_tags.get(int(_item), "")
                            for _t in _parse_tags(tag_str):
                                per_user_counts[_t] += 1
                        for _t, _cnt in per_user_counts.items():
                            tag_sum_freq[_t] += _cnt / float(k)

                    for _t, _sum in tag_sum_freq.items():
                        tag_frequency_results.append([
                            run_id, args.epoch, args.base_harm, args.beta,
                            args.score_type, args.method,
                            float(np.mean(gamma_values)), alpha, k,
                            "ConformalPerUser", args.collective,
                            args.flag_strategy, args.random_flag_pct, args.likes_temp, args.topk_n,
                            args.min_user_videos, _t, _sum / float(num_eval_users)
                        ])

                full_evaluation_results.append([
                    run_id, args.epoch, args.base_harm, args.beta,
                    args.score_type, args.method,
                    float(np.mean(gamma_values)), alpha, k,
                    ndcg_with_gamma, loss, size_of_recommendation,
                    num_items_replaced, recall_with_gamma, item_used_for_this_user,
                    "ConformalPerUser", args.collective,
                    args.flag_strategy, args.random_flag_pct, args.likes_temp, args.topk_n,
                    args.min_user_videos, float(np.mean(gamma_values)), float(np.min(gamma_values)), float(np.max(gamma_values)), len(candidate_ids)
                ])

        del train_data

    if len(full_evaluation_results) > 0:
        df = pd.DataFrame(
            full_evaluation_results,
            columns=[
                "run_id", "epoch", "base_harm", "beta",
                "conformal_score", "conformal_method",
                "gamma", "alpha", "k",
                "nDCG @ k", "H(S,X)", "|S|",
                "random_items", "Recall @ k", "items_exhaustes",
                "Method", "Collective", "Report Strategy", "Report Fraction", "likes_softmax_temperature", "topk_n",
                "min_user_videos", "avg_user_lambda", "min_user_lambda", "max_user_lambda", "num_calibrated_users",
            ],
        )
        df.to_csv(
            f"results_per_user_{args.dataset}_{args.method}_{args.score_model}_{args.score_type}_{args.runs}_{args.beta}_{args.base_harm}_{args.epoch}_{args.use_single_stage_ranker}_{args.users}_{args.collective}_{args.flag_strategy}_{args.random_flag_pct}_{args.likes_temp}_{args.topk_n}_{args.min_user_videos}_{TARGET_TAG}.csv",
            index=None,
        )

    if len(tag_frequency_results) > 0:
        tag_df = pd.DataFrame(
            tag_frequency_results,
            columns=[
                "run_id", "epoch", "base_harm", "beta",
                "conformal_score", "conformal_method",
                "gamma", "alpha", "k",
                "Method", "Collective", "Report Strategy", "Report Fraction", "likes_softmax_temperature", "topk_n",
                "min_user_videos", "tag", "avg_frequency_in_topk",
            ],
        )
        tag_df.to_csv(
            f"tagfreq_per_user_{args.dataset}_{args.method}_{args.score_model}_{args.score_type}_{args.runs}_{args.beta}_{args.base_harm}_{args.epoch}_{args.use_single_stage_ranker}_{args.users}_{args.collective}_{args.flag_strategy}_{args.random_flag_pct}_{args.likes_temp}_{args.topk_n}_{args.min_user_videos}_{TARGET_TAG}.csv",
            index=None,
        )

    if len(user_lambda_results) > 0:
        user_lambda_df = pd.DataFrame(
            user_lambda_results,
            columns=[
                "run_id", "epoch", "base_harm", "beta",
                "score_type", "conformal_method", "user_id", "alpha", "user_lambda", "k", "min_user_videos", "num_gamma_candidates",
            ],
        )
        user_lambda_df.to_csv(
            f"user_lambdas_{args.dataset}_{args.method}_{args.score_model}_{args.score_type}_{args.runs}_{args.beta}_{args.base_harm}_{args.epoch}_{args.use_single_stage_ranker}_{args.users}_{args.collective}_{args.flag_strategy}_{args.random_flag_pct}_{args.likes_temp}_{args.topk_n}_{args.min_user_videos}_{TARGET_TAG}.csv",
            index=False,
        )
