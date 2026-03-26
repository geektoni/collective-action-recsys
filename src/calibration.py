import concurrent.futures
import torch
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import gaussian_kde
from tqdm import tqdm


def adaptive_linspace(data, a, b, N):
    """
    Splits the interval [a, b] into N sections based on the density of data points.
    
    Parameters:
        data (array-like): The existing data points within [a, b].
        a (float): Start of the interval.
        b (float): End of the interval.
        N (int): Number of desired intervals.
    
    Returns:
        np.ndarray: The adaptive split points.
    """
    data = np.asarray(data)
    
    # Estimate density using KDE
    kde = gaussian_kde(data, bw_method='scott')  # Bandwidth selection
    x_vals = np.linspace(a, b, 500)  # Fine grid over the range
    density = kde(x_vals)
    
    # Normalize density to get a probability density function
    pdf = density / np.trapz(density, x_vals)  # Normalize with integral
    
    # Compute the cumulative density function (CDF)
    cdf = np.cumsum(pdf) / np.sum(pdf)
    
    # Interpolate to get evenly spaced points in the CDF domain
    uniform_cdf_vals = np.linspace(0, 1, N)
    adaptive_points = np.interp(uniform_cdf_vals, cdf, x_vals)
    
    return adaptive_points

def adaptive_threshold(user_ratings, percentile=75):
    if user_ratings is None:
        return 0  # No ratings available
    return np.percentile(user_ratings, percentile)

def recall(recommended_items, relevant_items, k=20):

    if len(recommended_items) == 0 or len(relevant_items) == 0:
        return 0.0

    recommended_set = set(recommended_items[:k])
    relevant_set = set(relevant_items[:k])
    
    true_positives = len(recommended_set & relevant_set)  # Intersection of recommended and relevant
    total_relevant = len(relevant_set)
    
    if total_relevant == 0:
        return 0.0  # Avoid division by zero
    
    return true_positives / total_relevant

# Compute item x item similarity
def compute_similarity(items):
    return cosine_similarity(items)

# Harmfulness loss used in the experiments.
def harmfulness_loss(feedback, k=None):

    assert k >= 1

    tmp_feedback = feedback[:k]

    if len(tmp_feedback) == 0:
        return 0

    #if k is None or k<len(tmp_feedback):
    #    k = len(tmp_feedback)
    return np.sum(tmp_feedback) / k

def process_gammas(
    gammas_chunk,
    user_ids,
    items_ids,
    predictions,
    feedbacks,
    scoring_method,
    score,
    k,
    method,
    return_per_user=False,
    mc_samples=100,
    mc_fraction=0.7,
    random_state=2025,
):
    results = []

    for gamma in tqdm(gammas_chunk):
        scores = []  # keep original aggregation list
        
        for user_id in np.unique(user_ids):
            
            # Generator
            rng = np.random.default_rng(random_state+user_id)

            mask_ids = user_ids == user_id

            user_predictions = predictions[mask_ids]
            user_feedbacks = feedbacks[mask_ids]
            user_items_ids = items_ids[mask_ids]

            # Keep the mc scores if we need them
            mc_scores = []

            # For each MC sample, we compute the score for the given user
            # In case return_per_user == False, then we just run it once
            for _ in range(mc_samples if return_per_user else 1):
                
                if mc_fraction < 1.0 and return_per_user:
                    subset_size = int(np.ceil(len(user_predictions) * mc_fraction))
                    sampled_idx = rng.choice(len(user_predictions), size=subset_size, replace=False)

                    sampled_predictions = user_predictions[sampled_idx]
                    sampled_feedbacks = user_feedbacks[sampled_idx]
                    sampled_items_ids = user_items_ids[sampled_idx]
                else:
                    sampled_predictions = user_predictions
                    sampled_feedbacks = user_feedbacks
                    sampled_items_ids = user_items_ids
                    sampled_idx=None # In case we do not need them

                if method == "replace" or method == "hybrid":

                    # We replace all items in the mask with random "trusted" items.
                    # In this synthetic case, we replace them with random items picked
                    # from the remaining user items, by ensuring they were not flagged.
                    # This might be akin to provide "trusted" curated videos, or previously
                    # seen videos the user did not flag.
                    new_scores, replaced_items_mask, _, index_replaced_items, _, _ = scoring_method.replace_low_scores(
                        sampled_predictions,
                        user_id,
                        k,
                        gamma,
                        item_ids=sampled_items_ids, # we give only the ids for the current user
                        score=score, # type of scoring function we want
                        is_calibrating=True, # specify we are calibrating (only for harm scorer)
                        sampled_idx=sampled_idx # Specify only a subset of items
                    )
                    num_item_replaced = len(index_replaced_items)
                    
                    # Replace with the correct harmfulness given the feedback
                    # We copy it to avoid issues when re-sampling later on
                    harmful_items = sampled_feedbacks.copy()
                    if len(index_replaced_items) > 0:
                        harmful_items[index_replaced_items] = 0

                    # We might have replaced less items than needed, so we drop
                    # the others if needed by the strategy
                    if method == "hybrid":

                        if num_item_replaced > 0:
                            # We mask all the items we replaced so far, and we
                            # leave the others for removal
                            replaced_items_mask[index_replaced_items] = 0

                        # We get the negative, these now are the elements we want to keep
                        # which are basically replaced items and not touched
                        replaced_items_mask = ~replaced_items_mask
                        
                        # Compute the harmfulness of this recommendation
                        new_scores = np.array(new_scores)[replaced_items_mask]
                        sorted_indices = np.argsort(-new_scores)
                        harmful_items = harmful_items[replaced_items_mask][sorted_indices]
            
                        user_score = harmfulness_loss(harmful_items, k=k)

                    else:                     

                        # We sort the new scores, the harmful items and the replaced items mask
                        sorted_indices = np.argsort(-new_scores)
                        harmful_items = harmful_items[sorted_indices]

                        # We compute the harmfulness only of items we did not replace since we assume
                        # replacing an item does not increase the harmfulness
                        user_score = harmfulness_loss(harmful_items, k=k)

                else:
                    # We sort the indeces based on the top-k prediction
                    # but then we recompute the harmfulness loss based on
                    # the updated score (if we use a different strategy that "simple")
                    sorted_indices = np.argsort(-sampled_predictions)

                    mask = scoring_method.score_items(
                            sampled_predictions,
                            gamma,
                            user_id=user_id,
                            item_ids=sampled_items_ids, # we give only the ids for the current user
                            score=score, # type of scoring function we want,
                            is_calibrating=True, # specify we are calibrating (only for harm scorer),
                            sampled_idx=sampled_idx # Specify only a subset of items
                    )[sorted_indices]
                
                    user_score = harmfulness_loss(feedbacks[mask_ids][sorted_indices][mask], k=k)

                # Save the MC score for this user
                mc_scores.append(
                    float(user_score)
                )

            # NEW: switch behavior
            if return_per_user:
                
                # We apply anyway the finite-sample correction
                N = len(mc_scores)
                corrected_user_score = np.mean(mc_scores)
                corrected_user_score *= N / (N + 1)
                corrected_user_score += 1 / (N + 1)

                results.append((gamma, user_id, float(corrected_user_score)))
            else:
                # Sanity check to avoid issues 
                assert len(mc_scores) == 1, f"We have MC samples while we want to return aggregate scores: len(mc_scores)= {len(mc_scores)}"
                scores.append(user_score)
        
        # We aggregate the results if we do not care about per-user value
        if not return_per_user:
            N = len(scores)
            scores = np.mean(scores)
            scores *= N / (N + 1)
            scores += 1 / (N + 1)
            
            results.append((gamma, scores))
        
    return results


# Calibrate threshold gamma
def calibrate_gamma(model, calibration_loader, alpha, k,
                    scoring_method, max_score=5, min_score=0, score="subtraction", method="remove",
                    cores=4,
                    device="cpu",
                    num_gammas= 100):
    model.eval()

    gammas = list(np.linspace(min_score, max_score, num=num_gammas))
    calibration_evals = []

    user_ids = []
    items_ids = []
    item_features = []
    predictions = []
    feedbacks = []

    with torch.no_grad():
        for user, item, genre, item_feature, binary_feedback in calibration_loader:
            prediction, _ = model.predict(user, item, genre, device=device)

            user_ids.extend(user.numpy())
            predictions.extend(prediction.numpy(force=True))
            feedbacks.extend(binary_feedback.numpy())
            items_ids.extend(item.numpy())
            item_features.extend(item_feature.numpy())
    
    user_ids = np.array(user_ids)
    predictions = np.array(predictions)
    feedbacks = np.array(feedbacks)
    items_ids = np.array(items_ids)
    item_features = np.array(item_features)

    num_workers = min(cores, os.cpu_count())
    gamma_chunks = np.array_split(gammas, num_workers)

    calibration_evals = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        
        gamma_futures = [
            executor.submit(
                process_gammas,
                chunk,
                user_ids,
                items_ids,
                predictions,
                feedbacks,
                scoring_method,
                score,
                k,
                method) for chunk in gamma_chunks]
    
        for future in gamma_futures:
            calibration_evals.extend(future.result())

    for _, (gamma, score) in enumerate(calibration_evals):
        if score <= alpha:
            return gamma
    
    return calibration_evals[-1][0]

def calibrate_gamma_precomputed(
                    user_ids,
                    items_ids,
                    predictions,
                    feedbacks,
                    alpha, k,
                    scoring_method, max_score=5, min_score=0, score="subtraction", method="remove",
                    cores=4,
                    device="cpu",
                    num_gammas= 100,
                    return_per_user=False,
                    mc_samples=100
                    ):

    max_score_grid = np.nextafter(float(max_score), np.inf)
    print(f"[**] Previous max score: {max_score}, new max score: {max_score_grid}")
    gammas = list(np.linspace(min_score, max_score_grid, num=num_gammas))
    calibration_evals = []

    num_workers = min(cores, os.cpu_count())
    gamma_chunks = np.array_split(gammas, num_workers)

    calibration_evals = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        
        gamma_futures = [
            executor.submit(
                process_gammas,
                chunk,
                user_ids,
                items_ids,
                predictions,
                feedbacks,
                scoring_method,
                score,
                k,
                method,
                return_per_user=return_per_user,
                mc_samples=mc_samples) for chunk in gamma_chunks]
    
        for future in gamma_futures:
            calibration_evals.extend(future.result())
    
    # We return the calibration evals directly, if we do not care
    # about per user results.
    if not return_per_user:
        return calibration_evals

    # Build a dictionary to store the gammas and score for each user
    per_user_results = {}
    for gamma, user_id, user_score in calibration_evals:
        if user_id not in per_user_results:
            per_user_results[user_id] = []

        per_user_results[user_id].append(
            (gamma, user_score)
        )

    # For any dictionary, sort the gammas
    for user_id in per_user_results.keys():
        per_user_results[user_id] = sorted(
           per_user_results[user_id],
           key=lambda x: x[0] 
        )

    return per_user_results

def avg_harmfulness_for_users_raw(
    user_ids,
    items_ids,
    predictions,
    feedbacks,
    gamma,
    k,
    scoring_method,
    score,     # args.score_type
    method,    # args.method
    users_subset,
):
    users_subset = np.array(list(users_subset), dtype=user_ids.dtype)
    if users_subset.size == 0:
        return np.nan, 0

    per_user_scores = []
    for user_id in users_subset:
        mask_ids = (user_ids == user_id)
        if not np.any(mask_ids):
            continue

        if method in ("replace", "hybrid"):
            new_scores, replaced_items_mask, _, index_replaced_items, _, _ = scoring_method.replace_low_scores(
                predictions[mask_ids],
                user_id,
                k,
                gamma,
                item_ids=items_ids[mask_ids],
                score=score,
                is_calibrating=True,
            )

            harmful_items = feedbacks[mask_ids].copy()
            if len(index_replaced_items) > 0:
                harmful_items[index_replaced_items] = 0

            if method == "hybrid":
                if len(index_replaced_items) > 0:
                    replaced_items_mask[index_replaced_items] = 0
                replaced_items_mask = ~replaced_items_mask

                new_scores = np.array(new_scores)[replaced_items_mask]
                sorted_indices = np.argsort(-new_scores)
                harmful_items = harmful_items[replaced_items_mask][sorted_indices]
                per_user_scores.append(harmfulness_loss(harmful_items, k=k))
            else:
                sorted_indices = np.argsort(-np.array(new_scores))
                harmful_items = harmful_items[sorted_indices]
                per_user_scores.append(harmfulness_loss(harmful_items, k=k))

        else:
            sorted_indices = np.argsort(-predictions[mask_ids])
            keep_mask = scoring_method.score_items(
                predictions[mask_ids],
                gamma,
                user_id=user_id,
                item_ids=items_ids[mask_ids],
                score=score,
                is_calibrating=True,
            )[sorted_indices]

            per_user_scores.append(
                harmfulness_loss(feedbacks[mask_ids][sorted_indices][keep_mask], k=k)
            )

    if len(per_user_scores) == 0:
        return np.nan, 0

    return float(np.mean(per_user_scores)), int(len(per_user_scores))
