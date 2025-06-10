from model import *
from cosine_distance import *

import numpy as np
import torch
import torch.nn as nn

import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_evaluations(users_test, final_users_recommendations):
    threshold = 4

    hit_ratios, precisions, recalls = [], [], []

    grouped = users_test.groupby("user")
    for user, group in grouped:
        actual_ratings = group["rating"].to_numpy()
        positive = group["item"][actual_ratings >= threshold].to_numpy()
        try:
            recommended = list(set(final_users_recommendations[user]))
        except:
            pass

        positive_recommended = set(positive).intersection(set(recommended))

        if not len(recommended):
            precision = 0
        else:
            precision = len(positive_recommended) / len(recommended)

        if not len(positive):
            recall = 0
        else:
            recall = len(positive_recommended) / len(positive)

        if len(positive_recommended) > 0:
            hit_ratio = 1
        else:
            hit_ratio = 0

        hit_ratios.append(hit_ratio)
        precisions.append(precision)
        recalls.append(recall)

    return hit_ratios, precisions, recalls

def acc_metric(model,test_data,k = 10,verbose=False):

    test_data = pd.DataFrame(test_data, columns=["user", "item", "rating"])
    users = test_data["user"].to_numpy()
    items = test_data["item"].to_numpy()
    ratings = test_data["rating"].to_numpy()

    preds = model(torch.tensor(users, device=device, dtype=torch.int), torch.tensor(items, device=device, dtype=torch.int))
    # print(preds)
    mse = np.square(preds.cpu().detach().numpy() - ratings).mean()
    # print("mse",mse)
    test_data["preds"] = preds.cpu().detach().numpy()

    sorted_test_data = test_data.sort_values(by=["preds"], ascending=False)

    grouped = sorted_test_data.groupby("user")

    final_users_recommendations = []
    for user, grouped in grouped:
        items = grouped["item"][:k]
        final_users_recommendations.append(items)

    hit_ratios, precisions, recalls = compute_evaluations(test_data, final_users_recommendations)
    if verbose:
        print(f"MSE:", mse)
        print(f"HR@{k}: {np.mean(hit_ratios)}")
        print(f"Precision@{k}: {np.mean(precisions)}")
        print(f"Recall@{k}: {np.mean(recalls)}")
    
    return {"mse":mse, "hr":np.mean(hit_ratios), "precision":np.mean(precisions),"recall":np.mean(recalls)}
    
def average_genres_covered(item_genre_matrix, recommendations,k):
    # n_users = len(final_users_recommendations_items)
    if device =="cuda":
        recommendations = recommendations.cpu().numpy()
    else:
        recommendations = recommendations.numpy()
    genre_counts = []
    for user_recs in recommendations:
        # Get the genres of all recommended items
        user_recs = np.asarray(user_recs)
        rec_genres = item_genre_matrix[user_recs]  # shape (t, k)
        # OR over the rows to see which genres are covered
        genres_covered = np.any(rec_genres, axis=0)
        # Count how many genres are covered
        count = np.sum(genres_covered)
        genre_counts.append(count)
    average_covered = np.mean(genre_counts)
    return average_covered
    
def mask_topk_to_zero(R, k):
    # R: [num_users, num_items] rating matrix
    topk_indices = torch.topk(R, k=k, dim=1).indices  # [num_users, k]

    # Create a copy to avoid in-place modification
    masked_R = R.clone()

    # Prepare the row indices
    row_indices = torch.arange(R.size(0)).unsqueeze(1).expand(-1, k)  # [num_users, k]

    # Set top-k values to 0
    masked_R[row_indices, topk_indices] = 0

    return masked_R

def weighted_avg_pairwise_jaccard_distance(M, R, eps=1e-8):
    R = R.float().to(device)  # Cast to float32
    M = M.float().to(device)
    RM = torch.matmul(R, M)
    numerator = (RM * R).sum(dim=1)
    w_sum = R.sum(dim=1)
    w_sq_sum = (R ** 2).sum(dim=1)
    denominator = w_sum**2 - w_sq_sum
    avg_dist = numerator / (denominator + eps)
    avg_dist[denominator == 0] = 0
    return avg_dist

def retain_top_k(matrix,top_k):

    _, top_indices = torch.topk(matrix, top_k, dim=1)
    result = torch.zeros_like(matrix, dtype=torch.int32, device=device)
    result.scatter_(1, top_indices, 1)
    return result

def diversit_metric(model, dist_mat,item_genres_mat,num_users,num_items,data, topk=10,soft_tau=0.00001,verbose=False,mask=True, diversity_target="topk-mat",R=None,out_of_target_topk=False):
    diversity_func = CosineDistanceDiversity(top_k=topk,sort_tau=soft_tau)

    if R==None:
        user_indices = torch.arange(num_users, device=device).unsqueeze(1).repeat(1, num_items).view(-1)
        item_indices = torch.arange(num_items, device=device).repeat(num_users)
        predictions = model(user_indices, item_indices)  # Shape: [num_users * num_items]
        R = predictions.view(num_users, num_items)

    if mask:
        if diversity_target == "topk-val":
            # delete ratings not in data, for example, diversify recommendation only in validation datasets.
            mask = torch.zeros((num_users, num_items), dtype=torch.float).to(device)
            for u, i, _ in data:
                mask[u, i] = 1
        elif diversity_target== "topk-mat":
            # delete ratings in data.
            mask = torch.ones((num_users, num_items), dtype=torch.float).to(device)
            for u, i, _ in data:
                mask[u, i] = 0
    
        R  = mask*R

    if out_of_target_topk == True:
        R = mask_topk_to_zero(R,topk)

    real_ranks_top_k = retain_top_k(R,topk)
    real_distance  = weighted_avg_pairwise_jaccard_distance(dist_mat, real_ranks_top_k)
    acc_diversity_score = torch.mean(real_distance)

    
    # approx_div,true_div = diversity_func(R,dist_mat)
    # approx_diversity_score = torch.mean(approx_div)
    # acc_diversity_score = torch.mean(true_div)

    _, recommendations = torch.topk(R, topk, dim=1)
    if item_genres_mat is None:
        average_covered_genres = None
    else:
        average_covered_genres = average_genres_covered(item_genres_mat, recommendations,k=topk)
    pop_bias = popularity_bias(num_users, num_items, data,recommendations)


    if verbose:
        k=topk
        print(f"HR@{k}: {np.mean(hit_ratios)}")
        print(f"Precision@{k}: {np.mean(precisions)}")
        print(f"Recall@{k}: {np.mean(recalls)}")
        print("Jaccard distance: ","approx_div:",approx_diversity_score.item()," acc_div:",acc_diversity_score.item() )
        print(f"Avg_genres@{k}:{average_covered_genres}")

    return { "approx_diversity":None, "acc_diversity":acc_diversity_score.item(), "genres":average_covered_genres,"pop_bias":pop_bias}


def popularity_bias(num_users, num_items, train_data,recommendations):
    item_popularity = np.zeros(num_items)
    for _,item,_ in train_data:
        item_popularity[item] += 1
    
    arp = 0
    
    for recs in recommendations:
        # recs = np.asarray(recs)
     
        avg_pop = sum(item_popularity[i] for i in recs) / len(recs)
        arp += avg_pop
    
    return arp / num_users