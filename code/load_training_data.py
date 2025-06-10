from cosine_distance import *
from model import *
from algo import *

from tqdm import tqdm
import os
import sklearn
import sklearn.model_selection
import torch.utils.data as data
from utility import get_ratings, get_dictionaries,get_structures
from visualization import plot_loss
from results import compute_evaluations
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
from collections import defaultdict
from typing import List
device = "cuda" if torch.cuda.is_available() else "cpu"




def lode_datasets(dataset_name = "movielens",is_finetune=False,device=device,bias_type=None,b=0.5,random_state=0):
    """
       This method is used to get train, validation and test datasets, if the bias_type is not None, we polute the training datsets.
    """
    dataset_folder = os.path.join("RawData/", dataset_name)
    ratings = get_ratings(dataset_name, dataset_folder)

    # mapping userid, item id.
    users_dictionary, items_dictionary = get_dictionaries(ratings, "outputs", dataset_name)
    ratings_dataset = []
    

    print("Load item embedding dataset...")
    if dataset_name == "arxiv":
        item_genres_mat = None
    else:
        item_genres_mat = np.load("outputs/items_genres_matrix_{}.npy".format(dataset_name))
    dist_mat = np.load("outputs/jaccard_genres_distances_{}.npy".format(dataset_name))
    dist_mat = torch.tensor(dist_mat, dtype=torch.float, requires_grad=False)
    # items_genres_matrix = np.load("outputs/items_genres_matrix_{}.npy".format(dataset_name))
    print("Creating ratings dataset...")
    for (user, item, rating) in ratings:
        reindexed_user = users_dictionary[user]
        reindexed_item = items_dictionary[item]

        ratings_dataset.append([reindexed_user, reindexed_item, rating])

    print(f"Ratings Length: {len(ratings_dataset)}")


    if is_finetune:
        batch_sizes_dict = {"movielens": 8096*10, "KuaiRec": 8096*40, "coat": 32*10,
                        "yahoo": 8096*10, "netflix": 8096*10, "arxiv":8096*10}
    else:
        batch_sizes_dict = {"movielens": 8096, "KuaiRec": 8096, "coat": 32,
                        "yahoo": 2048, "netflix": 2048, "arxiv":8096}
        
    batch_size = batch_sizes_dict[dataset_name]

    # ratings_dataset = biasd_ratings_data
    df = pd.DataFrame(ratings_dataset, columns=["user", "item", "rating"])
    train_data, test_data = sklearn.model_selection.train_test_split(ratings_dataset, test_size=0.2,
                                                                     stratify=df["user"],
                                                                     random_state=random_state)
    df =  pd.DataFrame(train_data, columns=["user", "item", "rating"])
    train_data, val_data = sklearn.model_selection.train_test_split(train_data, test_size=0.2,
                                                                     stratify=df["user"],
                                                                     random_state=random_state)
    df_train = pd.DataFrame(train_data, columns=["user", "item", "rating"])
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data))
    n_users, n_items = len(users_dictionary), len(items_dictionary)

    result = {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "n_users": n_users,
        "n_items": n_items,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,

        "dist_mat": dist_mat
        }

    return train_loader, test_loader,val_loader, n_users, n_items,train_data,val_data, test_data, dist_mat,item_genres_mat
    

