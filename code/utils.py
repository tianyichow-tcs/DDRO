
import numpy as np
import torch
from itertools import combinations, permutations
import  pandas._libs.lib as lib


def expectation_value_MHP(permutations, distance_matrix, items_continue_p):
    matrix_p = items_continue_p[permutations]  # size is num of permutations * k_param

    accumulate_p = np.cumprod(matrix_p, axis=1)[:, 1:]  # size is num of permutations * k_param-1

    # wi = p1p2..p_{i+1} + ...+ p1p2..p_n
    weights = np.cumsum(accumulate_p[:, ::-1], axis=1)[:, ::-1]

    indices = (permutations[:, :-1], permutations[:, 1:])

    # size is num of permutations * k_param-1
    distances = distance_matrix[indices]

    expectations = np.sum(weights * distances, axis=1, keepdims=True)

    return expectations

def expectation_MHP_incremental(permutations, distance_matrix, items_continue_p):
    expectations = 0

    accu_p= 0
    accu_dist = 0

    for idx in range(1, permutations.shape[1]):

        if idx == 1:  # start of incrementation
            ######################### incrementally calculate MHP expectation ################################
            top2 = permutations[:,:2]
            matrix_p = items_continue_p[top2]

            accu_p = np.multiply(matrix_p[:,0], matrix_p[:,1])

            indices = (permutations[:,0], permutations[:,1])
            # size is num of permutations * k_param-1
            accu_dist = distance_matrix[indices]

            expectations = np.multiply(accu_p , accu_dist) # num_permu * 1

            ########################################################################################

        else:  # when seq_len is >=3, do incrementation

            new_item = permutations[:,idx]
            last_item_p = items_continue_p[new_item]
            accu_p = np.multiply(last_item_p, accu_p)

            indices = (permutations[:,idx-1], permutations[:,idx])
            increment_dist = distance_matrix[indices]
            accu_dist += increment_dist

            increment_exp = np.multiply(accu_p, accu_dist)
            expectations += increment_exp

    return expectations

def separate_to_chunks(lst, k):
    chunk_size = len(lst) // k
    remainder = len(lst) % k

    start_index = 0
    end_index = 0
    chunk_indices = []
    for i in range(k):
        if i < remainder:
            end_index += chunk_size + 1
        else:
            end_index += chunk_size

        if end_index > len(lst):
            end_index = len(lst)

        chunk_indices.append((start_index, end_index))
        start_index = end_index

    return chunk_indices

def expectation_MHP_incremental_chunk(permutations, distance_matrix, items_continue_p, num_thred=5000):

    def chunk_computation(permutations_chunk):
        expectations = 0

        accu_p = 0
        accu_dist = 0

        for idx in range(1, permutations_chunk.shape[1]):

            if idx == 1:  # start of incrementation
                ######################### incrementally calculate MHP expectation ################################
                top2 = permutations_chunk[:,:2]
                matrix_p = items_continue_p[top2]

                accu_p = np.multiply(matrix_p[:, 0], matrix_p[:, 1])

                indices = (permutations_chunk[:, 0], permutations_chunk[:, 1])
                # size is num of permutations_chunk * k_param-1
                accu_dist = distance_matrix[indices]

                expectations = np.multiply(accu_p, accu_dist)  # num_permu * 1

                ########################################################################################

            else:  # when seq_len is >=3, do incrementation

                new_item = permutations_chunk[:, idx]
                last_item_p = items_continue_p[new_item]
                accu_p = np.multiply(last_item_p, accu_p)

                indices = (permutations_chunk[:, idx - 1], permutations_chunk[:, idx])
                increment_dist = distance_matrix[indices]
                accu_dist += increment_dist

                increment_exp = np.multiply(accu_p, accu_dist)
                expectations += increment_exp

        return expectations


    n_thread = num_thred
    num_permu = permutations.shape[0]
    chunk_range = separate_to_chunks(np.arange(num_permu), n_thread)

    results = []
    for i in range(len(chunk_range)):
        chunk = chunk_range[i]
        chunk_permutations = permutations[chunk[0]:chunk[1]]
        result = chunk_computation(chunk_permutations)

        results.append(result)

    expectation_concat = np.concatenate(results)

    assert len(expectation_concat) == permutations.shape[0]

    return expectation_concat

def generate_combinations_and_permutations(all_items, k_param):
    result = []
    # print ('items before permutation', all_items)
    for c in combinations(all_items, k_param):
        result.extend(permutations(c))

    all_permutations_array =  lib.to_object_array(result).astype(int)
    return all_permutations_array

def expectation_value_MSD_torch(permutations, distance_matrix, items_continue_p):

    permutations = torch.tensor(permutations)
    permutations = permutations.to(torch.int64)
    distance_matrix = torch.tensor(distance_matrix)
    items_continue_p = torch.tensor(items_continue_p)


    num_permutation, num_items = permutations.shape
    matrix_p = items_continue_p.gather(1, permutations)  # size is num of permutations * k_param

    # [p1p2, p1p2p3, p1p2p3p4, \cdots, p1...p_n]
    double_matrix_p = matrix_p.double()
    accumulate_p = torch.cumprod(double_matrix_p, dim=1)[:, 1:]  # size is num of permutations * k_param-1

    # [d(1,2), d(3,{1,2}), d(4,{1,2,3}), ...d(n, {1,2,...,n-1})], the i-th element is d(i+1, {1,2,...i})
    distances = torch.zeros((num_permutation, num_items-1), dtype=torch.double)
    for i in range(1, num_items):
        new_node = permutations[:, i]
        existing_nodes = permutations[:, :i]

        matrix_row_slices = distance_matrix[new_node]
        matrix_column_slices = matrix_row_slices.gather(1, existing_nodes)

        distance_new2exist = torch.sum(matrix_column_slices, dim=1)
        distances[:, i-1] = distance_new2exist

    expectations = torch.einsum('ij,ij->i', accumulate_p, distances)

    return expectations

def expectation_value_MSD_incremental(permutations, distance_matrix, items_continue_p):
    num_permutation, num_items = permutations.shape

    # Initialize results and accumulate initial probabilities
    first_2_p = items_continue_p[permutations[:, 0]] * items_continue_p[permutations[:, 1]]
    dist = distance_matrix[permutations[:, 0], permutations[:, 1]]
    expectation = first_2_p * dist

    # For storing accumulated probabilities
    accum_p = first_2_p

    for idx in range(2, num_items):
        new_item = permutations[:, idx]
        new_p = items_continue_p[new_item]
        accum_p *= new_p  # Update the accumulated probabilities

        # Compute the sum of distances from the new item to all previously considered items
        dist_increment = np.zeros(num_permutation)
        for prev_idx in range(idx):  # Loop over all previous items
            prev_item = permutations[:, prev_idx]
            dist_increment += distance_matrix[new_item, prev_item]

        # Update expectation
        expectation += accum_p * dist_increment

    return expectation

def greedy_msd_vectorized(items_continue_p_, distance_matrix_, lambda_factor):
    n_items = distance_matrix_.shape[0]
    U = set(np.arange(n_items))
    S = []
    ss = np.zeros(n_items)

    phi_u_S = items_continue_p_
    best_u = np.argmax(phi_u_S)
    S.append(best_u)
    U -= {best_u}
    items_continue_p_[best_u] = 0
    distance_matrix_[best_u, :] = np.zeros(n_items)

    while len(U) > 0:
        ss = ss + np.squeeze(distance_matrix_[:,best_u])
        ss[best_u] = 0
        phi_u_S = items_continue_p_/2 + lambda_factor * ss
        best_u = np.argmax(phi_u_S)
        S.append(best_u)
        U -= {best_u}

        # UPDATE THE P AND D SO YOU DONT CHOSE THE PREVIOUS ITEMS AGAIN
        items_continue_p_[best_u] = 0
        distance_matrix_[best_u,:] = np.zeros(n_items)

    return S


def expectation_value_MSD(permutations, distance_matrix, items_continue_p):
    # permutations  is a 1d array
    # item_continue_p is a 1d array

    num_items  = len(permutations)
    matrix_p = items_continue_p[permutations]  # size is 1 * k_param

    # [p1p2, p1p2p3, p1p2p3p4, \cdots, p1...p_n]
    accumulate_p = np.cumprod(matrix_p)[1:]  # size is 1 * k_param-1

    # [d(1,2), d(3,{1,2}), d(4,{1,2,3}), ...d(n, {1,2,...,n-1})], the i-th element is d(i+1, {1,2,...i})
    distances = np.zeros(num_items-1)
    for i in range(1, num_items):
        new_node = permutations[i]
        existing_nodes = permutations[:i]

        matrix_row_slices = distance_matrix[new_node][existing_nodes]

        distance_new2exist = np.sum(matrix_row_slices)
        distances[i-1] = distance_new2exist

    expectations = np.dot(accumulate_p, distances)

    return expectations # a scalar


def save_ranking_2file(folderpath, dataset_name, k_param, regime, ranking, strategy):
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = f'{abs_dir}/{folderpath}/'
    ensure_folder_exists(filepath)
    filename = filepath + dataset_name + '_'+ str(k_param) + '_'+ regime + '_'+ strategy + '.txt'
    with open(filename, 'a+') as writer:
        for row in ranking:
            row_string = ','.join(map(str, row))
            writer.write(row_string + '\n')


def save_qid_ranking_2file(folderpath, dataset_name, k_param, regime, ranking_list, strategy):
    abs_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = f'{abs_dir}/{folderpath}/'
    ensure_folder_exists(filepath)
    filename = filepath + dataset_name + '_'+ str(k_param) + '_'+ regime + f'_{strategy}.txt'
    with open(filename, 'a+') as writer:
        for (qid, ranking) in ranking_list:
            row_string = ','.join(map(str, ranking))
            writer.write(str(qid) + ' ' +  row_string + '\n')




def map2range(dataset_name, regime, ratings, regimes_mapping_vocab):
    mapping_range = regimes_mapping_vocab[regime]
    if dataset_name in  {"movielens", "KuaiRec", "coat", "yahoo", "netflix"}:
        if regime in ["small", "medium", "large"]:
            rating_mapped = np.interp(ratings, (1, 5), mapping_range)
        elif regime == "full":
            rating_mapped = np.interp(ratings, (1, 5), [0.1, 0.9])
        else:
            raise ValueError(f"Invalid regime: {regime}")

    elif dataset_name in ['LETOR']: # 0 1 2 mapped to 0.3, 0.5, 0.7
        if regime in ["small", "medium", "large"]:
            rating_mapped = np.interp(ratings, (0, 2), mapping_range)
        elif regime == "full":
            rating_mapped = one2one_mapping(ratings, dataset_name)

    elif dataset_name in ["LTRC", "LTRCB"]:
        if regime in ["small", "medium", "large"]:
            rating_mapped = np.interp(ratings, (0, 4), mapping_range)
        elif regime == "full":
            rating_mapped = one2one_mapping(ratings, dataset_name)

    return rating_mapped


def one2one_mapping(matrix_before_mapping, dataset_name):
    if len(matrix_before_mapping.shape) == 1: # input is array
        matrix_before_mapping = matrix_before_mapping.reshape(1,-1)

    matrix_after_mapping = np.zeros((matrix_before_mapping.shape[0], matrix_before_mapping.shape[1]))
    if dataset_name == 'LETOR': # 012 map to .3 .5 .7
        for i in range(matrix_before_mapping.shape[0]):
            for j in range(matrix_before_mapping.shape[1]):
                ele = matrix_before_mapping[i, j]
                if ele == 0:
                    matrix_after_mapping[i, j] = 0.3
                elif ele == 1:
                    matrix_after_mapping[i, j] = 0.5
                elif ele == 2:
                    matrix_after_mapping[i, j] = 0.7
                else:
                    raise ValueError('Unexpected element {}'.format(ele))
    elif dataset_name in ['LTRC', 'LTRCB']: # 01234 to .1 .3 .5 .7 .9
        for i in range(matrix_before_mapping.shape[0]):
            for j in range(matrix_before_mapping.shape[1]):
                ele = matrix_before_mapping[i, j]
                if ele == 0:
                    matrix_after_mapping[i, j] = 0.1
                elif ele == 1:
                    matrix_after_mapping[i, j] = 0.3
                elif ele == 2:
                    matrix_after_mapping[i, j] = 0.5
                elif ele == 3:
                    matrix_after_mapping[i, j] = 0.7
                elif ele == 4:
                    matrix_after_mapping[i,j] = 0.9
                else:
                    raise ValueError('Unexpected element {}'.format(ele))
    else:
        raise ValueError('Unexpected dataset_name')
    if matrix_after_mapping.shape[0] == 1:
        matrix_after_mapping = np.squeeze(matrix_after_mapping)
    return matrix_after_mapping


def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        pass


# expected number of items that user will check
# number of new items user will explore, compared to given rating matrix.
# ohhhhh old raw data needed...large amount of work...

import os

def expect_num_acceptance(dataset_name, method_list, regime_list,topk):

    rating_interploted_fp = f'../OMSD/rating_{dataset_name}.npy'
    rating_interploted = np.load(rating_interploted_fp)
    n_users, n_items = rating_interploted.shape

    def single_acceptance(ranking, item_continue_p, mapping_range):
        item_continue_p = np.interp(item_continue_p, (1,5), mapping_range) # 0.1-0.3 for small regime, 0.4-0.6 for medium and 0.7-0.9 for large
        ranking_prob = item_continue_p[ranking]
        accu_continue_prob = np.cumprod(ranking_prob)
        next_reject_prob = np.ones(len(ranking_prob))
        next_reject_prob[:-1] = 1-ranking_prob[1:]

        # prob of accepting 1,2,3,...items
        acc_num_prob = np.multiply(accu_continue_prob, next_reject_prob)
        # accepting 1,2,3...items
        acc_num_arr = np.arange(len(ranking))

        acc_num_exp = np.sum(np.multiply(acc_num_prob, acc_num_arr))
        return acc_num_exp



def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted.")
    else:
        print(f"File '{file_path}' does not exist.")

