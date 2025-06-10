
import torch
from torch import nn
import torch.nn.functional as F
# from fast_soft_sort.pytorch_ops import soft_rank
import torchsort 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CosineDistanceDiversity(nn.Module):

    def __init__(self, top_k: int =10, sort_tau: float=0.0001, device=device):
        """_summary_
        Args:
            top_k (int, optional): top k items for compute diversity. Defaults to 10.
            sort_tau (float, optional): regulariztion strenth for soft-rank. Defaults to 0.0001.
        """
        super(CosineDistanceDiversity,self).__init__()
        self.top_k = top_k
        self.sort_tau = sort_tau
        self.device = device

    def retain_top_x(self, matrix):
        """_summary_
        Args:
            matrix (_type_): prediction matrix

        Returns:
            torch.tensor: indicator matrix
        """
        _, top_indices = torch.topk(matrix, self.top_k, dim=1)
        result = torch.zeros_like(matrix, dtype=torch.int32, device=self.device)
        result.scatter_(1, top_indices, 1)
        return result
    
    def weighted_avg_pairwise_jaccard_distance(self, M, R, eps=1e-8):
        R = R.float().to(self.device)  # Cast to float32
        M = M.float().to(self.device)
        RM = torch.matmul(R, M)
        numerator = (RM * R).sum(dim=1)
        w_sum = R.sum(dim=1)
        w_sq_sum = (R ** 2).sum(dim=1)
        denominator = w_sum**2 - w_sq_sum
        avg_dist = numerator / (denominator + eps)
        avg_dist[denominator == 0] = 0
        return avg_dist

    def forward(self,R:torch.tensor, dist_mat:torch.tensor):
        """_summary_

        Args:
            R (torch.tensor): prediciton matrix
            dist_mat (torch.tensor): jaccard distance matrix, diagonal elements are zero.

        Returns:
            torch.tensor : approximate average distance of users, accurate average distance of users.
        """
        R = R.to(self.device)
        dist_mat = dist_mat.to(self.device)

        # soft_rank returns the rank of each elements in prediction matrix R, in ascending order.
        approx_ranks = torchsort.soft_rank(R,regularization_strength=self.sort_tau)
        approx_top_k = torch.sigmoid(1 * (approx_ranks - (R.shape[1] - self.top_k)))
        # compute the real rank vector 
        real_ranks_top_k = self.retain_top_x(R)

        approx_distance = self.weighted_avg_pairwise_jaccard_distance(dist_mat, approx_top_k)
        real_distance  = self.weighted_avg_pairwise_jaccard_distance(dist_mat, real_ranks_top_k)

        return approx_distance, real_distance