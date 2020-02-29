# -*- coding: utf-8 -*-

'''
Label propagation for modeling relations in instance-level graphs.
'''
import numpy as np
import torch

def label_propagation(embeddings, sigma, top_k, support_labels_onehot, n_classes):
    '''
    Build instance-level graph with Gaussian similarity function,
    then perform dual label propagation for modeling relations among instances.
    The closed-form solution of label propagationn is: `F = (I-\alpha S)^{-1}Y`

    Args:
        embeddings:
            the embeddings of the support set and query set.
            N x d, where d is the dimension of each instance embedding,
            N = n_support + n_query
        sigma:
            N x 1, learned by the LP_mlp module in graphsage.
        top_k:
            keep only the most important K neighbors for building the final instance-level graph.
        support_labels_onehot:
            label matrix of the support set (one-hot).
            size: n_support x n_classes
        n_classes:
            the number of the classes during label propagation. i.e., label space.
            which is used to build `Yu` in label propagation algorithm.
    Return:
        F:
            the updated label matrix.
            size: N x n_classes
            where F[n_support:, :] is the predicted labels of the query set.
    '''
    eps = np.finfo(float).eps
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = embeddings.size(0)
    n_support = support_labels_onehot.size(0)
    n_query = N - n_support
    assert embeddings.size(0) == sigma.size(0)
    embeddings = embeddings / (sigma + eps)         # N x d
    embed1     = torch.unsqueeze(embeddings, 1)     # N x 1 x d
    embed2     = torch.unsqueeze(embeddings, 0)     # 1 x N x d
    ig_adj     = ((embed1 - embed2)**2).mean(2)     # N x N x d -> N x N, "ig" for "instance-level graph"
    ig_adj     = torch.exp(-ig_adj / 2)
    
    ## pruning
    if top_k > 0:
        topk, indices = torch.topk(ig_adj, top_k)
        mask = torch.zeros_like(ig_adj)
        mask = mask.scatter(1, indices, 1)
        mask = ((mask + torch.t(mask)) > 0).type(torch.float32)
        ig_adj = ig_adj * mask
    
    ## normalize
    D  = ig_adj.sum(0)
    D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
    D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)
    D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
    S  = D1 * ig_adj * D2

    ## label propagation: F = (I-\alpha S)^{-1}Y
    Ys = support_labels_onehot
    Yu = torch.zeros(n_query, n_classes).to(device)
    Y  = torch.cat((Ys, Yu), 0)
    alpha = torch.tensor([0.99], requires_grad=False).to(device)
    F  = torch.matmul(torch.inverse(torch.eye(N).to(device) - alpha * S + eps), Y)
    Fq = F[n_support:, :]

    return F, Fq