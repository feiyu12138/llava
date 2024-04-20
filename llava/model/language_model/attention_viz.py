import torch
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA

def generate_attention_map(query:torch.tensor, key:torch.tensor) -> torch.tensor:
    """
    Generate attention map from query and key tensors.
    """
    # concatenate multiple heads
    query = query.transpose(1, 2)
    query = query.reshape(query.size(0), query.size(1), -1)
    key = key.transpose(1, 2)
    key = key.reshape(key.size(0), key.size(1), -1)
    
    # compute attention map
    attention_map = torch.einsum('bqd,bkd->bqk', query, key) / (query.size(-1) ** 0.5)
    attention_map = torch.softmax(attention_map, dim=-1)
    
    
    return attention_map

def minmax_normalize(tensor:torch.tensor) -> torch.tensor:
    """
    Min-max normalize a tensor.
    """
    min_val = tensor.min(dim=1).values.unsqueeze(1)
    max_val = tensor.max(dim=1).values.unsqueeze(1)
    return (tensor - min_val) / (max_val - min_val)
    
def kmeans_clustering(tokens, num_centroid, iterations=10):
    stride = tokens.size(1) // num_centroid
    tokens = tokens.detach()
    centroids = tokens[:,::stride,:]
    for _ in range(iterations):
        # Compute squared distances between each point and each centroid
        distances = torch.einsum('bqd,bkd->bqk', tokens, centroids) / centroids.norm(dim=-1) / tokens.norm(dim=-1).unsqueeze(2)
        indices = torch.argmax(distances, dim=2)
        # Update centroids as the weighted mean of data points
        centroids = torch.stack([tokens[:,indices[0] == k,:].mean(dim=1) for k in range(num_centroid)], dim=1)
    subset = [tokens[:,indices[0] == k] for k in range(num_centroid)]
    return centroids,subset

def calc_cluster_std(states: torch.tensor,num_centroid:int) -> torch.tensor:
    """
    Calculate centroids of query, key, and value tensors.
    """
    with torch.no_grad():
        centroids, subsets = kmeans_clustering(states, num_centroid)
    stds = torch.zeros(num_centroid)
    for i in range(num_centroid):
        stds[i] = calc_similarity_std(subsets[i], centroids[:,i:i+1])
    return stds.mean()

def calc_segment_std(states: torch.tensor, segment_length:int=4) -> torch.tensor:
    """
    Calculate standard deviation of query, key, and value tensors.
    """
    num_segment = states.size(1) // segment_length
    not_divided = int(states.size(1) % segment_length != 0)
    stds = torch.zeros(num_segment+not_divided)
    for i in range(num_segment+not_divided):
        stds[i] = calc_similarity_std(states[:,i*segment_length:(i+1)*segment_length,:])
    return stds.mean()

def calc_pca(states: torch.tensor) -> torch.tensor:
    """
    Calculate PCA of query, key, and value tensors.
    """
    pca = PCA(n_components=0.95)
    pca.fit(states.cpu().squeeze(0))
    return pca.explained_variance_ratio_

def calc_similarity_std(states: torch.tensor, centroid:torch.tensor=None) -> torch.tensor:
    """
    Calculate standard deviation of query, key, and value tensors.
    """
    if centroid is None:
        centroid = states.mean(dim=1).unsqueeze(1)
    centroid = centroid.repeat(1, states.shape[1], 1)
    similarity = torch.einsum('bqd,bqd->bq', states, centroid) / (centroid.norm(dim=-1) + 1e-6) / (states.norm(dim=-1) + 1e-6)
    if similarity.size(1) == 1 or similarity.size(1) == 0:
        return torch.zeros(1)
    return similarity.std(dim=1)
        

def calc_qkvs_std(query:torch.tensor, key:torch.tensor, value:torch.tensor, state:torch.tensor, method:str='segment') -> dict[str, torch.tensor]:
    """
    Calculate standard deviation of query, key, and value tensors.
    """
    query = minmax_normalize(query.transpose(1,2).view(query.shape[0],query.shape[2],-1))
    key = minmax_normalize(key.transpose(1,2).view(key.shape[0],key.shape[2],-1))
    value = minmax_normalize(value.transpose(1,2).view(value.shape[0],value.shape[2],-1))
    state = minmax_normalize(state)
    if method == 'plain':
        q_std = calc_similarity_std(query)
        k_std = calc_similarity_std(key)
        v_std = calc_similarity_std(value)
        s_std = calc_similarity_std(state)
    elif method == 'cluster':
        q_std = calc_cluster_std(query, num_centroid=10)
        k_std = calc_cluster_std(key, num_centroid=10)
        v_std = calc_cluster_std(value, num_centroid=10)
        s_std = calc_cluster_std(state, num_centroid=10)
    elif method == 'segment':
        q_std = calc_segment_std(query)
        k_std = calc_segment_std(key)
        v_std = calc_segment_std(value)
        s_std = calc_segment_std(state)
        
    result = {
        'query': q_std,
        'key': k_std,
        'value': v_std,
        'state': s_std
    }
    
    return result
    