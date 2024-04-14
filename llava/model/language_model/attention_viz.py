import torch

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
    