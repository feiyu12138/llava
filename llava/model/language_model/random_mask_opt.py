import torch

def random_drop(tokens, position_ids, stride):
    K = tokens.size(2) // stride
    # Randomly keep K tokens
    keep_ids = torch.randperm(tokens.size(2))[:K]    
    tokens = tokens[:,:,keep_ids]
    position_ids = position_ids[:,keep_ids]
    
    return tokens, position_ids
def random_mask(tokens, position_ids, stride):
    K = tokens.size(2) // stride
    
    return tokens, position_ids

if __name__ == '__main__':
    tokens = torch.randn(1,4096,576)
    position_ids = torch.arange(576).unsqueeze(0)
    tokens, position_ids = random_drop(tokens, position_ids, stride=4)
    print(tokens.size(), position_ids.size())