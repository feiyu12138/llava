import torch

def random_drop(tokens, position_ids, stride):
    K = tokens.size(2) // stride
    # Randomly keep K tokens
    keep_ids = torch.randperm(tokens.size(2))[:K]    
    tokens = tokens[:,:,keep_ids]
    position_ids = position_ids[:,keep_ids]
    
    return tokens, position_ids
def random_drop_block(tokens, position_ids, stride):
    K = tokens.size(2) // stride
    # Randomly keep K continuous tokens
    start_ids = torch.randint(0,tokens.size(2)-K+1,(1,))
    tokens = tokens[:,:,start_ids:start_ids+K]
    position_ids = position_ids[:,start_ids:start_ids+K]
    
    return tokens, position_ids

if __name__ == '__main__':
    tokens = torch.randn(1,4096,VISUAL_LENGTH)
    position_ids = torch.arange(VISUAL_LENGTH).unsqueeze(0)
    tokens, position_ids = random_drop_block(tokens, position_ids, stride=4)
    print(tokens.size(), position_ids)