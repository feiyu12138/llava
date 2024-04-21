import torch

def from_vcc(vcc_states):
    shape = vcc_states["important_token_mask"].shape
    dtype = vcc_states["partial_coarse_token_scores"].dtype
    device = vcc_states["partial_coarse_token_scores"].device

    hidden_states = [
        vcc_states["partial_fine_token_states"],
        vcc_states["partial_coarse_token_states"],
        vcc_states["important_token_states"]
    ]
    mask = [
        vcc_states["partial_fine_token_mask"],
        vcc_states["partial_coarse_token_mask"],
        vcc_states["important_token_mask"]
    ]
    position_ids = [
        vcc_states["partial_fine_token_positions"],
        vcc_states["partial_coarse_token_positions"],
        vcc_states["important_token_positions"]
    ]
    
    hidden_states = torch.cat(hidden_states, dim = 1)
    mask = torch.cat(mask, dim = 1)
    position_ids = torch.cat(position_ids, dim = 1)
    
    return hidden_states.half(), mask, position_ids