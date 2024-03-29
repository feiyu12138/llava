import torch

def generate_attention_map(visual_tokens, language_tokens, attention_layer, save_path):
    B, Nv, C = visual_tokens.size()
    B, Nl, L = language_tokens.size()
    
    attention_map = torch.zeros(Nv, Nl)
    visual_query = 1
    