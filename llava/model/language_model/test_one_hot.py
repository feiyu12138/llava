import torch
import torch.nn.functional as F
detach_tokens = torch.randn(2, 3, 4)
distances = torch.randn(2,4,2)
weights = torch.argmax(-distances, dim=2)
one_hot_weights = F.one_hot(weights, num_classes=distances.size(2)).to(detach_tokens.dtype)
detach_centroids = torch.einsum('bcl,blq->bcq', detach_tokens, one_hot_weights)
print(detach_centroids.size())