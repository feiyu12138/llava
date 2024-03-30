import torch

def soft_kmeans(tokens, stride, iterations=10):
    batched_centroids = []
    for B in range(tokens.size(0)):
        cur_tokens = tokens[B]
        centroids = cur_tokens[:,::stride]
        for _ in range(iterations):
            from ipdb import set_trace; set_trace()
            
            # Compute squared distances between each point and each centroid
            distances = torch.sum((cur_tokens.unsqueeze(2) - centroids.unsqueeze(1)) ** 2, dim=0)
            
            # Soft assignment of points to centroids
            weights = torch.softmax(-distances, dim=1)
            # Update centroids as the weighted mean of data points
            centroids = cur_tokens @ weights / weights.sum(dim=1)
        
        batched_centroids.append(centroids)
    batched_centroids = torch.stack(batched_centroids, dim=0)
    return batched_centroids

if __name__ == '__main__':
    tokens = torch.randn(1,4096,576)
    centroids = soft_kmeans(tokens, stride=4, iterations=10)
    print(centroids.size())