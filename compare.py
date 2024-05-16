import numpy as np

a = np.load("llava-v1.5-7b-progressive_pred_score.npy")
b = np.load("pool8layer2prog2600hl_pred_score.npy")

compare = np.where(a > b, 1, 0)
np.save("compare.npy", compare)

# print indices of the elements that are 1
print(np.where(compare == 1))