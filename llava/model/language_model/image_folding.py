from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def fold_images(images):
        B,C,H,W = images.shape
        images = images.view(B,C,2,H//2,2,W//2)
        images = images.permute(0, 2, 4, 1, 3, 5).contiguous().view(4*B,C,H//2,W//2)
        return images
    
def unfold_features(self,features):
        from ipdb import set_trace; set_trace()
        B_,L,C = features.shape
        B = B_ // 5
        H = W = int(L ** 0.5) * 2
        feature_maps = torch.zeros(B,H,W,C).to(features.device).to(features.dtype)
        global_maps = features[::5]
        features = torch.cat([x.unsqueeze(0) for x in features.split(5,0)],dim=0)  
        for i in range(2):
            for j in range(2):
                feature_maps[:,i*H//2:(i+1)*H//2,j*W//2:(j+1)*W//2] = features[:,i*2+j].reshape(B,H//2,W//2,C)    # TODO: check order for x and y  
        feature_maps = feature_maps.view(B,H*W,C)
        feature_maps = torch.cat([feature_maps,global_maps],dim=1)
        return feature_maps

if "__main__" == __name__:
    img_path1 = '/data/datasets/jchen293/data/llava_datasets/eval_luoxin/eval/mm-vet/images/v1_87.jpg'
    img_path2 = '/data/datasets/jchen293/data/llava_datasets/eval_luoxin/eval/mm-vet/images/v1_88.jpg'
    images = Image.open(img_path1)
    images2 = Image.open(img_path2)
    img_tensor = torch.tensor(np.array(images)).permute(2,0,1).unsqueeze(0).float()
    img_tensor = F.interpolate(img_tensor, (448,448),mode='bilinear')
    img_tensor2 = torch.tensor(np.array(images2)).permute(2,0,1).unsqueeze(0).float()
    img_tensor2 = F.interpolate(img_tensor2, (448,448),mode='bilinear')
    Img = Image.fromarray(img_tensor[0].permute(1,2,0).numpy().astype(np.uint8))
    Img2 = Image.fromarray(img_tensor2[0].permute(1,2,0).numpy().astype(np.uint8))
    from ipdb import set_trace; set_trace()
    Img.save('original_image.jpg')
    Img2.save('original_image2.jpg')
    img_tensor_all = torch.cat([img_tensor, img_tensor2], dim=0)
    imgs = fold_images(img_tensor_all)
    image = imgs.permute(0,2,3,1).reshape(-1,448,448,3).permute(0,3,1,2)[0].permute(1, 2, 0).numpy().astype(np.uint8)
    Image_ = Image.fromarray(image)
    Image_.save('folded_image.jpg')
    # Save folded images
    for i in range(imgs.shape[0]):
        img = imgs[i].permute(1, 2, 0).numpy().astype(np.uint8)
        folded_img = Image.fromarray(img)
        folded_img.save(f'folded_image_{i}.jpg')
    