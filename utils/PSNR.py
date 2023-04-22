import torch
import numpy as np

"""
******************************  PSNR   ******************************
"""
def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps

def torchPSNR(tar_img, prd_img):
    b = tar_img.shape[0]
    tar_imgset = torch.chunk(tar_img, int(b), dim=0)
    prd_imgset = torch.chunk(prd_img, int(b), dim=0)
    all = 0.0
    for i in range(b):
        imdff = torch.clamp(tar_imgset[i],0,1) - torch.clamp(prd_imgset[i],0,1)
        rmse = (imdff**2).mean().sqrt()
        ps = 20*torch.log10(1/rmse)
        all += ps
    ps = all/b
    return ps
