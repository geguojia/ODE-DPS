import numpy as np
import torch
import shutil
import pytorch_ssim

num=100
MAE_sum = 0
RMSE_sum = 0
SSIM_sum = 0
re_dis_sum = 0

for i in range(num):
    n = i+1
    v_pre = torch.tensor(np.load('./storage/{}/data/v_pre.npy'.format(n)))
    v_true = torch.tensor(np.load('./storage/{}/data/v_true.npy'.format(n)))
    re_dis = torch.linalg.norm(torch.relu(v_pre*1500+3000) - (v_true*1500+3000))/torch.linalg.norm(v_true*1500+3000)
    v_pre = v_pre.clamp(-1,1)
    v_true = v_true.clamp(-1,1)
    MAE = (torch.sum(abs(v_pre - v_true))/64**2).numpy()
    RMSE = (torch.sqrt(torch.sum((v_pre - v_true)**2)/64**2)).numpy()
    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    SSIM= ssim_loss(v_pre/2+0.5, v_true/2+0.5)
    MAE_sum += MAE
    RMSE_sum += RMSE
    SSIM_sum += SSIM
    re_dis_sum += re_dis

MAE_avg = MAE_sum/num
RMSE_avg = RMSE_sum/num
SSIM_avg = SSIM_sum/num
re_dis_avg = re_dis_sum/num

print(MAE_avg, RMSE_avg, SSIM_avg, re_dis_avg)