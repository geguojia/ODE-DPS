import numpy as np
import matplotlib.pyplot as plt
from guided_diffusion.v_to_u.solver import receiver
import torch
import yaml
import os
import re

folder_path = './storage'
folder_name = os.listdir(folder_path)
num_list = []
for name in folder_name:
    num_list.append(int(name))
dir_num = str(max(num_list))
save_root = os.path.join(folder_path, dir_num)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('./configs/forward_process.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

with open('./configs/diffusion_config.yaml') as f:
    config2 = yaml.load(f, Loader=yaml.FullLoader)
dif_para = config2

with open('./configs/model_config.yaml') as f:
    config3 = yaml.load(f, Loader=yaml.FullLoader)
model_para = config3

storation = {}
storation['steps'] = dif_para['steps']
storation['timestep_respacing'] = dif_para['timestep_respacing']
storation['normalize'] = config['data']['normalize']
storation['initialize'] = config['measurement']['opt']['initialize']
storation['scale_grad'] = config['conditioning']['params']['scale']
storation['decline'] = config['conditioning']['params']['decline']

n_x = 64
n_y = 64
n_t = 100

X = torch.linspace(0, 1, n_x)
Y = torch.linspace(0, 1, n_y)
X_u = torch.linspace(0, 1, 5)
Y_u = torch.linspace(0, 1, 64)

XX, YY = torch.meshgrid(X, Y, indexing='xy')
XX, YY = XX.numpy(), YY.numpy()
XX_u, YY_u = torch.meshgrid(Y_u, X_u, indexing='xy')
XX_u, YY_u = XX_u.numpy(), YY_u.numpy()

f = np.load(os.path.join(save_root, 'data', 'v_pre.npy'))
f_hat = torch.tensor(f).to(device)
f_pre = f_hat.cpu().detach().numpy()
temp = torch.ones(f_hat.shape[0], f_hat.shape[1], 1, 1).to(device)

f_hat = torch.relu(1500*f_hat+3000)
f_pre= f_hat.cpu().detach().numpy()
mid_res = receiver(f_hat,device)

u=mid_res
u_pre = u.cpu().detach().numpy()

# store u_pre
np.save(os.path.join(save_root, 'data', 'u_pre.npy'), u_pre)

plt.subplot(231)
plt.contourf(XX,YY,f_pre[-1,-1,:,:])
plt.xticks([])
plt.yticks([])
plt.colorbar().formatter.set_powerlimits((0,0))
plt.title('v_pre')
plt.subplot(234)
plt.contourf(XX_u,YY_u,u_pre[:,:,-1])
plt.xticks([])
plt.yticks([])
plt.colorbar().formatter.set_powerlimits((0,0))
plt.title('u_pre',y=-0.3)

f_true = np.load(os.path.join(save_root, 'data', 'v_true.npy'))
f_real = torch.tensor(f_true[-1,-1,:,:]).to(device)
f_real = temp * f_real
f_real = torch.relu(1500*f_real+3000)
f_true = f_real.cpu().detach().numpy()
mid_res = receiver(f_real,device)
u = mid_res
u_true = u.cpu().detach().numpy()

# store u_true
np.save(os.path.join(save_root, 'data', 'u_true.npy'), u_true)

plt.subplot(232)
plt.contourf(XX,YY,f_true[-1,-1,:,:])
plt.xticks([])
plt.yticks([])
plt.colorbar().formatter.set_powerlimits((0,0))
plt.title('v_true')
plt.subplot(235)
plt.contourf(XX_u,YY_u,u_true[:,:,-1])
plt.xticks([])
plt.yticks([])  
plt.colorbar().formatter.set_powerlimits((0,0))
plt.title('u_true',y=-0.3)
storation['scale_v'] = np.max(np.abs(f_true))
storation['scale_u'] = np.max(np.abs(u_true))

diff_f = abs(f_true[-1,-1,:,:]-f_pre[-1,-1,:,:])
diff_u = abs(u_true-u_pre)

# store diff_f and diff_u
np.save(os.path.join(save_root, 'data', 'diff_v.npy'), diff_f)
np.save(os.path.join(save_root, 'data', 'diff_u.npy'), diff_u)

plt.subplot(233)
plt.contourf(XX,YY,diff_f)
plt.xticks([])
plt.yticks([])
plt.colorbar().formatter.set_powerlimits((0,0))
plt.title('diff_a')
plt.subplot(236)
plt.contourf(XX_u,YY_u,diff_u[:,:,-1])
plt.xticks([])
plt.yticks([])
plt.colorbar().formatter.set_powerlimits((0,0))
plt.title('diff_u', y=-0.3)
f_diff = torch.tensor(f_true - f_pre[-1, -1, :, :])
difference = torch.linalg.norm(f_diff)
difference_f = torch.linalg.norm(torch.tensor(f_true))
difference_re = (difference / difference_f)
storation['difference_re'] = difference_re.cpu().detach().numpy()

# store the resolution
plt.savefig(os.path.join(save_root, 'figure', 'resolution.png'))
plt.close()

storation['noise_sigma'] = config['measurement']['noise']['sigma']
storation['regular_scale'] = config['conditioning']['params']['regular_scale']
storation['model'] = model_para['model_path']
storation['diffusion_whether'] = config['measurement']['diffusion_whether']
storation['lam']=config['measurement']['lam']

# store the process
'''os.makedirs(os.path.join(save_root,'process','figure'), exist_ok=True)
folder_name = os.listdir(os.path.join(save_root,'process','data'))
print(len(folder_name))
for file in folder_name:
    data = np.load(os.path.join(save_root, 'process', 'data', file))
    parttern = r'\d+'
    num = re.findall(parttern, file)[0]+'.png'
    plt.contourf(XX, YY, data[-1,-1,:,:])
    plt.colorbar()
    plt.savefig(os.path.join(save_root,'process','figure', num))
    plt.close()'''

# store the label_compare:
label = np.load(os.path.join(save_root, 'data', 'label.npy'))
label_noise = np.load(os.path.join(save_root, 'data', 'label_noise.npy'))
plt.subplot(121)
plt.contourf(XX_u, YY_u, label[:,:,-1])
plt.xticks([])
plt.yticks([])
plt.colorbar().formatter.set_powerlimits((0,0))
plt.title('label')
plt.subplot(122)
plt.contourf(XX_u, YY_u, label_noise[:,:,-1])
plt.xticks([])
plt.yticks([])
plt.colorbar().formatter.set_powerlimits((0,0))
plt.title('label_noise')
plt.savefig(os.path.join(save_root, 'figure', 'label_compare.png'))
plt.close()

# store the initial:
initial = np.load(os.path.join(save_root, 'data', 'initial.npy'))
plt.contourf(XX, YY, initial[-1,-1,:,:])
plt.xticks([])
plt.yticks([])
plt.colorbar().formatter.set_powerlimits((0,0))
plt.title('initial')
plt.savefig(os.path.join(save_root, 'figure', 'initial.png'))
plt.close()

# store the parameter
with open(os.path.join(save_root, 'text', 'parameter.txt'),'a') as op:
    for key in storation:
        try:
            op.write(key+': '+str(storation[key])+'\n')
        except:
            op.write(key+': '+(storation[key])+'\n')