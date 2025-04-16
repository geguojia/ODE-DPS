from functools import partial
import os
import argparse
import yaml
import shutil
import re

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger

import warnings
warnings.filterwarnings("ignore")

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    lam = measure_config['lam']

    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn, diffusion_whether=measure_config['diffusion_whether'])

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
        
    # Do Inference

    for i, ref_img in enumerate(loader):

        logger.info(f"Inference for image {i}")
        ref_img = ref_img.to(device)

        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)
        else: 
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)  
            y_n = noiser(y)

        initialize = measure_config['opt']['initialize']
        if initialize == 0:
            x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        else:
            x_start = initialize * torch.ones_like(ref_img).float().requires_grad_()
        
        label_dis = torch.linalg.norm(y-y_n)

        folder_path = './storage'
        folder_name = os.listdir(folder_path)
        dir_num = str(len(folder_name)+1)
        save_root = os.path.join(folder_path, dir_num)
        os.makedirs(save_root)
        for name in ['data', 'figure', 'text', 'process']:
            os.makedirs(os.path.join(save_root, name))
        
        sample = sample_fn(x_start=x_start, measurement=y_n, measurement_f = ref_img, record=True, save_root=save_root, delta=label_dis, lam=lam, tau=task_config['stop']['tau'], barrier=task_config['stop']['barrier'])
        
        y_n2 = y_n.cpu().detach().numpy()
        y2 = y.cpu().detach().numpy()
        ref_img2 = ref_img.cpu().detach().numpy()
        sample2 = sample.cpu().detach().numpy()
        x_start2 = x_start.cpu().detach().numpy()
        
        np.save(os.path.join(save_root, 'data', 'label.npy'), y2)
        np.save(os.path.join(save_root, 'data', 'label_noise.npy'), y_n2)
        np.save(os.path.join(save_root, 'data', 'v_true.npy'), ref_img2)
        np.save(os.path.join(save_root, 'data', 'v_pre.npy'), sample2)
        np.save(os.path.join(save_root, 'data','initial.npy'), x_start2)

        os.system('python3 plot.py')

if __name__ == '__main__':
    main()
