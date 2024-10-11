import os
import logging
import time
import glob
import torchvision.transforms as transforms
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils import get_root_logger, imwrite, tensor2img, load_imgDDNM, load_img_LearningDegradation, load_imgExplicit
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.nn.functional as F

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from functions.svd_ddnm import ddnm_diffusion, ddnm_plus_diffusion
from functions.conv_util import *
from functions.plot_util import *
from functions.eval import *

import torchvision.utils as tvu

from guided_diffusion.models import Model
from guided_diffusion.unet import create_model_ffhq
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

from lpips import LPIPS
from scipy.linalg import orth
from torch.cuda.amp import custom_bwd, custom_fwd
from guided_diffusion.kernel_estimator import Estimator
from diffusers import DDIMScheduler, DDPMScheduler


import torch
import torch.nn as nn
import torch.nn.functional as F

def PatchUpsample(x, scale):
    n, c, h, w = x.shape
    x = torch.zeros(n,c,h,scale,w,scale).to('cuda') + x.view(n,c,h,1,w,1)
    return x.view(n,c,scale*h,scale*w)


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.repeat(1,3,1,1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)    

def cal_img_psnr(gt, pred, max_value=1):
    mse = torch.mean((pred[0].to('cuda') - gt[0].to('cuda')) ** 2)
    psnr = 10 * torch.log10(max_value / mse)  
    return psnr 

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, IRmodel=None, IRopt=None, device=None, txt_logger=None, IRmodel2=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.np_beta = betas
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # load kernel estimator model
        if self.args.kernel_estimator:
            self.kernel_estimator = Estimator(self.args).to(self.device)
            model_path = self.args.kernel_model_path
            print(f"Loading model checkpoint from {model_path}\n")
            self.kernel_estimator.load_state_dict(
                torch.load(model_path)
            )
        if IRmodel is not None:
            self.IRmodel = IRmodel
        if IRopt is not None:
            self.IRopt = IRopt
        if txt_logger is not None:
            self.txt_logger = txt_logger
        self.IRmodel2 = IRmodel2
        
        
    # def cal_loss(self, model, input_img):#, noise, t):
    #     noise = torch.randn_like(input_img)
    #     t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
    #     x = self.scheduler.add_noise(input_img, noise, t)
    #     with torch.no_grad():
    #         noise_pred = model(x, t)
    #     w = (1 - self.alphas[t])
    #     grad = w * (noise_pred - noise)
    #     grad = torch.nan_to_num(grad)
    #     input_img.backward(gradient=grad, retain_graph=True)
    #     # loss = SpecifyGradient.apply(input_img, grad)
    #     return 0
          
    # def sds_train(self, model, Apy, x0_t, ApAx0_t_img, iter=10):
    #     # set_seed(42)
    #     self.scheduler = DDPMScheduler.from_pretrained('google/ddpm-celebahq-256')
    #     self.scheduler.set_timesteps(100)

    #     self.num_train_timesteps = self.scheduler.config.num_train_timesteps
    #     self.min_step = int(self.num_train_timesteps * 0.02)
    #     self.max_step = int(self.num_train_timesteps * 0.98)
    #     self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience


    #     para_alpha = torch.nn.Parameter(torch.tensor([0.5], device=self.device))
    #     optimizer = torch.optim.AdamW([para_alpha], lr=1e-2)
    #     tmp = []
    #     for _ in range(iter):
    #         optimizer.zero_grad()
    #         output = para_alpha * Apy + x0_t - para_alpha * ApAx0_t_img
    #         _ = self.cal_loss(model, output)
    #         # loss.backward()
    #         optimizer.step()
    #         tmp.append(round(para_alpha.item(), 4))
    #         # print(para_alpha.item())
    #     # print(tmp)
    #     return para_alpha.item(), tmp
                
    # def sds(self):
    #     # self.device = 'cpu'
    #     set_seed(42)
    #     args, config = self.args, self.config
    #     model = Model(self.config)
    #     ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
    #     model.load_state_dict(torch.load(ckpt, map_location=self.device))
    #     model.to(self.device)
        
    #     self.scheduler = DDPMScheduler.from_pretrained('google/ddpm-celebahq-256')
    #     self.scheduler.set_timesteps(100)

    #     self.num_train_timesteps = self.scheduler.config.num_train_timesteps
    #     self.min_step = int(self.num_train_timesteps * 0.02)
    #     self.max_step = int(self.num_train_timesteps * 0.98)
    #     self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        
    #     input_img_path = '/eva_data1/shlu2240/Dataset/celeba_hq_02_4/test/hq/00010.jpg'
    #     input_img = load_imgDDNM(input_img_path).to(self.device)
    #     input_img = data_transform(self.config, input_img)
    #     # t = torch.randint(500, 500 + 1, [1], dtype=torch.long, device=self.device)
    #     # noise_list = [torch.randn_like(input_img) for i in range(10)]
    #     noise = torch.randn_like(input_img)
    #     # print(self.alphas)
    #     def cal_loss(input_img, noise, t):
    #         x = self.scheduler.add_noise(input_img, noise, t)
    #         with torch.no_grad():
    #             noise_pred = model(x, t)
    #             w = (1 - self.alphas[t])
    #             grad = w * (noise_pred - noise)
    #             return grad.pow(2).sum().sqrt()
        
        
        
        
    #     folder_name = 'LD_scheduler_a03'
    #     time_step_list = [800, 500, 200, 100, 0]
    #     image_list = []
    #     # same img differnet timestep
    #     for t in time_step_list:    
    #         img_path = f'/eva_data1/shlu2240/DDNM/exp/image_samples/{folder_name}/x0_t_hat/00105/x0_{t}_hat.png'
    #         input_img = load_imgDDNM(img_path).to(self.device)
    #         input_img = data_transform(self.config, input_img)
    #         image_list.append(input_img)
        
    #     # img_path2 = f'/eva_data1/shlu2240/DDNM/exp/image_samples/{folder_name}/x0_t_hat/00105/x0_800_hat.png'
    #     # input_img2 = load_imgDDNM(img_path2).to(self.device)
    #     # input_img2 = data_transform(self.config, input_img2)
        
    #     # y1 = []
    #     # y2 = []
    #     result = [[] for _ in image_list]
    #     for i in range(0, 991, 10):
    #         t = torch.randint(i, i + 1, [1], dtype=torch.long, device=self.device)
    #         for idx, input_img in enumerate(image_list):
                
    #             result[idx].append(cal_loss(input_img, noise, t).cpu())
    #             # y2.append(cal_loss(input_img2, noise, t).cpu())
    #     colors = [
    #         '#FF0000',  # Red
    #         '#00FF00',  # Green
    #         '#0000FF',  # Blue
    #         '#FF00FF',  # Magenta
    #         '#00FFFF',  # Cyan
    #     ]
        
    #     plt.figure(figsize=(28, 5))
    #     for idx, r in enumerate(result):
    #         plt.plot(range(0, 991, 10), r, c=colors[idx], label=f'x0_{time_step_list[idx]}_hat {folder_name}')
    #     # plt.plot(range(0, 991, 10), y2, c='blue', label='x0_500_hat wo scaler')
    #     plt.legend()
    #     plt.savefig(os.path.join(self.args.image_folder, f"output.png"))
        
            
    #     # cal x0_t_hat loss
    #     # a03_loss = []
    #     # baseline_loss = []
    #     # for i in range(0, 991, 10):
    #     #     print(i)
    #     #     a03_img_path = f'/eva_data1/shlu2240/DDNM/exp/image_samples/LD_scheduler_a03/x0_t_hat/00997/x0_{i}_hat.png'
    #     #     baseline_img_path = f'/eva_data1/shlu2240/DDNM/exp/image_samples/LD_wo_scheduler/x0_t_hat/00997/x0_{i}_hat.png'
    #     #     input_img = load_imgDDNM(a03_img_path).to(self.device)
    #     #     input_img = data_transform(self.config, input_img)
    #     #     t = torch.randint(i, i + 1, [1], dtype=torch.long, device=self.device)
    #     #     a03_loss.append(cal_loss(input_img, noise, t).cpu())
    #     #     del input_img
    #     #     input_img = load_imgDDNM(baseline_img_path).to(self.device)
    #     #     input_img = data_transform(self.config, input_img)
    #     #     baseline_loss.append(cal_loss(input_img, noise, t).cpu())
    #     #     # print(cal_loss(input_img, noise, t))
    #     #     del input_img
    #     #     del t
    #     # # plot
    #     # plt.plot(range(0, 991, 10), a03_loss, c='red', label='a03')
    #     # plt.plot(range(0, 991, 10), baseline_loss, c='blue', label='baseline')
    #     # plt.legend()
    #     # plt.savefig(os.path.join(self.args.image_folder, f"output.png"))
        
    #     # add t step noise noise
    #     # x = self.scheduler.add_noise(input_img, noise, t)
    #     # noise_pred = model(x, t)
    #     # # cal grad
    #     # w = (1 - self.alphas[t])
    #     # grad = w * (noise_pred - noise)
    #     # print(t)
    #     # print(grad.pow(2).sum().sqrt() )
        
        
        
        
        
    #     # for idx in range(10):
    
    #     #     x = torch.randn(
    #     #         1,
    #     #         config.data.channels,
    #     #         config.data.image_size,
    #     #         config.data.image_size,
    #     #         device=self.device,
    #     #     )

    #     # with torch.no_grad():

    #     #     xs = [x]
    #     #     # reverse diffusion sampling
    #     #     for i, timestep in enumerate(self.scheduler.timesteps):
    #     #         if timestep > 500 :
    #     #             continue
    #     #         #self inplement DDNM
    #     #         t = (torch.ones(1) * timestep).to(x.device)
    #     #         # next_t = (torch.ones(n) * j).to(x.device)
    #     #         # at = compute_alpha(self.betas, t.long())
    #     #         # at_next = compute_alpha(self.betas, next_t.long())
    #     #         xt = xs[-1].to('cuda')

    #     #         et = model(xt, t)
    #     #         xt_next = self.scheduler.step(et, timestep, xt)['prev_sample']
    #     #         # Eq. 12
    #     #         # x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() 
                
    #     #         # eta = self.args.eta

    #     #         # c1 = (1 - at_next).sqrt() * eta
    #     #         # c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

    #     #         # # different from the paper, we use DDIM here instead of DDPM
    #     #         # xt_next = at_next.sqrt() * x0_t + (c1 * torch.randn_like(x0_t) + c2 * et)

    #     #         # x0_preds.append(x0_t.to('cpu'))
    #     #         xs.append(xt_next.to('cpu'))   
                    

    #     #     x = xs[-1]
            
    #     # x = [inverse_data_transform(config, xi) for xi in x]

    #     # tvu.save_image(
    #     #     x[0], os.path.join(self.args.image_folder, f"output.png")
    #     # )

    def sample(self, simplified):
        cls_fn = None
        if self.config.model.type == 'simple':
            model = Model(self.config)

            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            else:
                raise ValueError
            if name != 'celeba_hq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                # ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
                ckpt = '/eva_data3/shlu2240/checkpoints/diffusion/celeba_hq.ckpt'
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt',
                             ckpt)
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (
                        self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = '/eva_data3/shlu2240/checkpoints/diffusion/256x256_diffusion_uncond.pt'
                # ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
                        ckpt)

            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_classifier.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt' % image_size,
                        ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = torch.nn.DataParallel(classifier)

                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale

                cls_fn = cond_fn
        elif self.config.model.type == 'ffhq':
            config_dict = vars(self.config.model)
            model = create_model_ffhq(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()

            ckpt = config_dict['model_path']
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

        #generate x0_t dataset
        if self.args.sample_x0t:
            self.sample_x0t(model)
        else:
            # if simplified:
            print('Run Simplified DDNM, without SVD.',
                    f'{self.config.time_travel.T_sampling} sampling steps.',
                    f'travel_length = {self.config.time_travel.travel_length},',
                    f'travel_repeat = {self.config.time_travel.travel_repeat}.'
                    )
            if self.args.dataset == 'celeba':
                self.ddnm_plus_final(model)
                # self.ddnm_plus_celeba(model)
            elif self.args.dataset == 'div2k':
                self.ddnm_plus_div2k(model)
            # elif self.args.dataset == 'div2k_tmp':
            #     self.ddnm_plus_div2k_tmp(model)
            # elif self.args.dataset == 'ffhq':
            #     self.ddnm_plus_ffhq(model)
            # elif self.args.dataset == 'imagenet':
            #     self.ddnm_plus_final_imagenet(model)
            else:
                print('dataset not supported')
                
            # self.ddnm_plus_(model)
            # if self.args.mode == "implicit":
            #     self.implicit_simplified_ddnm_plus(model, cls_fn)
            # elif self.args.mode == "explicit":
            #     self.explicit_simplified_ddnm_plus(model, cls_fn)
            # elif self.args.mode == "explicit_gt":
            #     self.explicit_gt_simplified_ddnm_plus(model, cls_fn)
            # elif self.args.mode == "combine":
            #     self.combine_simplified_ddnm_plus(model, cls_fn)
            # else:
            #     print("unknown mode, exit.")
                # self.simplified_ddnm_plus(model, cls_fn)
                # self.IRmodel_simplified_ddnm_plus(model, cls_fn)

            
        # else:
        #     print('Run SVD-based DDNM.',
        #           f'{self.config.time_travel.T_sampling} sampling steps.',
        #           f'travel_length = {self.config.time_travel.travel_length},',
        #           f'travel_repeat = {self.config.time_travel.travel_repeat}.',
        #           f'Task: {self.args.deg}.'
        #          )
        #     self.svd_based_ddnm_plus(model, cls_fn)
    def sample_x0t(self, model):
        args, config = self.args, self.config
        g = torch.Generator()
        g.manual_seed(args.seed)
        set_seed(args.seed)
        
        dataset_root = self.args.path_y
        all_files = os.listdir(os.path.join(dataset_root, 'hq'))
        if self.args.dataset == 'imagenet':
            filename_list = sorted([file[:-4] for file in all_files])
        else:
            filename_list = sorted([file[:-4] for file in all_files])
        os.makedirs(os.path.join(dataset_root, "x0_t"), exist_ok=True)


        self.scheduler = DDPMScheduler.from_pretrained('google/ddpm-celebahq-256')
        self.scheduler.set_timesteps(100)
        
        
        for filename in tqdm.tqdm(filename_list):
            os.makedirs(os.path.join(dataset_root, "x0_t", f'{filename}'), exist_ok=True)
            if self.args.dataset == 'imagenet':
                hq_path = os.path.join(dataset_root, 'hq', f'{filename}.jpg')
            elif self.args.dataset == 'ffhq':
                hq_path = os.path.join(dataset_root, 'hq', f'{filename}.png')
            else:
                hq_path = os.path.join(dataset_root, 'hq', f'{filename}.jpg')    
                
            HQ = load_imgDDNM(hq_path).to(self.device)
            HQ = data_transform(self.config, HQ)
            
            for i in range(0, 1000, 10):
                t = torch.randint(i, i + 1, [1], dtype=torch.long, device=self.device)
                if os.path.exists(os.path.join(dataset_root, "x0_t", f'{filename}', f'x0_{int(t.item())}.jpg')):
                    continue  
                noise = torch.randn_like(HQ)
                #add t step of noise
                x = self.scheduler.add_noise(HQ, noise, t)
                noise_pred = model(x, t)
                if noise_pred.size(1) == 6:
                    noise_pred = noise_pred[:, :3]
                # print(noise_pred.is_cuda, t.is_cuda, x.is_cuda)
                x0_t = self.scheduler.step(noise_pred.cpu(), t.cpu(), x.cpu())['pred_original_sample']
                # print(x0_t.shape)
                tvu.save_image(
                    inverse_data_transform(config, x0_t[0]),
                    os.path.join(dataset_root, "x0_t", f'{filename}', f'x0_{int(t.item())}.jpg')
                )
        return   



    # def ddnm_plus_final_imagenet(self, model):
    #     args, config = self.args, self.config
    #     g = torch.Generator()
    #     g.manual_seed(args.seed)
    #     set_seed(args.seed)
        
    #     # mkdirs
    #     print(f'result save to {self.args.image_folder}')
    #     os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
        
    #     # get all test image filename
    #     test_dataset_root = self.args.path_y
    #     all_files = os.listdir(os.path.join(test_dataset_root, 'hq'))
    #     filename_list = sorted([os.path.splitext(file)[0] for file in all_files])
    #     filename_list = filename_list[:self.args.sample_number]
            
    #     avg_output_psnr = 0.0
    #     avg_kernel_psnr = 0.0
    #     avg_apy_psnr = 0.0
    #     with torch.no_grad():
    #         #init A, Ap
    #         if self.args.DDNM_A == "implicit" or self.args.DDNM_Ap == "implicit" or self.args.perturb_A == "implicit" or self.args.perturb_Ap == "implicit":
    #             #version 1
    #             implicit_A = lambda z : self.IRmodel.net_g(z, z, LD_LQ)[1] #lQ, HQ, lq_condition or lq, x0t_gt, oirginal_lq for sametarget
    #             implicit_Ap = lambda z : self.IRmodel.net_g(z, z, LD_LQ)[0]

    #         for filename in tqdm.tqdm(filename_list):
    #             if os.path.exists(os.path.join(self.args.image_folder, f"{filename}.png")):
    #                 continue
                
    #             hq_path = os.path.join(test_dataset_root, 'hq', f'{filename}.JPEG')
    #             lq_path = os.path.join(test_dataset_root, 'lq', f'{int(self.args.deg_scale)}', f'{filename}.jpg')
    #             gt_kernel_path = os.path.join(test_dataset_root, 'kernel', f'{filename}.npy')
                
    #             HQ = load_imgDDNM(hq_path).to(self.device)
    #             HQ = data_transform(self.config, HQ)
                
    #             if self.args.mode == 'explicit' or self.args.mode == 'explicit_gt' or self.args.mode == 'combine':
    #                 gt_kernel = np.load(gt_kernel_path)
    #                 gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
    #                 gt_A = lambda z : convolution2d(z, gt_kernel, stride=int(self.args.deg_scale), padding=gt_kernel.size(1)//2) 
    #                 DDNM_LQ = gt_A(HQ)
    #                 tvu.save_image(
    #                     inverse_data_transform(config, DDNM_LQ[0]),
    #                     os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")
    #                 )
    #                 lq_path = os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")
    #             DDNM_LQ = load_imgDDNM(lq_path).to(self.device)
    #             DDNM_LQ = data_transform(self.config, DDNM_LQ)
                
    #             if self.args.DDNM_A == "implicit" or self.args.DDNM_Ap == "implicit" or self.args.perturb_A == "implicit" or self.args.perturb_Ap == "implicit":
    #                 LD_LQ = load_imgDDNM(lq_path).to(self.device)
    #                 LD_LQ = transforms.Compose([transforms.Resize((256,256), antialias=None),])(LD_LQ.squeeze()).unsqueeze(0)
                
    #             # get A, Ap for current input
    #             if self.args.DDNM_A == "explicit_gt" or self.args.DDNM_Ap == "explicit_gt" or self.args.perturb_A == "explicit_gt" or self.args.perturb_Ap == "explicit_gt":
    #                 gt_kernel = np.load(gt_kernel_path)
    #                 gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
    #                 gt_A = lambda z : convolution2d(z, gt_kernel, stride=int(self.args.deg_scale), padding=gt_kernel.size(1)//2) 

    #                 gt_padding = gt_kernel.shape[0] // 2
    #                 gt_matrix_A = convolution_to_A(gt_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=gt_padding).cuda()
    #                 gt_matrix_A_pinverse = torch.pinverse(gt_matrix_A)
    #                 gt_A = lambda z: convolution_with_A(gt_matrix_A, z, padding=gt_padding)
    #                 gt_Ap = lambda z: convolution_with_A(gt_matrix_A_pinverse, z, padding=0)[:, :, gt_padding:-gt_padding, gt_padding:-gt_padding]
    #             if self.args.DDNM_A == "explicit" or self.args.DDNM_Ap == "explicit" or self.args.perturb_A == "explicit" or self.args.perturb_Ap == "explicit":
    #                 gt_kernel = np.load(gt_kernel_path)
    #                 gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
    #                 # padding gt_kenel to 21 * 21
    #                 gt_kernel = gt_kernel.unsqueeze(0)
    #                 padding21 = (self.args.kernel_size - gt_kernel.size(1)) //2
    #                 gt_kernel = F.pad(gt_kernel, (padding21, padding21, padding21, padding21)).squeeze()
                    
    #                 predict_kernel = None
    #                 with torch.no_grad():
    #                     predict_kernel = self.kernel_estimator(DDNM_LQ).squeeze()
    #                     kernel_psnr = calculate_psnr(predict_kernel, gt_kernel)
    #                     avg_kernel_psnr += kernel_psnr
    #                     self.txt_logger.info(f'{filename} kernel psnr: {kernel_psnr}')
    #                 plot_kernel(gt_kernel.unsqueeze(0).unsqueeze(0), predict_kernel.unsqueeze(0).unsqueeze(0), filename, self.args.image_folder)
    #                 if self.args.DDNM_Ap == "explicit" or self.args.perturb_Ap == "explicit":
    #                     explicit_padding = predict_kernel.shape[0] // 2
    #                     matrix_A = convolution_to_A(predict_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=explicit_padding).cuda()
    #                     matrix_A_pinverse = torch.pinverse(matrix_A) 
    #                     explicit_A = lambda z: convolution_with_A(matrix_A, z, padding=explicit_padding)
    #                     explicit_Ap = lambda z: convolution_with_A(matrix_A_pinverse, z, padding=0)[:, :, explicit_padding:-explicit_padding, explicit_padding:-explicit_padding]
    #                 else:
    #                     explicit_A = lambda z : convolution2d(z, predict_kernel, stride=int(self.args.deg_scale), padding=predict_kernel.size(1)//2) 

    #             #check perturb Y, store the perturbed y 
    #             if self.args.perturb_y:
    #                 if self.args.perturb_A == "explicit" and self.args.perturb_Ap == "implicit":
    #                     implicit_Apy = implicit_Ap(LD_LQ)
    #                     implicit_Apy = data_transform(self.config, implicit_Apy)
    #                     DDNM_LQ = explicit_A(implicit_Apy)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, DDNM_LQ[0]),
    #                         os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")
    #                     )
    #                 elif self.args.perturb_A == "explicit_gt" and self.args.perturb_Ap == "implicit":
    #                     implicit_Apy = implicit_Ap(LD_LQ)
    #                     implicit_Apy = data_transform(self.config, implicit_Apy)
    #                     DDNM_LQ = gt_A(implicit_Apy)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, DDNM_LQ[0]),
    #                         os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")
    #                     )
    #                 elif self.args.perturb_A == "implicit" and self.args.perturb_Ap == "implicit":    
    #                     LD_LQ = implicit_A(implicit_Ap(LD_LQ))
    #                     LQ_img = tensor2img(LD_LQ, rgb2bgr=True)
    #                     imwrite(LQ_img, os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png"))
    #                     DDNM_LQ = data_transform(self.config, LD_LQ)
    #                 else:
    #                     raise ValueError(f"perturb mode {self.args.perturb_A}, {self.args.perturb_Ap} not supported")
                    
    #             # perform Ap(y)
    #             if self.args.DDNM_Ap == "explicit_gt":
    #                 Apy = gt_Ap(DDNM_LQ)
    #                 tvu.save_image(
    #                     inverse_data_transform(config, Apy[0]),
    #                     os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")
    #                 )
    #             elif self.args.DDNM_Ap == "explicit":
    #                 Apy = explicit_Ap(DDNM_LQ)
    #                 # print(Apy.min()) # < -1
    #                 # print(Apy.max()) # > 1
    #                 tvu.save_image(
    #                     inverse_data_transform(config, Apy[0]),
    #                     os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")
    #                 )
    #             elif self.args.DDNM_Ap == "implicit":
    #                 Apy = implicit_Ap(LD_LQ)
    #                 Apy_img = tensor2img(Apy, rgb2bgr=True)
    #                 imwrite(Apy_img, os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png"))
    #             else:
    #                 raise ValueError("DDNM Ap mode {self.args.DDNM_Ap} not supported")  

    #             if self.args.DDNM_Ap == "implicit": 
    #                 Apy = data_transform(self.config, Apy) # don't know why 
                    

    #             tvu.save_image(
    #                 inverse_data_transform(config, HQ[0]),
    #                 os.path.join(self.args.image_folder, f"Apy/orig_{filename}.png")
    #             )
    #             Apy_psnr = cal_img_psnr([inverse_data_transform(config, HQ[0])], [inverse_data_transform(config, Apy[0])])
    #             self.txt_logger.info(f'{filename} per sample apy psnr: {Apy_psnr}')
    #             avg_apy_psnr += Apy_psnr
                
    #             if self.args.DDNM_Ap == "implicit": 
    #                 Apy = inverse_data_transform(self.config, Apy) # dopn't know why, equalivant to alpha = 0.5
                
    #             # init x_T
    #             x = torch.randn(
    #                 HQ.shape[0],
    #                 config.data.channels,
    #                 config.data.image_size,
    #                 config.data.image_size,
    #                 device=self.device,
    #             )
    #             # with torch.no_grad():
    #             skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
    #             n = x.size(0)
    #             x0_preds = []
    #             xs = [x]
                
    #             times = get_schedule_jump(config.time_travel.T_sampling, 
    #                                             config.time_travel.travel_length, 
    #                                             config.time_travel.travel_repeat,
    #                                             )
    #             time_pairs = list(zip(times[:-1], times[1:])) 
                
    #             # reverse diffusion sampling
    #             for i, j in time_pairs:
    #                 i, j = i*skip, j*skip
    #                 if j<0: 
    #                     j=-1 
    #                 if j < i: # normal sampling 
    #                     t = (torch.ones(n) * i).to(x.device)
    #                     next_t = (torch.ones(n) * j).to(x.device)
    #                     at = compute_alpha(self.betas, t.long())
    #                     at_next = compute_alpha(self.betas, next_t.long())
    #                     xt = xs[-1].to('cuda')
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "x_t", f'{filename}'), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, xt.to('cpu')),
    #                             os.path.join(self.args.image_folder, "x_t", f"{filename}/x_{int(t[0])}.png")
    #                         )     
                        
    #                     with torch.no_grad():
    #                         et = model(xt, t)
                            
    #                     if et.size(1) == 6:
    #                         et = et[:, :3]
    #                     # Eq. 12
    #                     x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() 
                        
    #                     # save x0_t
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "x0_t", f'{filename}'), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, x0_t.to('cpu')),
    #                             os.path.join(self.args.image_folder, "x0_t", f"{filename}/x0_{int(t[0])}.png")
    #                         )

    #                     # Eq. 13
    #                     # x0_t_hat = ApA(Apy) + x0_t - ApA(x0_t)
    #                     #get ApA(x0_t)
    #                     if self.args.DDNM_A == 'implicit' and self.args.DDNM_Ap == 'implicit':
    #                         x0_tL = inverse_data_transform(config, x0_t)
    #                         ApAx0_t = implicit_Ap(implicit_A(x0_tL))
                                            
    #                         if self.args.save_img:
    #                             ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True) 
    #                             os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                             imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, "ApA", f"{filename}/ApAx0_{int(t[0])}.png"))
    #                         # ApAx0_t = data_transform(config, ApAx0_t) # equalivant to alpha = 0.5
    #                     elif self.args.DDNM_A == 'explicit' and self.args.DDNM_Ap == 'implicit':
    #                         Ax0_t = explicit_A(x0_t)
    #                         if self.args.save_img:
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, Ax0_t.to('cpu')),
    #                                 os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")
    #                             )
    #                         Ax0_t = inverse_data_transform(config, Ax0_t)
    #                         Ax0_t = transforms.Compose([transforms.Resize((256,256), antialias=None),])(Ax0_t.squeeze()).unsqueeze(0)
    #                         ApAx0_t = implicit_Ap(Ax0_t)
    #                         if self.args.save_img:
    #                             ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)
    #                             os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                             imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png"))
    #                         # ApAx0_t = data_transform(self.config, ApAx0_t) # equalivant to alpha = 0.5
                            
    #                     elif self.args.DDNM_A == 'explicit_gt' and self.args.DDNM_Ap == 'implicit':
    #                         Ax0_t = gt_A(x0_t)
    #                         if self.args.save_img:
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, Ax0_t.to('cpu')),
    #                                 os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")
    #                             )
    #                         Ax0_t = inverse_data_transform(config, Ax0_t) 
    #                         Ax0_t = transforms.Compose([transforms.Resize((256,256), antialias=None),])(Ax0_t.squeeze()).unsqueeze(0)
    #                         ApAx0_t = implicit_Ap(Ax0_t)
                            
    #                         if self.args.save_img:
    #                             ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)
    #                             os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                             imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png"))
    #                         # ApAx0_t = data_transform(self.config, ApAx0_t) # equalivant to alpha = 0.5
                            
    #                     elif self.args.DDNM_A == 'explicit' and self.args.DDNM_Ap == 'explicit':
    #                         ApAx0_t = explicit_Ap(explicit_A(x0_t))
    #                         if self.args.save_img:
    #                             os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, ApAx0_t[0].to('cpu')),
    #                                 os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")
    #                             )
                                
    #                     elif self.args.DDNM_A == 'explicit_gt' and self.args.DDNM_Ap == 'explicit_gt':
    #                         ApAx0_t = gt_Ap(gt_A(x0_t))
    #                         if self.args.save_img:
    #                             os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, ApAx0_t[0]),
    #                                 os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")
    #                             )
    #                     else:
    #                         raise ValueError("DDNM A mode {self.args.DDNM_A} with Ap mode {self.args.DDNM_Ap} not supported")  
                        
    #                     alpha = self.args.alpha
    #                     if self.args.posterior_formula == "DDNM" and self.args.DDNM_Ap == 'implicit':
    #                         alpha = alpha * 2
    #                     x0_t_hat = x0_t + alpha * (Apy - ApAx0_t) # DDNM formula
                        
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "x0_t_hat", f'{filename}'), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, x0_t_hat),
    #                             os.path.join(self.args.image_folder, "x0_t_hat", f"{filename}/x0_{int(t[0])}_hat.png")
    #                         ) 
                        
    #                     eta = self.args.eta
    #                     if self.args.posterior_formula == "DDIM":
    #                         sigma = (
    #                             eta
    #                             * torch.sqrt((1 - at_next) / (1 - at))
    #                             * torch.sqrt(1 - at / at_next)
    #                         )
    #                         mean_pred = (
    #                             x0_t_hat * torch.sqrt(at_next)
    #                             + torch.sqrt(1 - at_next - sigma ** 2) * et
    #                         )
    #                         xt_next = mean_pred + sigma * torch.randn_like(x0_t)
    #                     elif self.args.posterior_formula == "DDNM":
    #                         c1 = (1 - at_next).sqrt() * eta
    #                         c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
    #                         # sigma_t = (1 - at_next**2).sqrt()
    #                         # different from the paper, we use DDIM here instead of DDPM
    #                         xt_next = at_next.sqrt() * x0_t_hat + (c1 * torch.randn_like(x0_t) + c2 * et)

    #                     x0_preds.append(x0_t.to('cpu'))
    #                     xs.append(xt_next.to('cpu'))   
    #                 else: # time-travel back
    #                     raise NotImplementedError

    #             x = xs[-1]
    #             x = [inverse_data_transform(config, xi) for xi in x]

    #             tvu.save_image(
    #                 x[0], os.path.join(self.args.image_folder, f"{filename}.png")
    #             )
                
    #             output_psnr = cal_img_psnr([inverse_data_transform(config, HQ[0])], x)
    #             self.txt_logger.info(f'{filename} per sample psnr: {output_psnr}')
    #             avg_output_psnr += output_psnr

    #         avg_output_psnr = avg_output_psnr / len(filename_list)
    #         avg_apy_psnr = avg_apy_psnr / len(filename_list)
    #         avg_kernel_psnr = avg_kernel_psnr / len(filename_list)
    #         self.txt_logger.info(f"Total Average kernel PSNR: {avg_kernel_psnr}")
    #         self.txt_logger.info(f"Apy Total Average PSNR: {avg_apy_psnr}")
    #         self.txt_logger.info(f"Output Total Average PSNR: {avg_output_psnr}")
    #         self.txt_logger.info(f"Number of samples: {len(filename_list)}")

    #         #eval
    #         lpips_model = LPIPS(net='alex').to('cuda')
    #         folder = self.args.image_folder
    #         print('eval folder', folder)
    #         N = len(filename_list)
    #         total_output_psnr = 0.0
    #         total_output_ssim = 0.0
    #         total_output_lpips = 0.0
            
    #         total_apy_psnr = 0.0
    #         total_apy_ssim = 0.0
    #         total_apy_lpips = 0.0
    #         for filename in tqdm.tqdm(filename_list):
    #             predict_path = os.path.join(folder, f'{filename}.png')   
    #             gt_path = os.path.join(folder, 'Apy', f'orig_{filename}.png')
    #             apy_path = os.path.join(folder, 'Apy', f'Apy_{filename}.png')

    #             predict_img = Image.open(predict_path).convert("RGB")
    #             apy_img = Image.open(apy_path).convert("RGB")
    #             gt_img = Image.open(gt_path).convert("RGB")
                
    #             total_output_psnr += calculate_psnr_eval(predict_img, gt_img)
    #             total_output_ssim += calculate_ssim_eval(predict_img, gt_img)
    #             total_output_lpips += calculate_lpips_eval(lpips_model, predict_img, gt_img)

    #             total_apy_psnr += calculate_psnr_eval(apy_img, gt_img)
    #             total_apy_ssim += calculate_ssim_eval(apy_img, gt_img)
    #             total_apy_lpips += calculate_lpips_eval(lpips_model, apy_img, gt_img)
    #         print('output avg psnr:', total_output_psnr / N) 
    #         print('output avg ssim:', total_output_ssim / N) 
    #         print('output avg lpips:', total_output_lpips / N) 

    #         print('apy avg psnr:', total_apy_psnr / N) 
    #         print('apy avg ssim:', total_apy_ssim / N) 
    #         print('apy avg lpips:', total_apy_lpips / N) 

    #         with open(os.path.join(folder, '_log_val_.txt'), 'w') as f:
    #             f.writelines(f'output avg psnr: {total_output_psnr / N} \n')
    #             f.writelines(f'output ssim psnr: {total_output_ssim / N} \n')
    #             f.writelines(f'output lpips psnr: {total_output_lpips / N} \n')
    #             f.writelines(f'apy avg psnr: {total_apy_psnr / N} \n')
    #             f.writelines(f'apy ssim psnr: {total_apy_ssim / N} \n')
    #             f.writelines(f'apy lpips psnr: {total_apy_lpips / N} \n')

    def ddnm_plus_final(self, model):
        args, config = self.args, self.config
        g = torch.Generator()
        g.manual_seed(args.seed)
        set_seed(args.seed)
        
        # mkdirs
        print(f'result save to {self.args.image_folder}')
        os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
        
        # get all test image filename
        test_dataset_root = self.args.path_y
        all_files = os.listdir(os.path.join(test_dataset_root, 'HR'))
        filename_list = sorted([os.path.splitext(file)[0] for file in all_files])
        if self.args.sample_number == -1:
            filename_list = filename_list
        else:
            filename_list = filename_list[:self.args.sample_number]
            
        avg_output_psnr = 0.0
        avg_kernel_psnr = 0.0
        avg_apy_psnr = 0.0
        with torch.no_grad():
            #init A, Ap
            if self.args.DDNM_A == "implicit" or self.args.DDNM_Ap == "implicit" or self.args.perturb_A == "implicit" or self.args.perturb_Ap == "implicit":
                #version 1
                implicit_A = lambda z : self.IRmodel.net_g(z, z, LD_LQ)[1] #lQ, HQ, lq_condition or lq, x0t_gt, oirginal_lq for sametarget
                implicit_Ap = lambda z : self.IRmodel.net_g(z, z, LD_LQ)[0]

            for filename in tqdm.tqdm(filename_list):
                if os.path.exists(os.path.join(self.args.image_folder, f"{filename}.png")):
                    continue
                
                hq_path = os.path.join(test_dataset_root, 'HR', f'{filename}.jpg')
                lq_path = os.path.join(test_dataset_root, 'LR', f'{int(self.args.deg_scale)}', f'{filename}.jpg')
                gt_kernel_path = os.path.join(test_dataset_root, 'kernel', f'{filename}.npy')
                
                HQ = load_imgDDNM(hq_path).to(self.device)
                HQ = data_transform(self.config, HQ)
                if self.args.mode == 'explicit' or self.args.mode == 'explicit_gt' or self.args.mode == 'combine':
                    gt_kernel = np.load(gt_kernel_path)
                    gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
                    gt_A = lambda z : convolution2d(z, gt_kernel, stride=int(self.args.deg_scale), padding=gt_kernel.size(1)//2) 
                    DDNM_LQ = gt_A(HQ)
                    tvu.save_image(
                        inverse_data_transform(config, DDNM_LQ[0]),
                        os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")
                    )
                    lq_path = os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")
                DDNM_LQ = load_imgDDNM(lq_path).to(self.device)
                DDNM_LQ = data_transform(self.config, DDNM_LQ)
                
                if self.args.DDNM_A == "implicit" or self.args.DDNM_Ap == "implicit" or self.args.perturb_A == "implicit" or self.args.perturb_Ap == "implicit":
                    LD_LQ = load_imgDDNM(lq_path).to(self.device)
                    LD_LQ = transforms.Compose([transforms.Resize((256,256), antialias=None),])(LD_LQ.squeeze()).unsqueeze(0)
                
                # get A, Ap for current input
                if self.args.DDNM_A == "explicit_gt" or self.args.DDNM_Ap == "explicit_gt" or self.args.perturb_A == "explicit_gt" or self.args.perturb_Ap == "explicit_gt":
                    gt_kernel = np.load(gt_kernel_path)
                    gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
                    gt_A = lambda z : convolution2d(z, gt_kernel, stride=int(self.args.deg_scale), padding=gt_kernel.size(1)//2) 

                    gt_padding = gt_kernel.shape[0] // 2
                    gt_matrix_A = convolution_to_A(gt_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=gt_padding).cuda()
                    gt_matrix_A_pinverse = torch.pinverse(gt_matrix_A)
                    gt_A = lambda z: convolution_with_A(gt_matrix_A, z, padding=gt_padding)
                    gt_Ap = lambda z: convolution_with_A(gt_matrix_A_pinverse, z, padding=0)[:, :, gt_padding:-gt_padding, gt_padding:-gt_padding]
                if self.args.DDNM_A == "explicit" or self.args.DDNM_Ap == "explicit" or self.args.perturb_A == "explicit" or self.args.perturb_Ap == "explicit":
                    gt_kernel = np.load(gt_kernel_path)
                    gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
                    # padding gt_kenel to 21 * 21
                    gt_kernel = gt_kernel.unsqueeze(0)
                    padding21 = (self.args.kernel_size - gt_kernel.size(1)) //2
                    gt_kernel = F.pad(gt_kernel, (padding21, padding21, padding21, padding21)).squeeze()
                    
                    predict_kernel = None
                    with torch.no_grad():
                        predict_kernel = self.kernel_estimator(DDNM_LQ).squeeze()
                        kernel_psnr = calculate_psnr(predict_kernel, gt_kernel)
                        avg_kernel_psnr += kernel_psnr
                        self.txt_logger.info(f'{filename} kernel psnr: {kernel_psnr}')
                    plot_kernel(gt_kernel.unsqueeze(0).unsqueeze(0), predict_kernel.unsqueeze(0).unsqueeze(0), filename, self.args.image_folder)
                    if self.args.DDNM_Ap == "explicit" or self.args.perturb_Ap == "explicit":
                        explicit_padding = predict_kernel.shape[0] // 2
                        matrix_A = convolution_to_A(predict_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=explicit_padding).cuda()
                        matrix_A_pinverse = torch.pinverse(matrix_A) 
                        explicit_A = lambda z: convolution_with_A(matrix_A, z, padding=explicit_padding)
                        explicit_Ap = lambda z: convolution_with_A(matrix_A_pinverse, z, padding=0)[:, :, explicit_padding:-explicit_padding, explicit_padding:-explicit_padding]
                    else:
                        explicit_A = lambda z : convolution2d(z, predict_kernel, stride=int(self.args.deg_scale), padding=predict_kernel.size(1)//2) 

                #check perturb Y, store the perturbed y 
                if self.args.perturb_y:
                    if self.args.perturb_A == "explicit" and self.args.perturb_Ap == "implicit":
                        implicit_Apy = implicit_Ap(LD_LQ)
                        implicit_Apy = data_transform(self.config, implicit_Apy)
                        DDNM_LQ = explicit_A(implicit_Apy)
                        tvu.save_image(
                            inverse_data_transform(config, DDNM_LQ[0]),
                            os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")
                        )
                    elif self.args.perturb_A == "explicit_gt" and self.args.perturb_Ap == "implicit":
                        implicit_Apy = implicit_Ap(LD_LQ)
                        implicit_Apy = data_transform(self.config, implicit_Apy)
                        DDNM_LQ = gt_A(implicit_Apy)
                        tvu.save_image(
                            inverse_data_transform(config, DDNM_LQ[0]),
                            os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")
                        )
                    elif self.args.perturb_A == "implicit" and self.args.perturb_Ap == "implicit":    
                        LD_LQ = implicit_A(implicit_Ap(LD_LQ))
                        LQ_img = tensor2img(LD_LQ, rgb2bgr=True)
                        imwrite(LQ_img, os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png"))
                        DDNM_LQ = data_transform(self.config, LD_LQ)
                    else:
                        raise ValueError(f"perturb mode {self.args.perturb_A}, {self.args.perturb_Ap} not supported")
                    
                # perform Ap(y)
                if self.args.DDNM_Ap == "explicit_gt":
                    Apy = gt_Ap(DDNM_LQ)
                    tvu.save_image(
                        inverse_data_transform(config, Apy[0]),
                        os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")
                    )
                elif self.args.DDNM_Ap == "explicit":
                    Apy = explicit_Ap(DDNM_LQ)
                    # print(Apy.min()) # < -1
                    # print(Apy.max()) # > 1
                    tvu.save_image(
                        inverse_data_transform(config, Apy[0]),
                        os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")
                    )
                elif self.args.DDNM_Ap == "implicit":
                    Apy = implicit_Ap(LD_LQ)
                    Apy_img = tensor2img(Apy, rgb2bgr=True)
                    imwrite(Apy_img, os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png"))
                else:
                    raise ValueError("DDNM Ap mode {self.args.DDNM_Ap} not supported")  

                if self.args.DDNM_Ap == "implicit": 
                    Apy = data_transform(self.config, Apy) # don't know why 
                    

                tvu.save_image(
                    inverse_data_transform(config, HQ[0]),
                    os.path.join(self.args.image_folder, f"Apy/orig_{filename}.png")
                )
                Apy_psnr = cal_img_psnr([inverse_data_transform(config, HQ[0])], [inverse_data_transform(config, Apy[0])])
                self.txt_logger.info(f'{filename} per sample apy psnr: {Apy_psnr}')
                avg_apy_psnr += Apy_psnr
                
                if self.args.DDNM_Ap == "implicit": 
                    Apy = inverse_data_transform(self.config, Apy) # dopn't know why, equalivant to alpha = 0.5
                
                # init x_T
                x = torch.randn(
                    HQ.shape[0],
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                # with torch.no_grad():
                skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
                n = x.size(0)
                x0_preds = []
                xs = [x]
                
                times = get_schedule_jump(config.time_travel.T_sampling, 
                                                config.time_travel.travel_length, 
                                                config.time_travel.travel_repeat,
                                                )
                time_pairs = list(zip(times[:-1], times[1:])) 
                
                # reverse diffusion sampling
                for i, j in time_pairs:
                    i, j = i*skip, j*skip
                    if j<0: 
                        j=-1 
                    if j < i: # normal sampling 
                        t = (torch.ones(n) * i).to(x.device)
                        next_t = (torch.ones(n) * j).to(x.device)
                        at = compute_alpha(self.betas, t.long())
                        at_next = compute_alpha(self.betas, next_t.long())
                        xt = xs[-1].to('cuda')
                        if self.args.save_img:
                            os.makedirs(os.path.join(self.args.image_folder, "x_t", f'{filename}'), exist_ok=True)
                            tvu.save_image(
                                inverse_data_transform(config, xt.to('cpu')),
                                os.path.join(self.args.image_folder, "x_t", f"{filename}/x_{int(t[0])}.png")
                            )     
                        
                        with torch.no_grad():
                            et = model(xt, t)
                            
                        if et.size(1) == 6:
                            et = et[:, :3]
                        # Eq. 12
                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() 
                        
                        # save x0_t
                        if self.args.save_img:
                            os.makedirs(os.path.join(self.args.image_folder, "x0_t", f'{filename}'), exist_ok=True)
                            tvu.save_image(
                                inverse_data_transform(config, x0_t.to('cpu')),
                                os.path.join(self.args.image_folder, "x0_t", f"{filename}/x0_{int(t[0])}.png")
                            )

                        # Eq. 13
                        # x0_t_hat = ApA(Apy) + x0_t - ApA(x0_t)
                        #get ApA(x0_t)
                        if self.args.DDNM_A == 'implicit' and self.args.DDNM_Ap == 'implicit':
                            x0_tL = inverse_data_transform(config, x0_t)
                            ApAx0_t = implicit_Ap(implicit_A(x0_tL))
                                            
                            if self.args.save_img:
                                ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True) 
                                os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
                                imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, "ApA", f"{filename}/ApAx0_{int(t[0])}.png"))
                            # ApAx0_t = data_transform(config, ApAx0_t) # equalivant to alpha = 0.5
                        elif self.args.DDNM_A == 'explicit' and self.args.DDNM_Ap == 'implicit':
                            Ax0_t = explicit_A(x0_t)
                            if self.args.save_img:
                                tvu.save_image(
                                    inverse_data_transform(config, Ax0_t.to('cpu')),
                                    os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")
                                )
                            Ax0_t = inverse_data_transform(config, Ax0_t)
                            Ax0_t = transforms.Compose([transforms.Resize((256,256), antialias=None),])(Ax0_t.squeeze()).unsqueeze(0)
                            ApAx0_t = implicit_Ap(Ax0_t)
                            if self.args.save_img:
                                ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)
                                os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
                                imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png"))
                            # ApAx0_t = data_transform(self.config, ApAx0_t) # equalivant to alpha = 0.5
                            
                        elif self.args.DDNM_A == 'explicit_gt' and self.args.DDNM_Ap == 'implicit':
                            Ax0_t = gt_A(x0_t)
                            if self.args.save_img:
                                tvu.save_image(
                                    inverse_data_transform(config, Ax0_t.to('cpu')),
                                    os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")
                                )
                            Ax0_t = inverse_data_transform(config, Ax0_t) 
                            Ax0_t = transforms.Compose([transforms.Resize((256,256), antialias=None),])(Ax0_t.squeeze()).unsqueeze(0)
                            ApAx0_t = implicit_Ap(Ax0_t)
                            
                            if self.args.save_img:
                                ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)
                                os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
                                imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png"))
                            # ApAx0_t = data_transform(self.config, ApAx0_t) # equalivant to alpha = 0.5
                            
                        elif self.args.DDNM_A == 'explicit' and self.args.DDNM_Ap == 'explicit':
                            ApAx0_t = explicit_Ap(explicit_A(x0_t))
                            if self.args.save_img:
                                os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
                                tvu.save_image(
                                    inverse_data_transform(config, ApAx0_t[0].to('cpu')),
                                    os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")
                                )
                                
                        elif self.args.DDNM_A == 'explicit_gt' and self.args.DDNM_Ap == 'explicit_gt':
                            ApAx0_t = gt_Ap(gt_A(x0_t))
                            if self.args.save_img:
                                os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
                                tvu.save_image(
                                    inverse_data_transform(config, ApAx0_t[0]),
                                    os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")
                                )
                        else:
                            raise ValueError("DDNM A mode {self.args.DDNM_A} with Ap mode {self.args.DDNM_Ap} not supported")  
                        
                        alpha = self.args.alpha
                        if self.args.posterior_formula == "DDNM" and self.args.DDNM_Ap == 'implicit':
                            alpha = alpha * 2
                        x0_t_hat = x0_t + alpha * (Apy - ApAx0_t) # DDNM formula
                        
                        if self.args.save_img:
                            os.makedirs(os.path.join(self.args.image_folder, "x0_t_hat", f'{filename}'), exist_ok=True)
                            tvu.save_image(
                                inverse_data_transform(config, x0_t_hat),
                                os.path.join(self.args.image_folder, "x0_t_hat", f"{filename}/x0_{int(t[0])}_hat.png")
                            ) 
                        
                        eta = self.args.eta
                        if self.args.posterior_formula == "DDIM":
                            sigma = (
                                eta
                                * torch.sqrt((1 - at_next) / (1 - at))
                                * torch.sqrt(1 - at / at_next)
                            )
                            mean_pred = (
                                x0_t_hat * torch.sqrt(at_next)
                                + torch.sqrt(1 - at_next - sigma ** 2) * et
                            )
                            xt_next = mean_pred + sigma * torch.randn_like(x0_t)
                        elif self.args.posterior_formula == "DDNM":
                            c1 = (1 - at_next).sqrt() * eta
                            c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                            sigma_t = (1 - at_next**2).sqrt()
                            # different from the paper, we use DDIM here instead of DDPM
                            xt_next = at_next.sqrt() * x0_t_hat + (c1 * torch.randn_like(x0_t) + c2 * et)

                        x0_preds.append(x0_t.to('cpu'))
                        xs.append(xt_next.to('cpu'))   
                    else: # time-travel back
                        raise NotImplementedError

                x = xs[-1]
                x = [inverse_data_transform(config, xi) for xi in x]

                tvu.save_image(
                    x[0], os.path.join(self.args.image_folder, f"{filename}.png")
                )
                
                output_psnr = cal_img_psnr([inverse_data_transform(config, HQ[0])], x)
                self.txt_logger.info(f'{filename} per sample psnr: {output_psnr}')
                avg_output_psnr += output_psnr

            avg_output_psnr = avg_output_psnr / len(filename_list)
            avg_apy_psnr = avg_apy_psnr / len(filename_list)
            avg_kernel_psnr = avg_kernel_psnr / len(filename_list)
            self.txt_logger.info(f"Total Average kernel PSNR: {avg_kernel_psnr}")
            self.txt_logger.info(f"Apy Total Average PSNR: {avg_apy_psnr}")
            self.txt_logger.info(f"Output Total Average PSNR: {avg_output_psnr}")
            self.txt_logger.info(f"Number of samples: {len(filename_list)}")

            #eval
            lpips_model = LPIPS(net='alex').to('cuda')
            folder = self.args.image_folder
            print('eval folder', folder)
            N = len(filename_list)
            total_output_psnr = 0.0
            total_output_ssim = 0.0
            total_output_lpips = 0.0
            
            total_apy_psnr = 0.0
            total_apy_ssim = 0.0
            total_apy_lpips = 0.0
            for filename in tqdm.tqdm(filename_list):
                predict_path = os.path.join(folder, f'{filename}.png')   
                gt_path = os.path.join(folder, 'Apy', f'orig_{filename}.png')
                apy_path = os.path.join(folder, 'Apy', f'Apy_{filename}.png')

                predict_img = Image.open(predict_path).convert("RGB")
                apy_img = Image.open(apy_path).convert("RGB")
                gt_img = Image.open(gt_path).convert("RGB")
                
                total_output_psnr += calculate_psnr_eval(predict_img, gt_img)
                total_output_ssim += calculate_ssim_eval(predict_img, gt_img)
                total_output_lpips += calculate_lpips_eval(lpips_model, predict_img, gt_img)

                total_apy_psnr += calculate_psnr_eval(apy_img, gt_img)
                total_apy_ssim += calculate_ssim_eval(apy_img, gt_img)
                total_apy_lpips += calculate_lpips_eval(lpips_model, apy_img, gt_img)
            print('output avg psnr:', total_output_psnr / N) 
            print('output avg ssim:', total_output_ssim / N) 
            print('output avg lpips:', total_output_lpips / N) 

            print('apy avg psnr:', total_apy_psnr / N) 
            print('apy avg ssim:', total_apy_ssim / N) 
            print('apy avg lpips:', total_apy_lpips / N) 

            with open(os.path.join(folder, '_log_val_.txt'), 'w') as f:
                f.writelines(f'output avg psnr: {total_output_psnr / N} \n')
                f.writelines(f'output ssim psnr: {total_output_ssim / N} \n')
                f.writelines(f'output lpips psnr: {total_output_lpips / N} \n')
                f.writelines(f'apy avg psnr: {total_apy_psnr / N} \n')
                f.writelines(f'apy ssim psnr: {total_apy_ssim / N} \n')
                f.writelines(f'apy lpips psnr: {total_apy_lpips / N} \n')


    def ddnm_plus_div2k(self, model):
        args, config = self.args, self.config
        g = torch.Generator()
        g.manual_seed(args.seed)
        set_seed(args.seed)
        
        # mkdirs
        print(f'result save to {self.args.image_folder}')
        os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
        
        # get all test image filename
        test_dataset_root = self.args.path_y
        all_files = os.listdir(os.path.join(test_dataset_root, 'HR'))
        filename_list = sorted([os.path.splitext(file)[0] for file in all_files])
        if self.args.sample_number != -1:
            filename_list = filename_list[:self.args.sample_number]
            
        avg_output_psnr = 0.0
        avg_kernel_psnr = 0.0
        avg_apy_psnr = 0.0
        with torch.no_grad():
            #init A, Ap
            if self.args.DDNM_A == "implicit" or self.args.DDNM_Ap == "implicit" or self.args.perturb_A == "implicit" or self.args.perturb_Ap == "implicit":
                #version 1
                implicit_A = lambda z : self.IRmodel.net_g(z, z, LD_LQ)[1] #lQ, HQ, lq_condition or lq, x0t_gt, oirginal_lq for sametarget
                implicit_Ap = lambda z : self.IRmodel.net_g(z, z, LD_LQ)[0]

            for filename in tqdm.tqdm(filename_list):
                if os.path.exists(os.path.join(self.args.image_folder, f"{filename}.png")):
                    continue
                
                hq_path = os.path.join(test_dataset_root, 'HR', f'{filename}.png')
                lq_path = os.path.join(test_dataset_root, 'LR', f'{int(self.args.deg_scale)}', f'{filename}.png')
                gt_kernel_path = os.path.join(test_dataset_root, 'kernel', f'{filename}.npy')
                
                HQ = load_imgDDNM(hq_path).to(self.device)
                HQ = data_transform(self.config, HQ)
                if os.path.exists(gt_kernel_path) and self.args.mode == 'explicit' or self.args.mode == 'explicit_gt' or self.args.mode == 'combine':
                    gt_kernel = np.load(gt_kernel_path)
                    gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
                    gt_A = lambda z : convolution2d(z, gt_kernel, stride=int(self.args.deg_scale), padding=gt_kernel.size(1)//2) 
                    DDNM_LQ = gt_A(HQ)
                    tvu.save_image(
                        inverse_data_transform(config, DDNM_LQ[0]),
                        os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")
                    )
                    lq_path = os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")
                DDNM_LQ = load_imgDDNM(lq_path).to(self.device)
                DDNM_LQ = data_transform(self.config, DDNM_LQ)
                
                if self.args.DDNM_A == "implicit" or self.args.DDNM_Ap == "implicit" or self.args.perturb_A == "implicit" or self.args.perturb_Ap == "implicit":
                    LD_LQ = load_imgDDNM(lq_path).to(self.device)
                    LD_LQ = transforms.Compose([transforms.Resize((256,256), antialias=None),])(LD_LQ.squeeze()).unsqueeze(0)
                
                # get A, Ap for current input
                if self.args.DDNM_A == "explicit_gt" or self.args.DDNM_Ap == "explicit_gt" or self.args.perturb_A == "explicit_gt" or self.args.perturb_Ap == "explicit_gt":
                    gt_kernel = np.load(gt_kernel_path)
                    gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
                    gt_A = lambda z : convolution2d(z, gt_kernel, stride=int(self.args.deg_scale), padding=gt_kernel.size(1)//2) 

                    gt_padding = gt_kernel.shape[0] // 2
                    gt_matrix_A = convolution_to_A(gt_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=gt_padding).cuda()
                    gt_matrix_A_pinverse = torch.pinverse(gt_matrix_A)
                    gt_A = lambda z: convolution_with_A(gt_matrix_A, z, padding=gt_padding)
                    gt_Ap = lambda z: convolution_with_A(gt_matrix_A_pinverse, z, padding=0)[:, :, gt_padding:-gt_padding, gt_padding:-gt_padding]
                if self.args.DDNM_A == "explicit" or self.args.DDNM_Ap == "explicit" or self.args.perturb_A == "explicit" or self.args.perturb_Ap == "explicit":

                    predict_kernel = None
                    with torch.no_grad():
                        predict_kernel = self.kernel_estimator(DDNM_LQ).squeeze()
                        
                    if os.path.exists(gt_kernel_path):
                        gt_kernel = np.load(gt_kernel_path)
                        gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
                        # padding gt_kenel to 21 * 21
                        gt_kernel = gt_kernel.unsqueeze(0)
                        padding21 = (self.args.kernel_size - gt_kernel.size(1)) //2
                        gt_kernel = F.pad(gt_kernel, (padding21, padding21, padding21, padding21)).squeeze()
                        kernel_psnr = calculate_psnr(predict_kernel, gt_kernel)
                        avg_kernel_psnr += kernel_psnr
                        self.txt_logger.info(f'{filename} kernel psnr: {kernel_psnr}')
                        plot_kernel(gt_kernel.unsqueeze(0).unsqueeze(0), predict_kernel.unsqueeze(0).unsqueeze(0), filename, self.args.image_folder)
                    if self.args.DDNM_Ap == "explicit" or self.args.perturb_Ap == "explicit":
                        explicit_padding = predict_kernel.shape[0] // 2
                        matrix_A = convolution_to_A(predict_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=explicit_padding).cuda()
                        matrix_A_pinverse = torch.pinverse(matrix_A) 
                        explicit_A = lambda z: convolution_with_A(matrix_A, z, padding=explicit_padding)
                        explicit_Ap = lambda z: convolution_with_A(matrix_A_pinverse, z, padding=0)[:, :, explicit_padding:-explicit_padding, explicit_padding:-explicit_padding]
                    else:
                        explicit_A = lambda z : convolution2d(z, predict_kernel, stride=int(self.args.deg_scale), padding=predict_kernel.size(1)//2) 

                #check perturb Y, store the perturbed y 
                if self.args.perturb_y:
                    if self.args.perturb_A == "explicit" and self.args.perturb_Ap == "implicit":
                        implicit_Apy = implicit_Ap(LD_LQ)
                        implicit_Apy = data_transform(self.config, implicit_Apy)
                        DDNM_LQ = explicit_A(implicit_Apy)
                        tvu.save_image(
                            inverse_data_transform(config, DDNM_LQ[0]),
                            os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")
                        )
                    elif self.args.perturb_A == "explicit_gt" and self.args.perturb_Ap == "implicit":
                        implicit_Apy = implicit_Ap(LD_LQ)
                        implicit_Apy = data_transform(self.config, implicit_Apy)
                        DDNM_LQ = gt_A(implicit_Apy)
                        tvu.save_image(
                            inverse_data_transform(config, DDNM_LQ[0]),
                            os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")
                        )
                    elif self.args.perturb_A == "implicit" and self.args.perturb_Ap == "implicit":    
                        LD_LQ = implicit_A(implicit_Ap(LD_LQ))
                        LQ_img = tensor2img(LD_LQ, rgb2bgr=True)
                        imwrite(LQ_img, os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png"))
                        DDNM_LQ = data_transform(self.config, LD_LQ)
                    else:
                        raise ValueError(f"perturb mode {self.args.perturb_A}, {self.args.perturb_Ap} not supported")
                    
                # perform Ap(y)
                if self.args.DDNM_Ap == "explicit_gt":
                    Apy = gt_Ap(DDNM_LQ)
                    tvu.save_image(
                        inverse_data_transform(config, Apy[0]),
                        os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")
                    )
                elif self.args.DDNM_Ap == "explicit":
                    Apy = explicit_Ap(DDNM_LQ)
                    # print(Apy.min()) # < -1
                    # print(Apy.max()) # > 1
                    tvu.save_image(
                        inverse_data_transform(config, Apy[0]),
                        os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")
                    )
                elif self.args.DDNM_Ap == "implicit":
                    Apy = implicit_Ap(LD_LQ)
                    Apy_img = tensor2img(Apy, rgb2bgr=True)
                    imwrite(Apy_img, os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png"))
                else:
                    raise ValueError("DDNM Ap mode {self.args.DDNM_Ap} not supported")  

                if self.args.DDNM_Ap == "implicit": 
                    Apy = data_transform(self.config, Apy) # don't know why 
                    

                tvu.save_image(
                    inverse_data_transform(config, HQ[0]),
                    os.path.join(self.args.image_folder, f"Apy/orig_{filename}.png")
                )
                Apy_psnr = cal_img_psnr([inverse_data_transform(config, HQ[0])], [inverse_data_transform(config, Apy[0])])
                self.txt_logger.info(f'{filename} per sample apy psnr: {Apy_psnr}')
                avg_apy_psnr += Apy_psnr
                
                if self.args.DDNM_Ap == "implicit": 
                    Apy = inverse_data_transform(self.config, Apy) # dopn't know why, equalivant to alpha = 0.5
                
                # init x_T
                x = torch.randn(
                    HQ.shape[0],
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                # with torch.no_grad():
                skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
                n = x.size(0)
                x0_preds = []
                xs = [x]
                
                times = get_schedule_jump(config.time_travel.T_sampling, 
                                                config.time_travel.travel_length, 
                                                config.time_travel.travel_repeat,
                                                )
                time_pairs = list(zip(times[:-1], times[1:])) 
                
                # reverse diffusion sampling
                for i, j in time_pairs:
                    i, j = i*skip, j*skip
                    if j<0: 
                        j=-1 
                    if j < i: # normal sampling 
                        t = (torch.ones(n) * i).to(x.device)
                        next_t = (torch.ones(n) * j).to(x.device)
                        at = compute_alpha(self.betas, t.long())
                        at_next = compute_alpha(self.betas, next_t.long())
                        xt = xs[-1].to('cuda')
                        if self.args.save_img:
                            os.makedirs(os.path.join(self.args.image_folder, "x_t", f'{filename}'), exist_ok=True)
                            tvu.save_image(
                                inverse_data_transform(config, xt.to('cpu')),
                                os.path.join(self.args.image_folder, "x_t", f"{filename}/x_{int(t[0])}.png")
                            )     
                        
                        with torch.no_grad():
                            et = model(xt, t)
                            
                        if et.size(1) == 6:
                            et = et[:, :3]
                        # Eq. 12
                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() 
                        
                        # save x0_t
                        if self.args.save_img:
                            os.makedirs(os.path.join(self.args.image_folder, "x0_t", f'{filename}'), exist_ok=True)
                            tvu.save_image(
                                inverse_data_transform(config, x0_t.to('cpu')),
                                os.path.join(self.args.image_folder, "x0_t", f"{filename}/x0_{int(t[0])}.png")
                            )

                        # Eq. 13
                        # x0_t_hat = ApA(Apy) + x0_t - ApA(x0_t)
                        #get ApA(x0_t)
                        if self.args.DDNM_A == 'implicit' and self.args.DDNM_Ap == 'implicit':
                            x0_tL = inverse_data_transform(config, x0_t)
                            ApAx0_t = implicit_Ap(implicit_A(x0_tL))
                                            
                            if self.args.save_img:
                                ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True) 
                                os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
                                imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, "ApA", f"{filename}/ApAx0_{int(t[0])}.png"))
                            # ApAx0_t = data_transform(config, ApAx0_t) # equalivant to alpha = 0.5
                        elif self.args.DDNM_A == 'explicit' and self.args.DDNM_Ap == 'implicit':
                            Ax0_t = explicit_A(x0_t)
                            if self.args.save_img:
                                tvu.save_image(
                                    inverse_data_transform(config, Ax0_t.to('cpu')),
                                    os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")
                                )
                            Ax0_t = inverse_data_transform(config, Ax0_t)
                            Ax0_t = transforms.Compose([transforms.Resize((256,256), antialias=None),])(Ax0_t.squeeze()).unsqueeze(0)
                            ApAx0_t = implicit_Ap(Ax0_t)
                            if self.args.save_img:
                                ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)
                                os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
                                imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png"))
                            # ApAx0_t = data_transform(self.config, ApAx0_t) # equalivant to alpha = 0.5
                            
                        elif self.args.DDNM_A == 'explicit_gt' and self.args.DDNM_Ap == 'implicit':
                            Ax0_t = gt_A(x0_t)
                            if self.args.save_img:
                                tvu.save_image(
                                    inverse_data_transform(config, Ax0_t.to('cpu')),
                                    os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")
                                )
                            Ax0_t = inverse_data_transform(config, Ax0_t) 
                            Ax0_t = transforms.Compose([transforms.Resize((256,256), antialias=None),])(Ax0_t.squeeze()).unsqueeze(0)
                            ApAx0_t = implicit_Ap(Ax0_t)
                            
                            if self.args.save_img:
                                ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)
                                os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
                                imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png"))
                            # ApAx0_t = data_transform(self.config, ApAx0_t) # equalivant to alpha = 0.5
                            
                        elif self.args.DDNM_A == 'explicit' and self.args.DDNM_Ap == 'explicit':
                            ApAx0_t = explicit_Ap(explicit_A(x0_t))
                            if self.args.save_img:
                                os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
                                tvu.save_image(
                                    inverse_data_transform(config, ApAx0_t[0].to('cpu')),
                                    os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")
                                )
                                
                        elif self.args.DDNM_A == 'explicit_gt' and self.args.DDNM_Ap == 'explicit_gt':
                            ApAx0_t = gt_Ap(gt_A(x0_t))
                            if self.args.save_img:
                                os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
                                tvu.save_image(
                                    inverse_data_transform(config, ApAx0_t[0]),
                                    os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")
                                )
                        else:
                            raise ValueError("DDNM A mode {self.args.DDNM_A} with Ap mode {self.args.DDNM_Ap} not supported")  
                        
                        alpha = self.args.alpha
                        if self.args.posterior_formula == "DDNM" and self.args.DDNM_Ap == 'implicit':
                            alpha = alpha * 2
                        x0_t_hat = x0_t + alpha * (Apy - ApAx0_t) # DDNM formula
                        
                        if self.args.save_img:
                            os.makedirs(os.path.join(self.args.image_folder, "x0_t_hat", f'{filename}'), exist_ok=True)
                            tvu.save_image(
                                inverse_data_transform(config, x0_t_hat),
                                os.path.join(self.args.image_folder, "x0_t_hat", f"{filename}/x0_{int(t[0])}_hat.png")
                            ) 
                        
                        eta = self.args.eta
                        if self.args.posterior_formula == "DDIM":
                            sigma = (
                                eta
                                * torch.sqrt((1 - at_next) / (1 - at))
                                * torch.sqrt(1 - at / at_next)
                            )
                            mean_pred = (
                                x0_t_hat * torch.sqrt(at_next)
                                + torch.sqrt(1 - at_next - sigma ** 2) * et
                            )
                            xt_next = mean_pred + sigma * torch.randn_like(x0_t)
                        elif self.args.posterior_formula == "DDNM":
                            c1 = (1 - at_next).sqrt() * eta
                            c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                            sigma_t = (1 - at_next**2).sqrt()
                            # different from the paper, we use DDIM here instead of DDPM
                            xt_next = at_next.sqrt() * x0_t_hat + (c1 * torch.randn_like(x0_t) + c2 * et)

                        x0_preds.append(x0_t.to('cpu'))
                        xs.append(xt_next.to('cpu'))   
                    else: # time-travel back
                        raise NotImplementedError

                x = xs[-1]
                x = [inverse_data_transform(config, xi) for xi in x]

                tvu.save_image(
                    x[0], os.path.join(self.args.image_folder, f"{filename}.png")
                )
                
                output_psnr = cal_img_psnr([inverse_data_transform(config, HQ[0])], x)
                self.txt_logger.info(f'{filename} per sample psnr: {output_psnr}')
                avg_output_psnr += output_psnr

            avg_output_psnr = avg_output_psnr / len(filename_list)
            avg_apy_psnr = avg_apy_psnr / len(filename_list)
            avg_kernel_psnr = avg_kernel_psnr / len(filename_list)
            self.txt_logger.info(f"Total Average kernel PSNR: {avg_kernel_psnr}")
            self.txt_logger.info(f"Apy Total Average PSNR: {avg_apy_psnr}")
            self.txt_logger.info(f"Output Total Average PSNR: {avg_output_psnr}")
            self.txt_logger.info(f"Number of samples: {len(filename_list)}")

            #eval
            lpips_model = LPIPS(net='alex').to('cuda')
            folder = self.args.image_folder
            print('eval folder', folder)
            N = len(filename_list)
            total_output_psnr = 0.0
            total_output_ssim = 0.0
            total_output_lpips = 0.0
            
            total_apy_psnr = 0.0
            total_apy_ssim = 0.0
            total_apy_lpips = 0.0
            for filename in tqdm.tqdm(filename_list):
                predict_path = os.path.join(folder, f'{filename}.png')   
                gt_path = os.path.join(folder, 'Apy', f'orig_{filename}.png')
                apy_path = os.path.join(folder, 'Apy', f'Apy_{filename}.png')

                predict_img = Image.open(predict_path).convert("RGB")
                apy_img = Image.open(apy_path).convert("RGB")
                gt_img = Image.open(gt_path).convert("RGB")
                
                total_output_psnr += calculate_psnr_eval(predict_img, gt_img)
                total_output_ssim += calculate_ssim_eval(predict_img, gt_img)
                total_output_lpips += calculate_lpips_eval(lpips_model, predict_img, gt_img)

                total_apy_psnr += calculate_psnr_eval(apy_img, gt_img)
                total_apy_ssim += calculate_ssim_eval(apy_img, gt_img)
                total_apy_lpips += calculate_lpips_eval(lpips_model, apy_img, gt_img)
            print('output avg psnr:', total_output_psnr / N) 
            print('output avg ssim:', total_output_ssim / N) 
            print('output avg lpips:', total_output_lpips / N) 

            print('apy avg psnr:', total_apy_psnr / N) 
            print('apy avg ssim:', total_apy_ssim / N) 
            print('apy avg lpips:', total_apy_lpips / N) 

            with open(os.path.join(folder, '_log_val_.txt'), 'w') as f:
                f.writelines(f'output avg psnr: {total_output_psnr / N} \n')
                f.writelines(f'output ssim psnr: {total_output_ssim / N} \n')
                f.writelines(f'output lpips psnr: {total_output_lpips / N} \n')
                f.writelines(f'apy avg psnr: {total_apy_psnr / N} \n')
                f.writelines(f'apy ssim psnr: {total_apy_ssim / N} \n')
                f.writelines(f'apy lpips psnr: {total_apy_lpips / N} \n')
                
                            
    # def ddnm_plus_celeba(self, model):
    #     args, config = self.args, self.config
    #     g = torch.Generator()
    #     g.manual_seed(args.seed)
    #     set_seed(args.seed)
        
    #     # mkdirs
    #     os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
        
    #     # get all test image filename
    #     test_dataset_root = self.args.path_y
    #     all_files = os.listdir(os.path.join(test_dataset_root, 'hq'))
    #     filename_list = sorted([file[:-4] for file in all_files])
    #     filename_list = filename_list[:self.args.sample_number]
            
    #     avg_output_psnr = 0.0
    #     avg_kernel_psnr = 0.0
    #     avg_apy_psnr = 0.0
        
    #     #init A, Ap
    #     if self.args.DDNM_A == "implicit" or self.args.DDNM_Ap == "implicit" or self.args.perturb_A == "implicit" or self.args.perturb_Ap == "implicit":
    #         #version 1
    #         implicit_A = lambda z : self.IRmodel.net_g(z, z, LD_LQ_gt)[1] #lQ, HQ, lq_condition or lq, x0t_gt, oirginal_lq for sametarget
    #         implicit_Ap = lambda z : self.IRmodel.net_g(z, z, LD_LQ_gt)[0]

    #         # #version 4 with timestep
    #         # implicit_A = lambda z : self.IRmodel.net_g(z, z, LD_LQ, LD_timestep, LD_timestep)[1] #lQ, HQ, lq_condition, timestepA, timestepAp
    #         # implicit_Ap = lambda z : self.IRmodel.net_g(z, z, LD_LQ, LD_timestep, LD_timestep)[0]

        
    #     for filename in tqdm.tqdm(filename_list):
    #         LD_timestep = torch.tensor([0], dtype=torch.long).to('cuda').unsqueeze(0)
    #         if os.path.exists(os.path.join(self.args.image_folder, f"{filename}.png")):
    #             continue
            
    #         hq_path = os.path.join(test_dataset_root, 'hq', f'{filename}.jpg')
    #         lq_path = os.path.join(test_dataset_root, 'lq', f'{int(self.args.deg_scale)}', f'{filename}.jpg')
    #         gt_kernel_path = os.path.join(test_dataset_root, 'kernel', f'{filename}.npy')
            
    #         HQ = load_imgDDNM(hq_path).to(self.device)
    #         HQ = data_transform(self.config, HQ)
    #         # gt_kernel = np.load(gt_kernel_path)
    #         # gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
    #         # gt_A = lambda z : convolution2d(z, gt_kernel, stride=int(self.args.deg_scale), padding=gt_kernel.size(1)//2) 
    #         # DDNM_LQ = gt_A(HQ)
    #         # tvu.save_image(
    #         #     inverse_data_transform(config, DDNM_LQ[0]),
    #         #     os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")
    #         # )
    #         # lq_path_gt = os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")
            
    #         #AHQ input
    #         lq_path = os.path.join('/eva_data4/shlu2240/Dataset/celeba_AHQ/x4_v2', f'{filename}.png')
    #         DDNM_LQ = load_imgDDNM(lq_path).to(self.device)
            
    #         if self.args.mode == "explicit_gt" or self.args.mode == "explicit":
    #             DDNM_LQ = transforms.Compose([transforms.Resize((64,64), antialias=None),])(DDNM_LQ.squeeze(0)).unsqueeze(0)
    #         DDNM_LQ = data_transform(self.config, DDNM_LQ)
            
    #         if self.args.DDNM_A == "implicit" or self.args.DDNM_Ap == "implicit" or self.args.perturb_A == "implicit" or self.args.perturb_Ap == "implicit":
    #             lq_path_gt = os.path.join(test_dataset_root, 'lq', f'{int(self.args.deg_scale)}', f'{filename}.jpg')
    #             LD_LQ_gt = load_img_LearningDegradation(lq_path_gt).to('cuda')
    #             LD_LQ = load_img_LearningDegradation(lq_path).to('cuda')
            
    #         # get A, Ap for current input
    #         if self.args.DDNM_A == "explicit_gt" or self.args.DDNM_Ap == "explicit_gt" or self.args.perturb_A == "explicit_gt" or self.args.perturb_Ap == "explicit_gt":
    #             gt_kernel = np.load(gt_kernel_path)
    #             gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
    #             gt_A = lambda z : convolution2d(z, gt_kernel, stride=int(self.args.deg_scale), padding=gt_kernel.size(1)//2) 

    #             gt_padding = gt_kernel.shape[0] // 2
    #             gt_matrix_A = convolution_to_A(gt_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=gt_padding).cuda()
    #             gt_matrix_A_pinverse = torch.pinverse(gt_matrix_A)
    #             gt_A = lambda z: convolution_with_A(gt_matrix_A, z, padding=gt_padding)
    #             gt_Ap = lambda z: convolution_with_A(gt_matrix_A_pinverse, z, padding=0)[:, :, gt_padding:-gt_padding, gt_padding:-gt_padding]
    #         if self.args.DDNM_A == "explicit" or self.args.DDNM_Ap == "explicit" or self.args.perturb_A == "explicit" or self.args.perturb_Ap == "explicit":
    #             gt_kernel = np.load(gt_kernel_path)
    #             gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
    #             # padding gt_kenel to 21 * 21
    #             gt_kernel = gt_kernel.unsqueeze(0)
    #             padding21 = (self.args.kernel_size - gt_kernel.size(1)) //2
    #             gt_kernel = F.pad(gt_kernel, (padding21, padding21, padding21, padding21)).squeeze()
                
    #             predict_kernel = None
    #             explicit_LQ = load_imgDDNM(lq_path).to(self.device)
                
    #             #handle AHQ input 256*256
    #             explicit_LQ = transforms.Compose([transforms.Resize((64,64), antialias=None),])(explicit_LQ.squeeze(0)).unsqueeze(0)
                
    #             explicit_LQ = data_transform(self.config, explicit_LQ)
    #             with torch.no_grad():
    #                 predict_kernel = self.kernel_estimator(explicit_LQ).squeeze()
    #                 kernel_psnr = calculate_psnr(predict_kernel, gt_kernel)
    #                 avg_kernel_psnr += kernel_psnr
    #                 self.txt_logger.info(f'{filename} kernel psnr: {kernel_psnr}')
    #             plot_kernel(gt_kernel.unsqueeze(0).unsqueeze(0), predict_kernel.unsqueeze(0).unsqueeze(0), filename, self.args.image_folder)
    #             if self.args.DDNM_Ap == "explicit" or self.args.perturb_Ap == "explicit":
    #                 explicit_padding = predict_kernel.shape[0] // 2
    #                 matrix_A = convolution_to_A(predict_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=explicit_padding).cuda()
    #                 matrix_A_pinverse = torch.pinverse(matrix_A) 
    #                 # print("AAp, I p2 norm ", torch.dist(matrix_A @ matrix_A_pinverse, torch.eye(matrix_A.shape[0]).cuda()).item())
    #                 explicit_A = lambda z: convolution_with_A(matrix_A, z, padding=explicit_padding)
    #                 explicit_Ap = lambda z: convolution_with_A(matrix_A_pinverse, z, padding=0)[:, :, explicit_padding:-explicit_padding, explicit_padding:-explicit_padding]
    #             else:
    #                 explicit_A = lambda z : convolution2d(z, predict_kernel, stride=int(self.args.deg_scale), padding=predict_kernel.size(1)//2) 

    #         #check perturb Y, store the perturbed y 
    #         if self.args.perturb_y:
    #             if self.args.perturb_A == "explicit" and self.args.perturb_Ap == "implicit":
    #                 implicit_Apy = implicit_Ap(LD_LQ)
    #                 implicit_Apy_img = tensor2img(implicit_Apy, rgb2bgr=True)
    #                 imwrite(implicit_Apy_img, os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png"))
    #                 implicit_Apy = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")).to(self.device)
    #                 implicit_Apy = data_transform(self.config, implicit_Apy)
    #                 LQ = explicit_A(implicit_Apy)
    #                 tvu.save_image(
    #                     inverse_data_transform(config, LQ[0]),
    #                     os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")
    #                 )
    #             elif self.args.perturb_A == "explicit_gt" and self.args.perturb_Ap == "implicit":
    #                 implicit_Apy = implicit_Ap(LD_LQ)
    #                 implicit_Apy_img = tensor2img(implicit_Apy, rgb2bgr=True)
    #                 imwrite(implicit_Apy_img, os.path.join(self.args.image_folder, f"Apy/implicit_Apy_{filename}.png"))
    #                 implicit_Apy = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/implicit_Apy_{filename}.png")).to(self.device)
    #                 implicit_Apy = data_transform(self.config, implicit_Apy)
    #                 LQ = gt_A(implicit_Apy)
    #                 tvu.save_image(
    #                     inverse_data_transform(config, LQ[0]),
    #                     os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")
    #                 )
    #             elif self.args.perturb_A == "implicit" and self.args.perturb_Ap == "implicit":    
    #                 LQ = implicit_A(implicit_Ap(LD_LQ))
    #                 LQ_img = tensor2img(LQ, rgb2bgr=True)
    #                 imwrite(LQ_img, os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png"))
    #             else:
    #                 raise ValueError(f"perturb mode {self.args.perturb_A}, {self.args.perturb_Ap} not supported")
                
    #             # load perturbed y as new LQ
    #             DDNM_LQ = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")).to(self.device)
    #             DDNM_LQ = data_transform(self.config, DDNM_LQ)

    #         # perform Ap(y)
    #         if self.args.DDNM_Ap == "explicit_gt":
    #             Apy = gt_Ap(DDNM_LQ)
    #             tvu.save_image(
    #                 inverse_data_transform(config, Apy[0]),
    #                 os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")
    #             )
    #         elif self.args.DDNM_Ap == "explicit":
    #             Apy = explicit_Ap(DDNM_LQ)
    #             # print(Apy.min()) # < -1
    #             # print(Apy.max()) # > 1
    #             tvu.save_image(
    #                 inverse_data_transform(config, Apy[0]),
    #                 os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")
    #             )
    #         elif self.args.DDNM_Ap == "implicit":
    #             if self.args.perturb_y:
    #                 LD_LQ = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")).to('cuda')
    #             Apy = implicit_Ap(LD_LQ)
    #             Apy_img = tensor2img(Apy, rgb2bgr=True)
    #             imwrite(Apy_img, os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png"))
    #         else:
    #             raise ValueError("DDNM Ap mode {self.args.DDNM_Ap} not supported")  

    #         if self.args.DDNM_Ap == "implicit": 
    #             Apy = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")).to(self.device)
    #             Apy = data_transform(self.config, Apy) # don't know why 
                 
    #         # HQ = load_imgDDNM(hq_path).to(self.device)
    #         # HQ = data_transform(self.config, HQ)
    #         tvu.save_image(
    #             inverse_data_transform(config, HQ[0]),
    #             os.path.join(self.args.image_folder, f"Apy/orig_{filename}.png")
    #         )
    #         Apy_psnr = cal_img_psnr([inverse_data_transform(config, HQ[0])], [inverse_data_transform(config, Apy[0])])
    #         self.txt_logger.info(f'{filename} per sample apy psnr: {Apy_psnr}')
    #         avg_apy_psnr += Apy_psnr
            
    #         if self.args.DDNM_Ap == "implicit": 
    #             Apy = inverse_data_transform(self.config, Apy) # dopn't know why equalivant to alpha = 0.5
            
    #         # init x_T
    #         x = torch.randn(
    #             HQ.shape[0],
    #             config.data.channels,
    #             config.data.image_size,
    #             config.data.image_size,
    #             device=self.device,
    #         )
    #         # with torch.no_grad():
    #         skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
    #         n = x.size(0)
    #         x0_preds = []
    #         xs = [x]
            
    #         times = get_schedule_jump(config.time_travel.T_sampling, 
    #                                         config.time_travel.travel_length, 
    #                                         config.time_travel.travel_repeat,
    #                                         )
    #         time_pairs = list(zip(times[:-1], times[1:])) 
            
    #         # reverse diffusion sampling
    #         for i, j in time_pairs:
    #             i, j = i*skip, j*skip
    #             if j<0: 
    #                 j=-1 
    #             if j < i: # normal sampling 
    #                 t = (torch.ones(n) * i).to(x.device)
    #                 LD_timestep = torch.tensor([i], dtype=torch.long).to('cuda').unsqueeze(0)
    #                 next_t = (torch.ones(n) * j).to(x.device)
    #                 at = compute_alpha(self.betas, t.long())
    #                 at_next = compute_alpha(self.betas, next_t.long())
    #                 xt = xs[-1].to('cuda')
    #                 if self.args.save_img:
    #                     os.makedirs(os.path.join(self.args.image_folder, "x_t", f'{filename}'), exist_ok=True)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, xt.to('cpu')),
    #                         os.path.join(self.args.image_folder, "x_t", f"{filename}/x_{int(t[0])}.png")
    #                     )     
                    
    #                 with torch.no_grad():
    #                     et = model(xt, t)
                        
    #                 if et.size(1) == 6:
    #                     et = et[:, :3]
    #                 # Eq. 12
    #                 x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() 
                    
    #                 # save x0_t
    #                 if self.args.save_img:
    #                     os.makedirs(os.path.join(self.args.image_folder, "x0_t", f'{filename}'), exist_ok=True)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, x0_t.to('cpu')),
    #                         os.path.join(self.args.image_folder, "x0_t", f"{filename}/x0_{int(t[0])}.png")
    #                     )
    #                 else:
    #                     os.makedirs(os.path.join(self.args.image_folder, "tmp"), exist_ok=True)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, x0_t.to('cpu')),
    #                         os.path.join(self.args.image_folder, "tmp", "x0_t.png")
    #                     )        
                        
    #                 # Eq. 13
    #                 # x0_t_hat = ApA(Apy) + x0_t - ApA(x0_t)
    #                 #get ApA(x0_t)
    #                 if self.args.DDNM_A == 'implicit' and self.args.DDNM_Ap == 'implicit':
    #                     if self.args.save_img:
    #                         x0_tL = load_img_LearningDegradation(os.path.join(self.args.image_folder, "x0_t", f"{filename}/x0_{int(t[0])}.png")).to('cuda')
    #                     else:
    #                         x0_tL = load_img_LearningDegradation(os.path.join(self.args.image_folder, "tmp", "x0_t.png")).to('cuda')
    #                     ApAx0_t = implicit_Ap(implicit_A(x0_tL))
    #                     ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)                    
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, "ApA", f"{filename}/ApAx0_{int(t[0])}.png"))
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, "ApA", f"{filename}/ApAx0_{int(t[0])}.png")).to('cuda')
    #                     else:
    #                         os.makedirs(os.path.join(self.args.image_folder, "tmp"), exist_ok=True)
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, "tmp", "ApAx0_t.png"))
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, "tmp", "ApAx0_t.png")).to('cuda')
    #                     # ApAx0_t = data_transform(config, ApAx0_t) # equalivant to alpha = 0.5
    #                 elif self.args.DDNM_A == 'explicit' and self.args.DDNM_Ap == 'implicit':
    #                     explicit_Ax0_t = explicit_A(x0_t)
    #                     if self.args.save_img:
    #                         tvu.save_image(
    #                             inverse_data_transform(config, explicit_Ax0_t.to('cpu')),
    #                             os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")
    #                         )
    #                         Ax0t = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")).to('cuda')
    #                     else:
    #                         os.makedirs(os.path.join(self.args.image_folder, "tmp"), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, explicit_Ax0_t.to('cpu')),
    #                             os.path.join(self.args.image_folder, f"tmp/Ax0t.png")
    #                         )
    #                         Ax0t = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"tmp/Ax0t.png")).to('cuda')
    #                     ApAx0_t = implicit_Ap(Ax0t)
    #                     ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png"))
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")).to('cuda')
    #                     else:
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"tmp/ApAx0t.png"))
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, f"tmp/ApAx0t.png")).to('cuda')
    #                     # ApAx0_t = data_transform(self.config, ApAx0_t) # equalivant to alpha = 0.5
                        
    #                 elif self.args.DDNM_A == 'explicit_gt' and self.args.DDNM_Ap == 'implicit':
    #                     explicit_Ax0_t = gt_A(x0_t)
    #                     if self.args.save_img:
    #                         tvu.save_image(
    #                             inverse_data_transform(config, explicit_Ax0_t.to('cpu')),
    #                             os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")
    #                         )
    #                         Ax0t = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")).to('cuda')
    #                     else:
    #                         os.makedirs(os.path.join(self.args.image_folder, "tmp"), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, explicit_Ax0_t.to('cpu')),
    #                             os.path.join(self.args.image_folder, f"tmp/Ax0t.png")
    #                         )
    #                         Ax0t = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"tmp/Ax0t.png")).to('cuda')
    #                     ApAx0_t = implicit_Ap(Ax0t)
    #                     ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png"))
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")).to('cuda')
    #                     else:
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"tmp/ApAx0t.png"))
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, f"tmp/ApAx0t.png")).to('cuda')
    #                     # ApAx0_t = data_transform(self.config, ApAx0_t) # equalivant to alpha = 0.5
                        
    #                 elif self.args.DDNM_A == 'explicit' and self.args.DDNM_Ap == 'explicit':
    #                     ApAx0_t = explicit_Ap(explicit_A(x0_t))
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, ApAx0_t[0].to('cpu')),
    #                             os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")
    #                         )
                            
    #                 elif self.args.DDNM_A == 'explicit_gt' and self.args.DDNM_Ap == 'explicit_gt':
    #                     ApAx0_t = gt_Ap(gt_A(x0_t))
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, ApAx0_t[0]),
    #                             os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")
    #                         )
    #                 else:
    #                     raise ValueError("DDNM A mode {self.args.DDNM_A} with Ap mode {self.args.DDNM_Ap} not supported")  
                    
    #                 alpha = self.args.alpha
    #                 if self.args.posterior_formula == "DDNM":
    #                     alpha = alpha * 2
    #                 x0_t_hat = x0_t + alpha * (Apy - ApAx0_t) # DDNM formula
                     
    #                 if self.args.save_img:
    #                     os.makedirs(os.path.join(self.args.image_folder, "x0_t_hat", f'{filename}'), exist_ok=True)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, x0_t_hat),
    #                         os.path.join(self.args.image_folder, "x0_t_hat", f"{filename}/x0_{int(t[0])}_hat.png")
    #                     ) 
                    
    #                 eta = self.args.eta
    #                 if self.args.posterior_formula == "DDIM":
    #                     sigma = (
    #                         eta
    #                         * torch.sqrt((1 - at_next) / (1 - at))
    #                         * torch.sqrt(1 - at / at_next)
    #                     )
    #                     mean_pred = (
    #                         x0_t_hat * torch.sqrt(at_next)
    #                         + torch.sqrt(1 - at_next - sigma ** 2) * et
    #                     )
    #                     xt_next = mean_pred + sigma * torch.randn_like(x0_t)
    #                 elif self.args.posterior_formula == "DDNM":
    #                     c1 = (1 - at_next).sqrt() * eta
    #                     c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
    #                     sigma_t = (1 - at_next**2).sqrt()
    #                     # different from the paper, we use DDIM here instead of DDPM
    #                     xt_next = at_next.sqrt() * x0_t_hat + sigma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

    #                 x0_preds.append(x0_t.to('cpu'))
    #                 xs.append(xt_next.to('cpu'))   
    #             else: # time-travel back
    #                 raise NotImplementedError

    #         x = xs[-1]
    #         x = [inverse_data_transform(config, xi) for xi in x]

    #         tvu.save_image(
    #             x[0], os.path.join(self.args.image_folder, f"{filename}.png")
    #         )
            
    #         output_psnr = cal_img_psnr([inverse_data_transform(config, HQ[0])], x)
    #         self.txt_logger.info(f'{filename} per sample psnr: {output_psnr}')
    #         avg_output_psnr += output_psnr

    #     avg_output_psnr = avg_output_psnr / len(filename_list)
    #     avg_apy_psnr = avg_apy_psnr / len(filename_list)
    #     avg_kernel_psnr = avg_kernel_psnr / len(filename_list)
    #     self.txt_logger.info(f"Total Average kernel PSNR: {avg_kernel_psnr}")
    #     self.txt_logger.info(f"Apy Total Average PSNR: {avg_apy_psnr}")
    #     self.txt_logger.info(f"Output Total Average PSNR: {avg_output_psnr}")
    #     self.txt_logger.info(f"Number of samples: {len(filename_list)}")

    #     #eval
    #     lpips_model = LPIPS(net='alex').to('cuda')
    #     folder = self.args.image_folder
    #     print('eval folder', folder)
    #     N = len(filename_list)
    #     total_output_psnr = 0.0
    #     total_output_ssim = 0.0
    #     total_output_lpips = 0.0
        
    #     total_apy_psnr = 0.0
    #     total_apy_ssim = 0.0
    #     total_apy_lpips = 0.0
    #     for filename in tqdm.tqdm(filename_list):
    #         predict_path = os.path.join(folder, f'{filename}.png')   
    #         gt_path = os.path.join(folder, 'Apy', f'orig_{filename}.png')
    #         apy_path = os.path.join(folder, 'Apy', f'Apy_{filename}.png')

    #         predict_img = Image.open(predict_path).convert("RGB")
    #         apy_img = Image.open(apy_path).convert("RGB")
    #         gt_img = Image.open(gt_path).convert("RGB")
            
    #         total_output_psnr += calculate_psnr_eval(predict_img, gt_img)
    #         total_output_ssim += calculate_ssim_eval(predict_img, gt_img)
    #         total_output_lpips += calculate_lpips_eval(lpips_model, predict_img, gt_img)

    #         total_apy_psnr += calculate_psnr_eval(apy_img, gt_img)
    #         total_apy_ssim += calculate_ssim_eval(apy_img, gt_img)
    #         total_apy_lpips += calculate_lpips_eval(lpips_model, apy_img, gt_img)
    #     print('output avg psnr:', total_output_psnr / N) 
    #     print('output avg ssim:', total_output_ssim / N) 
    #     print('output avg lpips:', total_output_lpips / N) 

    #     print('apy avg psnr:', total_apy_psnr / N) 
    #     print('apy avg ssim:', total_apy_ssim / N) 
    #     print('apy avg lpips:', total_apy_lpips / N) 

    #     with open(os.path.join(folder, '_log_val_.txt'), 'w') as f:
    #         f.writelines(f'output avg psnr: {total_output_psnr / N} \n')
    #         f.writelines(f'output ssim psnr: {total_output_ssim / N} \n')
    #         f.writelines(f'output lpips psnr: {total_output_lpips / N} \n')
    #         f.writelines(f'apy avg psnr: {total_apy_psnr / N} \n')
    #         f.writelines(f'apy ssim psnr: {total_apy_ssim / N} \n')
    #         f.writelines(f'apy lpips psnr: {total_apy_lpips / N} \n')



    # def ddnm_plus_div2k_tmp(self, model):
    #     args, config = self.args, self.config
    #     g = torch.Generator()
    #     g.manual_seed(args.seed)
    #     set_seed(args.seed)
        
    #     # mkdirs
    #     print(f'result save to {self.args.image_folder}')
    #     os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
        
    #     # get all test image filename
    #     test_dataset_root = self.args.path_y
    #     all_files = os.listdir(os.path.join(test_dataset_root, 'HR'))
    #     filename_list = sorted([os.path.splitext(file)[0] for file in all_files])
    #     filename_list = filename_list[:self.args.sample_number]
            
    #     avg_output_psnr = 0.0
    #     avg_kernel_psnr = 0.0
    #     avg_apy_psnr = 0.0
    #     with torch.no_grad():
    #         #init A, Ap
    #         if self.args.DDNM_A == "implicit" or self.args.DDNM_Ap == "implicit" or self.args.perturb_A == "implicit" or self.args.perturb_Ap == "implicit":
    #             #version 1
    #             implicit_A = lambda z : self.IRmodel.net_g(z, z, LD_LQ)[1] #lQ, HQ, lq_condition or lq, x0t_gt, oirginal_lq for sametarget
    #             implicit_Ap = lambda z : self.IRmodel.net_g(z, z, LD_LQ)[0]

    #         for filename in tqdm.tqdm(filename_list):
    #             if os.path.exists(os.path.join(self.args.image_folder, f"{filename}.png")):
    #                 continue
                
    #             hq_path = os.path.join(test_dataset_root, 'HR', f'{filename}.png')
    #             lq_path = os.path.join(test_dataset_root, 'LR', f'{filename}.png')
    #             gt_kernel_path = os.path.join(test_dataset_root, 'kernel', f'{filename}.npy')
                
    #             HQ = load_imgDDNM(hq_path).to(self.device)
    #             HQ = data_transform(self.config, HQ)
    #             gt_kernel = np.load(gt_kernel_path)
    #             gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
    #             gt_A = lambda z : convolution2d(z, gt_kernel, stride=int(self.args.deg_scale), padding=gt_kernel.size(1)//2) 
    #             DDNM_LQ = gt_A(HQ)
    #             tvu.save_image(
    #                 inverse_data_transform(config, DDNM_LQ[0]),
    #                 os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")
    #             )
    #             lq_path = os.path.join(self.args.image_folder, f"Apy/y_{filename}.png") 
    #             DDNM_LQ = load_imgDDNM(lq_path).to(self.device)
    #             DDNM_LQ = data_transform(self.config, DDNM_LQ)
                
    #             if self.args.DDNM_A == "implicit" or self.args.DDNM_Ap == "implicit" or self.args.perturb_A == "implicit" or self.args.perturb_Ap == "implicit":
    #                 LD_LQ = load_imgDDNM(lq_path).to(self.device)
    #                 LD_LQ = transforms.Compose([transforms.Resize((256,256), antialias=None),])(LD_LQ.squeeze()).unsqueeze(0)
                
    #             # get A, Ap for current input
    #             if self.args.DDNM_A == "explicit_gt" or self.args.DDNM_Ap == "explicit_gt" or self.args.perturb_A == "explicit_gt" or self.args.perturb_Ap == "explicit_gt":
    #                 gt_kernel = np.load(gt_kernel_path)
    #                 gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
    #                 gt_A = lambda z : convolution2d(z, gt_kernel, stride=int(self.args.deg_scale), padding=gt_kernel.size(1)//2) 

    #                 gt_padding = gt_kernel.shape[0] // 2
    #                 gt_matrix_A = convolution_to_A(gt_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=gt_padding).cuda()
    #                 gt_matrix_A_pinverse = torch.pinverse(gt_matrix_A)
    #                 gt_A = lambda z: convolution_with_A(gt_matrix_A, z, padding=gt_padding)
    #                 gt_Ap = lambda z: convolution_with_A(gt_matrix_A_pinverse, z, padding=0)[:, :, gt_padding:-gt_padding, gt_padding:-gt_padding]
    #             if self.args.DDNM_A == "explicit" or self.args.DDNM_Ap == "explicit" or self.args.perturb_A == "explicit" or self.args.perturb_Ap == "explicit":
    #                 gt_kernel = np.load(gt_kernel_path)
    #                 gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
    #                 # padding gt_kenel to 21 * 21
    #                 gt_kernel = gt_kernel.unsqueeze(0)
    #                 padding21 = (self.args.kernel_size - gt_kernel.size(1)) //2
    #                 gt_kernel = F.pad(gt_kernel, (padding21, padding21, padding21, padding21)).squeeze()
                    
    #                 predict_kernel = None
    #                 with torch.no_grad():
    #                     predict_kernel = self.kernel_estimator(DDNM_LQ).squeeze()
    #                     kernel_psnr = calculate_psnr(predict_kernel, gt_kernel)
    #                     avg_kernel_psnr += kernel_psnr
    #                     self.txt_logger.info(f'{filename} kernel psnr: {kernel_psnr}')
    #                 plot_kernel(gt_kernel.unsqueeze(0).unsqueeze(0), predict_kernel.unsqueeze(0).unsqueeze(0), filename, self.args.image_folder)
    #                 if self.args.DDNM_Ap == "explicit" or self.args.perturb_Ap == "explicit":
    #                     explicit_padding = predict_kernel.shape[0] // 2
    #                     matrix_A = convolution_to_A(predict_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=explicit_padding).cuda()
    #                     matrix_A_pinverse = torch.pinverse(matrix_A) 
    #                     explicit_A = lambda z: convolution_with_A(matrix_A, z, padding=explicit_padding)
    #                     explicit_Ap = lambda z: convolution_with_A(matrix_A_pinverse, z, padding=0)[:, :, explicit_padding:-explicit_padding, explicit_padding:-explicit_padding]
    #                 else:
    #                     explicit_A = lambda z : convolution2d(z, predict_kernel, stride=int(self.args.deg_scale), padding=predict_kernel.size(1)//2) 

    #             #check perturb Y, store the perturbed y 
    #             if self.args.perturb_y:
    #                 if self.args.perturb_A == "explicit" and self.args.perturb_Ap == "implicit":
    #                     implicit_Apy = implicit_Ap(LD_LQ)
    #                     implicit_Apy = data_transform(self.config, implicit_Apy)
    #                     DDNM_LQ = explicit_A(implicit_Apy)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, DDNM_LQ[0]),
    #                         os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")
    #                     )
    #                 elif self.args.perturb_A == "explicit_gt" and self.args.perturb_Ap == "implicit":
    #                     implicit_Apy = implicit_Ap(LD_LQ)
    #                     implicit_Apy = data_transform(self.config, implicit_Apy)
    #                     DDNM_LQ = gt_A(implicit_Apy)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, DDNM_LQ[0]),
    #                         os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")
    #                     )
    #                 elif self.args.perturb_A == "implicit" and self.args.perturb_Ap == "implicit":    
    #                     LD_LQ = implicit_A(implicit_Ap(LD_LQ))
    #                     LQ_img = tensor2img(LD_LQ, rgb2bgr=True)
    #                     imwrite(LQ_img, os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png"))
    #                     DDNM_LQ = data_transform(self.config, LD_LQ)
    #                 else:
    #                     raise ValueError(f"perturb mode {self.args.perturb_A}, {self.args.perturb_Ap} not supported")
                    
    #             # perform Ap(y)
    #             if self.args.DDNM_Ap == "explicit_gt":
    #                 Apy = gt_Ap(DDNM_LQ)
    #                 tvu.save_image(
    #                     inverse_data_transform(config, Apy[0]),
    #                     os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")
    #                 )
    #             elif self.args.DDNM_Ap == "explicit":
    #                 Apy = explicit_Ap(DDNM_LQ)
    #                 # print(Apy.min()) # < -1
    #                 # print(Apy.max()) # > 1
    #                 tvu.save_image(
    #                     inverse_data_transform(config, Apy[0]),
    #                     os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")
    #                 )
    #             elif self.args.DDNM_Ap == "implicit":
    #                 Apy = implicit_Ap(LD_LQ)
    #                 Apy_img = tensor2img(Apy, rgb2bgr=True)
    #                 imwrite(Apy_img, os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png"))
    #             else:
    #                 raise ValueError("DDNM Ap mode {self.args.DDNM_Ap} not supported")  

    #             if self.args.DDNM_Ap == "implicit": 
    #                 Apy = data_transform(self.config, Apy) # don't know why 
                    

    #             tvu.save_image(
    #                 inverse_data_transform(config, HQ[0]),
    #                 os.path.join(self.args.image_folder, f"Apy/orig_{filename}.png")
    #             )
    #             Apy_psnr = cal_img_psnr([inverse_data_transform(config, HQ[0])], [inverse_data_transform(config, Apy[0])])
    #             self.txt_logger.info(f'{filename} per sample apy psnr: {Apy_psnr}')
    #             avg_apy_psnr += Apy_psnr
                
    #             if self.args.DDNM_Ap == "implicit": 
    #                 Apy = inverse_data_transform(self.config, Apy) # dopn't know why, equalivant to alpha = 0.5
                
    #             # init x_T
    #             x = torch.randn(
    #                 HQ.shape[0],
    #                 config.data.channels,
    #                 config.data.image_size,
    #                 config.data.image_size,
    #                 device=self.device,
    #             )
    #             # with torch.no_grad():
    #             skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
    #             n = x.size(0)
    #             x0_preds = []
    #             xs = [x]
                
    #             times = get_schedule_jump(config.time_travel.T_sampling, 
    #                                             config.time_travel.travel_length, 
    #                                             config.time_travel.travel_repeat,
    #                                             )
    #             time_pairs = list(zip(times[:-1], times[1:])) 
                
    #             # reverse diffusion sampling
    #             for i, j in time_pairs:
    #                 i, j = i*skip, j*skip
    #                 if j<0: 
    #                     j=-1 
    #                 if j < i: # normal sampling 
    #                     t = (torch.ones(n) * i).to(x.device)
    #                     next_t = (torch.ones(n) * j).to(x.device)
    #                     at = compute_alpha(self.betas, t.long())
    #                     at_next = compute_alpha(self.betas, next_t.long())
    #                     xt = xs[-1].to('cuda')
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "x_t", f'{filename}'), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, xt.to('cpu')),
    #                             os.path.join(self.args.image_folder, "x_t", f"{filename}/x_{int(t[0])}.png")
    #                         )     
                        
    #                     with torch.no_grad():
    #                         et = model(xt, t)
                            
    #                     if et.size(1) == 6:
    #                         et = et[:, :3]
    #                     # Eq. 12
    #                     x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() 
                        
    #                     # save x0_t
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "x0_t", f'{filename}'), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, x0_t.to('cpu')),
    #                             os.path.join(self.args.image_folder, "x0_t", f"{filename}/x0_{int(t[0])}.png")
    #                         )

    #                     # Eq. 13
    #                     # x0_t_hat = ApA(Apy) + x0_t - ApA(x0_t)
    #                     #get ApA(x0_t)
    #                     if self.args.DDNM_A == 'implicit' and self.args.DDNM_Ap == 'implicit':
    #                         x0_tL = inverse_data_transform(config, x0_t)
    #                         ApAx0_t = implicit_Ap(implicit_A(x0_tL))
                                            
    #                         if self.args.save_img:
    #                             ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True) 
    #                             os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                             imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, "ApA", f"{filename}/ApAx0_{int(t[0])}.png"))
    #                         # ApAx0_t = data_transform(config, ApAx0_t) # equalivant to alpha = 0.5
    #                     elif self.args.DDNM_A == 'explicit' and self.args.DDNM_Ap == 'implicit':
    #                         Ax0_t = explicit_A(x0_t)
    #                         if self.args.save_img:
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, Ax0_t.to('cpu')),
    #                                 os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")
    #                             )
    #                         Ax0_t = inverse_data_transform(config, Ax0_t)
    #                         Ax0_t = transforms.Compose([transforms.Resize((256,256), antialias=None),])(Ax0_t.squeeze()).unsqueeze(0)
    #                         ApAx0_t = implicit_Ap(Ax0_t)
    #                         if self.args.save_img:
    #                             ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)
    #                             os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                             imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png"))
    #                         # ApAx0_t = data_transform(self.config, ApAx0_t) # equalivant to alpha = 0.5
                            
    #                     elif self.args.DDNM_A == 'explicit_gt' and self.args.DDNM_Ap == 'implicit':
    #                         Ax0_t = gt_A(x0_t)
    #                         if self.args.save_img:
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, Ax0_t.to('cpu')),
    #                                 os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")
    #                             )
    #                         Ax0_t = inverse_data_transform(config, Ax0_t) 
    #                         Ax0_t = transforms.Compose([transforms.Resize((256,256), antialias=None),])(Ax0_t.squeeze()).unsqueeze(0)
    #                         ApAx0_t = implicit_Ap(Ax0_t)
                            
    #                         if self.args.save_img:
    #                             ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)
    #                             os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                             imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png"))
    #                         # ApAx0_t = data_transform(self.config, ApAx0_t) # equalivant to alpha = 0.5
                            
    #                     elif self.args.DDNM_A == 'explicit' and self.args.DDNM_Ap == 'explicit':
    #                         ApAx0_t = explicit_Ap(explicit_A(x0_t))
    #                         if self.args.save_img:
    #                             os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, ApAx0_t[0].to('cpu')),
    #                                 os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")
    #                             )
                                
    #                     elif self.args.DDNM_A == 'explicit_gt' and self.args.DDNM_Ap == 'explicit_gt':
    #                         ApAx0_t = gt_Ap(gt_A(x0_t))
    #                         if self.args.save_img:
    #                             os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, ApAx0_t[0]),
    #                                 os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")
    #                             )
    #                     else:
    #                         raise ValueError("DDNM A mode {self.args.DDNM_A} with Ap mode {self.args.DDNM_Ap} not supported")  
                        
    #                     alpha = self.args.alpha
    #                     if self.args.posterior_formula == "DDNM" and self.args.DDNM_Ap == 'implicit':
    #                         alpha = alpha * 2
    #                     x0_t_hat = x0_t + alpha * (Apy - ApAx0_t) # DDNM formula
                        
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "x0_t_hat", f'{filename}'), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, x0_t_hat),
    #                             os.path.join(self.args.image_folder, "x0_t_hat", f"{filename}/x0_{int(t[0])}_hat.png")
    #                         ) 
                        
    #                     eta = self.args.eta
    #                     if self.args.posterior_formula == "DDIM":
    #                         sigma = (
    #                             eta
    #                             * torch.sqrt((1 - at_next) / (1 - at))
    #                             * torch.sqrt(1 - at / at_next)
    #                         )
    #                         mean_pred = (
    #                             x0_t_hat * torch.sqrt(at_next)
    #                             + torch.sqrt(1 - at_next - sigma ** 2) * et
    #                         )
    #                         xt_next = mean_pred + sigma * torch.randn_like(x0_t)
    #                     elif self.args.posterior_formula == "DDNM":
    #                         c1 = (1 - at_next).sqrt() * eta
    #                         c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
    #                         sigma_t = (1 - at_next**2).sqrt()
    #                         # different from the paper, we use DDIM here instead of DDPM
    #                         xt_next = at_next.sqrt() * x0_t_hat + sigma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

    #                     x0_preds.append(x0_t.to('cpu'))
    #                     xs.append(xt_next.to('cpu'))   
    #                 else: # time-travel back
    #                     raise NotImplementedError

    #             x = xs[-1]
    #             x = [inverse_data_transform(config, xi) for xi in x]

    #             tvu.save_image(
    #                 x[0], os.path.join(self.args.image_folder, f"{filename}.png")
    #             )
                
    #             output_psnr = cal_img_psnr([inverse_data_transform(config, HQ[0])], x)
    #             self.txt_logger.info(f'{filename} per sample psnr: {output_psnr}')
    #             avg_output_psnr += output_psnr

    #         avg_output_psnr = avg_output_psnr / len(filename_list)
    #         avg_apy_psnr = avg_apy_psnr / len(filename_list)
    #         avg_kernel_psnr = avg_kernel_psnr / len(filename_list)
    #         self.txt_logger.info(f"Total Average kernel PSNR: {avg_kernel_psnr}")
    #         self.txt_logger.info(f"Apy Total Average PSNR: {avg_apy_psnr}")
    #         self.txt_logger.info(f"Output Total Average PSNR: {avg_output_psnr}")
    #         self.txt_logger.info(f"Number of samples: {len(filename_list)}")

    #         #eval
    #         lpips_model = LPIPS(net='alex').to('cuda')
    #         folder = self.args.image_folder
    #         print('eval folder', folder)
    #         N = len(filename_list)
    #         total_output_psnr = 0.0
    #         total_output_ssim = 0.0
    #         total_output_lpips = 0.0
            
    #         total_apy_psnr = 0.0
    #         total_apy_ssim = 0.0
    #         total_apy_lpips = 0.0
    #         for filename in tqdm.tqdm(filename_list):
    #             predict_path = os.path.join(folder, f'{filename}.png')   
    #             gt_path = os.path.join(folder, 'Apy', f'orig_{filename}.png')
    #             apy_path = os.path.join(folder, 'Apy', f'Apy_{filename}.png')

    #             predict_img = Image.open(predict_path).convert("RGB")
    #             apy_img = Image.open(apy_path).convert("RGB")
    #             gt_img = Image.open(gt_path).convert("RGB")
                
    #             total_output_psnr += calculate_psnr_eval(predict_img, gt_img)
    #             total_output_ssim += calculate_ssim_eval(predict_img, gt_img)
    #             total_output_lpips += calculate_lpips_eval(lpips_model, predict_img, gt_img)

    #             total_apy_psnr += calculate_psnr_eval(apy_img, gt_img)
    #             total_apy_ssim += calculate_ssim_eval(apy_img, gt_img)
    #             total_apy_lpips += calculate_lpips_eval(lpips_model, apy_img, gt_img)
    #         print('output avg psnr:', total_output_psnr / N) 
    #         print('output avg ssim:', total_output_ssim / N) 
    #         print('output avg lpips:', total_output_lpips / N) 

    #         print('apy avg psnr:', total_apy_psnr / N) 
    #         print('apy avg ssim:', total_apy_ssim / N) 
    #         print('apy avg lpips:', total_apy_lpips / N) 

    #         with open(os.path.join(folder, '_log_val_.txt'), 'w') as f:
    #             f.writelines(f'output avg psnr: {total_output_psnr / N} \n')
    #             f.writelines(f'output ssim psnr: {total_output_ssim / N} \n')
    #             f.writelines(f'output lpips psnr: {total_output_lpips / N} \n')
    #             f.writelines(f'apy avg psnr: {total_apy_psnr / N} \n')
    #             f.writelines(f'apy ssim psnr: {total_apy_ssim / N} \n')
    #             f.writelines(f'apy lpips psnr: {total_apy_lpips / N} \n')

    # def ddnm_plus_ffhq(self, model):
    #     args, config = self.args, self.config
    #     g = torch.Generator()
    #     g.manual_seed(args.seed)
    #     set_seed(args.seed)
        
    #     # mkdirs
    #     os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
        
    #     # get all test image filename
    #     test_dataset_root = self.args.path_y
    #     all_files = os.listdir(os.path.join(test_dataset_root, 'hq'))
    #     filename_list = sorted([file[:-4] for file in all_files])
    #     filename_list = filename_list[:self.args.sample_number]
            
    #     avg_output_psnr = 0.0
    #     avg_kernel_psnr = 0.0
    #     avg_apy_psnr = 0.0
        
    #     #init A, Ap
    #     if self.args.DDNM_A == "implicit" or self.args.DDNM_Ap == "implicit" or self.args.perturb_A == "implicit" or self.args.perturb_Ap == "implicit":
    #         #version 1, 2
    #         implicit_A = lambda z : self.IRmodel.net_g(z, z, LD_LQ)[1] #lQ, HQ, lq_condition
    #         implicit_Ap = lambda z : self.IRmodel.net_g(z, z, LD_LQ)[0]

    #         # #version 3
    #         # implicit_A = lambda z : self.IRmodel.net_g(z, z, LD_LQ, LD_timestep, LD_timestep)[1] #lQ, HQ, lq_condition, timestepA, timestepAp
    #         # implicit_Ap = lambda z : self.IRmodel.net_g(z, z, LD_LQ, LD_timestep, LD_timestep)[0]
            
    #     for filename in tqdm.tqdm(filename_list):
    #         LD_timestep = torch.tensor([0], dtype=torch.long).to('cuda').unsqueeze(0)
    #         if os.path.exists(os.path.join(self.args.image_folder, f"{filename}.png")):
    #             continue
            
    #         hq_path = os.path.join(test_dataset_root, 'hq', f'{filename}.jpg')
    #         lq_path = os.path.join(test_dataset_root, 'lq', f'{int(self.args.deg_scale)}', f'{filename}.jpg')
    #         gt_kernel_path = os.path.join(test_dataset_root, 'kernel', f'{filename}.npy')
            
    #         HQ = load_imgDDNM(hq_path).to(self.device)
    #         HQ = data_transform(self.config, HQ)
    #         gt_kernel = np.load(gt_kernel_path)
    #         gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
    #         gt_A = lambda z : convolution2d(z, gt_kernel, stride=int(self.args.deg_scale), padding=gt_kernel.size(1)//2) 
    #         DDNM_LQ = gt_A(HQ)
    #         tvu.save_image(
    #             inverse_data_transform(config, DDNM_LQ[0]),
    #             os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")
    #         )
    #         lq_path_gt = os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")  
    #         DDNM_LQ = load_imgDDNM(lq_path_gt).to(self.device)
    #         DDNM_LQ = data_transform(self.config, DDNM_LQ)
            
    #         if self.args.DDNM_A == "implicit" or self.args.DDNM_Ap == "implicit" or self.args.perturb_A == "implicit" or self.args.perturb_Ap == "implicit":
    #             LD_LQ = load_img_LearningDegradation(lq_path).to('cuda')
            
    #         # get A, Ap for current input
    #         if self.args.DDNM_A == "explicit_gt" or self.args.DDNM_Ap == "explicit_gt" or self.args.perturb_A == "explicit_gt" or self.args.perturb_Ap == "explicit_gt":
    #             gt_kernel = np.load(gt_kernel_path)
    #             gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
    #             gt_A = lambda z : convolution2d(z, gt_kernel, stride=int(self.args.deg_scale), padding=gt_kernel.size(1)//2) 

    #             gt_padding = gt_kernel.shape[0] // 2
    #             gt_matrix_A = convolution_to_A(gt_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=gt_padding).cuda()
    #             gt_matrix_A_pinverse = torch.pinverse(gt_matrix_A)
    #             gt_A = lambda z: convolution_with_A(gt_matrix_A, z, padding=gt_padding)
    #             gt_Ap = lambda z: convolution_with_A(gt_matrix_A_pinverse, z, padding=0)[:, :, gt_padding:-gt_padding, gt_padding:-gt_padding]
    #         if self.args.DDNM_A == "explicit" or self.args.DDNM_Ap == "explicit" or self.args.perturb_A == "explicit" or self.args.perturb_Ap == "explicit":
    #             gt_kernel = np.load(gt_kernel_path)
    #             gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
    #             # padding gt_kenel to 21 * 21
    #             gt_kernel = gt_kernel.unsqueeze(0)
    #             padding21 = (self.args.kernel_size - gt_kernel.size(1)) //2
    #             gt_kernel = F.pad(gt_kernel, (padding21, padding21, padding21, padding21)).squeeze()
                
    #             predict_kernel = None
    #             explicit_LQ = load_imgDDNM(lq_path).to(self.device)
    #             explicit_LQ = data_transform(self.config, explicit_LQ)
    #             with torch.no_grad():
    #                 predict_kernel = self.kernel_estimator(explicit_LQ).squeeze()
    #                 kernel_psnr = calculate_psnr(predict_kernel, gt_kernel)
    #                 avg_kernel_psnr += kernel_psnr
    #                 self.txt_logger.info(f'{filename} kernel psnr: {kernel_psnr}')
    #             plot_kernel(gt_kernel.unsqueeze(0).unsqueeze(0), predict_kernel.unsqueeze(0).unsqueeze(0), filename, self.args.image_folder)
    #             if self.args.DDNM_Ap == "explicit" or self.args.perturb_Ap == "explicit":
    #                 explicit_padding = predict_kernel.shape[0] // 2
    #                 matrix_A = convolution_to_A(predict_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=explicit_padding).cuda()
    #                 matrix_A_pinverse = torch.pinverse(matrix_A) 
    #                 # print("AAp, I p2 norm ", torch.dist(matrix_A @ matrix_A_pinverse, torch.eye(matrix_A.shape[0]).cuda()).item())
    #                 explicit_A = lambda z: convolution_with_A(matrix_A, z, padding=explicit_padding)
    #                 explicit_Ap = lambda z: convolution_with_A(matrix_A_pinverse, z, padding=0)[:, :, explicit_padding:-explicit_padding, explicit_padding:-explicit_padding]
    #             else:
    #                 explicit_A = lambda z : convolution2d(z, predict_kernel, stride=int(self.args.deg_scale), padding=predict_kernel.size(1)//2) 

    #         #check perturb Y, store the perturbed y 
    #         if self.args.perturb_y:
    #             if self.args.perturb_A == "explicit" and self.args.perturb_Ap == "implicit":
    #                 implicit_Apy = implicit_Ap(LD_LQ)
    #                 implicit_Apy_img = tensor2img(implicit_Apy, rgb2bgr=True)
    #                 imwrite(implicit_Apy_img, os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png"))
    #                 implicit_Apy = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")).to(self.device)
    #                 implicit_Apy = data_transform(self.config, implicit_Apy)
    #                 LQ = explicit_A(implicit_Apy)
    #                 tvu.save_image(
    #                     inverse_data_transform(config, LQ[0]),
    #                     os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")
    #                 )
    #             elif self.args.perturb_A == "explicit_gt" and self.args.perturb_Ap == "implicit":
    #                 implicit_Apy = implicit_Ap(LD_LQ)
    #                 implicit_Apy_img = tensor2img(implicit_Apy, rgb2bgr=True)
    #                 imwrite(implicit_Apy_img, os.path.join(self.args.image_folder, f"Apy/implicit_Apy_{filename}.png"))
    #                 implicit_Apy = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/implicit_Apy_{filename}.png")).to(self.device)
    #                 implicit_Apy = data_transform(self.config, implicit_Apy)
    #                 LQ = gt_A(implicit_Apy)
    #                 tvu.save_image(
    #                     inverse_data_transform(config, LQ[0]),
    #                     os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")
    #                 )
    #             elif self.args.perturb_A == "implicit" and self.args.perturb_Ap == "implicit":    
    #                 LQ = implicit_A(implicit_Ap(LD_LQ))
    #                 LQ_img = tensor2img(LQ, rgb2bgr=True)
    #                 imwrite(LQ_img, os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png"))
    #             else:
    #                 raise ValueError(f"perturb mode {self.args.perturb_A}, {self.args.perturb_Ap} not supported")
                
    #             # load perturbed y as new LQ
    #             DDNM_LQ = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")).to(self.device)
    #             DDNM_LQ = data_transform(self.config, DDNM_LQ)

    #         # perform Ap(y)
    #         if self.args.DDNM_Ap == "explicit_gt":
    #             Apy = gt_Ap(DDNM_LQ)
    #             tvu.save_image(
    #                 inverse_data_transform(config, Apy[0]),
    #                 os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")
    #             )
    #         elif self.args.DDNM_Ap == "explicit":
    #             Apy = explicit_Ap(DDNM_LQ)
    #             # print(Apy.min()) # < -1
    #             # print(Apy.max()) # > 1
    #             tvu.save_image(
    #                 inverse_data_transform(config, Apy[0]),
    #                 os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")
    #             )
    #         elif self.args.DDNM_Ap == "implicit":
    #             if self.args.perturb_y:
    #                 # tmp_path = convert_jpg(os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png"))
    #                 # LD_LQ = load_img_LearningDegradation(tmp_path).to('cuda')
    #                 LD_LQ_input = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")).to('cuda')
    #                 Apy = implicit_Ap(LD_LQ_input)
    #             else:
    #                 Apy = implicit_Ap(LD_LQ)
    #             Apy_img = tensor2img(Apy, rgb2bgr=True)
    #             imwrite(Apy_img, os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png"))
    #         else:
    #             raise ValueError("DDNM Ap mode {self.args.DDNM_Ap} not supported")  

    #         if self.args.DDNM_Ap == "implicit": 
    #             Apy = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")).to(self.device)
    #             Apy = data_transform(self.config, Apy) # don't know why 
                 
    #         # HQ = load_imgDDNM(hq_path).to(self.device)
    #         # HQ = data_transform(self.config, HQ)
    #         tvu.save_image(
    #             inverse_data_transform(config, HQ[0]),
    #             os.path.join(self.args.image_folder, f"Apy/orig_{filename}.png")
    #         )
    #         Apy_psnr = cal_img_psnr([inverse_data_transform(config, HQ[0])], [inverse_data_transform(config, Apy[0])])
    #         self.txt_logger.info(f'{filename} per sample apy psnr: {Apy_psnr}')
    #         avg_apy_psnr += Apy_psnr
            
    #         if self.args.DDNM_Ap == "implicit": 
    #             Apy = inverse_data_transform(self.config, Apy) # dopn't know why equalivant to alpha = 0.5
            
    #         # init x_T
    #         x = torch.randn(
    #             HQ.shape[0],
    #             config.data.channels,
    #             config.data.image_size,
    #             config.data.image_size,
    #             device=self.device,
    #         )
    #         # with torch.no_grad():
    #         skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
    #         n = x.size(0)
    #         x0_preds = []
    #         xs = [x]
            
    #         times = get_schedule_jump(config.time_travel.T_sampling, 
    #                                         config.time_travel.travel_length, 
    #                                         config.time_travel.travel_repeat,
    #                                         )
    #         time_pairs = list(zip(times[:-1], times[1:])) 
            
    #         # reverse diffusion sampling
    #         for i, j in time_pairs:
    #             i, j = i*skip, j*skip
    #             if j<0: 
    #                 j=-1 
    #             if j < i: # normal sampling 
    #                 t = (torch.ones(n) * i).to(x.device)
    #                 LD_timestep = torch.tensor([i], dtype=torch.long).to('cuda').unsqueeze(0)
    #                 next_t = (torch.ones(n) * j).to(x.device)
    #                 at = compute_alpha(self.betas, t.long())
    #                 at_next = compute_alpha(self.betas, next_t.long())
    #                 xt = xs[-1].to('cuda')
    #                 if self.args.save_img:
    #                     os.makedirs(os.path.join(self.args.image_folder, "x_t", f'{filename}'), exist_ok=True)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, xt.to('cpu')),
    #                         os.path.join(self.args.image_folder, "x_t", f"{filename}/x_{int(t[0])}.png")
    #                     )     
                    
    #                 with torch.no_grad():
    #                     et = model(xt, t)
                        
    #                 if et.size(1) == 6:
    #                     et = et[:, :3]
    #                 # Eq. 12
    #                 x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() 
                    
    #                 # save x0_t
    #                 if self.args.save_img:
    #                     os.makedirs(os.path.join(self.args.image_folder, "x0_t", f'{filename}'), exist_ok=True)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, x0_t.to('cpu')),
    #                         os.path.join(self.args.image_folder, "x0_t", f"{filename}/x0_{int(t[0])}.png")
    #                     )
    #                 else:
    #                     os.makedirs(os.path.join(self.args.image_folder, "tmp"), exist_ok=True)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, x0_t.to('cpu')),
    #                         os.path.join(self.args.image_folder, "tmp", "x0_t.png")
    #                     )        
                        
    #                 # Eq. 13
    #                 # x0_t_hat = ApA(Apy) + x0_t - ApA(x0_t)
    #                 #get ApA(x0_t)
    #                 if self.args.DDNM_A == 'implicit' and self.args.DDNM_Ap == 'implicit':
    #                     if self.args.save_img:
    #                         x0_tL = load_img_LearningDegradation(os.path.join(self.args.image_folder, "x0_t", f"{filename}/x0_{int(t[0])}.png")).to('cuda')
    #                     else:
    #                         # tmp_path = convert_jpg(os.path.join(self.args.image_folder, "tmp", "x0_t.png"))
    #                         # x0_tL = load_img_LearningDegradation(tmp_path).to('cuda')
    #                         x0_tL = load_img_LearningDegradation(os.path.join(self.args.image_folder, "tmp", "x0_t.png")).to('cuda')
    #                     ApAx0_t = implicit_Ap(implicit_A(x0_tL))
    #                     ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)                    
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, "ApA", f"{filename}/ApAx0_{int(t[0])}.png"))
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, "ApA", f"{filename}/ApAx0_{int(t[0])}.png")).to('cuda')
    #                     else:
    #                         os.makedirs(os.path.join(self.args.image_folder, "tmp"), exist_ok=True)
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, "tmp", "ApAx0_t.png"))
                            
    #                         # tmp_path = convert_jpg(os.path.join(self.args.image_folder, "tmp", "ApAx0_t.png"))
    #                         # ApAx0_t = load_imgDDNM(tmp_path).to('cuda')
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, "tmp", "ApAx0_t.png")).to('cuda')
    #                     # ApAx0_t = data_transform(config, ApAx0_t) # equalivant to alpha = 0.5
    #                 elif self.args.DDNM_A == 'explicit' and self.args.DDNM_Ap == 'implicit':
    #                     explicit_Ax0_t = explicit_A(x0_t)
    #                     if self.args.save_img:
    #                         tvu.save_image(
    #                             inverse_data_transform(config, explicit_Ax0_t.to('cpu')),
    #                             os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")
    #                         )
    #                         Ax0t = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")).to('cuda')
    #                     else:
    #                         os.makedirs(os.path.join(self.args.image_folder, "tmp"), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, explicit_Ax0_t.to('cpu')),
    #                             os.path.join(self.args.image_folder, f"tmp/Ax0t.png")
    #                         )
    #                         # tmp_path = convert_jpg(os.path.join(self.args.image_folder, f"tmp/Ax0t.png"))
    #                         # Ax0t = load_img_LearningDegradation(tmp_path).to('cuda')
    #                         Ax0t = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"tmp/Ax0t.png")).to('cuda')
    #                     ApAx0_t = implicit_Ap(Ax0t)
    #                     ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png"))
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")).to('cuda')
    #                     else:
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"tmp/ApAx0t.png"))
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, f"tmp/ApAx0t.png")).to('cuda')
    #                     # ApAx0_t = data_transform(self.config, ApAx0_t) # equalivant to alpha = 0.5
                        
    #                 elif self.args.DDNM_A == 'explicit_gt' and self.args.DDNM_Ap == 'implicit':
    #                     explicit_Ax0_t = gt_A(x0_t)
    #                     if self.args.save_img:
    #                         tvu.save_image(
    #                             inverse_data_transform(config, explicit_Ax0_t.to('cpu')),
    #                             os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")
    #                         )
    #                         Ax0t = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")).to('cuda')
    #                     else:
    #                         os.makedirs(os.path.join(self.args.image_folder, "tmp"), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, explicit_Ax0_t.to('cpu')),
    #                             os.path.join(self.args.image_folder, f"tmp/Ax0t.png")
    #                         )
    #                         Ax0t = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"tmp/Ax0t.png")).to('cuda')
    #                     ApAx0_t = implicit_Ap(Ax0t)
    #                     ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png"))
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")).to('cuda')
    #                     else:
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"tmp/ApAx0t.png"))
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, f"tmp/ApAx0t.png")).to('cuda')
    #                     # ApAx0_t = data_transform(self.config, ApAx0_t) # equalivant to alpha = 0.5
                        
    #                 elif self.args.DDNM_A == 'explicit' and self.args.DDNM_Ap == 'explicit':
    #                     ApAx0_t = explicit_Ap(explicit_A(x0_t))
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, ApAx0_t[0].to('cpu')),
    #                             os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")
    #                         )
                            
    #                 elif self.args.DDNM_A == 'explicit_gt' and self.args.DDNM_Ap == 'explicit_gt':
    #                     ApAx0_t = gt_Ap(gt_A(x0_t))
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, ApAx0_t[0]),
    #                             os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")
    #                         )
    #                 else:
    #                     raise ValueError("DDNM A mode {self.args.DDNM_A} with Ap mode {self.args.DDNM_Ap} not supported")  
                    
    #                 alpha = self.args.alpha
    #                 x0_t_hat = x0_t + alpha * (Apy - ApAx0_t) # DDNM formula
                     
    #                 if self.args.save_img:
    #                     os.makedirs(os.path.join(self.args.image_folder, "x0_t_hat", f'{filename}'), exist_ok=True)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, x0_t_hat),
    #                         os.path.join(self.args.image_folder, "x0_t_hat", f"{filename}/x0_{int(t[0])}_hat.png")
    #                     ) 
                    
    #                 eta = self.args.eta
    #                 if self.args.posterior_formula == "DDIM":
    #                     sigma = (
    #                         eta
    #                         * torch.sqrt((1 - at_next) / (1 - at))
    #                         * torch.sqrt(1 - at / at_next)
    #                     )
    #                     mean_pred = (
    #                         x0_t_hat * torch.sqrt(at_next)
    #                         + torch.sqrt(1 - at_next - sigma ** 2) * et
    #                     )
    #                     xt_next = mean_pred + sigma * torch.randn_like(x0_t)
    #                 elif self.args.posterior_formula == "DDNM":
    #                     c1 = (1 - at_next).sqrt() * eta
    #                     c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
    #                     sigma_t = (1 - at_next**2).sqrt()
    #                     # different from the paper, we use DDIM here instead of DDPM
    #                     xt_next = at_next.sqrt() * x0_t_hat + sigma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

    #                 x0_preds.append(x0_t.to('cpu'))
    #                 xs.append(xt_next.to('cpu'))   
    #             else: # time-travel back
    #                 raise NotImplementedError

    #         x = xs[-1]
    #         x = [inverse_data_transform(config, xi) for xi in x]

    #         tvu.save_image(
    #             x[0], os.path.join(self.args.image_folder, f"{filename}.png")
    #         )
            
    #         output_psnr = cal_img_psnr([inverse_data_transform(config, HQ[0])], x)
    #         self.txt_logger.info(f'{filename} per sample psnr: {output_psnr}')
    #         avg_output_psnr += output_psnr

    #     avg_output_psnr = avg_output_psnr / len(filename_list)
    #     avg_apy_psnr = avg_apy_psnr / len(filename_list)
    #     avg_kernel_psnr = avg_kernel_psnr / len(filename_list)
    #     self.txt_logger.info(f"Total Average kernel PSNR: {avg_kernel_psnr}")
    #     self.txt_logger.info(f"Apy Total Average PSNR: {avg_apy_psnr}")
    #     self.txt_logger.info(f"Output Total Average PSNR: {avg_output_psnr}")
    #     self.txt_logger.info(f"Number of samples: {len(filename_list)}")

    #     #eval
    #     lpips_model = LPIPS(net='alex').to('cuda')
    #     folder = self.args.image_folder
    #     print('eval folder', folder)
    #     N = len(filename_list)
    #     total_output_psnr = 0.0
    #     total_output_ssim = 0.0
    #     total_output_lpips = 0.0
        
    #     total_apy_psnr = 0.0
    #     total_apy_ssim = 0.0
    #     total_apy_lpips = 0.0
    #     for filename in tqdm.tqdm(filename_list):
    #         predict_path = os.path.join(folder, f'{filename}.png')   
    #         gt_path = os.path.join(folder, 'Apy', f'orig_{filename}.png')
    #         apy_path = os.path.join(folder, 'Apy', f'Apy_{filename}.png')

    #         predict_img = Image.open(predict_path).convert("RGB")
    #         apy_img = Image.open(apy_path).convert("RGB")
    #         gt_img = Image.open(gt_path).convert("RGB")
            
    #         total_output_psnr += calculate_psnr_eval(predict_img, gt_img)
    #         total_output_ssim += calculate_ssim_eval(predict_img, gt_img)
    #         total_output_lpips += calculate_lpips_eval(lpips_model, predict_img, gt_img)

    #         total_apy_psnr += calculate_psnr_eval(apy_img, gt_img)
    #         total_apy_ssim += calculate_ssim_eval(apy_img, gt_img)
    #         total_apy_lpips += calculate_lpips_eval(lpips_model, apy_img, gt_img)
    #     print('output avg psnr:', total_output_psnr / N) 
    #     print('output avg ssim:', total_output_ssim / N) 
    #     print('output avg lpips:', total_output_lpips / N) 

    #     print('apy avg psnr:', total_apy_psnr / N) 
    #     print('apy avg ssim:', total_apy_ssim / N) 
    #     print('apy avg lpips:', total_apy_lpips / N) 

    #     with open(os.path.join(folder, '_log_val_.txt'), 'w') as f:
    #         f.writelines(f'output avg psnr: {total_output_psnr / N} \n')
    #         f.writelines(f'output ssim psnr: {total_output_ssim / N} \n')
    #         f.writelines(f'output lpips psnr: {total_output_lpips / N} \n')
    #         f.writelines(f'apy avg psnr: {total_apy_psnr / N} \n')
    #         f.writelines(f'apy ssim psnr: {total_apy_ssim / N} \n')
    #         f.writelines(f'apy lpips psnr: {total_apy_lpips / N} \n')
        
    # def ddnm_plus(self, model):
    #     args, config = self.args, self.config
    #     g = torch.Generator()
    #     g.manual_seed(args.seed)
    #     set_seed(args.seed)
        
    #     self.scheduler = DDPMScheduler.from_pretrained('google/ddpm-celebahq-256')
    #     self.scheduler.set_timesteps(100)
    #     # mkdirs
    #     os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
        
    #     # get all test image filename
    #     test_dataset_root = self.args.path_y
    #     all_files = os.listdir(os.path.join(test_dataset_root, 'hq'))
    #     if self.args.dataset == 'imagenet':
    #         filename_list = sorted([file[:-5] for file in all_files])
    #     else:
    #         filename_list = sorted([file[:-4] for file in all_files])
            
    #     avg_output_psnr = 0.0
    #     avg_kernel_psnr = 0.0
    #     avg_apy_psnr = 0.0
    #     avg_consistency = 0.0
        
    #     #init A, Ap
    #     if self.args.DDNM_A == "implicit" or self.args.DDNM_Ap == "implicit" or self.args.perturb_A == "implicit" or self.args.perturb_Ap == "implicit":
    #         implicit_A = lambda z : self.IRmodel.net_g(z, z, LD_LQ)[1] #lq, x0t_gt, oirginal_lq
    #         if self.IRmodel2 is not None:
    #             implicit_Ap = lambda z : self.IRmodel2.net_g(z, lq=LD_LQ)
    #         else:     
    #             implicit_Ap = lambda z : self.IRmodel.net_g(z, z, LD_LQ)[0]
            

    #     filename_list = filename_list[:self.args.sample_number]
    #     for filename in tqdm.tqdm(filename_list):
    #         if os.path.exists(os.path.join(self.args.image_folder, f"{filename}.png")):
    #             continue
    #         if self.args.dataset == 'imagenet':
    #             hq_path = os.path.join(test_dataset_root, 'hq', f'{filename}.JPEG')
    #             lq_path = os.path.join(test_dataset_root, 'lq', f'{int(self.args.deg_scale)}', f'{filename}.jpg')
    #         elif self.args.dataset == 'benchmark':
    #             hq_path = os.path.join(test_dataset_root, 'hq', f'{filename}.png')
    #             lq_path = os.path.join(test_dataset_root, 'lq', f'{int(self.args.deg_scale)}', f'{filename}.png')
    #         else:
    #             hq_path = os.path.join(test_dataset_root, 'hq', f'{filename}.jpg')
    #             lq_path = os.path.join(test_dataset_root, 'lq', f'{int(self.args.deg_scale)}', f'{filename}.jpg')
    #         gt_kernel_path = os.path.join(test_dataset_root, 'kernel', f'{filename}.npy')
            
    #         # if self.args.input_mode == 'LQ':
    #         #     DDNM_LQ = load_imgDDNM(lq_path).to(self.device)
    #         #     DDNM_LQ = data_transform(self.config, DDNM_LQ)
    #         #     tvu.save_image(
    #         #         inverse_data_transform(config, DDNM_LQ[0]),
    #         #         os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")
    #         #     )
    #         #     # print(DDNM_LQ.min()) # -1
    #         #     # print(DDNM_LQ.max()) # 1
    #         # elif self.args.input_mode == 'AHQ':
    #         # permformance will be better if we use below code to get LQ, rather than load image from LQ_path 
    #         HQ = load_imgDDNM(hq_path).to(self.device)
    #         HQ = data_transform(self.config, HQ)
    #         gt_kernel = np.load(gt_kernel_path)
    #         gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
    #         gt_A = lambda z : convolution2d(z, gt_kernel, stride=int(self.args.deg_scale), padding=gt_kernel.size(1)//2) 
    #         DDNM_LQ = gt_A(HQ)
    #         tvu.save_image(
    #             inverse_data_transform(config, DDNM_LQ[0]),
    #             os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")
    #         )
    #         lq_path = os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")
    #         # lq_path = convert_jpg(lq_path)
    #         DDNM_LQ = load_imgDDNM(lq_path).to(self.device)
    #         DDNM_LQ = data_transform(self.config, DDNM_LQ)
            
    #         if self.args.DDNM_A == "implicit" or self.args.DDNM_Ap == "implicit" or self.args.perturb_A == "implicit" or self.args.perturb_Ap == "implicit":
    #             LD_LQ = load_img_LearningDegradation(lq_path).to('cuda')
    #         # get A, Ap for current input
    #         if self.args.DDNM_A == "explicit_gt" or self.args.DDNM_Ap == "explicit_gt" or self.args.perturb_A == "explicit_gt" or self.args.perturb_Ap == "explicit_gt":
    #             gt_kernel = np.load(gt_kernel_path)
    #             gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
    #             gt_A = lambda z : convolution2d(z, gt_kernel, stride=int(self.args.deg_scale), padding=gt_kernel.size(1)//2) 

    #             gt_padding = gt_kernel.shape[0] // 2
    #             gt_matrix_A = convolution_to_A(gt_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=gt_padding).cuda()
    #             gt_matrix_A_pinverse = torch.pinverse(gt_matrix_A)
    #             gt_A = lambda z: convolution_with_A(gt_matrix_A, z, padding=gt_padding)
    #             gt_Ap = lambda z: convolution_with_A(gt_matrix_A_pinverse, z, padding=0)[:, :, gt_padding:-gt_padding, gt_padding:-gt_padding]
    #         if self.args.DDNM_A == "explicit" or self.args.DDNM_Ap == "explicit" or self.args.perturb_A == "explicit" or self.args.perturb_Ap == "explicit":
    #             gt_kernel = np.load(gt_kernel_path)
    #             gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
    #             # padding gt_kenel to 21 * 21
    #             gt_kernel = gt_kernel.unsqueeze(0)
    #             padding21 = (self.args.kernel_size - gt_kernel.size(1)) //2
    #             gt_kernel = F.pad(gt_kernel, (padding21, padding21, padding21, padding21)).squeeze()
                
    #             predict_kernel = None
    #             explicit_LQ = load_imgDDNM(lq_path).to(self.device)
    #             explicit_LQ = data_transform(self.config, explicit_LQ)
    #             with torch.no_grad():
    #                 predict_kernel = self.kernel_estimator(explicit_LQ).squeeze()
    #                 if self.args.save_kernel:
    #                     os.makedirs(os.path.join(self.args.image_folder, "kernel"), exist_ok=True)
    #                     np_kernel = predict_kernel.cpu().numpy()
    #                     np.save(os.path.join(self.args.image_folder, "kernel", f"{filename}.npy"), np_kernel)
                        
    #                 kernel_psnr = calculate_psnr(predict_kernel, gt_kernel)
    #                 avg_kernel_psnr += kernel_psnr
    #                 self.txt_logger.info(f'{filename} kernel psnr: {kernel_psnr}')
    #             plot_kernel(gt_kernel.unsqueeze(0).unsqueeze(0), predict_kernel.unsqueeze(0).unsqueeze(0), filename, self.args.image_folder)
    #             if self.args.DDNM_Ap == "explicit" or self.args.perturb_Ap == "explicit":
    #                 explicit_padding = predict_kernel.shape[0] // 2
    #                 matrix_A = convolution_to_A(predict_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=explicit_padding).cuda()
    #                 matrix_A_pinverse = torch.pinverse(matrix_A) 
    #                 # print("AAp, I p2 norm ", torch.dist(matrix_A @ matrix_A_pinverse, torch.eye(matrix_A.shape[0]).cuda()).item())
    #                 explicit_A = lambda z: convolution_with_A(matrix_A, z, padding=explicit_padding)
    #                 explicit_Ap = lambda z: convolution_with_A(matrix_A_pinverse, z, padding=0)[:, :, explicit_padding:-explicit_padding, explicit_padding:-explicit_padding]
    #             else:
    #                 if self.args.consistency_kernel != "":
    #                     predict_kernel = np.load(self.args.consistency_kernel)
    #                     predict_kernel = torch.from_numpy(predict_kernel).float().cuda()
    #                     predict_kernel = predict_kernel.squeeze()
    #                 explicit_A = lambda z : convolution2d(z, predict_kernel, stride=int(self.args.deg_scale), padding=predict_kernel.size(1)//2) 
    #             # tt_y = torch.randn((1, 3, 64, 64)).cuda()
    #             # tt_AApy = explicit_A(explicit_Ap(tt_y))
    #             # print("random float check consistenct:", torch.dist(tt_y, tt_AApy) / len(tt_y.flatten()))
                

    #         #check perturb Y, store the perturbed y 
    #         if self.args.perturb_y:
    #             if self.args.perturb_A == "explicit" and self.args.perturb_Ap == "implicit":
    #                 implicit_Apy = implicit_Ap(LD_LQ)
    #                 implicit_Apy_img = tensor2img(implicit_Apy, rgb2bgr=True)
    #                 imwrite(implicit_Apy_img, os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png"))
    #                 implicit_Apy = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")).to(self.device)
    #                 implicit_Apy = data_transform(self.config, implicit_Apy)
    #                 LQ = explicit_A(implicit_Apy)
    #                 tvu.save_image(
    #                     inverse_data_transform(config, LQ[0]),
    #                     os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")
    #                 )
    #             elif self.args.perturb_A == "explicit_gt" and self.args.perturb_Ap == "implicit":
    #                 implicit_Apy = implicit_Ap(LD_LQ)
    #                 implicit_Apy_img = tensor2img(implicit_Apy, rgb2bgr=True)
    #                 imwrite(implicit_Apy_img, os.path.join(self.args.image_folder, f"Apy/implicit_Apy_{filename}.png"))
    #                 implicit_Apy = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/implicit_Apy_{filename}.png")).to(self.device)
    #                 implicit_Apy = data_transform(self.config, implicit_Apy)
    #                 LQ = gt_A(implicit_Apy)
    #                 tvu.save_image(
    #                     inverse_data_transform(config, LQ[0]),
    #                     os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")
    #                 )
    #             elif self.args.perturb_A == "implicit" and self.args.perturb_Ap == "implicit":    
    #                 LQ = implicit_A(implicit_Ap(LD_LQ))
    #                 LQ_img = tensor2img(LQ, rgb2bgr=True)
    #                 imwrite(LQ_img, os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png"))
    #             else:
    #                 raise ValueError(f"perturb mode {self.args.perturb_A}, {self.args.perturb_Ap} not supported")
                
    #             # load perturbed y as new LQ
    #             DDNM_LQ = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")).to(self.device)
    #             DDNM_LQ = data_transform(self.config, DDNM_LQ)


    #         # #cal high pass filter l2 norm (how noisy the input is)
    #         # Up_N = lambda z: F.interpolate(z, size=(64, 64), mode='bilinear', align_corners=False)
    #         # Down_N = lambda z: F.interpolate(z, size=(64 // 4, 64 // 4), mode='bilinear', align_corners=False)
    #         # lowpassfilter = lambda z : Up_N(Down_N(z))
    #         # filtered_output = DDNM_LQ - lowpassfilter(DDNM_LQ)
    #         # l2_norm = torch.norm(filtered_output, dim=(1, 2, 3), p=2)
    #         # self.txt_logger.info(f"{filename} LQ apply high pass filter l2 norm : {l2_norm.item()}")
            
            
    #         # perform Ap(y)
    #         if self.args.DDNM_Ap == "explicit_gt":
    #             Apy = gt_Ap(DDNM_LQ)
    #             tvu.save_image(
    #                 inverse_data_transform(config, Apy[0]),
    #                 os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")
    #             )
    #         elif self.args.DDNM_Ap == "explicit":
    #             Apy = explicit_Ap(DDNM_LQ)
    #             # print(Apy.min()) # < -1
    #             # print(Apy.max()) # > 1
    #             tvu.save_image(
    #                 inverse_data_transform(config, Apy[0]),
    #                 os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")
    #             )
    #         elif self.args.DDNM_Ap == "implicit":
    #             if self.args.perturb_y:
    #                 LD_LQ = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")).to('cuda')
    #             Apy = implicit_Ap(LD_LQ)
    #             Apy_img = tensor2img(Apy, rgb2bgr=True)
    #             imwrite(Apy_img, os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png"))
    #         else:
    #             raise ValueError("DDNM Ap mode {self.args.DDNM_Ap} not supported")  

    #         if self.args.DDNM_Ap == "implicit": 
    #             Apy = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")).to(self.device)
    #             Apy = data_transform(self.config, Apy) # don't know why 
            
    #         # check_consistency:
    #         # if self.args.DDNM_A == "explicit_gt":    
    #         #     con_Apy = inverse_data_transform(config, Apy)              
    #         #     AApy = gt_A(con_Apy)
    #         #     tvu.save_image(
    #         #         AApy[0],
    #         #         os.path.join(self.args.image_folder, f"Apy/AApy_{filename}.png")
    #         #     )
    #         # elif self.args.DDNM_A == "explicit":
    #         #     con_Apy = inverse_data_transform(config, Apy)
    #         #     AApy = explicit_A(con_Apy)
    #         #     tvu.save_image(
    #         #         AApy[0],
    #         #         os.path.join(self.args.image_folder, f"Apy/AApy_{filename}.png")
    #         #     ) 
    #         # elif self.args.DDNM_A == "implicit":
    #         #     AApy = implicit_A(implicit_Ap(LD_LQ))
    #         #     AApy_img = tensor2img(AApy, rgb2bgr=True)
    #         #     imwrite(AApy_img, os.path.join(self.args.image_folder, f"Apy/AApy_{filename}.png"))
    #         # consistemcy_y = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")).to(self.device)
    #         # if self.args.DDNM_A == "implicit":
    #         #     consistemcy_y = transforms.Compose([transforms.Resize((256,256), antialias=None),])(consistemcy_y.squeeze()).unsqueeze(0)
    #         # consistemcy_AApy = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/AApy_{filename}.png")).to(self.device)
    #         # p2_dist = torch.mean((consistemcy_y-consistemcy_AApy)**2).item()
    #         # avg_consistency += p2_dist
    #         # self.txt_logger.info(f'{filename} per sample consistency mse: {p2_dist}')
                   
    #         HQ = load_imgDDNM(hq_path).to(self.device)
    #         HQ = data_transform(self.config, HQ)
    #         tvu.save_image(
    #             inverse_data_transform(config, HQ[0]),
    #             os.path.join(self.args.image_folder, f"Apy/orig_{filename}.png")
    #         )
    #         Apy_psnr = cal_img_psnr([inverse_data_transform(config, HQ[0])], [inverse_data_transform(config, Apy[0])])
    #         self.txt_logger.info(f'{filename} per sample apy psnr: {Apy_psnr}')
    #         avg_apy_psnr += Apy_psnr
            
    #         if self.args.DDNM_Ap == "implicit": 
    #             Apy = inverse_data_transform(self.config, Apy) # dopn't know why
            
    #         # init x_T
    #         x = torch.randn(
    #             HQ.shape[0],
    #             config.data.channels,
    #             config.data.image_size,
    #             config.data.image_size,
    #             device=self.device,
    #         )
    #         # with torch.no_grad():
    #         skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
    #         n = x.size(0)
    #         x0_preds = []
    #         xs = [x]
            
    #         times = get_schedule_jump(config.time_travel.T_sampling, 
    #                                         config.time_travel.travel_length, 
    #                                         config.time_travel.travel_repeat,
    #                                         )
    #         time_pairs = list(zip(times[:-1], times[1:])) 
            
    #         # reverse diffusion sampling
    #         for i, j in time_pairs:
    #             i, j = i*skip, j*skip
    #             if j<0: j=-1 

    #             if j < i: # normal sampling 
    #                 t = (torch.ones(n) * i).to(x.device)
    #                 next_t = (torch.ones(n) * j).to(x.device)
    #                 at = compute_alpha(self.betas, t.long())
    #                 at_next = compute_alpha(self.betas, next_t.long())
    #                 xt = xs[-1].to('cuda')
    #                 if self.args.save_img:
    #                     os.makedirs(os.path.join(self.args.image_folder, "x_t", f'{filename}'), exist_ok=True)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, xt.to('cpu')),
    #                         os.path.join(self.args.image_folder, "x_t", f"{filename}/x_{int(t[0])}.png")
    #                     )     
                    
    #                 with torch.no_grad():
    #                     et = model(xt, t)
                        
    #                 if et.size(1) == 6:
    #                     et = et[:, :3]
    #                 # Eq. 12
    #                 x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() 
                    
    #                 # save x0_t
    #                 if self.args.save_img:
    #                     os.makedirs(os.path.join(self.args.image_folder, "x0_t", f'{filename}'), exist_ok=True)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, x0_t.to('cpu')),
    #                         os.path.join(self.args.image_folder, "x0_t", f"{filename}/x0_{int(t[0])}.png")
    #                     )
    #                 else:
    #                     os.makedirs(os.path.join(self.args.image_folder, "tmp"), exist_ok=True)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, x0_t.to('cpu')),
    #                         os.path.join(self.args.image_folder, "tmp", "x0_t.png")
    #                     )        
    #                 # Eq. 13
    #                 # x0_t_hat = ApA(Apy) + x0_t - ApA(x0_t)
                    
    #                 #get ApA(x0_t)
    #                 if self.args.DDNM_A == 'implicit' and self.args.DDNM_Ap == 'implicit':
    #                     if self.args.save_img:
    #                         x0_tL = load_img_LearningDegradation(os.path.join(self.args.image_folder, "x0_t", f"{filename}/x0_{int(t[0])}.png")).to('cuda')
    #                     else:
    #                         x0_tL = load_img_LearningDegradation(os.path.join(self.args.image_folder, "tmp", "x0_t.png")).to('cuda')
    #                     ApAx0_t = implicit_Ap(implicit_A(x0_tL))
    #                     ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)                    
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, "ApA", f"{filename}/ApAx0_{int(t[0])}.png"))
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, "ApA", f"{filename}/ApAx0_{int(t[0])}.png")).to('cuda')
    #                     else:
    #                         os.makedirs(os.path.join(self.args.image_folder, "tmp"), exist_ok=True)
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, "tmp", "ApAx0_t.png"))
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, "tmp", "ApAx0_t.png")).to('cuda')
    #                     # ApAx0_t = data_transform(config, ApAx0_t)
    #                 elif self.args.DDNM_A == 'explicit' and self.args.DDNM_Ap == 'implicit':
    #                     explicit_Ax0_t = explicit_A(x0_t)
    #                     if self.args.save_img:
    #                         tvu.save_image(
    #                             inverse_data_transform(config, explicit_Ax0_t.to('cpu')),
    #                             os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")
    #                         )
    #                         Ax0t = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")).to('cuda')
    #                     else:
    #                         os.makedirs(os.path.join(self.args.image_folder, "tmp"), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, explicit_Ax0_t.to('cpu')),
    #                             os.path.join(self.args.image_folder, f"tmp/Ax0t.png")
    #                         )
    #                         Ax0t = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"tmp/Ax0t.png")).to('cuda')
    #                     ApAx0_t = implicit_Ap(Ax0t)
    #                     ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png"))
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")).to('cuda')
    #                     else:
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"tmp/ApAx0t.png"))
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, f"tmp/ApAx0t.png")).to('cuda')
    #                     # ApAx0_t = data_transform(self.config, ApAx0_t)
                        
    #                 elif self.args.DDNM_A == 'explicit_gt' and self.args.DDNM_Ap == 'implicit':
    #                     explicit_Ax0_t = gt_A(x0_t)
    #                     if self.args.save_img:
    #                         tvu.save_image(
    #                             inverse_data_transform(config, explicit_Ax0_t.to('cpu')),
    #                             os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")
    #                         )
    #                         Ax0t = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")).to('cuda')
    #                     else:
    #                         os.makedirs(os.path.join(self.args.image_folder, "tmp"), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, explicit_Ax0_t.to('cpu')),
    #                             os.path.join(self.args.image_folder, f"tmp/Ax0t.png")
    #                         )
    #                         Ax0t = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"tmp/Ax0t.png")).to('cuda')
    #                     ApAx0_t = implicit_Ap(Ax0t)
    #                     ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png"))
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")).to('cuda')
    #                     else:
    #                         imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"tmp/ApAx0t.png"))
    #                         ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, f"tmp/ApAx0t.png")).to('cuda')
    #                     # ApAx0_t = data_transform(self.config, ApAx0_t)
                        
    #                 elif self.args.DDNM_A == 'explicit' and self.args.DDNM_Ap == 'explicit':
    #                     ApAx0_t = explicit_Ap(explicit_A(x0_t))
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, ApAx0_t[0].to('cpu')),
    #                             os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")
    #                         )
                            
    #                 elif self.args.DDNM_A == 'explicit_gt' and self.args.DDNM_Ap == 'explicit_gt':
    #                     ApAx0_t = gt_Ap(gt_A(x0_t))
    #                     if self.args.save_img:
    #                         os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                         tvu.save_image(
    #                             inverse_data_transform(config, ApAx0_t[0]),
    #                             os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")
    #                         )
    #                 else:
    #                     raise ValueError("DDNM A mode {self.args.DDNM_A} with Ap mode {self.args.DDNM_Ap} not supported")  
                    
    #                 alpha = self.args.alpha
    #                 x0_t_hat = x0_t + alpha * (Apy - ApAx0_t) # DDNM formula
                    
    #                 # if self.args.check_dist:
    #                 #     #check wether x0_t is in diffusion distribution
    #                 #     T = torch.randint(999, 999 + 1, [1], dtype=torch.long, device='cpu')
    #                 #     noi = torch.randn_like(x0_t_hat.cpu())
    #                 #     xT = self.scheduler.add_noise(x0_t_hat.cpu(), noi, T)
    #                 #     self.txt_logger.info(f'timestep: {int(t[0])} {xT.mean().item()} {xT.var().item()}')
    #                 #     # print('timestep:', int(t[0]), xT.mean().item(), xT.var().item()) # should be 0, 1
                    
    #                 if self.args.save_img:
    #                     os.makedirs(os.path.join(self.args.image_folder, "x0_t_hat", f'{filename}'), exist_ok=True)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, x0_t_hat),
    #                         os.path.join(self.args.image_folder, "x0_t_hat", f"{filename}/x0_{int(t[0])}_hat.png")
    #                     ) 
                    
    #                 eta = self.args.eta
    #                 if self.args.posterior_formula == "DDIM":
    #                     sigma = (
    #                         eta
    #                         * torch.sqrt((1 - at_next) / (1 - at))
    #                         * torch.sqrt(1 - at / at_next)
    #                     )
    #                     mean_pred = (
    #                         x0_t_hat * torch.sqrt(at_next)
    #                         + torch.sqrt(1 - at_next - sigma ** 2) * et
    #                     )
    #                     xt_next = mean_pred + sigma * torch.randn_like(x0_t)
    #                 elif self.args.posterior_formula == "DDNM":
    #                     c1 = (1 - at_next).sqrt() * eta
    #                     c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
    #                     # sigma_t = (1 - at_next**2).sqrt()
    #                     # different from the paper, we use DDIM here instead of DDPM
    #                     xt_next = at_next.sqrt() * x0_t_hat + (c1 * torch.randn_like(x0_t) + c2 * et)

    #                 x0_preds.append(x0_t.to('cpu'))
    #                 xs.append(xt_next.to('cpu'))   
    #             else: # time-travel back
    #                 raise NotImplementedError
    #                 # next_t = (torch.ones(n) * j).to(x.device)
    #                 # at_next = compute_alpha(self.betas, next_t.long())
    #                 # x0_t = x0_preds[-1].to('cuda')

    #                 # xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

    #                 # xs.append(xt_next.to('cpu'))

    #         x = xs[-1]
    #         # check how noisy the output is
    #         # t =  x[0].unsqueeze(0)
    #         # Up_N = lambda z: F.interpolate(z, size=(256, 256), mode='bilinear', align_corners=False)
    #         # Down_N = lambda z: F.interpolate(z, size=(256 // 4, 256 // 4), mode='bilinear', align_corners=False)
    #         # lowpassfilter = lambda z : Up_N(Down_N(z))
    #         # filtered_output = t - lowpassfilter(t)
    #         # l2_norm = torch.norm(filtered_output, dim=(1, 2, 3), p=2)
    #         # self.txt_logger.info(f"{filename} output apply high pass filter l2 norm : {l2_norm.item()}")
            
    #         x = [inverse_data_transform(config, xi) for xi in x]

    #         tvu.save_image(
    #             x[0], os.path.join(self.args.image_folder, f"{filename}.png")
    #         )
            
    #         output_psnr = cal_img_psnr([inverse_data_transform(config, HQ[0])], x)
    #         self.txt_logger.info(f'{filename} per sample psnr: {output_psnr}')
    #         avg_output_psnr += output_psnr

    #     avg_output_psnr = avg_output_psnr / len(filename_list)
    #     avg_apy_psnr = avg_apy_psnr / len(filename_list)
    #     avg_kernel_psnr = avg_kernel_psnr / len(filename_list)
    #     avg_consistency = avg_consistency / len(filename_list)
    #     self.txt_logger.info(f"Total Average kernel PSNR: {avg_kernel_psnr}")
    #     self.txt_logger.info(f"Average AApy consistency in mse: {avg_consistency}")
    #     self.txt_logger.info(f"Apy Total Average PSNR: {avg_apy_psnr}")
    #     self.txt_logger.info(f"Output Total Average PSNR: {avg_output_psnr}")
    #     self.txt_logger.info(f"Number of samples: {len(filename_list)}")
            
            
            

        
    # def combine_simplified_ddnm_plus(self, model, cls_fn):
    #     args, config = self.args, self.config
        
    #     def seed_worker(worker_id):
    #         worker_seed = args.seed % 2 ** 32
    #         np.random.seed(worker_seed)
    #         random.seed(worker_seed)

    #     g = torch.Generator()
    #     g.manual_seed(args.seed)
    

    #     # get all test image filename
    #     test_dataset_root = self.args.path_y
    #     all_files = os.listdir(os.path.join(test_dataset_root, 'hq'))
    #     if self.config.data.dataset == 'ImageNet':
    #         filename_list = sorted([file[:-5] for file in all_files])
    #     else:
    #         filename_list = sorted([file[:-4] for file in all_files])
        
    #     avg_kernel_psnr = 0.0
    #     avg_output_psnr = 0.0
    #     avg_implicit_apy_psnr = 0.0
        
    #     # initial impicit IR model
    #     implicit_A = lambda z : self.IRmodel.net_g(z, z, lq=implicit_LQ)[1]        
    #     implicit_Ap = lambda z : self.IRmodel.net_g(z, z, lq=implicit_LQ)[0]
    #     implicit_ApA = lambda z: implicit_Ap(implicit_A(z))     
        
    #     # Up_N = lambda z: F.interpolate(z, size=(self.config.data.image_size, self.config.data.image_size), mode='bilinear', align_corners=False)
    #     # Down_N = lambda z: F.interpolate(z, size=(self.config.data.image_size // self.args.N, self.config.data.image_size // self.args.N), mode='bilinear', align_corners=False)
    #     # lowpassfilter = lambda z : Up_N(Down_N(z))
        
        
    #     for filename in tqdm.tqdm(filename_list):
    #         if os.path.exists(os.path.join(self.args.image_folder, f"output_{filename}.png")):
    #             continue  
    #         lq_path = os.path.join(test_dataset_root, 'lq', f'{int(self.args.deg_scale)}', f'{filename}.jpg')
    #         if self.config.data.dataset == 'ImageNet':
    #             hq_path = os.path.join(test_dataset_root, 'hq', f'{filename}.JPEG')
    #         else:
    #             hq_path = os.path.join(test_dataset_root, 'hq', f'{filename}.jpg')
    #         gt_kernel_path = os.path.join(test_dataset_root, 'kernel', f'{filename}.npy')
    #         gt_kernel = np.load(gt_kernel_path)
    #         gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
            
    #         HQ = load_imgDDNM(hq_path).to(self.device)

    #         implicit_LQ = load_img_LearningDegradation(lq_path).to('cuda')
    #         implicit_Apy = implicit_Ap(implicit_LQ)
    #         implicit_Apy_img = tensor2img(implicit_Apy, rgb2bgr=True)
    #         os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
    #         imwrite(implicit_Apy_img, os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png"))
    #         implicit_Apy = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")).to(self.device)
    #         # cal apy psnr 
    #         implicit_apy_psnr = cal_img_psnr(HQ, implicit_Apy)
    #         self.txt_logger.info(f'{filename} per sample apy psnr implicit: {implicit_apy_psnr}')
    #         # print(f'{filename} per sample apy psnr implicit:', implicit_apy_psnr)
    #         avg_implicit_apy_psnr += implicit_apy_psnr
    
    #         HQ = data_transform(self.config, HQ)
    #         LQ = convolution2d(HQ, gt_kernel, stride=int(self.args.deg_scale), padding=gt_kernel.size(1)//2) 
            
    #         # padding gt_kernel to 21 * 21
    #         gt_kernel = gt_kernel.unsqueeze(0)
    #         padding = (self.args.kernel_size - gt_kernel.size(1)) //2
    #         gt_kernel = F.pad(gt_kernel, (padding, padding, padding, padding)).squeeze()

            
    #         # predict explicit kernel and construct A, Ap
    #         predict_kernel = None
    #         with torch.no_grad():
    #             predict_kernel = self.kernel_estimator(LQ).squeeze()
    #             kernel_psnr = calculate_psnr(predict_kernel, gt_kernel)
    #             avg_kernel_psnr += kernel_psnr
    #             self.txt_logger.info(f'{filename} kernel psnr: {kernel_psnr}')
    #             # print(f'{filename} kernel psnr:', kernel_psnr)
    #         plot_kernel(gt_kernel.unsqueeze(0).unsqueeze(0), predict_kernel.unsqueeze(0).unsqueeze(0), filename, self.args.image_folder)
    #         padding = predict_kernel.shape[0] // 2
    #         matrix_A = convolution_to_A(predict_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=padding).cuda()
    #         matrix_A_pinverse = torch.pinverse(matrix_A) 
        
    #         # padding = gt_kernel.shape[0] // 2
    #         # matrix_A = convolution_to_A(gt_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=padding).cuda()
    #         # matrix_A_pinverse = torch.pinverse(matrix_A) 
                
            
    #         explicit_A = lambda z: convolution_with_A(matrix_A, z, padding=padding)
    #         explicit_Ap = lambda z: convolution_with_A(matrix_A_pinverse, z, padding=0)[:, :, padding:-padding, padding:-padding]
    #         explicit_ApA = lambda z: explicit_Ap(explicit_A(z))
            
    #         implicit_Apy = data_transform(self.config, implicit_Apy)
    #         if self.args.perturb_y:
    #             AApy = explicit_A(implicit_Apy)
    #             tvu.save_image(
    #                 inverse_data_transform(config, AApy[0]),
    #                 os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")
    #             )
    #             implicit_LQ = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"Apy/perturbedY_{filename}.png")).to('cuda')
    #             implicit_Apy = implicit_Ap(implicit_LQ)
    #             implicit_Apy_img = tensor2img(implicit_Apy, rgb2bgr=True)
    #             imwrite(implicit_Apy_img, os.path.join(self.args.image_folder, f"Apy/newApy_{filename}.png"))
    #             implicit_Apy = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/newApy_{filename}.png")).to(self.device)
    #             implicit_Apy = data_transform(self.config, implicit_Apy)
                
                
                
    #         # add perturb y
    #         # explicit_Apy = explicit_Ap(explicit_A(implicit_Apy)) 
    #         # explicit_Apy = explicit_Ap(LQ)
    #         for i in range(len(implicit_Apy)):
    #             # tvu.save_image(
    #             #     inverse_data_transform(config, explicit_Apy[i]),
    #             #     os.path.join(self.args.image_folder, f"Apy/Apy_explicit_{filename}.png")
    #             # )
    #             tvu.save_image(
    #                 inverse_data_transform(config, HQ[i]),
    #                 os.path.join(self.args.image_folder, f"Apy/orig_{filename}.png")
    #             )
    #             tvu.save_image(
    #                 inverse_data_transform(config, LQ[i]),
    #                 os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")
    #             )
                
    #         # init x_T
    #         x = torch.randn(
    #             HQ.shape[0],
    #             config.data.channels,
    #             config.data.image_size,
    #             config.data.image_size,
    #             device=self.device,
    #         )

    #         with torch.no_grad():
    #             skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
    #             n = x.size(0)
    #             x0_preds = []
    #             xs = [x]
                
    #             times = get_schedule_jump(config.time_travel.T_sampling, 
    #                                            config.time_travel.travel_length, 
    #                                            config.time_travel.travel_repeat,
    #                                           )
    #             time_pairs = list(zip(times[:-1], times[1:]))
                
                
    #             # reverse diffusion sampling
    #             for i, j in time_pairs:
    #                 i, j = i*skip, j*skip
    #                 if j<0: j=-1 

    #                 if j < i: # normal sampling 
    #                     if self.config.user.ddnm:
    #                         #self inplement DDNM
    #                         t = (torch.ones(n) * i).to(x.device)
    #                         next_t = (torch.ones(n) * j).to(x.device)
    #                         at = compute_alpha(self.betas, t.long())
    #                         at_next = compute_alpha(self.betas, next_t.long())
    #                         xt = xs[-1].to('cuda')

    #                         et = model(xt, t)
    #                         if et.size(1) == 6:
    #                             et = et[:, :3]
    #                         # Eq. 12
    #                         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() 
    #                         explicit_Ax0_t = explicit_A(x0_t)
                            
    #                         if self.args.save_img:
    #                             # save x0_t
    #                             os.makedirs(os.path.join(self.args.image_folder, "x0_t", f'{filename}'), exist_ok=True)
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, x0_t),
    #                                 os.path.join(self.args.image_folder, f"x0_t/{filename}/x0_{t[0]}.png")
    #                             )
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, explicit_Ax0_t),
    #                                 os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")
    #                             )
    #                         else:
    #                             os.makedirs(os.path.join(self.args.image_folder, "tmp"), exist_ok=True)
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, explicit_Ax0_t),
    #                                 os.path.join(self.args.image_folder, f"tmp/Ax0t.png")
    #                             )
                                
    #                         #implicit + DDNM
    #                         if self.args.save_img:
    #                             Ax0t = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"x0_t/{filename}/Ax0_{t[0]}.png")).to('cuda')
    #                         else:
    #                             Ax0t = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"tmp/Ax0t.png")).to('cuda')
                            
    #                         ApAx0t = implicit_Ap(Ax0t)
    #                         ApAx0t_img = tensor2img(ApAx0t, rgb2bgr=True)
    #                         if self.args.save_img:
    #                             imwrite(ApAx0t_img, os.path.join(self.args.image_folder, f"ApA/{filename}/ApA_x0_{t[0]}.png"))
    #                             ApAx0t = load_imgDDNM(os.path.join(self.args.image_folder, f"ApA/{filename}/ApA_x0_{t[0]}.png")).to('cuda')
    #                         else:
    #                             imwrite(ApAx0t_img, os.path.join(self.args.image_folder, f"tmp/ApAx0t.png"))
    #                             ApAx0t = load_imgDDNM(os.path.join(self.args.image_folder, f"tmp/ApAx0t.png")).to('cuda')
    #                         ApAx0t = data_transform(self.config, ApAx0t)
    #                         # Eq. 13
    #                         # x0_t_hat = ApA(Apy) + x0_t - ApA(x0_t)
    #                         # explicit + DDNM
    #                         # explicit_ApAx0_t = explicit_ApA(x0_t)
    #                         # if self.args.save_img:
    #                         #     os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                         #     tvu.save_image(
    #                         #         inverse_data_transform(config, explicit_ApAx0_t[0]),
    #                         #         os.path.join(self.args.image_folder, f"ApA/{filename}/ApA_explicit_x0_{t[0]}.png")
    #                         #     )
    #                             #implicit + DDNM
    #                             # x0_tL = load_img_LearningDegrad tion(os.path.join(self.args.image_folder, f"x0_t/{filename}/x0_{t[0]}.png")).to('cuda')
    #                             # implicit_ApAx0_t = implicit_ApA(x0_tL)
    #                             # ApAx0_t_img = tensor2img(implicit_ApAx0_t, rgb2bgr=True)
    #                             # imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"ApA/{filename}/ApA_implicit_x0_{t[0]}.png"))
    #                             # implicit_ApAx0_t = load_imgDDNM(os.path.join(self.args.image_folder, f"ApA/{filename}/ApA_implicit_x0_{t[0]}.png")).to('cuda')
    #                         alpha = self.args.alpha
    #                         x0_t_hat = alpha * implicit_Apy + x0_t - alpha * ApAx0t
    #                         # explicit_x0_t_hat = explicit_Apy + x0_t - explicit_ApAx0_t     
                            
    #                         # combine_x0_t_hat = lowpassfilter(explicit_x0_t_hat) + implicit_x0_t_hat - lowpassfilter(implicit_x0_t_hat)
    #                         # combine_x0_t_hat = lowpassfilter(implicit_x0_t_hat) + explicit_x0_t_hat - lowpassfilter(explicit_x0_t_hat)
    #                         # combine_x0_t_hat = (implicit_x0_t_hat + explicit_x0_t_hat) / 2                       
                            
                            
    #                         # x0_t_hat = Apy + x0_t - ApA(x0_t)
                            
    #                         eta = self.args.eta

    #                         c1 = (1 - at_next).sqrt() * eta
    #                         c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

    #                         # different from the paper, we use DDIM here instead of DDPM
    #                         xt_next = at_next.sqrt() * x0_t_hat + (c1 * torch.randn_like(x0_t) + c2 * et)

    #                         x0_preds.append(x0_t.to('cpu'))
    #                         xs.append(xt_next.to('cpu'))   
    #                     else:

    #                         t = (torch.ones(n) * i).to(x.device)
    #                         next_t = (torch.ones(n) * j).to(x.device)
    #                         at = compute_alpha(self.betas, t.long())
    #                         at_next = compute_alpha(self.betas, next_t.long())
    #                         sigma_t = (1 - at_next**2).sqrt()
    #                         xt = xs[-1].to('cuda')

    #                         et = model(xt, t)

    #                         if et.size(1) == 6:
    #                             et = et[:, :3]

    #                         # Eq. 12 
    #                         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

    #                         # Eq. 19
    #                         if sigma_t >= at_next*sigma_y:
    #                             lambda_t = 1.
    #                             gamma_t = (sigma_t**2 - (at_next*sigma_y)**2).sqrt()
    #                         else:
    #                             lambda_t = (sigma_t)/(at_next*sigma_y)
    #                             gamma_t = 0.

    #                         # Eq. 17
    #                         x0_t_hat = x0_t - lambda_t*Ap(A(x0_t) - y)

    #                         eta = self.args.eta

    #                         c1 = (1 - at_next).sqrt() * eta
    #                         c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

    #                         # different from the paper, we use DDIM here instead of DDPM
    #                         xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

    #                         x0_preds.append(x0_t.to('cpu'))
    #                         xs.append(xt_next.to('cpu'))  
    #                         x0_t = (xt - et * (1 - at).sqrt())  
    #                 else: # time-travel back
    #                     next_t = (torch.ones(n) * j).to(x.device)
    #                     at_next = compute_alpha(self.betas, next_t.long())
    #                     x0_t = x0_preds[-1].to('cuda')

    #                     xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

    #                     xs.append(xt_next.to('cpu'))

    #             x = xs[-1]
                
    #         x = [inverse_data_transform(config, xi) for xi in x]

    #         tvu.save_image(
    #             x[0], os.path.join(self.args.image_folder, f"output_{filename}.png")
    #         )
    #         orig = inverse_data_transform(config, HQ[0])
    #         mse = torch.mean((x[0].to(self.device) - orig) ** 2)
    #         psnr = 10 * torch.log10(1 / mse)
    #         avg_output_psnr += psnr
    #         self.txt_logger.info(f'{filename} output psnr: {psnr}')
    #         # print(f'{filename} output psnr:', psnr)

    #     avg_output_psnr = avg_output_psnr / len(filename_list)
    #     avg_kernel_psnr = avg_kernel_psnr / len(filename_list)
    #     self.txt_logger.info(f'Total Average Output PSNR: {avg_output_psnr}')
    #     self.txt_logger.info(f'Total Average Kernel PSNR: {avg_kernel_psnr}')
    #     self.txt_logger.info(f"Number of samples: {len(filename_list)}")
    #     # print("Total Average Output PSNR: %.2f" % avg_output_psnr)
    #     # print(""Total Average Output PSNR: %.2f" % avg_kernel_psnr)
                 
    # def IRmodel_simplified_ddnm_plus(self, model, cls_fn):
    #     args, config = self.args, self.config
        
    #     def seed_worker(worker_id):
    #         worker_seed = args.seed % 2 ** 32
    #         np.random.seed(worker_seed)
    #         random.seed(worker_seed)

    #     g = torch.Generator()
    #     g.manual_seed(args.seed)

    #     avg_output_psnr = 0.0
    #     avg_apy_psnr = 0.0
        
    #     test_dataset_root = self.args.path_y
    #     all_files = os.listdir(os.path.join(test_dataset_root, 'hq'))
    #     filename_list = sorted([file[:-4] for file in all_files])
    #     print(f'{len(filename_list)=}')
    #     implicit_Ap = lambda z : self.IRmodel.net_g(z, z, lq=implicit_LQ)[0]
        
    #     Ap = lambda z: F.interpolate(z, size=(self.config.data.image_size, self.config.data.image_size), mode='bilinear', align_corners=False)
    #     A = lambda z: F.interpolate(z, size=(self.config.data.image_size // self.args.N, self.config.data.image_size // self.args.N), mode='bilinear', align_corners=False)
    #     ApA = lambda z: Ap(A(z))     
        
        
          
    #     for filename in tqdm.tqdm(filename_list):
    #         lq_path = os.path.join(test_dataset_root, 'lq', '4', f'{filename}.jpg')
    #         hq_path = os.path.join(test_dataset_root, 'hq', f'{filename}.jpg')
    #         x_orig = load_imgDDNM(hq_path).to(self.device)
            
    #         implicit_LQ = load_img_LearningDegradation(lq_path).to('cuda')
    #         implicit_Apy = implicit_Ap(implicit_LQ)
    #         implicit_Apy_img = tensor2img(implicit_Apy, rgb2bgr=True)
    #         os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
    #         imwrite(implicit_Apy_img, os.path.join(self.args.image_folder, f"Apy/Apy_implicit_{filename}.png"))
    #         Apy = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/Apy_implicit_{filename}.png")).to(self.device)
    #         # cal apy psnr 
    #         implicit_apy_psnr = cal_img_psnr(x_orig, Apy)
    #         print(f'{filename} per sample apy psnr implicit:', implicit_apy_psnr)
    #         avg_apy_psnr += implicit_apy_psnr

    #         Apy = data_transform(self.config, Apy)
    #         x_orig = data_transform(self.config, x_orig)
    #         tvu.save_image(
    #             inverse_data_transform(config, x_orig[0]),
    #             os.path.join(self.args.image_folder, f"Apy/orig_{filename}.png")
    #         )
        
    #         if config.sampling.batch_size!=1:
    #             raise ValueError("please change the config file to set batch size as 1")

            
    #         # init x_T
    #         x = torch.randn(
    #             x_orig.shape[0],
    #             config.data.channels,
    #             config.data.image_size,
    #             config.data.image_size,
    #             device=self.device,
    #         )

    #         with torch.no_grad():
    #             skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
    #             n = x.size(0)
    #             x0_preds = []
    #             xs = [x]
                
    #             times = get_schedule_jump(config.time_travel.T_sampling, 
    #                                            config.time_travel.travel_length, 
    #                                            config.time_travel.travel_repeat,
    #                                           )
    #             time_pairs = list(zip(times[:-1], times[1:]))
                
                
    #             # reverse diffusion sampling
    #             for i, j in time_pairs:
    #                 i, j = i*skip, j*skip
    #                 if j<0: j=-1 

    #                 if j < i: # normal sampling 
    #                     #self inplement DDNM
    #                     t = (torch.ones(n) * i).to(x.device)
    #                     next_t = (torch.ones(n) * j).to(x.device)
    #                     at = compute_alpha(self.betas, t.long())
    #                     at_next = compute_alpha(self.betas, next_t.long())
    #                     xt = xs[-1].to('cuda')

    #                     et = model(xt, t)
    #                     # Eq. 12
    #                     x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() 

    #                     # Eq. 13
    #                     x0_t_hat = ApA(Apy) + x0_t - ApA(x0_t)

    #                     eta = self.args.eta

    #                     c1 = (1 - at_next).sqrt() * eta
    #                     c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

    #                     # different from the paper, we use DDIM here instead of DDPM
    #                     xt_next = at_next.sqrt() * x0_t_hat + (c1 * torch.randn_like(x0_t) + c2 * et)

    #                     x0_preds.append(x0_t.to('cpu'))
    #                     xs.append(xt_next.to('cpu'))   
    #                 else: # time-travel back
    #                     next_t = (torch.ones(n) * j).to(x.device)
    #                     at_next = compute_alpha(self.betas, next_t.long())
    #                     x0_t = x0_preds[-1].to('cuda')

    #                     xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

    #                     xs.append(xt_next.to('cpu'))

    #             x = xs[-1]
                
    #         x = [inverse_data_transform(config, xi) for xi in x]

    #         tvu.save_image(
    #             x[0], os.path.join(self.args.image_folder, f"output_{filename}.png")
    #         )
            
    #         orig = inverse_data_transform(config, x_orig[0])
    #         mse = torch.mean((x[0].to(self.device) - orig) ** 2)
    #         psnr = 10 * torch.log10(1 / mse)
    #         print(f'{filename} per sample psnr:', psnr)
    #         avg_output_psnr += psnr

    #     avg_output_psnr = avg_output_psnr / len(filename_list)
    #     avg_apy_psnr = avg_apy_psnr / len(filename_list)
    #     print("Apy Total Average PSNR: %.2f" % avg_apy_psnr)
    #     print("Output Total Average PSNR: %.2f" % avg_output_psnr)
    #     # print("Total Average kernel psnr: %.2f" % avg_kernel_psnr / (idx_so_far - idx_init))
    #     print("Number of samples: %d" % (len(filename_list)))     

    # def explicit_simplified_ddnm_plus(self, model, cls_fn):
    #     args, config = self.args, self.config
        
    #     def seed_worker(worker_id):
    #         worker_seed = args.seed % 2 ** 32
    #         np.random.seed(worker_seed)
    #         random.seed(worker_seed)

    #     g = torch.Generator()
    #     g.manual_seed(args.seed)
    
    #     avg_output_psnr = 0.0

    #     test_dataset_root = self.args.path_y
    #     all_files = os.listdir(os.path.join(test_dataset_root, 'hq'))
    #     if self.config.data.dataset == 'ImageNet':
    #         filename_list = sorted([file[:-5] for file in all_files])
    #     else:
    #         filename_list = sorted([file[:-4] for file in all_files])
        
    #     avg_kernel_psnr = 0.0
    #     avg_output_psnr = 0.0


    #     for filename in tqdm.tqdm(filename_list):
    #         if os.path.exists(os.path.join(self.args.image_folder, f"{filename}.png")):
    #             continue
    #         if self.config.data.dataset == 'ImageNet':
    #             hq_path = os.path.join(test_dataset_root, 'hq', f'{filename}.JPEG')
    #         else:
    #             hq_path = os.path.join(test_dataset_root, 'hq', f'{filename}.jpg')
    #         # lq_path = os.path.join(test_dataset_root, 'lq', f'{int(self.args.deg_scale)}', f'{filename}.jpg')
    #         # hq_path = os.path.join(test_dataset_root, 'hq', f'{filename}.JPEG')
    #         gt_kernel_path = os.path.join(test_dataset_root, 'kernel', f'{filename}.npy')
            
    #         HQ = load_imgDDNM(hq_path).to(self.device)
    #         # LQ = load_imgExplicit(lq_path).to(self.device)
    #         HQ = data_transform(self.config, HQ)
    #         # LQ = data_transform(self.config, LQ)

    #         gt_kernel = np.load(gt_kernel_path)
    #         gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
            

    #         gt_A = lambda z : convolution2d(z, gt_kernel, stride=int(self.args.deg_scale), padding=gt_kernel.size(1)//2) 
    #         LQ = gt_A(HQ)
    #         gt_kernel = gt_kernel.unsqueeze(0)
    #         padding = (self.args.kernel_size - gt_kernel.size(1)) //2
    #         # Use F.pad to perform zero-padding
    #         gt_kernel = F.pad(gt_kernel, (padding, padding, padding, padding)).squeeze()

            
    #         predict_kernel = None
            
    #         with torch.no_grad():
    #             predict_kernel = self.kernel_estimator(LQ).squeeze()
    #             kernel_psnr = calculate_psnr(predict_kernel, gt_kernel)
    #             avg_kernel_psnr += kernel_psnr
    #             self.txt_logger.info(f'{filename} kernel psnr: {kernel_psnr}')
    #             # print(f'{filename} kernel psnr:', kernel_psnr)
    #         plot_kernel(gt_kernel.unsqueeze(0).unsqueeze(0), predict_kernel.unsqueeze(0).unsqueeze(0), filename, self.args.image_folder)
    #         padding = predict_kernel.shape[0] // 2

    #         matrix_A = convolution_to_A(predict_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=padding).cuda()

    #         matrix_A_pinverse = torch.pinverse(matrix_A) 
    #         A = lambda z: convolution_with_A(matrix_A, z, padding=padding)
    #         Ap = lambda z: convolution_with_A(matrix_A_pinverse, z, padding=0)[:, :, padding:-padding, padding:-padding]
    #         ApA = lambda z: Ap(A(z))
            
    #         Apy = Ap(LQ)
            
    #         os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
    #         for i in range(len(Apy)):
    #             tvu.save_image(
    #                 inverse_data_transform(config, Apy[i]),
    #                 os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")
    #             )
    #             tvu.save_image(
    #                 inverse_data_transform(config, HQ[i]),
    #                 os.path.join(self.args.image_folder, f"Apy/orig_{filename}.png")
    #             )
    #             tvu.save_image(
    #                 inverse_data_transform(config, LQ[i]),
    #                 os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")
    #             )
                
    #         # init x_T
    #         x = torch.randn(
    #             HQ.shape[0],
    #             config.data.channels,
    #             config.data.image_size,
    #             config.data.image_size,
    #             device=self.device,
    #         )

    #         with torch.no_grad():
    #             skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
    #             n = x.size(0)
    #             x0_preds = []
    #             xs = [x]
                
    #             times = get_schedule_jump(config.time_travel.T_sampling, 
    #                                            config.time_travel.travel_length, 
    #                                            config.time_travel.travel_repeat,
    #                                           )
    #             time_pairs = list(zip(times[:-1], times[1:]))
                
                
    #             # reverse diffusion sampling
    #             for i, j in time_pairs:
    #                 i, j = i*skip, j*skip
    #                 if j<0: j=-1 

    #                 if j < i: # normal sampling 
    #                     if self.config.user.ddnm:
    #                         #self inplement DDNM
    #                         t = (torch.ones(n) * i).to(x.device)
    #                         next_t = (torch.ones(n) * j).to(x.device)
    #                         at = compute_alpha(self.betas, t.long())
    #                         at_next = compute_alpha(self.betas, next_t.long())
    #                         xt = xs[-1].to('cuda')

    #                         et = model(xt, t)
    #                         if et.size(1) == 6:
    #                             et = et[:, :3]
    #                         # Eq. 12
    #                         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() 
                            
    #                         # save x0_t
    #                         if self.args.save_img:
    #                             os.makedirs(os.path.join(self.args.image_folder, "x0_t", f'{filename}'), exist_ok=True)
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, x0_t),
    #                                 os.path.join(self.args.image_folder, f"x0_t/{filename}/x0_{t[0]}.png")
    #                             )

    #                         # Eq. 13
    #                         # x0_t_hat = ApA(Apy) + x0_t - ApA(x0_t)
    #                         ApAx0_t = ApA(x0_t)
    #                         if self.args.save_img:
    #                             os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, ApAx0_t[0]),
    #                                 os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")
    #                             )
                            
    #                         alpha = 1.0
    #                         x0_t_hat = alpha * Apy + x0_t - alpha * ApAx0_t # add scheduler
                            
    #                         # x0_t_hat = Apy + x0_t - ApAx0_t                                 
    #                         # x0_t_hat = Apy + x0_t - ApA(x0_t)
    #                         if self.args.save_img:
    #                             os.makedirs(os.path.join(self.args.image_folder, "x0_t_hat", f'{filename}'), exist_ok=True)
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, x0_t_hat),
    #                                 os.path.join(self.args.image_folder, "x0_t_hat", f"{filename}/x0_{int(t[0])}_hat.png")
    #                             )
                            
    #                         eta = self.args.eta
    #                         sigma = (
    #                             eta
    #                             * torch.sqrt((1 - at_next) / (1 - at))
    #                             * torch.sqrt(1 - at / at_next)
    #                         )
    #                         mean_pred = (
    #                             x0_t_hat * torch.sqrt(at_next)
    #                             + torch.sqrt(1 - at_next - sigma ** 2) * et
    #                         )
    #                         xt_next = mean_pred + sigma * torch.randn_like(x0_t)
    #                         # c1 = (1 - at_next).sqrt() * eta
    #                         # c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

    #                         # # different from the paper, we use DDIM here instead of DDPM
    #                         # xt_next = at_next.sqrt() * x0_t_hat + (c1 * torch.randn_like(x0_t) + c2 * et)

    #                         x0_preds.append(x0_t.to('cpu'))
    #                         xs.append(xt_next.to('cpu'))   
    #                     else:

    #                         t = (torch.ones(n) * i).to(x.device)
    #                         next_t = (torch.ones(n) * j).to(x.device)
    #                         at = compute_alpha(self.betas, t.long())
    #                         at_next = compute_alpha(self.betas, next_t.long())
    #                         sigma_t = (1 - at_next**2).sqrt()
    #                         xt = xs[-1].to('cuda')

    #                         et = model(xt, t)

    #                         if et.size(1) == 6:
    #                             et = et[:, :3]

    #                         # Eq. 12 
    #                         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

    #                         # Eq. 19
    #                         if sigma_t >= at_next*sigma_y:
    #                             lambda_t = 1.
    #                             gamma_t = (sigma_t**2 - (at_next*sigma_y)**2).sqrt()
    #                         else:
    #                             lambda_t = (sigma_t)/(at_next*sigma_y)
    #                             gamma_t = 0.

    #                         # Eq. 17
    #                         x0_t_hat = x0_t - lambda_t*Ap(A(x0_t) - y)

    #                         eta = self.args.eta
    #                         sigma = (
    #                             eta
    #                             * torch.sqrt((1 - at_next) / (1 - at))
    #                             * torch.sqrt(1 - at / at_next)
    #                         )
    #                         mean_pred = (
    #                             x0_t_hat * torch.sqrt(at_next)
    #                             + torch.sqrt(1 - at_next - sigma ** 2) * et
    #                         )
    #                         xt_next = mean_pred + sigma * torch.randn_like(x0_t)
    #                         # c1 = (1 - at_next).sqrt() * eta
    #                         # c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

    #                         # # different from the paper, we use DDIM here instead of DDPM
    #                         # xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

    #                         x0_preds.append(x0_t.to('cpu'))
    #                         xs.append(xt_next.to('cpu'))  
    #                         x0_t = (xt - et * (1 - at).sqrt())  
    #                 else: # time-travel back
    #                     next_t = (torch.ones(n) * j).to(x.device)
    #                     at_next = compute_alpha(self.betas, next_t.long())
    #                     x0_t = x0_preds[-1].to('cuda')

    #                     xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

    #                     xs.append(xt_next.to('cpu'))

    #             x = xs[-1]
                
    #         x = [inverse_data_transform(config, xi) for xi in x]

    #         tvu.save_image(
    #             x[0], os.path.join(self.args.image_folder, f"output_{filename}.png")
    #         )
    #         orig = inverse_data_transform(config, HQ[0])
    #         mse = torch.mean((x[0].to(self.device) - orig) ** 2)
    #         psnr = 10 * torch.log10(1 / mse)
    #         avg_output_psnr += psnr
    #         self.txt_logger.info(f'{filename} output psnr: {psnr}')
    #         # print(f'{filename} output psnr:', psnr)

    #     avg_output_psnr = avg_output_psnr / len(filename_list)
    #     avg_kernel_psnr = avg_kernel_psnr / len(filename_list)
    #     self.txt_logger.info(f'Total Average Output PSNR: {avg_output_psnr}')
    #     self.txt_logger.info(f'Total Average kernel PSNR: {avg_kernel_psnr}')
    #     self.txt_logger.info(f'Total sample: {len(filename_list)}')
    #     # print("Total Average Output PSNR: %.2f" % avg_output_psnr)
    #     # print("Total Average kernel PSNR: %.2f" % avg_kernel_psnr)
            
             
            
    # def implicit_simplified_ddnm_plus(self, model, cls_fn):
    #     args, config = self.args, self.config
        
    #     def seed_worker(worker_id):
    #         worker_seed = args.seed % 2 ** 32
    #         np.random.seed(worker_seed)
    #         random.seed(worker_seed)

    #     g = torch.Generator()
    #     g.manual_seed(args.seed)

    #     avg_output_psnr = 0.0
    #     avg_apy_psnr = 0.0
        
    #     test_dataset_root = self.args.path_y
    #     all_files = os.listdir(os.path.join(test_dataset_root, 'hq'))
    #     if self.config.data.dataset == 'ImageNet':
    #         filename_list = sorted([file[:-5] for file in all_files])
    #     else:
    #         filename_list = sorted([file[:-4] for file in all_files])
    #     print(f'{len(filename_list)=}')
    #     A = lambda z : self.IRmodel.net_g(z, z, lq=gt_lq)[1]        
    #     Ap = lambda z : self.IRmodel.net_g(z, z, lq=gt_lq)[0]
    #     ApA = lambda z: Ap(A(z))     
        
    #     # lineaer_schedule = np.linspace(1, 0, 1000).tolist() # v1 scheduler
    #     # lineaer_schedule.reverse()
        
    #     # first_half = [1] * 500
    #     # second_half = np.linspace(1, 0, 500).tolist()
    #     # lineaer_schedule = first_half + second_half # v2 scheduler
    #     # lineaer_schedule.reverse()

    #     # t = np.linspace(1, 0, 1000).tolist() 
    #     # first_half = [1] * 500
    #     # lineaer_schedule = first_half + t[500:] # v3 scheduler
    #     # lineaer_schedule.reverse()   
        

    #     # first_half = [1.0] * 700
    #     # lineaer_schedule = first_half + [0.3] * 300 # v4 scheduler
    #     # lineaer_schedule.reverse()   
        
          
    #     for filename in tqdm.tqdm(filename_list):
    #         if os.path.exists(os.path.join(self.args.image_folder, f"output_{filename}.png")):
    #             continue
    #         lq_path = os.path.join(test_dataset_root, 'lq', f'{int(args.deg_scale)}', f'{filename}.jpg')
    #         if self.config.data.dataset == 'ImageNet':
    #             hq_path = os.path.join(test_dataset_root, 'hq', f'{filename}.JPEG')
    #         else:
    #             hq_path = os.path.join(test_dataset_root, 'hq', f'{filename}.jpg')
    #         # implicit_lq_path = os.path.join(test_dataset_root, 'implicit_lq', f'{int(args.deg_scale)}', f'{filename}_reblur.png')
            
    #         # implicit_lq_img = load_img_LearningDegradation(implicit_lq_path).to('cuda')
    #         gt_lq = load_img_LearningDegradation(lq_path).to('cuda')
            
    #         if self.args.perturb_y:
    #             # perturb y into Ax distribution, y = A(Ap(y))
    #             Apy = Ap(A(Ap(gt_lq)))
    #         else:          
    #             # baseline
    #             Apy = Ap(gt_lq)
            
    #         # use implicit_A(HQ) as LQ
    #         # Apy = Ap(implicit_lq_img)
    #         # Apy = Ap(A(Ap(implicit_lq_img)))
            
    #         Apy_img = tensor2img(Apy, rgb2bgr=True)
    #         os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
    #         imwrite(Apy_img, os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png"))
        
    #         Apy = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")).to(self.device)
    #         x_orig = load_imgDDNM(hq_path).to(self.device)

    #         _mse = torch.mean((Apy[0].to(self.device) - x_orig[0]) ** 2)
    #         _psnr = 10 * torch.log10(1 / _mse)
    #         avg_apy_psnr += _psnr
    #         self.txt_logger.info(f'{filename} per sample apy psnr: {_psnr}')
    #         # print(f'{filename} per sample apy psnr:', _psnr)

    #         os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
    #         # Apy = data_transform(self.config, Apy)
    #         x_orig = data_transform(self.config, x_orig)
    #         tvu.save_image(
    #             inverse_data_transform(config, x_orig[0]),
    #             os.path.join(self.args.image_folder, f"Apy/orig_{filename}.png")
    #         )

        
    #         if config.sampling.batch_size!=1:
    #             raise ValueError("please change the config file to set batch size as 1")

            
                
    #         # init x_T
    #         x = torch.randn(
    #             x_orig.shape[0],
    #             config.data.channels,
    #             config.data.image_size,
    #             config.data.image_size,
    #             device=self.device,
    #         )
    #         # with torch.no_grad():
    #         skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
    #         n = x.size(0)
    #         x0_preds = []
    #         xs = [x]
            
    #         times = get_schedule_jump(config.time_travel.T_sampling, 
    #                                         config.time_travel.travel_length, 
    #                                         config.time_travel.travel_repeat,
    #                                         )
    #         time_pairs = list(zip(times[:-1], times[1:]))
                
                
    #         # reverse diffusion sampling
    #         for i, j in time_pairs:
    #             i, j = i*skip, j*skip
    #             if j<0: j=-1 

    #             if j < i: # normal sampling 
    #                 #self inplement DDNM
    #                 t = (torch.ones(n) * i).to(x.device)
    #                 next_t = (torch.ones(n) * j).to(x.device)
    #                 at = compute_alpha(self.betas, t.long())
    #                 at_next = compute_alpha(self.betas, next_t.long())
    #                 xt = xs[-1].to('cuda')
    #                 if self.args.save_img:
    #                     os.makedirs(os.path.join(self.args.image_folder, "x_t", f'{filename}'), exist_ok=True)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, xt),
    #                         os.path.join(self.args.image_folder, "x_t", f"{filename}/x_{int(t[0])}.png")
    #                     )     
                    
                    
    #                 with torch.no_grad():
    #                     et = model(xt, t)
                        
    #                 if et.size(1) == 6:
    #                     et = et[:, :3]
    #                     # Eq. 12
    #                 x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() 
                    
    #                 # save x0_t
    #                 if self.args.save_img:
    #                     os.makedirs(os.path.join(self.args.image_folder, "x0_t", f'{filename}'), exist_ok=True)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, x0_t),
    #                         os.path.join(self.args.image_folder, "x0_t", f"{filename}/x0_{int(t[0])}.png")
    #                     )
    #                 else:
    #                     os.makedirs(os.path.join(self.args.image_folder, "tmp"), exist_ok=True)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, x0_t),
    #                         os.path.join(self.args.image_folder, "tmp", "x0_t.png")
    #                     )        

    #                 # Eq. 13
    #                 # x0_t_hat = ApA(Apy) + x0_t - ApA(x0_t)
    #                 if self.args.save_img:
    #                     x0_tL = load_img_LearningDegradation(os.path.join(self.args.image_folder, "x0_t", f"{filename}/x0_{int(t[0])}.png")).to('cuda')
    #                 else:
    #                     x0_tL = load_img_LearningDegradation(os.path.join(self.args.image_folder, "tmp", "x0_t.png")).to('cuda')
    #                 ApAx0_t = ApA(x0_tL)
    #                 ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)

    #                 if self.args.save_img:
    #                     Apx0_t = Ap(x0_tL)
    #                     Apx0_t_img = tensor2img(Apx0_t, rgb2bgr=True)
    #                     os.makedirs(os.path.join(self.args.image_folder, "Ap", f'{filename}'), exist_ok=True)
    #                     imwrite(Apx0_t_img, os.path.join(self.args.image_folder, "Ap", f"{filename}/Apx0_{int(t[0])}.png"))

                    
    #                 if self.args.save_img:
    #                     os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                     imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, "ApA", f"{filename}/ApAx0_{int(t[0])}.png"))
    #                     ApAx0_t_img = load_imgDDNM(os.path.join(self.args.image_folder, "ApA", f"{filename}/ApAx0_{int(t[0])}.png")).to('cuda')
    #                 else:
    #                     os.makedirs(os.path.join(self.args.image_folder, "tmp"), exist_ok=True)
    #                     imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, "tmp", "ApAx0_t.png"))
    #                     ApAx0_t_img = load_imgDDNM(os.path.join(self.args.image_folder, "tmp", "ApAx0_t.png")).to('cuda')
    #                 # alpha = lineaer_schedule[i]
    #                 alpha = self.args.alpha
                    
    #                 # print('time step:', i)
    #                 # if i in range(200, 800):
    #                 #     if self.args.sds_optimize_alpha:
    #                 #         self.txt_logger.info(f"timestep: {i}")
    #                 #         alpha, tmp = self.sds_train(model=model, Apy=Apy, x0_t=x0_t, ApAx0_t_img=ApAx0_t_img, iter=10)
    #                 #         self.txt_logger.info(f"{tmp}")
    #                 # print(alpha)
    #                 # alpha = 0.3
    #                 x0_t_hat = alpha * Apy + x0_t - alpha * ApAx0_t_img # add linear scheduler
    #                 # x0_t_hat = Apy + x0_t - ApAx0_t_img                                
    #                 # x0_t_hat = Apy + x0_t - ApA(x0_t)
    #                 if self.args.save_img:
    #                     os.makedirs(os.path.join(self.args.image_folder, "x0_t_hat", f'{filename}'), exist_ok=True)
    #                     tvu.save_image(
    #                         inverse_data_transform(config, x0_t_hat),
    #                         os.path.join(self.args.image_folder, "x0_t_hat", f"{filename}/x0_{int(t[0])}_hat.png")
    #                     ) 
                    
    #                 eta = self.args.eta
    #                 # sigma = (
    #                 #     eta
    #                 #     * torch.sqrt((1 - at_next) / (1 - at))
    #                 #     * torch.sqrt(1 - at / at_next)
    #                 # )
    #                 # mean_pred = (
    #                 #     x0_t_hat * torch.sqrt(at_next)
    #                 #     + torch.sqrt(1 - at_next - sigma ** 2) * et
    #                 # )
    #                 # xt_next = mean_pred + sigma * torch.randn_like(x0_t)
                    
    #                 c1 = (1 - at_next).sqrt() * eta
    #                 c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
    #                 # sigma_t = (1 - at_next**2).sqrt()
    #                 # different from the paper, we use DDIM here instead of DDPM
    #                 xt_next = at_next.sqrt() * x0_t_hat + 1 * (c1 * torch.randn_like(x0_t) + c2 * et)

    #                 x0_preds.append(x0_t.to('cpu'))
    #                 xs.append(xt_next.to('cpu'))   
    #             else: # time-travel back
    #                 next_t = (torch.ones(n) * j).to(x.device)
    #                 at_next = compute_alpha(self.betas, next_t.long())
    #                 x0_t = x0_preds[-1].to('cuda')

    #                 xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

    #                 xs.append(xt_next.to('cpu'))

    #         x = xs[-1]
                
    #         x = [inverse_data_transform(config, xi) for xi in x]

    #         tvu.save_image(
    #             x[0], os.path.join(self.args.image_folder, f"output_{filename}.png")
    #         )
            
    #         orig = inverse_data_transform(config, x_orig[0])
    #         mse = torch.mean((x[0].to(self.device) - orig) ** 2)
    #         psnr = 10 * torch.log10(1 / mse)
    #         self.txt_logger.info(f'{filename} per sample psnr: {psnr}')
    #         # print(f'{filename} per sample psnr:', psnr)
    #         avg_output_psnr += psnr

    #     avg_output_psnr = avg_output_psnr / len(filename_list)
    #     avg_apy_psnr = avg_apy_psnr / len(filename_list)
    #     self.txt_logger.info(f"Apy Total Average PSNR: {avg_apy_psnr}")
    #     self.txt_logger.info(f"Output Total Average PSNR: {avg_output_psnr}")
    #     self.txt_logger.info(f"Number of samples: {len(filename_list)}")
    #     # print("Apy Total Average PSNR: %.2f" % avg_apy_psnr)
    #     # print("Output Total Average PSNR: %.2f" % avg_output_psnr)
    #     # print("Total Average kernel psnr: %.2f" % avg_kernel_psnr / (idx_so_far - idx_init))
    #     # print("Number of samples: %d" % (len(filename_list)))     
           
    # def explicit_gt_simplified_ddnm_plus(self, model, cls_fn):
    #     args, config = self.args, self.config
        
    #     def seed_worker(worker_id):
    #         worker_seed = args.seed % 2 ** 32
    #         np.random.seed(worker_seed)
    #         random.seed(worker_seed)

    #     g = torch.Generator()
    #     g.manual_seed(args.seed)


    #     avg_output_psnr = 0.0

    #     test_dataset_root = self.args.path_y
    #     all_files = os.listdir(os.path.join(test_dataset_root, 'hq'))
    #     if self.config.data.dataset == 'ImageNet':
    #         filename_list = sorted([file[:-5] for file in all_files])
    #     else:
    #         filename_list = sorted([file[:-4] for file in all_files])
    
    #     avg_kernel_psnr = 0.0
    #     avg_output_psnr = 0.0
        
        
    #     for filename in tqdm.tqdm(filename_list):
    #         if os.path.exists(os.path.join(self.args.image_folder, f"{filename}.png")):
    #             continue
    #         lq_path = os.path.join(test_dataset_root, 'lq', f'{int(self.args.deg_scale)}', f'{filename}.jpg')
    #         if self.config.data.dataset == 'ImageNet':
    #             hq_path = os.path.join(test_dataset_root, 'hq', f'{filename}.JPEG')
    #         elif self.config.data.dataset == 'benchmark':
    #             hq_path = os.path.join(test_dataset_root, 'hq', f'{filename}.png')
    #         else:
    #             hq_path = os.path.join(test_dataset_root, 'hq', f'{filename}.jpg')
    #         gt_kernel_path = os.path.join(test_dataset_root, 'kernel', f'{filename}.npy')
            
    #         HQ = load_imgDDNM(hq_path).to(self.device)
    #         # LQ = load_imgExplicit(lq_path).to(self.device)
    #         HQ = data_transform(self.config, HQ)
    #         # LQ = data_transform(self.config, LQ)

    #         # gt_kernel = np.load(gt_kernel_path)
    #         # gt_kernel = torch.from_numpy(gt_kernel).float().cuda()
    #         # gt_A = lambda z : convolution2d(z, gt_kernel, stride=int(self.args.deg_scale), padding=gt_kernel.size(1)//2) 
    #         # LQ = gt_A(HQ)
            
    #         # padding = gt_kernel.shape[0] // 2
    #         # matrix_A = convolution_to_A(gt_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=padding).cuda()
    #         # matrix_A_pinverse = torch.pinverse(matrix_A)
    #         # A = lambda z: convolution_with_A(matrix_A, z, padding=padding)
    #         # Ap = lambda z: convolution_with_A(matrix_A_pinverse, z, padding=0)[:, :, padding:-padding, padding:-padding]
    #         # ApA = lambda z: Ap(A(z))
            
    #         A = torch.nn.AdaptiveAvgPool2d((256//int(self.args.deg_scale),256//int(self.args.deg_scale)))
    #         Ap = lambda z: PatchUpsample(z, int(self.args.deg_scale))
    #         ApA = lambda z: Ap(A(z))
    #         LQ = A(HQ)
            
    #         Apy = Ap(LQ)
    #         os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
    #         for i in range(len(Apy)):
    #             tvu.save_image(
    #                 inverse_data_transform(config, Apy[i]),
    #                 os.path.join(self.args.image_folder, f"Apy/Apy_{filename}.png")
    #             )
    #             tvu.save_image(
    #                 inverse_data_transform(config, HQ[i]),
    #                 os.path.join(self.args.image_folder, f"Apy/orig_{filename}.png")
    #             )
    #             tvu.save_image(
    #                 inverse_data_transform(config, LQ[i]),
    #                 os.path.join(self.args.image_folder, f"Apy/y_{filename}.png")
    #             )
                
    #         # init x_T
    #         x = torch.randn(
    #             HQ.shape[0],
    #             config.data.channels,
    #             config.data.image_size,
    #             config.data.image_size,
    #             device=self.device,
    #         )

    #         with torch.no_grad():
    #             skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
    #             n = x.size(0)
    #             x0_preds = []
    #             xs = [x]
                
    #             times = get_schedule_jump(config.time_travel.T_sampling, 
    #                                            config.time_travel.travel_length, 
    #                                            config.time_travel.travel_repeat,
    #                                           )
    #             time_pairs = list(zip(times[:-1], times[1:]))
                
                
    #             # reverse diffusion sampling
    #             for i, j in time_pairs:
    #                 i, j = i*skip, j*skip
    #                 if j<0: j=-1 

    #                 if j < i: # normal sampling 
    #                     if self.config.user.ddnm:
    #                         #self inplement DDNM
    #                         t = (torch.ones(n) * i).to(x.device)
    #                         next_t = (torch.ones(n) * j).to(x.device)
    #                         at = compute_alpha(self.betas, t.long())
    #                         at_next = compute_alpha(self.betas, next_t.long())
    #                         sigma_t = (1 - at_next**2).sqrt()
    #                         xt = xs[-1].to('cuda')

    #                         et = model(xt, t)
    #                         if et.size(1) == 6:
    #                             et = et[:, :3]
    #                         # Eq. 12
    #                         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() 
                            
    #                         # save x0_t
    #                         if self.args.save_img:
    #                             os.makedirs(os.path.join(self.args.image_folder, "x0_t", f'{filename}'), exist_ok=True)
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, x0_t),
    #                                 os.path.join(self.args.image_folder, f"x0_t/{filename}/x0_{t[0]}.png")
    #                             )

    #                         # Eq. 13
    #                         # x0_t_hat = ApA(Apy) + x0_t - ApA(x0_t)
    #                         ApAx0_t = ApA(x0_t)
    #                         if self.args.save_img:
    #                             os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, ApAx0_t[0]),
    #                                 os.path.join(self.args.image_folder, f"ApA/{filename}/ApAx0_{t[0]}.png")
    #                             )
    #                         alpha = self.args.alpha
    #                         x0_t_hat = alpha * Apy + x0_t - alpha * ApAx0_t # add linear scheduler
    #                         # x0_t_hat = Apy + x0_t - ApAx0_t                                 
    #                         # x0_t_hat = Apy + x0_t - ApA(x0_t)
    #                         if self.args.save_img:
    #                             os.makedirs(os.path.join(self.args.image_folder, "ApA", f'{filename}'), exist_ok=True)
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, (x0_t - ApAx0_t)[0]),
    #                                 os.path.join(self.args.image_folder, f"ApA/{filename}/I-ApAx0_{t[0]}.png")
    #                             )
    #                         if self.args.save_img:
    #                             os.makedirs(os.path.join(self.args.image_folder, "x0_t_hat", f'{filename}'), exist_ok=True)
    #                             tvu.save_image(
    #                                 inverse_data_transform(config, x0_t_hat),
    #                                 os.path.join(self.args.image_folder, "x0_t_hat", f"{filename}/x0_{int(t[0])}_hat.png")
    #                             )
    #                         eta = self.args.eta
    #                         # sigma = (
    #                         #     eta
    #                         #     * torch.sqrt((1 - at_next) / (1 - at))
    #                         #     * torch.sqrt(1 - at / at_next)
    #                         # )
    #                         # mean_pred = (
    #                         #     x0_t_hat * torch.sqrt(at_next)
    #                         #     + torch.sqrt(1 - at_next - sigma ** 2) * et
    #                         # )
    #                         # xt_next = mean_pred + sigma * torch.randn_like(x0_t)
                            
    #                         c1 = (1 - at_next).sqrt() * eta
    #                         c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
    #                         # different from the paper, we use DDIM here instead of DDPM
    #                         xt_next = at_next.sqrt() * x0_t_hat + sigma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

    #                         x0_preds.append(x0_t.to('cpu'))
    #                         xs.append(xt_next.to('cpu'))   
    #                     else:

    #                         t = (torch.ones(n) * i).to(x.device)
    #                         next_t = (torch.ones(n) * j).to(x.device)
    #                         at = compute_alpha(self.betas, t.long())
    #                         at_next = compute_alpha(self.betas, next_t.long())
    #                         sigma_t = (1 - at_next**2).sqrt()
    #                         xt = xs[-1].to('cuda')

    #                         et = model(xt, t)

    #                         if et.size(1) == 6:
    #                             et = et[:, :3]

    #                         # Eq. 12 
    #                         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

    #                         # Eq. 19
    #                         if sigma_t >= at_next*sigma_y:
    #                             lambda_t = 1.
    #                             gamma_t = (sigma_t**2 - (at_next*sigma_y)**2).sqrt()
    #                         else:
    #                             lambda_t = (sigma_t)/(at_next*sigma_y)
    #                             gamma_t = 0.

    #                         # Eq. 17
    #                         x0_t_hat = x0_t - lambda_t*Ap(A(x0_t) - y)

    #                         eta = self.args.eta

    #                         c1 = (1 - at_next).sqrt() * eta
    #                         c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

    #                         # different from the paper, we use DDIM here instead of DDPM
    #                         xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

    #                         x0_preds.append(x0_t.to('cpu'))
    #                         xs.append(xt_next.to('cpu'))  
    #                         x0_t = (xt - et * (1 - at).sqrt())  
    #                 else: # time-travel back
    #                     next_t = (torch.ones(n) * j).to(x.device)
    #                     at_next = compute_alpha(self.betas, next_t.long())
    #                     x0_t = x0_preds[-1].to('cuda')

    #                     xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

    #                     xs.append(xt_next.to('cpu'))

    #             x = xs[-1]
                
    #         x = [inverse_data_transform(config, xi) for xi in x]

    #         tvu.save_image(
    #             x[0], os.path.join(self.args.image_folder, f"output_{filename}.png")
    #         )
    #         orig = inverse_data_transform(config, HQ[0])
    #         mse = torch.mean((x[0].to(self.device) - orig) ** 2)
    #         psnr = 10 * torch.log10(1 / mse)
    #         avg_output_psnr += psnr
    #         self.txt_logger.info(f'{filename} output psnr: {psnr}')
    #         # print(f'{filename} output psnr:', psnr)

    #     avg_output_psnr = avg_output_psnr / len(filename_list)
    #     avg_kernel_psnr = avg_kernel_psnr / len(filename_list)
    #     self.txt_logger.info(f"Total Average Output PSNR: {avg_output_psnr}")
    #     self.txt_logger.info(f"Total Average kernel PSNR: {avg_kernel_psnr}")
    #     self.txt_logger.info(f'sample total {len(filename_list)} image')
    #     # print("Total Average Output PSNR: %.2f" % avg_output_psnr)
    #     # print("Total Average kernel PSNR: %.2f" % avg_kernel_psnr)
 
    def simplified_ddnm_plus(self, model, cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)

        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # get degradation operator
        print("args.deg:",args.deg)
        if args.deg =='colorization':
            A = lambda z: color2gray(z)
            Ap = lambda z: gray2color(z)
        elif args.deg =='denoising':
            A = lambda z: z
            Ap = A
        elif args.deg =='sr_averagepooling':
            scale=round(args.deg_scale)
            A = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
            Ap = lambda z: MeanUpsample(z,scale)
            # A_real = F.interpolate() 
        elif args.deg =='inpainting':
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device)
            A = lambda z: z*mask
            Ap = A
        elif args.deg =='mask_color_sr':
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device)
            A1 = lambda z: z*mask
            A1p = A1
            
            A2 = lambda z: color2gray(z)
            A2p = lambda z: gray2color(z)
            
            scale=round(args.deg_scale)
            A3 = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
            A3p = lambda z: MeanUpsample(z,scale)
            
            A = lambda z: A3(A2(A1(z)))
            Ap = lambda z: A1p(A2p(A3p(z)))
        elif args.deg =='diy':
            # design your own degradation
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device)
            A1 = lambda z: z*mask
            A1p = A1
            
            A2 = lambda z: color2gray(z)
            A2p = lambda z: gray2color(z)
            
            scale=args.deg_scale
            A3 = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
            A3p = lambda z: MeanUpsample(z,scale)
            
            A = lambda z: A3(A2(A1(z)))
            Ap = lambda z: A1p(A2p(A3p(z)))
        else:
            raise NotImplementedError("degradation type not supported")

        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        sigma_y = args.sigma_y
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        # custom kernel degradation
        # custom_kernel = torch.tensor([[1/16, 1/16, 1/16, 1/16],
        #                               [1/16, 1/16, 1/16, 1/16],
        #                               [1/16, 1/16, 1/16, 1/16],
        #                               [1/16, 1/16, 1/16, 1/16]], dtype=torch.float32).cuda()
        # custom_kernel = torch.tensor([[2/60, 11/60, 2/60, 1/60],
        #                               [1/60, 3/60, 3/60, 1/60],
        #                               [1/60, 5/60, 12/60, 1/60],
        #                               [7/60, 7/60, 2/60, 1/60]], dtype=torch.float32).cuda()
        # custom_kernel = torch.tensor([[2/32, 3/32, 2/32, 2/32],
        #                               [2/32, 1/32, 3/32, 2/32],
        #                               [2/32, 3/32, 1/32, 2/32],
        #                               [2/32, 1/32, 2/32, 2/32]], dtype=torch.float32).cuda()
        # custom_kernel = torch.tensor([[2/40, 1/40, 2/40, 4/40],
        #                               [2/40, 3/40, 3/40, 5/40],
        #                               [3/40, 5/40, 2/40, 2/40],
        #                               [3/40, 7/40, 2/40, 3/40]], dtype=torch.float32).cuda()
        # custom_kernel = torch.tensor([[2/60, 11/60, 2/60, 1/60],
        #                                 [1/60, 3/60, 3/60, 1/60],
        #                                 [1/60, 5/60, 12/60, 1/60],
        #                                 [7/60, 7/60, 2/60, 1/60]], dtype=torch.float32).cuda()
        # custom_kernel = torch.tensor([[2/60, 11/60, 2/60, 1/60],
        #                                 [1/60, 3/60, 3/60, 5/60],
        #                                 [1/60, 5/60, 12/60, 1/60],
        #                                 [13/60, 7/60, 2/60, 3/60]], dtype=torch.float32).cuda()
        # custom_kernel = torch.tensor([[2/60, 3/60, 2/60, 1/60],
        #                               [1/60, 3/60, 3/60, 5/60],
        #                               [1/60, 5/60, 1/60, 1/60],
        #                               [3/60, 7/60, 2/60, 3/60]], dtype=torch.float32).cuda()
        # custom_kernel = torch.tensor([[2/60, 3/60, 2/60, 1/60],
        #                               [1/60, 1/60, 1/60, 2/60],
        #                               [1/60, 20/60, 1/60, 1/60],
        #                               [1/60, 20/60, 2/60, 1/60]], dtype=torch.float32).cuda()
        # custom_kernel = torch.tensor([[1, 0, 0, 0],
        #                               [0, 0, 0, 0],
        #                               [0, 0, 0, 0],
        #                               [0, 0, 0, 0]], dtype=torch.float32).cuda()
        # custom_kernel = torch.tensor([[2/40, 1/140, 2/40, 4/40, 1/60, 1/60, 1/60],
        #                               [2/40, 3/40, 3/40, 5/40, 1/60, 1/160, 20/60],
        #                               [2/40, 3/40, 3/40, 20/40, 1/60, 1/60, 1/60],
        #                               [2/40, 3/80, 3/40, 5/40, 1/160, 1/60, 1/60],
        #                               [2/40, 3/40, 3/40, 5/40, 1/60, 1/60, 1/60],
        #                               [10/40, 5/40, 2/140, 2/40, 1/60, 1/90, 1/70],
        #                               [3/40, 7/40, 2/40, 3/40, 1/60, 10/60, 1/60]], dtype=torch.float32).cuda()
        # custom_kernel = create_random_kernel(kernel_size=19).float().cuda()
        # if self.args.gt_kernel_path:
        #     custom_kernel = np.load(self.args.gt_kernel_path)
        #     custom_kernel = torch.from_numpy(custom_kernel).float().cuda()
        # stride = int(self.args.deg_scale)
        # padding = custom_kernel.shape[0] // 2
        # gt_A = lambda z : convolution2d(z, custom_kernel, stride=stride, padding=padding) 
        # # A = lambda z: convolution_with_A(matrix_A, z, padding=padding)
        # # Ap = lambda z: convolution_with_A(matrix_A_pinverse, z, padding=0)[:, :, padding:-padding, padding:-padding]
        # # Ap = lambda z: F.interpolate(z, scale_factor=4, mode='bicubic', align_corners=False)
        # # Ap = lambda z: MeanUpsample(z,4)
        # # Ap = lambda z: non_overlapConv_Ap(z, custom_kernel)
        # avg_kernel_psnr = 0.0
        
        # load gt_lq
        gt_lq = load_img_LearningDegradation(self.args.gt_lq).to('cuda')
        # file_client = FileClient('disk')
        # img_bytes = file_client.get(self.args.gt_lq, 'lq')
        # try:
        #     img_lq = imfrombytes(img_bytes, float32=True)
        # except:
        #     raise Exception("lq path {} not working".format(self.args.gt_lq))
        
        # gt_lq = img2tensor(img_lq,bgr2rgb=True,float32=True)
        # gt_lq = transforms.Compose([transforms.Resize((256,256), antialias=None),])(gt_lq).unsqueeze(0).to('cuda')

        A = lambda z : self.IRmodel.net_g(z, z, lq=gt_lq)[1]        
        Ap = lambda z : self.IRmodel.net_g(z, z, lq=gt_lq)[0]
        ApA = lambda z: Ap(A(z))      
        
        Apy = Ap(gt_lq)
        Apy_img = tensor2img(Apy, rgb2bgr=True)
        os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
        imwrite(Apy_img, os.path.join(self.args.image_folder, f"Apy/Apy.png"))
        
        Apy = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/Apy.png"))
    
        # A = lambda z: F.interpolate(z, size=(self.config.data.image_size // self.args.N, self.config.data.image_size // self.args.N), mode='bilinear', align_corners=False)
        # Ap = lambda z: F.interpolate(z, size=(self.config.data.image_size, self.config.data.image_size), mode='bilinear', align_corners=False)
        # ApA = lambda z: Ap(A(z))



        for idx, (x_orig, classes) in enumerate(pbar):
            x_orig = x_orig.to(self.device)

            _mse = torch.mean((Apy[0].to(self.device) - x_orig[0]) ** 2)
            _psnr = 10 * torch.log10(1 / _mse)
            print('psnr between IR image and HQ:', _psnr)


            x_orig = data_transform(self.config, x_orig)

            Apy = data_transform(self.config, Apy)
            # y = gt_A(x_orig)
            # # print(f'{y.shape=}')
            # predict_kernel = custom_kernel.clone()
            # if self.args.kernel_estimator:
            #     with torch.no_grad():
            #         predict_kernel = self.kernel_estimator(y).squeeze()
            #         kernel_psnr = calculate_psnr(predict_kernel, custom_kernel)
            #         avg_kernel_psnr += kernel_psnr
            #     print('per sample kernel psnr:', kernel_psnr)
            # plot_kernel(custom_kernel.unsqueeze(0).unsqueeze(0), predict_kernel.unsqueeze(0).unsqueeze(0), idx, self.args.image_folder)
            # padding = predict_kernel.shape[0] // 2
            # print('calculating matrix A...')
            # time1 = time.time()
            # matrix_A = convolution_to_A(predict_kernel, (1, 3, 256, 256), stride=int(self.args.deg_scale), padding=padding).cuda()
            # time2 = time.time()
            # matrix_A_pinverse = torch.pinverse(matrix_A) 
            # time3 = time.time()
            # print('get A:', time2 - time1)
            # print('get A pinverse:', time3 - time2)
            # print(f'{matrix_A.shape=}')
            # print(f'{matrix_A_pinverse.shape=}')
            # A = lambda z: convolution_with_A(matrix_A, z, padding=padding)
            # Ap = lambda z: convolution_with_A(matrix_A_pinverse, z, padding=0)[:, :, padding:-padding, padding:-padding]

            # y = F.interpolate(x_orig, size=(64, 64), mode='bicubic', align_corners=False) #bicubic degradation
            # y = convolution2d(x_orig, custom_kernel, stride=4)
            
            
            if config.sampling.batch_size!=1:
                raise ValueError("please change the config file to set batch size as 1")
            # Apy = IR_img
            # Apy = Ap(y)
            # print(f'{Apy.shape=}')

            os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
            for i in range(len(Apy)):
                # tvu.save_image(
                #     inverse_data_transform(config, Apy[i]),
                #     os.path.join(self.args.image_folder, f"Apy/Apy_{idx_so_far + i}.png")
                # )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]),
                    os.path.join(self.args.image_folder, f"Apy/orig_{idx_so_far + i}.png")
                )
                # tvu.save_image(
                #     inverse_data_transform(config, y[i]),
                #     os.path.join(self.args.image_folder, f"Apy/y_{idx_so_far + i}.png")
                # )
                
            # init x_T
            x = torch.randn(
                x_orig.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
                n = x.size(0)
                x0_preds = []
                xs = [x]
                
                times = get_schedule_jump(config.time_travel.T_sampling, 
                                               config.time_travel.travel_length, 
                                               config.time_travel.travel_repeat,
                                              )
                time_pairs = list(zip(times[:-1], times[1:]))
                
                
                # reverse diffusion sampling
                for i, j in time_pairs:
                    i, j = i*skip, j*skip
                    if j<0: j=-1 

                    if j < i: # normal sampling 
                        if self.config.user.ddnm:
                            #self inplement DDNM
                            t = (torch.ones(n) * i).to(x.device)
                            next_t = (torch.ones(n) * j).to(x.device)
                            at = compute_alpha(self.betas, t.long())
                            at_next = compute_alpha(self.betas, next_t.long())
                            xt = xs[-1].to('cuda')

                            et = model(xt, t)
                            # Eq. 12
                            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() 
                            
                            # save x0_t
                            os.makedirs(os.path.join(self.args.image_folder, "x0_t"), exist_ok=True)
                            tvu.save_image(
                                inverse_data_transform(config, x0_t),
                                os.path.join(self.args.image_folder, f"x0_t/x0_{t[0]}.png")
                            )

                            # Eq. 13
                            # x0_t_hat = ApA(Apy) + x0_t - ApA(x0_t)
                            x0_tL = load_img_LearningDegradation(os.path.join(self.args.image_folder, f"x0_t/x0_{t[0]}.png")).to('cuda')
                            ApAx0_t = ApA(x0_tL)
                            ApAx0_t_img = tensor2img(ApAx0_t, rgb2bgr=True)
                            os.makedirs(os.path.join(self.args.image_folder, "ApA"), exist_ok=True)
                            imwrite(ApAx0_t_img, os.path.join(self.args.image_folder, f"ApA/ApAx0_{t[0]}.png"))

                            Apy = load_imgDDNM(os.path.join(self.args.image_folder, f"Apy/Apy.png")).to('cuda') 
                            ApAx0_t_img = load_imgDDNM(os.path.join(self.args.image_folder, f"ApA/ApAx0_{t[0]}.png")).to('cuda')
                            
                            x0_t_hat = Apy + x0_t - ApAx0_t_img                                 
                            # x0_t_hat = Apy + x0_t - ApA(x0_t)
                            
                            eta = self.args.eta

                            c1 = (1 - at_next).sqrt() * eta
                            c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                            # different from the paper, we use DDIM here instead of DDPM
                            xt_next = at_next.sqrt() * x0_t_hat + (c1 * torch.randn_like(x0_t) + c2 * et)

                            x0_preds.append(x0_t.to('cpu'))
                            xs.append(xt_next.to('cpu'))   
                        else:

                            t = (torch.ones(n) * i).to(x.device)
                            next_t = (torch.ones(n) * j).to(x.device)
                            at = compute_alpha(self.betas, t.long())
                            at_next = compute_alpha(self.betas, next_t.long())
                            sigma_t = (1 - at_next**2).sqrt()
                            xt = xs[-1].to('cuda')

                            et = model(xt, t)

                            if et.size(1) == 6:
                                et = et[:, :3]

                            # Eq. 12 
                            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                            # Eq. 19
                            if sigma_t >= at_next*sigma_y:
                                lambda_t = 1.
                                gamma_t = (sigma_t**2 - (at_next*sigma_y)**2).sqrt()
                            else:
                                lambda_t = (sigma_t)/(at_next*sigma_y)
                                gamma_t = 0.

                            # Eq. 17
                            x0_t_hat = x0_t - lambda_t*Ap(A(x0_t) - y)

                            eta = self.args.eta

                            c1 = (1 - at_next).sqrt() * eta
                            c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                            # different from the paper, we use DDIM here instead of DDPM
                            xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

                            x0_preds.append(x0_t.to('cpu'))
                            xs.append(xt_next.to('cpu'))  
                            x0_t = (xt - et * (1 - at).sqrt())  
                    else: # time-travel back
                        next_t = (torch.ones(n) * j).to(x.device)
                        at_next = compute_alpha(self.betas, next_t.long())
                        x0_t = x0_preds[-1].to('cuda')

                        xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                        xs.append(xt_next.to('cpu'))

                x = xs[-1]
                
            x = [inverse_data_transform(config, xi) for xi in x]

            tvu.save_image(
                x[0], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{0}.png")
            )
            orig = inverse_data_transform(config, x_orig[0])
            mse = torch.mean((x[0].to(self.device) - orig) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            print('per sample psnr:', psnr)
            avg_psnr += psnr

            idx_so_far += x_orig.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        # print("Total Average kernel psnr: %.2f" % avg_kernel_psnr / (idx_so_far - idx_init))
        print("Number of samples: %d" % (idx_so_far - idx_init))
        
        

    def svd_based_ddnm_plus(self, model, cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)

        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # get degradation matrix
        deg = args.deg
        A_funcs = None
        if deg == 'cs_walshhadamard':
            compress_by = round(1/args.deg_scale)
            from functions.svd_operators import WalshHadamardCS
            A_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by,
                                      torch.randperm(self.config.data.image_size ** 2, device=self.device), self.device)
        elif deg == 'cs_blockbased':
            cs_ratio = args.deg_scale
            from functions.svd_operators import CS
            A_funcs = CS(config.data.channels, self.config.data.image_size, cs_ratio, self.device)
        elif deg == 'inpainting':
            from functions.svd_operators import Inpainting
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            A_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)
        elif deg == 'denoising':
            from functions.svd_operators import Denoising
            A_funcs = Denoising(config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'colorization':
            from functions.svd_operators import Colorization
            A_funcs = Colorization(config.data.image_size, self.device)
        elif deg == 'sr_averagepooling':
            blur_by = int(args.deg_scale)
            from functions.svd_operators import SuperResolution
            A_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
        elif deg == 'sr_bicubic':
            factor = int(args.deg_scale)
            from functions.svd_operators import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            A_funcs = SRConv(kernel / kernel.sum(), \
                             config.data.channels, self.config.data.image_size, self.device, stride=factor)
        elif deg == 'deblur_uni':
            from functions.svd_operators import Deblurring
            A_funcs = Deblurring(torch.Tensor([1 / 9] * 9).to(self.device), config.data.channels,
                                 self.config.data.image_size, self.device)
        elif deg == 'deblur_gauss':
            from functions.svd_operators import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            A_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_aniso':
            from functions.svd_operators import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            A_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels,
                                   self.config.data.image_size, self.device)
        else:
            raise ValueError("degradation type not supported")
        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        sigma_y = args.sigma_y
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y = A_funcs.A(x_orig)
            
            b, hwc = y.size()
            if 'color' in deg:
                hw = hwc / 1
                h = w = int(hw ** 0.5)
                y = y.reshape((b, 1, h, w))
            elif 'inp' in deg or 'cs' in deg:
                pass
            else:
                hw = hwc / 3
                h = w = int(hw ** 0.5)
                y = y.reshape((b, 3, h, w))
                
            if self.args.add_noise: # for denoising test
                y = get_gaussian_noisy_img(y, sigma_y) 
            
            y = y.reshape((b, hwc))

            Apy = A_funcs.A_pinv(y).view(y.shape[0], config.data.channels, self.config.data.image_size,
                                                self.config.data.image_size)

            if deg[:6] == 'deblur':
                Apy = y.view(y.shape[0], config.data.channels, self.config.data.image_size,
                                    self.config.data.image_size)
            elif deg == 'colorization':
                Apy = y.view(y.shape[0], 1, self.config.data.image_size, self.config.data.image_size).repeat(1,3,1,1)
            elif deg == 'inpainting':
                Apy += A_funcs.A_pinv(A_funcs.A(torch.ones_like(Apy))).reshape(*Apy.shape) - 1

            os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
            for i in range(len(Apy)):
                tvu.save_image(
                    inverse_data_transform(config, Apy[i]),
                    os.path.join(self.args.image_folder, f"Apy/Apy_{idx_so_far + i}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]),
                    os.path.join(self.args.image_folder, f"Apy/orig_{idx_so_far + i}.png")
                )

            #Start DDIM
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                if sigma_y==0.: # noise-free case, turn to ddnm
                    x, _ = ddnm_diffusion(x, model, self.betas, self.args.eta, A_funcs, y, cls_fn=cls_fn, classes=classes, config=config)
                else: # noisy case, turn to ddnm+
                    x, _ = ddnm_plus_diffusion(x, model, self.betas, self.args.eta, A_funcs, y, sigma_y, cls_fn=cls_fn, classes=classes, config=config)

            x = [inverse_data_transform(config, xi) for xi in x]


            for j in range(x[0].size(0)):
                tvu.save_image(
                    x[0][j], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{0}.png")
                )
                orig = inverse_data_transform(config, x_orig[j])
                mse = torch.mean((x[0][j].to(self.device) - orig) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                avg_psnr += psnr

            idx_so_far += y.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))

# Code form RePaint   
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


