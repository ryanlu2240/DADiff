# python val.py --fn_root /eva_data1/shlu2240/Dataset/ffhq/val --sample_number 50 --mode root_folder --root_folder_path /eva_data1/shlu2240/DDNM_LD/exp/output/scale_timestep --face_mask 0
import sys
import os
import torch
import csv
import facer
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from lpips import LPIPS
from pytorch_fid import fid_score
from skimage import io
import shutil
import torchvision
import torchvision.transforms as transforms
import argparse
# from tqdm import tqdm
import math
import cv2


def calculate_psnr_eval(img1, img2):
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim_eval(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
    

def calculate_lpips_eval(model, prediction, target):
    lpips_model = model
    # prediction = Image.open(prediction_path).convert("RGB")
    # target = Image.open(target_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    prediction = transform(prediction).unsqueeze(0).to('cuda')
    target = transform(target).unsqueeze(0).to('cuda')
    lpips_value = lpips_model(prediction, target)
    # print(lpips_value)
    return lpips_value.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', type=str, default="folder", help="folder or root_folder"
    )
    parser.add_argument(
        '--root_folder_path', type=str
    )
    parser.add_argument(
        "--run_FID", type=int, default=0
    )
    parser.add_argument(
        "--sample_number", type=int, default=None
    )
    parser.add_argument(
        "--face_mask", type=int, default=0
    )
    parser.add_argument(
        '--fn_root', type=str, help="folder for extract file name"
    )
    parser.add_argument(
        "--device", type=int, default=0
    )
    parser.add_argument(
        "--save_csv", type=int, default=0
    )
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    fn_list = get_fn_list(args)
    N = len(fn_list)
    
    if args.mode == 'folder':
        # result folder list, loop through each result folder
        result_folder = ['/eva_data1/shlu2240/DDNM_LD/exp/output/scale_timestep/combine_eta05_alpha03_step25_x4'
                        ]
        
        
        lpips_model = LPIPS(net='alex').to('cuda')


        for folder in result_folder:
            if args.run_FID:
                des_root = os.path.join(folder, 'result')
                os.makedirs(os.path.join(des_root, "output"), exist_ok=True)
                os.makedirs(os.path.join(des_root, "gt"), exist_ok=True)
                os.makedirs(os.path.join(des_root, "apy"), exist_ok=True)

            total_output_psnr = 0.0
            total_output_ssim = 0.0
            total_output_lpips = 0.0
            
            total_apy_psnr = 0.0
            total_apy_ssim = 0.0
            total_apy_lpips = 0.0
            
            for filename in tqdm(fn_list):
                predict_path = os.path.join(folder, f'{filename}.png')   
                gt_path = os.path.join(folder, 'Apy', f'orig_{filename}.png')
                apy_path = os.path.join(folder, 'Apy', f'Apy_{filename}.png')
                if args.run_FID:
                    des_pred = os.path.join(des_root, 'output', f'{filename}.png')
                    des_gt = os.path.join(des_root, 'gt', f'orig_{filename}.png')
                    des_apy = os.path.join(des_root, 'apy', f'Apy_{filename}.png')
                    shutil.copy(predict_path, des_pred)
                    shutil.copy(gt_path, des_gt)
                    shutil.copy(apy_path, des_apy)
                
                # load img
                predict_img = Image.open(predict_path).convert("RGB")
                apy_img = Image.open(apy_path).convert("RGB")
                gt_img = Image.open(gt_path).convert("RGB")
                
                if args.face_mask:
                    gt_img = get_masked_img(gt_path, device)
                    apy_img = get_masked_img(apy_path, device)
                    predict_img = get_masked_img(predict_path, device)
                    
                
                
                total_output_psnr += calculate_psnr(predict_img, gt_img)
                total_output_ssim += calculate_ssim(predict_img, gt_img)
                total_output_lpips += calculate_lpips(lpips_model, predict_img, gt_img)

                total_apy_psnr += calculate_psnr(apy_img, gt_img)
                total_apy_ssim += calculate_ssim(apy_img, gt_img)
                total_apy_lpips += calculate_lpips(lpips_model, apy_img, gt_img)
                
            print('output avg psnr:', total_output_psnr / N) 
            print('output ssim psnr:', total_output_ssim / N) 
            print('output lpips psnr:', total_output_lpips / N) 

            print('apy avg psnr:', total_apy_psnr / N) 
            print('apy ssim psnr:', total_apy_ssim / N) 
            print('apy lpips psnr:', total_apy_lpips / N) 
            
            if args.run_FID:
                output_FID = calculate_fid(os.path.join(des_root, "output"), os.path.join(des_root, "gt"))
                apy_FID = calculate_fid(os.path.join(des_root, "apy"), os.path.join(des_root, "gt"))
                print(f'output FID score:', output_FID)
                print(f'apy FID score:', apy_FID)
                
            with open(os.path.join(folder, '_log_val_.txt'), 'w') as f:
                f.writelines(f'output avg psnr: {total_output_psnr / N} \n')
                f.writelines(f'output ssim psnr: {total_output_ssim / N} \n')
                f.writelines(f'output lpips psnr: {total_output_lpips / N} \n')
                f.writelines(f'apy avg psnr: {total_apy_psnr / N} \n')
                f.writelines(f'apy ssim psnr: {total_apy_ssim / N} \n')
                f.writelines(f'apy lpips psnr: {total_apy_lpips / N} \n')
                if args.run_FID:
                    f.writelines(f'output FID score: {output_FID} \n')
                    f.writelines(f'apy FID score: {apy_FID} \n')
    else:
        # result all folder under folder root, loop through each result folder
        folder_root = args.root_folder_path
        result_folder = list_folders(folder_root)
    
        lpips_model = LPIPS(net='alex').to('cuda')

        log_fn = f'_val_masked{args.face_mask}.txt'
        csv_file = f'{folder_root}/_val_masked{args.face_mask}.csv'
        # Appending data to CSV file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([['method', 'PSNR', 'SSIM', 'LPIPS']])
            for folder in result_folder:
                print(folder)
                if args.face_mask or args.run_FID:
                    des_root = os.path.join(folder, 'result')
                    os.makedirs(os.path.join(des_root, "output"), exist_ok=True)
                    os.makedirs(os.path.join(des_root, "gt"), exist_ok=True)
                    os.makedirs(os.path.join(des_root, "apy"), exist_ok=True)
                total_output_psnr = 0.0
                total_output_ssim = 0.0
                total_output_lpips = 0.0
                
                total_apy_psnr = 0.0
                total_apy_ssim = 0.0
                total_apy_lpips = 0.0
                
                for filename in tqdm(fn_list):
                    predict_path = os.path.join(folder, f'{filename}.png')   
                    gt_path = os.path.join(folder, 'Apy', f'orig_{filename}.png')
                    apy_path = os.path.join(folder, 'Apy', f'Apy_{filename}.png')
                    if args.run_FID:
                        des_pred = os.path.join(des_root, 'output', f'{filename}.png')
                        des_gt = os.path.join(des_root, 'gt', f'orig_{filename}.png')
                        des_apy = os.path.join(des_root, 'apy', f'Apy_{filename}.png')
                        shutil.copy(predict_path, des_pred)
                        shutil.copy(gt_path, des_gt)
                        shutil.copy(apy_path, des_apy)
                    
                    # load img
                    predict_img = Image.open(predict_path).convert("RGB")
                    apy_img = Image.open(apy_path).convert("RGB")
                    gt_img = Image.open(gt_path).convert("RGB")
                    
                    if args.face_mask:
                        mask = get_mask(gt_path, device)
                        gt_img = get_masked_img(gt_path, mask)
                        apy_img = get_masked_img(apy_path, mask)
                        predict_img = get_masked_img(predict_path, mask)
                        gt_img.save(os.path.join(des_root, "gt", f'{filename}_masked.png'))
                        apy_img.save(os.path.join(des_root, "apy", f'{filename}_masked.png'))
                        predict_img.save(os.path.join(des_root, "output", f'{filename}_masked.png'))

                        predict_img = Image.open(os.path.join(des_root, "output", f'{filename}_masked.png')).convert("RGB")
                        apy_img = Image.open(os.path.join(des_root, "apy", f'{filename}_masked.png')).convert("RGB")
                        gt_img = Image.open(os.path.join(des_root, "gt", f'{filename}_masked.png')).convert("RGB")
                        
                    
                    
                    total_output_psnr += calculate_psnr(predict_img, gt_img)
                    total_output_ssim += calculate_ssim(predict_img, gt_img)
                    total_output_lpips += calculate_lpips(lpips_model, predict_img, gt_img)

                    total_apy_psnr += calculate_psnr(apy_img, gt_img)
                    total_apy_ssim += calculate_ssim(apy_img, gt_img)
                    total_apy_lpips += calculate_lpips(lpips_model, apy_img, gt_img)
                    
                print('output avg psnr:', total_output_psnr / N) 
                print('output ssim psnr:', total_output_ssim / N) 
                print('output lpips psnr:', total_output_lpips / N) 

                print('apy avg psnr:', total_apy_psnr / N) 
                print('apy ssim psnr:', total_apy_ssim / N) 
                print('apy lpips psnr:', total_apy_lpips / N) 
                
                if args.run_FID:
                    output_FID = calculate_fid(os.path.join(des_root, "output"), os.path.join(des_root, "gt"))
                    apy_FID = calculate_fid(os.path.join(des_root, "apy"), os.path.join(des_root, "gt"))
                    print(f'output FID score:', output_FID)
                    print(f'apy FID score:', apy_FID)
                
                with open(os.path.join(folder, log_fn), 'w') as f:
                    f.writelines(f'output avg psnr: {total_output_psnr / N} \n')
                    f.writelines(f'output ssim psnr: {total_output_ssim / N} \n')
                    f.writelines(f'output lpips psnr: {total_output_lpips / N} \n')
                    f.writelines(f'apy avg psnr: {total_apy_psnr / N} \n')
                    f.writelines(f'apy ssim psnr: {total_apy_ssim / N} \n')
                    f.writelines(f'apy lpips psnr: {total_apy_lpips / N} \n')
                    if args.run_FID:
                        f.writelines(f'output FID score: {output_FID} \n')
                        f.writelines(f'apy FID score: {apy_FID} \n')
            
                new_data = [[
                    os.path.basename(folder), 
                    str(round(total_output_psnr / N, 4)), 
                    str(round(total_output_ssim / N, 4)), 
                    str(round(total_output_lpips / N, 4))
                ]]
                # print(new_data)
                writer.writerows(new_data)
                
        
        
        
        
    
    
    