import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2
import torchvision.utils as tvu
    
def calculate_psnr(predicted_kernel, ground_truth_kernel, max_pixel_value=1):
    """
    Calculate PSNR between predicted and ground truth images.

    Parameters:
    - predicted: NumPy array representing the predicted image
    - ground_truth: NumPy array representing the ground truth image
    - max_pixel_value: Maximum possible pixel value (default is 255 for 8-bit images)

    Returns:
    - psnr: Peak Signal-to-Noise Ratio
    """
    mse = torch.mean((predicted_kernel - ground_truth_kernel) ** 2)
    psnr = 10 * torch.log10(1 / mse)  # The maximum possible value is 1 for kernels
    return psnr

def plot_kernel(gt_kernel, pred_kernel, idx, save_path):
    batch_size, _, kernel_size, _ = pred_kernel.shape

    pred_kernels_np = pred_kernel.detach().cpu().numpy()
    gt_kernels_np = gt_kernel.detach().cpu().numpy()
    os.makedirs(f'{save_path}/kernel_image', exist_ok=True)
    for i in range(batch_size):
        pred_kernel = pred_kernels_np[i, 0, :, :]
        gt_kernel = gt_kernels_np[i, 0, :, :]

        # Plotting
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(pred_kernel, cmap='viridis', interpolation='nearest')
        plt.title('Prediction Kernel')

        plt.subplot(1, 2, 2)
        plt.imshow(gt_kernel, cmap='viridis', interpolation='nearest')
        plt.title('Ground Truth Kernel')

        plt.tight_layout()
        plt.savefig(f'{save_path}/kernel_image/{idx}.png')
        plt.close()

def convert_jpg(png_path):
    jpg_path = png_path.replace('png', 'jpg')
    # image = cv2.imread(png_path)
    # # Save the image in JPG format
    # cv2.imwrite(jpg_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])    

    png_image = Image.open(png_path)
    # Convert to RGB mode if it's not already in that mode
    if png_image.mode != "RGB":
        png_image = png_image.convert("RGB")
    # Save as JPG
    png_image.save(jpg_path, "JPEG")
    os.remove(png_path)
    return jpg_path
