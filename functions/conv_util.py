import torch.nn.functional as F
import torch
import numpy as np
import time
import tqdm
import random


def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out


def convolution2d(input_tensor, custom_kernel, stride, padding=0):
    # input_tensor : [1, c, image_size, image_size]
    # custom_kernel : [kernel_size, kernel_size]
    # stride : convolution stride 
    # perform convolution on every channel with custom kernel
    kernel_size = custom_kernel.size(0)
    output_channels = []

    # Loop through each input channel and apply the custom convolution
    for channel in range(input_tensor.size(1)):
        input_channel = input_tensor[:, channel, :, :].unsqueeze(1)  # Select the channel
        output_channel = F.conv2d(input_channel, custom_kernel.view(1, 1, kernel_size, kernel_size), stride=stride, padding=padding)
        output_channels.append(output_channel)

    # Stack the output channels along the channel dimension
    return torch.cat(output_channels, dim=1)


def non_overlapConv_Ap(input_tensor, custom_kernel):
    # non overlap convolution Ap
    # return Ap(y), which will become y if we apply A
    # formula Ap(y)[i][j] = patch_upsample(y) / kernel_size / custom_kernel[i][j]
    # will have problem if kernel have 0 (devide by 0)

    kernel_size = custom_kernel.size(0)

    upsample_tensor = MeanUpsample(input_tensor, kernel_size)
    upsample_tensor = upsample_tensor / kernel_size / kernel_size
    image_size = upsample_tensor.size(2)
    kernel = 1 / custom_kernel
    kernel = kernel.repeat(image_size // kernel_size, image_size // kernel_size)
    
    return upsample_tensor * kernel.unsqueeze(0).unsqueeze(0)


# kernel : custom kernel
# input_size : input image size [B, c, h, w]
# stride : convolution stride
# return matrix A such that convolution operation can be perform by A*x.
def convolution_to_A(kernel, input_size, stride, padding):
    _, _, H, W = input_size
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    kernel_h, kernel_w = kernel.shape

    A = []
    for row in range(0, H_padded - kernel_h + 1, stride):
        for col in range(0, W_padded - kernel_w + 1, stride):
            target_region = torch.zeros((1, 1, H_padded, W_padded), dtype=kernel.dtype)
            target_region[:, :, row:row+kernel_h, col:col+kernel_w] = kernel
            A.append(target_region.view(-1))

    A = torch.stack(A)
    return A

# input : image input in shape [B, c, h, w], B default to 1
# A : 2d tensor for convolution operation
# return output = A * input. shape = [1, c, h, w]
def convolution_with_A(A, input_tensor, padding):
    _, c, h, w = input_tensor.size()
    output_size = int(A.size(0) ** 0.5)
    output_channels = []

    # Loop through each input channel and apply the custom convolution
    for channel in range(c):
        input_channel = input_tensor[0, channel, :, :].unsqueeze(0)  # Select the channel
        # print('before padding', input_channel.shape)
        input_channel = F.pad(input_channel, (padding, padding, padding, padding), mode='constant', value=0)
        # print('after padding', input_channel.shape)
        input_channel = input_channel.flatten()
        output = A @ input_channel
        output_channels.append(output.reshape(1, output_size, output_size))

    # Stack the output channels along the channel dimension
    return torch.cat(output_channels, dim=0).unsqueeze(0)

def create_gaussian_kernel(isize, sigma=1.0):
    # Create an empty array of zeros with the specified size
    kernel = np.zeros((isize, isize))

    # Calculate the center of the kernel
    center = isize // 2

    # Generate a 2D Gaussian distribution
    for x in range(isize):
        for y in range(isize):
            kernel[x, y] = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))

    # Normalize the kernel so that the sum of all values is 1
    kernel /= kernel.sum()

    return torch.tensor(kernel)

def create_random_kernel(kernel_size):
    kernel_size = kernel_size
    std_deviation_x = random.uniform(0.7, 5.0)
    std_deviation_y = random.uniform(0.7, 5.0)  # Different standard deviation for Y
    tilt_angle = np.random.uniform(0, 180)  # Random tilt angle in degrees

    # Create a 2D Gaussian kernel with different std deviations and tilt
    x = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    y = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    x, y = np.meshgrid(x, y)

    # Apply the tilt to the kernel
    rot_matrix = np.array([[np.cos(np.radians(tilt_angle)), -np.sin(np.radians(tilt_angle))],
                        [np.sin(np.radians(tilt_angle)), np.cos(np.radians(tilt_angle))]])

    x_rotated = x * rot_matrix[0, 0] + y * rot_matrix[0, 1]
    y_rotated = x * rot_matrix[1, 0] + y * rot_matrix[1, 1]

    kernel = np.exp(-0.5 * ((x_rotated / std_deviation_x)**2 + (y_rotated / std_deviation_y)**2))
    kernel /= kernel.sum()

    return torch.tensor(kernel)




if __name__ == "__main__":

    input_tensor = torch.randn((1, 3, 256, 256)).cuda()

    custom_kernel = torch.tensor([[2/60, 3/60, 2/60, 1/60],
                                    [1/60, 1/60, 1/60, 2/60],
                                    [1/60, 20/60, 1/60, 1/60],
                                    [1/60, 20/60, 2/60, 1/60]], dtype=torch.float32).cuda()
    
    start_time = time.time()

    gt_convolution = convolution2d(input_tensor, custom_kernel, stride=4, padding=0)

    conv2d_time = time.time()
    
    A = convolution_to_A(custom_kernel, input_tensor.size(), 4, 0).cuda()

    A_time = time.time()

    Ax = convolution_with_A(A, input_tensor, 0)


    Ax_time = time.time()

    print(f'{gt_convolution.shape=}')
    print(f'{Ax.shape=}')

    print('conv2d time:', conv2d_time - start_time)
    print('calculate matrix A time:', A_time - conv2d_time)
    print('Ax time:', Ax_time - A_time)

    tolerance = 0.00001
    # Check if the two tensors are almost the same within the tolerance
    are_almost_equal = torch.all(torch.abs(Ax - gt_convolution) < tolerance)

    if are_almost_equal:
        print("The two tensors are almost the same.")
    else:
        print("The two tensors are not almost the same.")

    ap_start = time.time()
    pseudo_inverse = torch.pinverse(A) 
    print('pseudo inverse time:', time.time() - ap_start)
    print(f'{A.shape}')
    print(f'{pseudo_inverse.shape}')

    # x = torch.arange(64).view(8, 8).float().cuda()
    # x = x.flatten()

    # print('x:(HQ)')
    # print(x)
    # print(f'{x.shape}')

    # y = A @ x
    # print('A(x):(LQ)')
    # print(y)
    # print(f'{y.shape=}')
    # apy = pseudo_inverse @ y
    # print('Ap(y):')
    # print(apy)
    # print(f'{apy.shape}')

    # aapy = A @ apy
    # print('A(Ap(y))')
    # print(aapy)
    # print(f'{aapy.shape=}')



