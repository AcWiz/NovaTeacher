import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from skimage import exposure, filters
from scipy.ndimage import median_filter
from astropy.io import fits


root_dir = 'data/300simages20241222/0'
filename_list = os.listdir(root_dir)

for file_ in filename_list:
    # 读取FITS文件
    image_data = fits.getdata(os.path.join(root_dir, file_))
    


    # 打印图像数据的最小值和最大值
    # print(np.min(image_data))
    # print(np.max(image_data))

    # image_data = median_filter(image_data, size=2) 

    # 直方均衡
    # image_data = exposure.equalize_hist(image_data)

    # 使用Zscale算法计算显示范围
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image_data)

    # 对图像进行缩放
    scaled_image_data = np.clip(image_data, vmin, vmax)

    # 归一化
    normalized_image_data = (scaled_image_data - vmin) / (vmax - vmin)

    # CLAHE处理
    clahe = exposure.equalize_adapthist(normalized_image_data, clip_limit=0.10)


    image_data = median_filter(clahe, size=3) 
    clahe = image_data

    txt_file = file_[:-4] + '.png'

    # 保存图像
    plt.imsave(os.path.join('data/exposure/0_zscale_z', txt_file), clahe, cmap='gray', format='png')
    # # plt.imsave('data/new_galaxy/data/images/CSST_1702_V01.png', edges, cmap='gray', format='png')



# 另外一种zscale
# import numpy as np

# def zscale(image, nsamples=1000, contrast=0.25):
#     """
#     Implement the ZSCALE algorithm for image contrast adjustment.
    
#     Parameters:
#         image (numpy.ndarray): 2D array of image pixel values.
#         nsamples (int): Number of pixel samples to use for fitting.
#         contrast (float): Contrast parameter to control the scale.

#     Returns:
#         tuple: (z1, z2) where z1 is the minimum intensity and z2 is the maximum intensity.
#     """
#     # Flatten the image and remove NaN/Inf values
#     pixels = image.flatten()
#     pixels = pixels[np.isfinite(pixels)]
    
#     # Select a random sample of pixels if there are too many
#     if len(pixels) > nsamples:
#         np.random.seed(42)  # For reproducibility
#         pixels = np.random.choice(pixels, nsamples, replace=False)
    
#     # Sort the pixel values
#     pixels.sort()
    
#     # Estimate the median and slope using a robust fit
#     midpoint = len(pixels) // 2
#     medval = pixels[midpoint]
#     slope = (pixels[-1] - pixels[0]) / len(pixels)
    
#     # Adjust slope for contrast
#     slope /= contrast
    
#     # Calculate z1 and z2
#     z1 = medval - slope * midpoint
#     z2 = medval + slope * (len(pixels) - midpoint)
#     return z1, z2





# '''
# 方法二
# 取对数量化
# '''

# from astropy.io import fits
# import matplotlib.pyplot as plt
# import numpy as np
# import imageio
# from matplotlib.colors import LogNorm
# from skimage import exposure, transform
# from PIL import Image
# import os
# from scipy.ndimage import median_filter

# log_norm = LogNorm()

# root_dir = 'data/300simages20241222/0/'
# filename_list = os.listdir(root_dir)

# for file_ in filename_list:
    
    
#     # 读取示例FITS文件
#     filename = os.path.join(root_dir , file_)

#     image_data = fits.getdata(filename)
#     print(np.min(image_data))
#     print(np.max(image_data))
#     image_data = median_filter(image_data, size=10) 
#     normalized_image_data = log_norm(image_data) 
#     normalized_image_data = (normalized_image_data - np.min(normalized_image_data)) / (np.max(normalized_image_data) - np.min(normalized_image_data))

#     clahe = exposure.equalize_adapthist(normalized_image_data, clip_limit=0.2)

#     # txt_file = 'CSST_' + file_[-17:-12] + '_V01.png'
#     txt_file = file_[:-4] + '1.png'
    
#     # print(np.min(normalized_image_data))
#     # print(np.max(normalized_image_data))
#     plt.imsave(os.path.join('data/exposure/', txt_file), clahe, cmap='gray', format='png')
#     # plt.imsave(os.path.join('data/exposure/', txt_file), normalized_image_data, cmap= 'gray', format='png')




 

# '''
# 方法三


# '''

# import os
# import numpy as np
# from astropy.io import fits
# import matplotlib.pyplot as plt
# from scipy.ndimage import median_filter

# def zscale(image, nsamples=600, contrast=0.25):
#     samples = zscale_sample(image, nsamples)
#     samples = np.sort(samples)
#     center_pixel = (len(samples) - 1) // 2
#     median = samples[center_pixel]
    
#     if len(samples) % 2 == 0:
#         median = (median + samples[center_pixel + 1]) / 2
    
#     fit = np.polyfit(range(len(samples)), samples, 1)
#     slope = fit[0]
#     intercept = fit[1]
    
#     z1 = max(image.min(), median - center_pixel * slope / contrast)
#     z2 = min(image.max(), median + (len(samples) - center_pixel - 1) * slope / contrast)
    
#     return z1, z2

# def zscale_sample(image, nsamples):
#     stride = max(1, min(image.shape) // nsamples)
#     samples = image[::stride, ::stride].flatten()
#     if len(samples) > nsamples:
#         samples = np.random.choice(samples, nsamples, replace=False)
#     return samples



# root_dir = 'data/300simages20241222/0'
# filename_list = os.listdir(root_dir)

# for file_ in filename_list:
    
    
#     # 读取示例FITS文件
#     filename = os.path.join(root_dir , file_)
    


#     image_data = fits.getdata(filename)

#     image_data = median_filter(image_data, size=2) 

#     # 计算ZScale
#     z1, z2 = zscale(image_data)
#     # txt_file = 'CSST_' + file_[-17:-12] + '_V01.png'
#     txt_file = 'CSST' + file_[:-4] + '.png'
#     print(txt_file)
#     plt.imsave(os.path.join('data/exposure/0', txt_file), image_data, cmap='gray', vmin=z1, vmax=z2)
# #     # plt.colorbar()
# #     # plt.show()
