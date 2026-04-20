import numpy as np
import sep
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage import exposure, filters
from astropy.visualization import ZScaleInterval
from skimage import exposure, filters
from astropy.io import fits


def zscale(image, nsamples=1000, contrast=0.25):
    """
    Implement the ZSCALE algorithm for image contrast adjustment.
    
    Parameters:
        image (numpy.ndarray): 2D array of image pixel values.
        nsamples (int): Number of pixel samples to use for fitting.
        contrast (float): Contrast parameter to control the scale.

    Returns:
        tuple: (z1, z2) where z1 is the minimum intensity and z2 is the maximum intensity.
    """
    # Flatten the image and remove NaN/Inf values
    pixels = image.flatten()
    pixels = pixels[np.isfinite(pixels)]
    
    # Select a random sample of pixels if there are too many
    if len(pixels) > nsamples:
        np.random.seed(42)  # For reproducibility
        pixels = np.random.choice(pixels, nsamples, replace=False)
    
    # Sort the pixel values
    pixels.sort()
    
    # Estimate the median and slope using a robust fit
    midpoint = len(pixels) // 2
    medval = pixels[midpoint]
    slope = (pixels[-1] - pixels[0]) / len(pixels)
    
    # Adjust slope for contrast
    slope /= contrast
    
    # Calculate z1 and z2
    z1 = medval - slope * midpoint
    z2 = medval + slope * (len(pixels) - midpoint)
    
    return z1, z2


# 加载 FITS 文件
filename = "data/300simages20241222/0/image162.fit"
with fits.open(filename) as hdulist:
    data = hdulist[0].data

# 将数据转换为 float32 格式（sep 需要此格式）
data = data.astype(np.float32)

# 提取背景
bkg = sep.Background(data)

# 背景数据和RMS
background = bkg.back()  # 背景值
rms = bkg.rms()          # 背景 RMS（噪声）




# 显示原始图像和背景
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].imshow(data, origin='lower', cmap='gray', vmin=np.percentile(data, 5), vmax=np.percentile(data, 95))
# ax[0].set_title("Original Image")

# ax[1].imshow(background, origin='lower', cmap='gray', vmin=np.percentile(background, 5), vmax=np.percentile(background, 95))
# ax[1].set_title("Extracted Background")
# plt.show()




epsilon = 0
background_min = np.min(background)
background_max = np.max(background)
data_min = np.min(data)
data_max = np.max(data)
print(background_max,background_min)
print(data_max,data_min)

# print(background)

# 对背景进行归一化
normalized_background = (background - background_min) / (background_max - background_min) + epsilon


data_subtracted = data / normalized_background

print(np.min(data_subtracted))



zscale = ZScaleInterval()
vmin, vmax = zscale.get_limits(data_subtracted)

# 对图像进行缩放
scaled_image_data = np.clip(data_subtracted, vmin, vmax)

# 归一化
normalized_image_data = (scaled_image_data - vmin) / (vmax - vmin)


# 绘制亮度值直方图
plt.figure(figsize=(8, 6))
plt.hist(normalized_background.flatten(), bins=50, color='green', edgecolor='black', alpha=0.7)
plt.savefig('data/exposure/data_background/0/histogram.png', format='png')



# plt.imsave('data/exposure/data_background/0/normal_background.png', normalized_background, cmap='gray', format='png')
# plt.imsave('data/exposure/data_background/0/zscale_new.png', data_subtracted, cmap='gray', format='png', vmin=z1, vmax=z2)
# plt.imsave('data/exposure/data_background/0/zscaleMe_data.png', normalized_image_data, cmap='gray', format='png')
# plt.hist(normalized_background.flatten(), bins=50, color='gray', edgecolor='black', alpha=0.7)
# plt.savefig('data/exposure/data_background/0/histogram.png', format='png')


