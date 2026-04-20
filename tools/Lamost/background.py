# import numpy as np
# import sep
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# from astropy.io import fits
# from skimage import exposure, filters
# from astropy.visualization import ZScaleInterval
# from skimage import exposure, filters
# from astropy.io import fits



# # 加载 FITS 文件
# filename = "data/exposure/long_exposure/image162.fit"
# with fits.open(filename) as hdulist:
#     data = hdulist[0].data

# # 将数据转换为 float32 格式（sep 需要此格式）
# data = data.astype(np.float32)

# # 提取背景
# bkg = sep.Background(data)

# # 背景数据和RMS
# background = bkg.back()  # 背景值
# rms = bkg.rms()          # 背景 RMS（噪声）



# epsilon = 0
# background_min = np.min(background)
# background_max = np.max(background)
# data_min = np.min(data)
# data_max = np.max(data)
# print(background_max,background_min)
# print(data_max,data_min)


# normalized_background = (background - background_min) / (background_max - background_min) + epsilon




# # 绘制亮度值直方图
# plt.figure(figsize=(8, 6))
# plt.hist(normalized_background.flatten(), bins=50, color='green', edgecolor='black', alpha=0.7)
# plt.savefig('data/exposure/data_background/0/histogram.png', format='png')



# # plt.imsave('data/exposure/data_background/0/normal_background.png', normalized_background, cmap='gray', format='png')
# # plt.imsave('data/exposure/data_background/0/zscale_new.png', data_subtracted, cmap='gray', format='png', vmin=z1, vmax=z2)
# # plt.imsave('data/exposure/data_background/0/zscaleMe_data.png', normalized_image_data, cmap='gray', format='png')
# # plt.hist(normalized_background.flatten(), bins=50, color='gray', edgecolor='black', alpha=0.7)
# # plt.savefig('data/exposure/data_background/0/histogram.png', format='png')




"""
train文件夹下一千幅图的均值为 0.97213155  0.14726804 1.017939   0.14445712  1-2*std = 0.7290247599999999

"""


import numpy as np
import sep
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from astropy.io import fits
import os
from scipy.ndimage import median_filter



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


# 计算所有图像的均值亮度
def compute_average_histogram(src_dir, output_dir, bins=50):
    # 存储所有图像的背景数据
    all_backgrounds = []
    
    # 遍历源文件夹中的所有 FITS 文件
    for filename in os.listdir(src_dir):
        if filename.endswith(".fit"):
            file_path = os.path.join(src_dir, filename)
            
            # 加载 FITS 文件
            with fits.open(file_path) as hdulist:
                data = hdulist[0].data

            # 将数据转换为 float32 格式（sep 需要此格式）
            data = data.astype(np.float32)

            # 提取背景
            bkg = sep.Background(data)

            # 获取背景数据
            background = bkg.back()

            # 归一化背景数据
            background_min = np.min(background)
            background_max = np.max(background)
            normalized_background = (background - background_min) / (background_max - background_min)

            print(np.mean(normalized_background))
            print(np.std(normalized_background))
            single_mean = np.mean(normalized_background)
            scale_ = all_mean / single_mean 
            mut_normalized_background = normalized_background * scale_
            print(np.mean(mut_normalized_background))
            print(np.std(mut_normalized_background))


            print(file_path)

            # 将当前背景数据添加到列表
            all_backgrounds.append(mut_normalized_background.flatten())
            


    # 合并所有图像的背景数据
    all_backgrounds = np.concatenate(all_backgrounds)

    # 计算均值和标准差
    median = np.median(all_backgrounds)
    std_dev = np.std(all_backgrounds)

    # 计算中值减去三倍标准差
    median_minus_3sigma = median - 3 * std_dev
    
    print(median)
    print(std_dev)
    print(median_minus_3sigma)



    # 计算所有图像的平均背景直方图
    plt.figure(figsize=(10, 6))
    plt.hist(all_backgrounds, bins=bins, color='green', edgecolor='black', alpha=0.7)

    # 设置横坐标范围
    plt.xlim(0.5, 1.1)

    # 保存直方图到文件
    output_path = os.path.join(output_dir, 'average_histogram.png')
    # plt.savefig(output_path, format='png')

    print(f"Average histogram saved at: {output_path}")
    plt.close()




# 得到超级平场。整幅图的均值。

def compute_average_fig(src_dir, output_dir, bins=50):
    # 存储所有图像的背景数据
    all_backgrounds = []
    
    # 遍历源文件夹中的所有 FITS 文件
    for filename in os.listdir(src_dir):
        if filename.endswith(".fit"):
            file_path = os.path.join(src_dir, filename)
            
            # 加载 FITS 文件
            with fits.open(file_path) as hdulist:
                data = hdulist[0].data

            # 将数据转换为 float32 格式（sep 需要此格式）
            data = data.astype(np.float32)

            # 提取背景
            bkg = sep.Background(data)

            # 获取背景数据
            background = bkg.back()

            # 归一化背景数据
            background_min = np.min(background)
            background_max = np.max(background)
            normalized_background = (background - background_min) / (background_max - background_min)

            single_mean = np.mean(normalized_background)
            scale_ = all_mean / single_mean 
            mut_normalized_background = normalized_background * scale_
            all_backgrounds.append(mut_normalized_background)

            print(f"Processed: {file_path}")


    # 合并所有图像的背景数据
    all_backgrounds = np.array(all_backgrounds)

    avg_background = np.mean(all_backgrounds, axis=0)


    avg_background[avg_background < (mean - 1.5 * std) ] = 99


    sub_avg_background = avg_background.flatten()
    

    output_fits_path = os.path.join(output_dir, 'average_background_image_1.5std.fits')
    hdu = fits.PrimaryHDU(avg_background)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(output_fits_path, overwrite=True)
    print(f"Average background image saved as FITS file at: {output_fits_path}")


    # 计算所有图像的平均背景直方图
    plt.figure(figsize=(10, 6))
    plt.hist(sub_avg_background, bins=bins, color='green', edgecolor='black', alpha=0.7)

    # 设置横坐标范围
    plt.xlim(0.7, 1.1)

    # 保存直方图到文件
    output_path = os.path.join(output_dir, 'mean_average_histogram_1.5std.png')
    plt.savefig(output_path, format='png')
    plt.imsave('data/exposure/data_background/1.5/mean_background_1.5std.png',  avg_background, cmap='gray', format='png')

    print(f"Average histogram saved at: {output_path}")
    plt.close()



# 得到每幅图像
def get_png(src_dir, output_dir):
    # 存储所有图像的背景数据
    all_backgrounds = []
    
    with fits.open('data/exposure/data_background/1.5/average_background_image_1.5std.fits') as hdulist:
                average_background = hdulist[0].data
    
    # 遍历源文件夹中的所有 FITS 文件
    for filename in os.listdir(src_dir):
        if filename.endswith(".fit"):
            file_path = os.path.join(src_dir, filename)
            
            # 加载 FITS 文件
            with fits.open(file_path) as hdulist:
                data = hdulist[0].data

            # 中值滤波
            data = median_filter(data, size=3) 

            data_min = np.min(data)
            data_max = np.max(data)
            normalized_data = (data - data_min) / (data_max - data_min)

            
            sub_data = normalized_data / average_background

            # 中值滤波
            # sub_data = median_filter(sub_data, size=3) 

            zscale = ZScaleInterval()
            vmin, vmax = zscale.get_limits(sub_data)




            # 对图像进行缩放
            scaled_image_data = np.clip(sub_data, vmin, vmax)

            # # 保存去除背景后的fit文件。
            # output_fits_path = os.path.join('/home/flh/datasets/gs_imgs/data/ex_meanBackground_zscale_1.5std/', filename)
            # hdu = fits.PrimaryHDU(scaled_image_data)
            # hdulist = fits.HDUList([hdu])
            # hdulist.writeto(output_fits_path, overwrite=True)
            # print(f"fit saved at: {output_fits_path}")

            # 保存png图片
            png_path = filename[:-4] + '.png'
            out_path = os.path.join(output_dir, png_path)
            plt.imsave(out_path,  scaled_image_data, cmap='gray', format='png')
            print(f"png saved at: {out_path}")
            



# 使用示例
source_directory = '/home/flh/datasets/gs_imgs/data/train'  # 这里指定包含 FITS 文件的文件夹
output_directory = 'data/exposure/data_background/1.5'  # 这里指定保存直方图的文件夹
train_data_directory = '/home/flh/datasets/gs_imgs/data/1.5std_png_mean_test/'
all_mean = 0.97213155

mean, std =1.017939 , 0.14445712
compute_average_fig(source_directory, output_directory)
# compute_average_histogram(source_directory, output_directory)
# get_png(source_directory, train_data_directory)





