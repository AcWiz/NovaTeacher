# import numpy as np
# from astropy.io import fits
# import os 

# # 加载图像


# # 目标区域的像素坐标 (x, y) 和半径
# x, y, radius = 76.058, 4914.124, 1.824
# target_region = image_data[y-radius:y+radius, x-radius:x+radius]

# # 计算信号
# signal = np.sum(target_region)

# # 背景区域的像素坐标 (x_bg, y_bg) 和半径
# x_bg, y_bg, bg_radius = 4640.82, 4623.7, 10
# background_region = image_data[y_bg-bg_radius:y_bg+bg_radius, x_bg-bg_radius:x_bg+bg_radius]

# # 计算背景噪声
# background_mean = np.mean(background_region)
# background_noise = np.std(background_region)

# # 读出噪声（假设已知）
# readout_noise = 5.0

# # 计算总噪声
# total_noise = np.sqrt((background_noise**2) + (readout_noise**2))

# # 计算信噪比
# snr = signal / total_noise

# print(f"Signal-to-Noise Ratio (SNR): {snr}")




import numpy as np
from scipy.ndimage import map_coordinates
from astropy.io import fits
import matplotlib.pyplot as plt
import os



background_noise = 0.025252203

fits_dir = 'data/split_ss_/train_wh4_fits/images/'
pre_dir = 'work_dir/star_norm/test_img/epoch195/det/'
pre_file_list = os.listdir(pre_dir)

snr_list = []

for pre_file in pre_file_list:
    txt_file = os.path.join(pre_dir, pre_file)
    with open(txt_file, 'r+') as rd:
        
        img_file = fits_dir + pre_file[:-8] + '.fits'
        image_data = fits.getdata(img_file)
        # img_data = fits.getdata(img_file)
        # for li in img_list:
        #     image_data = fits.getdata(li)
        bboxes = rd.readlines()[1:]
        for bbox in bboxes:
            

            pre_bbox = bbox.split(' ')
            
            pre_bbox = pre_bbox[:4]
            x_center, y_center, a, b = map(float,pre_bbox)

            y, x = np.ogrid[-b:b:complex(0, 2*b), -a:a:complex(0, 2*a)]
            x_coords = x + x_center
            y_coords = y + y_center
            coords = np.array([y_coords.ravel(), x_coords.ravel()])
            interp_values = map_coordinates(image_data, coords, order=1).reshape(y.shape)
            
            # 计算信号强度
            signal = np.sum(interp_values)
            
            
            # 计算目标区域的背景噪声
            num_pixels = interp_values.size
            total_background_noise = background_noise * np.sqrt(num_pixels) + 1e-6
            total_background_noise = np.clip(total_background_noise, 1e-6, None)
            # 计算信噪比
            snr = 20 * np.log10(signal / total_background_noise)
            if  snr < 0:
                print(snr)
                continue
            snr_list.append(snr)
            # print(snr)
            


# 绘制 SNR 分布直方图
plt.hist(snr_list, bins=50, alpha=0.75, edgecolor='black')
plt.title('SNR Distribution')
plt.xlabel('SNR (dB)')
plt.ylabel('Frequency')
plt.grid(True)

# 保存直方图为图片
output_image_path = 'snr_distribution.png'
plt.savefig(output_image_path)


            

# image_data = fits.getdata('data/CSST_MSC_MS_SCI_20250212214432_20250212214702_100000000_06_img.fits')
# # 假设你有浮点数坐标和半径
# x_float, y_float, radius_float = 76.058, 4914.124, 1.824

# # 生成目标区域的网格坐标
# y, x = np.ogrid[-radius_float:radius_float:complex(0, 2*radius_float), -radius_float:radius_float:complex(0, 2*radius_float)]
# x_coords = x + x_float
# y_coords = y + y_float

# # 使用 map_coordinates 进行插值获取目标区域像素值
# coords = np.array([y_coords.ravel(), x_coords.ravel()])
# interp_values = map_coordinates(image_data, coords, order=1).reshape(y.shape)

# # 计算信号强度
# signal = np.sum(interp_values)

# # 选择背景区域进行噪声估计
# # 可以选择图像中一个没有天体的区域
# x_bg_float, y_bg_float, bg_radius_float = 4640.82, 4623.7, 10.0
# y_bg, x_bg = np.ogrid[-bg_radius_float:bg_radius_float:complex(0, 2*bg_radius_float), -bg_radius_float:bg_radius_float:complex(0, 2*bg_radius_float)]
# x_bg_coords = x_bg + x_bg_float
# y_bg_coords = y_bg + y_bg_float
# bg_coords = np.array([y_bg_coords.ravel(), x_bg_coords.ravel()])
# bg_interp_values = map_coordinates(image_data, bg_coords, order=1).reshape(y_bg.shape)

# # 计算背景噪声
# background_mean = np.mean(bg_interp_values)
# background_noise = np.std(bg_interp_values)

# # 计算目标区域的背景噪声
# num_pixels = interp_values.size
# total_background_noise = background_noise * np.sqrt(num_pixels)

# # # 假设已知读出噪声
# # readout_noise = 5.0



# # 计算总噪声
# # total_noise = np.sqrt((background_noise**2) + (readout_noise**2))
# total_noise = np.sqrt((total_background_noise**2))
# print('background_noise', total_background_noise)
# # print('readout_noise', readout_noise)
# print('total_noise', total_noise)

# # 计算信噪比
# snr = 20 * np.log10(signal / total_noise)

# print(f"Signal-to-Noise Ratio (SNR): {snr}")


