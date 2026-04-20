import numpy as np
from astropy.io import fits
import sep
import matplotlib.pyplot as plt
import os
from pathlib import Path
from matplotlib.colors import LogNorm
from astropy.visualization import ZScaleInterval
from scipy.ndimage import median_filter

def extract_stars(image_path,
                 detection_threshold=5.0,    # 检测阈值
                 minarea=20,                  # 最小连通像素数
                 deblend_nthresh=32,         # 用于分离重叠源的阈值数量
                 deblend_cont=0.002,         # 分离对比度阈值
                 clean=True,                 # 是否清理误检
                 clean_param=1.0,            # 清理参数
                 filter_kernel=None):        # 匹配滤波核
    """

    
    参数：
    image_path: str, FITS文件路径
    detection_threshold: float, 检测阈值
    minarea: int, 最小连通像素数
    deblend_nthresh: int, 用于分离重叠源的阈值数量
    deblend_cont: float, 分离对比度阈值
    clean: bool, 是否进行清理
    clean_param: float, 清理参数
    filter_kernel: ndarray, 可选的匹配滤波核
    """
    # 读取FITS图像
    hdul = fits.open(image_path)
    data = hdul[0].data.astype(np.float64)

    # 中值滤波
    data = median_filter(data, size=3) 
    
    # 确保数据是连续的内存块
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
        
    # 创建匹配滤波核（如果未指定）
    if filter_kernel is None:
        filter_kernel = np.array([[1., 2., 1.],
                                [2., 4., 2.],
                                [1., 2., 1.]])
        filter_kernel /= filter_kernel.sum()
    
    # 测量背景
    bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3)
    
    # 减去背景
    data_sub = data - bkg.back()
    
    # 进行源检测
    objects = sep.extract(data_sub, detection_threshold,
                         err=bkg.globalrms,
                         minarea=minarea,
                         deblend_nthresh=deblend_nthresh,
                         deblend_cont=deblend_cont,
                         clean=clean,
                         clean_param=clean_param,
                         filter_kernel=filter_kernel)
    
    


    flux, fluxerr, flag = sep.sum_circle(data_sub, 
                                       objects['x'], 
                                       objects['y'], 
                                       3.0,
                                       err=bkg.globalrms,
                                       gain=1.0)
    
    # 整理结果
    results = {
        'x': objects['x'],  # X坐标
        'y': objects['y'],  # Y坐标
        'flux': flux,       # 流量
        'flux_err': fluxerr,# 流量误差
        'a': objects['a'],  # 半长轴
        'b': objects['b'],  # 半短轴
        'theta': objects['theta'], # 位置角
        'ellipticity': 1.0 - objects['b']/objects['a'], # 椭率
        'flags': flag       # 检测标志
    }

    filtered_results = {
    'x': [],
    'y': [],
    'a': [],
    'b': [],
    'theta': [],
    'ellipticity': [],
    'flux': [],
    'flux_err': [],
    'flags': []
}
    
    # 过滤左上角的星
    for i in range(len(results['x'])):
        # 过滤掉以下两种情况的星体 
        # if not (results['x'][i] <= 256 and results['y'][i] <= 256) and not (results['x'][i] < 50 or results['y'][i] < 20):
            filtered_results['x'].append(results['x'][i])
            filtered_results['y'].append(results['y'][i])
            filtered_results['a'].append(results['a'][i])
            filtered_results['b'].append(results['b'][i])
            filtered_results['theta'].append(results['theta'][i])
            filtered_results['ellipticity'].append(results['ellipticity'][i])
            filtered_results['flux'].append(results['flux'][i])
            filtered_results['flux_err'].append(results['flux_err'][i])
            filtered_results['flags'].append(results['flags'][i])
    
    hdul.close()
    return filtered_results, data_sub
    # return filtered_results, data

def visualize_results(image_data, results, save_path=None):
    """
    可视化检测结果
    
    参数:
    image_data: ndarray, 图像数据
    results: dict, 检测结果
    save_path: str, 可选，保存路径
    """
    fig, ax = plt.subplots()
    
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image_data)

            # 对图像进行缩放
    scaled_image_data = np.clip(image_data, vmin, vmax)

    # # 可视化显示图像
    # im = ax.imshow(scaled_image_data, cmap='gray')
    # # plt.colorbar(im)
    
    # # 绘制检测到的天体
    # # 用圆圈标记位置
    # for i in range(len(results['x'])):
    #     e = plt.Circle((results['x'][i], results['y'][i]), 
    #                   radius=results['a'][i],
    #                   fill=False,
    #                   color='red')
    #     ax.add_patch(e)

    # ax.set_axis_off()
    

    
    if save_path:
        # plt.savefig(save_path)
        plt.imsave(save_path,  scaled_image_data, cmap='gray', format='png')
        plt.close()
    else:
        plt.show()

def process_directory(input_dir, output_dir, threshold=5):
    """
    批量处理目录下的所有FITS文件
    
    参数:
    input_dir: str, 输入目录路径
    output_dir: str, 输出目录路径
    threshold: float, 检测阈值 默认3
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有FITS文件
    input_dir = Path(input_dir)
    fits_files = list(input_dir.glob('*.fits')) + list(input_dir.glob('*.fit'))

    
    print(f"找到 {len(fits_files)} 个FITS文件")
    
    # 处理每个文件
    for fits_file in fits_files:
        print(f"\n处理文件: {fits_file.name}")
        try:
            # 提取星源
            results, data_sub = extract_stars(str(fits_file), threshold)
            
            # 准备输出文件名
            base_name = fits_file.stem
            plot_path = output_dir / f"image/{base_name}.png"
            catalog_path = output_dir / f"txt/{base_name}.txt"
            if not os.path.exists(output_dir / "image"):
                os.makedirs(output_dir / "image")

            if not os.path.exists(output_dir / "txt"):
                os.makedirs(output_dir / "txt")
            
            # 保存可视化结果
            visualize_results(data_sub, results, str(plot_path))
            
            # 保存星表
            with open(catalog_path, 'w') as f:
                
                f.write(f"{len(results['x'])}\n")
                for i in range(len(results['x'])):
                    f.write(f"{results['x'][i]:.2f}\t{results['y'][i]:.2f}\t"
                           f"{results['a'][i]:.2f}\t{results['b'][i]:.2f}\t"
                           f"{results['theta'][i]:.5f}\n"
                        #    'star\n'
                           )


            print(f"完成！已保存检测图和星表")
            
        except Exception as e:
            print(f"处理 {fits_file.name} 时出错: {e}")

# 使用示例
def main():
    # 设置输入输出目录
    input_dir = "/home/flh/datasets/LAMOST_new/images_1k"                     # 替换为你的输入目录
    output_dir = "/home/flh/datasets/LAMOST_new/without_back/all"            # 替换为你想要的输出目录
    
    # 设置检测阈值
    threshold = 5.0
    
    try:
        process_directory(input_dir, output_dir, threshold)
        print("\n所有文件处理完成！")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

if __name__ == "__main__":
    main()