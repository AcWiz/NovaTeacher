# from astropy.io import fits
# import numpy as np
# import os
# import sep
# import re
# from pathlib import Path


# def calculate_snr(fit_file, star_params):
#     """
#     Calculate the Signal-to-Noise Ratio (SNR) for a single star.
    
#     Parameters:
#     fit_file: str - path to the .fit file
#     star_params: list - [x, y, a, b, theta] star parameters (center x, center y, semi-major axis a, semi-minor axis b, and rotation angle theta)
    
#     Returns:
#     snr: float - calculated signal-to-noise ratio
#     """
    
#     # 读取 FIT 文件
#     hdu_list = fits.open(fit_file)
#     image_data = hdu_list[0].data
    
#     # 使用sep函数来计算背景噪声
#     image_data = image_data.astype(np.float32)
#     background = sep.Background(image_data)

#     # 提取星体的参数
#     x, y, a, b, theta = star_params
    
#     # 根据椭圆的长短轴和角度构建星体的掩码
#     yy, xx = np.indices(image_data.shape)
    
#     # 旋转坐标系，使椭圆轴与图像坐标对齐
#     x_rot = (xx - x) * np.cos(np.radians(theta)) + (yy - y) * np.sin(np.radians(theta))
#     y_rot = -(xx - x) * np.sin(np.radians(theta)) + (yy - y) * np.cos(np.radians(theta))
    
#     # 计算椭圆掩码
#     star_mask = ((x_rot**2) / a**2 + (y_rot**2) / b**2) <= 1
    
#     # 提取与星体对应的区域
#     star_region = image_data[star_mask]
    
#     # 信号：恒星区域的总亮度（像素值之和）
#     # signal = np.sum(star_region)
#     signal = np.sum(star_region - background.back()[star_mask])
    
#     # # 计算背景区域的标准差作为噪声
#     # # 计算背景区域的标准差：这里需要通过sep来计算背景模型
#     noise_std = np.std(image_data[~star_mask] - background.back()[~star_mask])

#     # 计算恒星区域的像素数
#     N = np.sum(star_mask)
    
#     # 计算 SNR（信号除以噪声标准差乘以恒星区域的像素数的平方根）
#     snr = signal / (np.sqrt(N) * noise_std)
    
#     print(signal, np.sqrt(N) * noise_std, np.sqrt(N))
#     print(f"SNR: {snr}")
#     return snr


# def process_all_stars(txt_folder, fit_folder, save_txt_folder):
#     """Process SNR for all stars in the txt files in the folder."""
#     txt_files = os.listdir(txt_folder)
    
#     for txt_file in txt_files:
#         # Match the corresponding FITS file using regex
#         pattern = r'image\d+'
#         match = re.search(pattern, txt_file)
        
#         if match:
#             fit_name = txt_file[:-4] + '.fit'
#             fit_file = os.path.join(fit_folder, fit_name)
#             if not os.path.exists(fit_file):
#                 print(f"Could not find the corresponding FITS file: {fit_file}")
#                 continue
#             txt_path = os.path.join(txt_folder, txt_file)
            
#             # 检查txt文件是否为空或无有效数据
#             try:
#                 # Load star parameters from txt file
#                 stars = np.loadtxt(txt_path, comments='#', skiprows=1, delimiter='\t', usecols=(0, 1, 2, 3, 4))
                
#                 # 检查是否读取到数据
#                 if stars.size == 0:
#                     print(f"Skipping {txt_file}: No star data found")
#                     continue
                    
#                 if stars.ndim == 1:  # Only one star
#                     stars = stars.reshape(1, -1)
                    
#             except (ValueError, OSError) as e:
#                 print(f"Skipping {txt_file}: Error reading file - {e}")
#                 continue
            
#             # Calculate SNR for each star
#             snr_results = []
#             for star in stars:
#                 try:
#                     snr = calculate_snr(fit_file, star)
#                     snr_results.append(snr)
#                 except Exception as e:
#                     print(f"Error calculating SNR for star in {txt_file}: {e}")
#                     continue
            
#             # 只有当有有效结果时才保存和显示统计信息
#             if len(snr_results) > 0:
#                 # Save results to a new text file
#                 results_file = os.path.join(save_txt_folder, txt_file)
#                 np.savetxt(results_file, snr_results)
#                 print(f"Processed {txt_file}: {len(snr_results)} stars")
#                 print(f"Average SNR: {np.mean(snr_results):.2f}")
#                 print(f"Max SNR: {np.max(snr_results):.2f}")
#                 print(f"Min SNR: {np.min(snr_results):.2f}")
#             else:
#                 print(f"Skipping {txt_file}: No valid SNR calculations")


# if __name__ == "__main__":
#     # Input folder paths
#     txt_folder = "/home/flh/datasets/LAMOST_new/dataset_ori/train/gt_norm"
#     fit_folder = "/home/flh/datasets/LAMOST_new/images_1k"
#     save_txt_folder = "/home/flh/datasets/LAMOST_new/dataset_ori/train/snr_txt"
#     save_txt_folder = Path(save_txt_folder)
#     save_txt_folder.mkdir(parents=True, exist_ok=True)
#     # Process SNR for all stars
#     process_all_stars(txt_folder, fit_folder, save_txt_folder)



from astropy.io import fits
import numpy as np
import os
import sep
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time


def calculate_snr(fit_file, star_params):
    """
    Calculate the Signal-to-Noise Ratio (SNR) for a single star.
    """
    # 读取 FIT 文件
    hdu_list = fits.open(fit_file)
    image_data = hdu_list[0].data
    hdu_list.close()
    
    # 使用sep函数来计算背景噪声
    image_data = image_data.astype(np.float32)
    background = sep.Background(image_data)

    # 提取星体的参数
    x, y, a, b, theta = star_params
    
    # 根据椭圆的长短轴和角度构建星体的掩码
    yy, xx = np.indices(image_data.shape)
    
    # 旋转坐标系，使椭圆轴与图像坐标对齐
    x_rot = (xx - x) * np.cos(np.radians(theta)) + (yy - y) * np.sin(np.radians(theta))
    y_rot = -(xx - x) * np.sin(np.radians(theta)) + (yy - y) * np.cos(np.radians(theta))
    
    # 计算椭圆掩码
    star_mask = ((x_rot**2) / a**2 + (y_rot**2) / b**2) <= 1
    
    # 提取与星体对应的区域
    star_region = image_data[star_mask]
    
    # 信号：恒星区域的总亮度（像素值之和）
    signal = np.sum(star_region - background.back()[star_mask])
    
    # 计算背景区域的标准差作为噪声
    noise_std = np.std(image_data[~star_mask] - background.back()[~star_mask])

    # 计算恒星区域的像素数
    N = np.sum(star_mask)
    
    # 计算 SNR
    snr = signal / (np.sqrt(N) * noise_std)
    
    return snr


def process_single_file_mp(args):
    """
    多进程版本的单文件处理函数
    """
    txt_file, txt_folder, fit_folder, save_txt_folder = args
    
    # Match the corresponding FITS file using regex
    pattern = r'image\d+'
    match = re.search(pattern, txt_file)
    
    if not match:
        return {"file": txt_file, "status": "skipped", "reason": "no image pattern found"}
    
    fit_name = txt_file[:-4] + '.fit'
    fit_file = os.path.join(fit_folder, fit_name)
    
    if not os.path.exists(fit_file):
        return {"file": txt_file, "status": "skipped", "reason": f"FITS file not found"}
    
    txt_path = os.path.join(txt_folder, txt_file)
    
    # 检查txt文件是否为空或无有效数据
    try:
        stars = np.loadtxt(txt_path, comments='#', skiprows=1, delimiter='\t', usecols=(0, 1, 2, 3, 4))
        
        if stars.size == 0:
            return {"file": txt_file, "status": "skipped", "reason": "No star data found"}
            
        if stars.ndim == 1:
            stars = stars.reshape(1, -1)
            
    except (ValueError, OSError) as e:
        return {"file": txt_file, "status": "error", "reason": f"Error reading file: {e}"}
    
    # Calculate SNR for each star
    snr_results = []
    failed_stars = 0
    
    for star in stars:
        try:
            snr = calculate_snr(fit_file, star)
            snr_results.append(snr)
        except Exception as e:
            failed_stars += 1
            continue
    
    if len(snr_results) > 0:
        # Save results to a new text file
        results_file = os.path.join(save_txt_folder, txt_file)
        np.savetxt(results_file, snr_results)
        
        return {
            "file": txt_file,
            "status": "success",
            "star_count": len(snr_results),
            "failed_stars": failed_stars,
            "avg_snr": np.mean(snr_results),
            "max_snr": np.max(snr_results),
            "min_snr": np.min(snr_results)
        }
    else:
        return {"file": txt_file, "status": "skipped", "reason": "No valid SNR calculations"}


def process_all_stars_mp(txt_folder, fit_folder, save_txt_folder, num_processes=None):
    """
    多进程版本的批处理函数
    """
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    
    if not txt_files:
        print("No txt files found in the specified folder.")
        return
    
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"Found {len(txt_files)} txt files to process.")
    print(f"Using {num_processes} processes.")
    
    # 准备参数列表
    args_list = [(txt_file, txt_folder, fit_folder, save_txt_folder) 
                 for txt_file in txt_files]
    
    start_time = time.time()
    
    # 使用多进程处理
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_file_mp, args_list)
    
    end_time = time.time()
    
    # 统计结果
    successful_files = 0
    skipped_files = 0
    error_files = 0
    total_stars = 0
    all_snr_values = []
    
    for result in results:
        if result["status"] == "success":
            successful_files += 1
            total_stars += result["star_count"]
            
            # 收集SNR值
            results_file = os.path.join(save_txt_folder, result["file"])
            try:
                snr_data = np.loadtxt(results_file)
                if snr_data.size > 0:
                    if snr_data.ndim == 0:
                        all_snr_values.append(float(snr_data))
                    else:
                        all_snr_values.extend(snr_data.flatten())
            except:
                pass
            
            print(f"✓ {result['file']}: {result['star_count']} stars, "
                  f"avg SNR: {result['avg_snr']:.2f}")
            
        elif result["status"] == "skipped":
            skipped_files += 1
            print(f"⚠ Skipped {result['file']}: {result['reason']}")
            
        elif result["status"] == "error":
            error_files += 1
            print(f"✗ Error {result['file']}: {result['reason']}")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Total files processed: {len(txt_files)}")
    print(f"Successful: {successful_files}")
    print(f"Skipped: {skipped_files}")
    print(f"Errors: {error_files}")
    print(f"Total stars processed: {total_stars}")
    
    if all_snr_values:
        print(f"\nGLOBAL SNR STATISTICS:")
        print(f"Average SNR: {np.mean(all_snr_values):.2f}")
        print(f"Median SNR: {np.median(all_snr_values):.2f}")
        print(f"Max SNR: {np.max(all_snr_values):.2f}")
        print(f"Min SNR: {np.min(all_snr_values):.2f}")
        print(f"Std SNR: {np.std(all_snr_values):.2f}")


if __name__ == "__main__":
    # Input folder paths
    txt_folder = "/home/flh/datasets/LAMOST_new/dataset_ori/train/gt_norm"
    fit_folder = "/home/flh/datasets/LAMOST_new/images_1k"
    save_txt_folder = "/home/flh/datasets/LAMOST_new/dataset_ori/train/snr_txt"
    save_txt_folder = Path(save_txt_folder)
    save_txt_folder.mkdir(parents=True, exist_ok=True)
    
    # 使用多进程处理（建议用于计算密集型任务）
    process_all_stars_mp(txt_folder, fit_folder, save_txt_folder, num_processes=cpu_count())