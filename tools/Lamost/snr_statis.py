import os
import numpy as np

def read_snr_data(folder_path):
    """读取文件夹下所有txt文件中的信噪比数据"""
    snr_values = []
    invalid_count = 0
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # 跳过空行
                            try:
                                value = float(line)
                                if value >= 0:  # 只接受非负值
                                    snr_values.append(value)
                                else:
                                    invalid_count += 1
                            except ValueError:
                                invalid_count += 1
            except Exception as e:
                print(f"警告：无法读取文件 {filename}: {str(e)}")
                continue
    
    if invalid_count > 0:
        print(f"警告：跳过 {invalid_count} 个无效数据")
    
    if not snr_values:
        raise ValueError("未找到有效数据")
    
    return np.array(snr_values)

def calculate_distribution(snr_values, bins=5):
    """计算信噪比分布"""
    # 创建区间边界
    bin_edges = np.linspace(0, 20, bins)
    bin_edges = np.append(bin_edges, np.inf)  # 添加最后一个区间 >10
    
    # 计算每个区间的计数
    counts, _ = np.histogram(snr_values, bins=bin_edges)
    
    # 创建区间标签
    labels = []
    for i in range(len(bin_edges)-1):
        if i == len(bin_edges)-2:
            labels.append(f'>{bin_edges[i]:.1f}')
        else:
            labels.append(f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}')
    
    # 计算统计量
    min_snr = np.min(snr_values)
    max_snr = np.max(snr_values)
    mean_snr = np.mean(snr_values)
    median_snr = np.median(snr_values)
    
    return dict(zip(labels, counts)), min_snr, max_snr, mean_snr, median_snr

def main():
    # 输入文件夹路径
    input_folder = '/home/flh/datasets/LAMOST_new/dataset_ori/test/snr_txt_mutil'
    
    try:
        # 读取数据
        snr_values = read_snr_data(input_folder)
        
        # 计算分布
        distribution, min_snr, max_snr, mean_snr, median_snr = calculate_distribution(snr_values)
        
        # 输出结果
        print("信噪比统计结果：")
        print(f"最小值: {min_snr:.2f}")
        print(f"最大值: {max_snr:.2f}")
        print(f"平均值: {mean_snr:.2f}")
        print(f"中位数: {median_snr:.2f}")
        print("\n信噪比分布统计：")
        for interval, count in distribution.items():
            print(f"{interval}: {count} 颗恒星")
    
    except Exception as e:
        print(f"错误：{str(e)}")

if __name__ == '__main__':
    main()