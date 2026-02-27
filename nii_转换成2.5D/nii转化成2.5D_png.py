import nibabel as nib
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# 定义路径变量
image_path = r"D:\workSpace\dataset\Task03_Liver\val_2d\imagesVal"
label_path = r"D:\workSpace\dataset\Task03_Liver\val_2d\labelsVal"
output_image_dir = r"D:\workSpace\dataset\Task03_Liver\train_2.5d_GT\images_2.5d_2"
output_label_dir = r"D:\workSpace\dataset\Task03_Liver\train_2.5d_GT\masks_2.5d_2"

# 定义截断范围和窗宽窗位
trunc_min = -200   # HU值截断下限
trunc_max = 250    # HU值截断上限
window_width = 150  # 肝癌分割推荐窗宽
window_center = 40  # 肝癌分割推荐窗位


def preprocess_image(image):
    """预处理图像：截断+窗宽窗位调整"""
    # 1. 截断到指定范围
    truncated = np.clip(image, trunc_min, trunc_max)
    
    # 2. 应用窗宽窗位
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed = np.clip(truncated, img_min, img_max)
    
    return windowed

def convert_3d_to_2_5d(image_path, label_path, output_image_dir, output_label_dir):
    # 确保输出目录存在
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    # 列出所有nii文件
    image_files = [f for f in os.listdir(image_path) if f.endswith('.nii.gz')]
    for image_file in tqdm(image_files, desc="Processing Images"):
        # 构建完整的文件路径
        image_filepath = os.path.join(image_path, image_file)
        label_filepath = os.path.join(label_path, image_file.replace('.nii.gz', '.nii.gz'))
        # 加载图像和标签
        image = nib.load(image_filepath).get_fdata()
        label = nib.load(label_filepath).get_fdata()
        # 处理每个切片
        num_slices = image.shape[2]
        for i in range(num_slices):
            # 创建2.5D图像：前一张，当前这张，后一张
            slices = []
            for j in (i-1, i, i+1):
                if 0 <= j < num_slices:
                    slices.append(image[:, :, j])
                else:
                    # 如果超出边界，则使用当前切片再添加一个通道
                    slices.append(image[:, :, i])
            # 预处理每个切片（截断+窗宽窗位）
            preprocessed_slices = [preprocess_image(slice) for slice in slices]
            multi_channel_image = np.stack(preprocessed_slices, axis=-1)
            
            # 归一化到0-255范围
            # 计算实际数据范围（避免使用理论范围）
            min_val = np.min(multi_channel_image)
            max_val = np.max(multi_channel_image)
            if max_val > min_val:  # 防止除零
                normalized = (multi_channel_image - min_val) / (max_val - min_val) * 255
            else:
                normalized = np.zeros_like(multi_channel_image)  # 全零图像
            final_image = normalized.astype(np.uint8)
            # 保存为PNG
            img = Image.fromarray(final_image)
            
            # 处理标签（确保二值化）
            label_slice = label[:, :, i]
            # 将非零值转为255（白色）
            lbl_data = np.where(label_slice > 0, 255, 0).astype(np.uint8)
            lbl = Image.fromarray(lbl_data)
            
            slice_filename = f"{os.path.splitext(image_file)[0]}_slice_{i}.png"
            img.save(os.path.join(output_image_dir, slice_filename))
            lbl.save(os.path.join(output_label_dir, slice_filename.replace('.png', '.png')))
# 使用函数
convert_3d_to_2_5d(image_path, label_path, output_image_dir, output_label_dir)