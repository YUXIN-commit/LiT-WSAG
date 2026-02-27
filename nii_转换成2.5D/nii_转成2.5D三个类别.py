import nibabel as nib
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# 定义路径变量
image_path = r"D:\workSpace\data\Task03_Liver\val_2d\image_nii"
label_path = r"D:\workSpace\data\Task03_Liver\val_2d\mask_nii"
output_image_dir = r"D:\workSpace\data\Task03_Liver\val_2.5d\images_2.5d"
output_liver_dir = r"D:\workSpace\data\Task03_Liver\val_2.5d\liver_2.5d"
output_tumor_dir = r"D:\workSpace\data\Task03_Liver\val_2.5d\tumor_2.5d"

# 定义窗宽和窗位
window_width = 1601
window_center = -300

def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed_image = np.clip(image, img_min, img_max)
    return windowed_image

def convert_3d_to_2_5d(image_path, label_path, output_image_dir, output_liver_dir, output_tumor_dir):
    # 确保输出目录存在
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_liver_dir, exist_ok=True)
    os.makedirs(output_tumor_dir, exist_ok=True)

    # 列出所有nii文件
    image_files = [f for f in os.listdir(image_path) if f.endswith('.nii.gz')]
    for image_file in tqdm(image_files, desc="Processing Images"):
        # 构建完整的文件路径
        image_filepath = os.path.join(image_path, image_file)
        
        # 解析出标签文件名
        label_filename = image_file.replace("._liver_", "")  # 根据命名规则去掉前缀"._liver_"
        label_filepath = os.path.join(label_path, label_filename)

        # 检查标签文件是否存在
        if not os.path.exists(label_filepath):
            print(f"Warning: Label file '{label_filepath}' not found for image '{image_file}'")
            continue

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

            # 应用窗宽窗位并转换成三通道图像
            windowed_slices = [window_image(slice, window_center, window_width) for slice in slices]
            windowed_image = np.stack(windowed_slices, axis=-1)
            # 检查最大值，避免除以零
            image_max = windowed_image.max()
            if image_max > 0:
                final_image = ((windowed_image - windowed_image.min()) / (image_max - windowed_image.min()) * 255).astype(np.uint8)
            else:
                final_image = windowed_image.astype(np.uint8)

            # 保存为PNG
            img = Image.fromarray(final_image)
            slice_filename = f"{os.path.splitext(image_file)[0]}_slice_{i}.png"
            img.save(os.path.join(output_image_dir, slice_filename))

            # 处理标签图像
            label_slice = label[:, :, i]

            # 生成肝脏（类别为1）的标签图像
            liver_label = np.where(label_slice == 1, 255, 0).astype(np.uint8)
            liver_img = Image.fromarray(liver_label)
            liver_img.save(os.path.join(output_liver_dir, slice_filename))

            # 生成肿瘤（类别为2）的标签图像
            tumor_label = np.where(label_slice == 2, 255, 0).astype(np.uint8)
            tumor_img = Image.fromarray(tumor_label)
            tumor_img.save(os.path.join(output_tumor_dir, slice_filename))

# 使用函数
convert_3d_to_2_5d(image_path, label_path, output_image_dir, output_liver_dir, output_tumor_dir)
