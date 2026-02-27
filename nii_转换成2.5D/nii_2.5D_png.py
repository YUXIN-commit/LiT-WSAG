import nibabel as nib
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# 路径变量
image_path = r"D:\workSpace\data\Task03_Liver\train_2d\image_nii"
label_path = r"D:\workSpace\data\Task03_Liver\train_2d\mask_nii"
output_image_dir = r"D:\workSpace\data\Task03_Liver\train_2.5d\images_2.5d"
output_label_dir = r"D:\workSpace\data\Task03_Liver\train_2.5d\masks_2.5d"
# 窗宽和窗位
window_width = 1601
window_center = -300

def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed_image = np.clip(image, img_min, img_max)
    return windowed_image

def convert_3d_to_2_5d(image_path, label_path, output_image_dir, output_label_dir):
    # 确保输出目录存在
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 列出所有nii.gz文件
    image_files = [f for f in os.listdir(image_path) if f.endswith('.nii.gz')]
    if not image_files:
        print("No .nii.gz files found in the specified image directory.")
        return
    
    for image_file in tqdm(image_files, desc="Processing Images"):
        # 构建完整的文件路径
        image_filepath = os.path.join(image_path, image_file)
        label_filepath = os.path.join(label_path, image_file.replace('_0000.nii', '.nii'))

        # 检查文件是否存在
        if not os.path.exists(image_filepath):
            print(f"Image file not found: {image_filepath}")
            continue
        if not os.path.exists(label_filepath):
            print(f"Label file not found: {label_filepath}")
            continue

        try:
            # 加载图像和标签
            image = nib.load(image_filepath).get_fdata()
            label = nib.load(label_filepath).get_fdata()
        except Exception as e:
            print(f"Error loading NIfTI files: {e}")
            continue

        # 处理每个切片
        num_slices = image.shape[2]
        for i in range(num_slices):
            # 创建2.5D图像：前一张，当前这张，后一张
            slices = []
            for j in (i-1, i, i+1):
                if 0 <= j < num_slices:
                    slices.append(image[:, :, j])
                else:
                    slices.append(image[:, :, i])

            # 应用窗宽窗位并转换成三通道图像
            windowed_slices = [window_image(slice, window_center, window_width) for slice in slices]
            windowed_image = np.stack(windowed_slices, axis=-1)
            image_max = windowed_image.max()
            if image_max > 0:
                final_image = ((windowed_image - windowed_image.min()) / (image_max - windowed_image.min()) * 255).astype(np.uint8)
            else:
                final_image = windowed_image.astype(np.uint8)

            # 保存为PNG
            try:
                img = Image.fromarray(final_image)
                lbl = Image.fromarray((label[:, :, i] * 255).astype(np.uint8)) if np.max(label[:, :, i]) > 0 else Image.fromarray(label[:, :, i].astype(np.uint8))
                slice_filename = f"{os.path.splitext(image_file)[0]}_slice_{i}.png"
                img.save(os.path.join(output_image_dir, slice_filename))
                lbl.save(os.path.join(output_label_dir, slice_filename.replace('.png', '.png')))
            except Exception as e:
                print(f"Error saving PNG files: {e}")

# 使用函数
convert_3d_to_2_5d(image_path, label_path, output_image_dir, output_label_dir)
