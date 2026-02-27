import os

def remove_string_from_filenames(directory, target_string):
    # 获取指定目录中的所有文件名
    files = os.listdir(directory)

    # 遍历每个文件
    for filename in files:
        # 检查文件名中是否包含目标字符串
        if target_string in filename:
            # 新的文件名，将目标字符串移除
            new_filename = filename.replace(target_string, "")

            # 获取旧的文件路径和新的文件路径
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)

            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    # 指定要重命名文件的目录路径
    target_directory = r"D:\workSpace\data\LiverTumor_Clinical\XNYK\data_2.5d\mask_GTV"# 修改为你的文件夹路径

    # 指定要移除的字符串
    string_to_remove = ".nii"  # 修改为你想要移除的字符串

    # 调用函数移除字符串
    remove_string_from_filenames(target_directory, string_to_remove)
