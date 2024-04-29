from PIL import Image
import os

# 定义原始图片路径和保存缩小后图片的路径
input_dir = '../MINIST/nature/'
output_dir = '../MINIST/nature224/'

# 定义缩小后的目标尺寸
target_size = (224, 224)

# 遍历原始文件夹中的图片
for dataset_type in ['train', 'test', 'pred']:
    input_dataset_dir = os.path.join(input_dir, dataset_type)
    output_dataset_dir = os.path.join(output_dir, dataset_type)

    # 创建保存缩小后图片的文件夹
    if not os.path.exists(output_dataset_dir):
        os.makedirs(output_dataset_dir)

    # 遍历数据集文件夹中的子文件夹
    for folder_name in os.listdir(input_dataset_dir):
        folder_path = os.path.join(input_dataset_dir, folder_name)
        if os.path.isdir(folder_path):
            output_folder_path = os.path.join(output_dataset_dir, folder_name)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            # 处理子文件夹中的图片
            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)

                # 打开图片
                with Image.open(image_path) as img:
                    # 缩放图片
                    img_resized = img.resize(target_size)

                    # 保存缩小后的图片到目标文件夹
                    output_file_path = os.path.join(output_folder_path, filename)
                    img_resized.save(output_file_path)

                print(f"Processed: {image_path}")

print("All images resized and saved successfully.")
