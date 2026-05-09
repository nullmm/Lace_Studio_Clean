import os
import cv2
import numpy as np
import glob
from pathlib import Path
from skimage.morphology import skeletonize

def process_and_rename(input_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 递归查找所有支持的图像格式，防止有遗漏的子文件夹
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.heic", "*.HEIC")
    raw_files = []
    for ext in extensions:
        raw_files.extend(Path(input_dir).rglob(ext))
    
    # 将路径转为字符串并排序，确保每次运行的重命名顺序一致
    raw_files = sorted([str(p) for p in raw_files])
    print(f"在 {input_dir} 中共找到 {len(raw_files)} 张原始图像，开始自动化处理...")

    success_count = 0

    for i, file_path in enumerate(raw_files):
        try:
            # 1. 生成工业级标准流水号 (例如: 0001, 0002)
            idx_str = f"{i+1:04d}"
            
            # 2. 读取图像
            # 如果是 HEIC 格式，使用专门的读取方式；否则使用 OpenCV
            if file_path.lower().endswith('.heic'):
                import pillow_heif
                from PIL import Image
                pillow_heif.register_heif_opener()
                img_pil = Image.open(file_path).convert('RGB')
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            else:
                img = cv2.imread(file_path)
            
            if img is None:
                print(f"警告：无法读取图像 {file_path}，已跳过。")
                continue
                
            # 统一缩放至 VAR 模型友好的 512x512 分辨率
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            
            # 3. 提取空间结构掩膜 (Condition)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 提取主花 (Motif) - 存入红色通道
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            main_flower = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # 提取细微连筋并骨架化 - 存入绿色通道
            mesh = cv2.subtract(binary, main_flower)
            skeleton = (skeletonize(mesh > 127) * 255).astype(np.uint8)
            
            # 合成双通道解耦掩膜
            cond = np.zeros_like(img)
            cond[:, :, 2] = main_flower  # R 通道 (OpenCV 默认 BGR，索引 2 为 R)
            cond[:, :, 1] = skeleton     # G 通道
            
            # 4. 执行重命名与保存
            rgb_name = f"lace_{idx_str}_rgb.png"
            cond_name = f"lace_{idx_str}_cond.png"
            
            cv2.imwrite(os.path.join(output_dir, rgb_name), img)
            cv2.imwrite(os.path.join(output_dir, cond_name), cond)
            
            success_count += 1
            if (i+1) % 50 == 0:
                print(f"处理进度: {i+1}/{len(raw_files)}")
                
        except Exception as e:
            print(f"处理第 {i+1} 张图像 ({file_path}) 时发生错误: {str(e)}")

    print(f"\n全部处理完毕！成功转换并重命名了 {success_count} 组训练数据。")
    print(f"数据已妥善存放在: {output_dir}")

if __name__ == "__main__":
    # 指向我们建立的工作区路径
    RAW_DIR = "./data/raw"
    PROCESSED_DIR = "./data/processed"
    
    process_and_rename(RAW_DIR, PROCESSED_DIR)