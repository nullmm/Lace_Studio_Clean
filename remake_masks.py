import cv2
import numpy as np
import os
import glob

# 你的数据目录
DATA_DIR = "/root/autodl-tmp/Lace_Studio_Clean/data/processed"
rgb_files = glob.glob(os.path.join(DATA_DIR, "*_rgb.png"))

print(f"🔍 找到 {len(rgb_files)} 张原图，准备重制完美掩膜...")

for rgb_path in rgb_files:
    # 1. 读取原图并转灰度
    img = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. CLAHE 局部自适应对比度增强 (让蕾丝细节更凸显)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    
    # 3. Otsu 自动阈值二值化 (区分蕾丝和纯黑背景)
    _, binary = cv2.threshold(img_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. 提取主花 (红色区域)：使用形态学“开运算”抹去纤细的网眼，只保留粗壮的花纹
    kernel_large = np.ones((5, 5), np.uint8) # 这里的 5x5 可以根据蕾丝粗细微调
    main_flower = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_large)
    
    # 5. 提取底网 (绿色区域)：完整的蕾丝减去主花，剩下的就是网眼
    net_mesh = cv2.subtract(binary, main_flower)
    
    # 6. 合成红绿条件图 (Cond)
    # 创建全黑画布 [H, W, 3]
    cond_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    # 红色通道 (B, G, R) -> 赋值给主花
    cond_img[main_flower > 0] = [0, 0, 255] 
    # 绿色通道 -> 赋值给底网
    cond_img[net_mesh > 0] = [0, 255, 0]    
    
    # 7. 覆盖保存旧的 cond 图
    cond_path = rgb_path.replace("_rgb.png", "_cond.png")
    cv2.imwrite(cond_path, cond_img)

print("✅ 所有掩膜图重制完成！现在的红绿骨架已经和原图像素级完美对齐了！")